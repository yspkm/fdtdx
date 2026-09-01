"""Plane source driven by externally supplied modal fields."""

from collections.abc import Callable
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx import constants
from fdtdx.core.grid import calculate_time_offset_yee
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.linalg import get_wave_vector_raw
from fdtdx.core.physics.metrics import normalize_by_poynting_flux
from fdtdx.dispersion import effective_complex_inv_permittivity, effective_inv_permittivity
from fdtdx.objects.sources.tfsf import TFSFPlaneSource, _build_dispersive_H_filter

ModeFunction = Callable[..., tuple[jax.Array, jax.Array]]


@autoinit
class CustomModePlaneSource(TFSFPlaneSource):
    """TFSF plane source using a caller-supplied complex mode profile.

    ``mode_function`` is evaluated once during :meth:`apply` with the detector
    callback arguments plus the source-plane inverse permeability::

        mode_function(
            coordinates,        # (X, Y, Z) cell-centre meshes
            frequency,          # carrier frequency in Hz
            propagation_axis,   # physical axis 0, 1, or 2
            inv_permittivity,   # carrier-frequency material slice
            inv_permeability,   # source-plane magnetic material slice
        ) -> (mode_E, mode_H)

    Both returned arrays must have shape ``(3, *source.grid_shape)``. Magnetic
    fields use FDTDX's eta0-normalized convention, exactly like
    :func:`fdtdx.core.physics.modes.compute_mode`. Complex profiles are injected
    through the existing in-phase/quadrature TFSF path.

    ``effective_index`` is explicit because it determines the component-specific
    Yee time offsets and cannot be inferred reliably from arbitrary fields.
    ``normalize=False`` preserves caller-owned amplitudes byte-for-byte; the
    default normalizes by FDTDX's integrated-flux convention.

    The callback is a setup boundary. FDTDX deliberately stops gradients through
    source setup, so design derivatives must enter through the FDTD material and
    time-stepping arrays rather than through the imported mode profile.
    """

    #: Callable returning external ``(E, eta0_H)`` fields on the placed plane.
    mode_function: ModeFunction = frozen_field()

    #: Complex effective index used for Yee time offsets and inspection.
    effective_index: complex | float = frozen_field()

    #: Renormalize the supplied fields with FDTDX's integrated-flux convention.
    normalize: bool = frozen_field(default=True)

    #: Permit overlap with a parameterized Device despite the static source-mode VJP.
    allow_device_overlap: bool = frozen_field(default=False)

    _inv_permittivity: jax.Array = private_field()
    _inv_permeability: jax.Array | float = private_field()
    _neff: jax.Array = private_field()

    def place_on_grid(
        self: Self,
        grid_slice_tuple,
        config,
        key: jax.Array,
    ) -> Self:
        """Place the source and validate its scalar effective index."""

        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        neff = np.asarray(self.effective_index)
        if neff.shape != () or not np.isfinite(neff).all() or float(np.real(neff)) <= 0.0:
            raise ValueError("effective_index must be a finite scalar with positive real part")
        return self

    def _local_edge_coordinates(self) -> tuple[jax.Array, jax.Array, jax.Array] | None:
        """Return source-local physical edge coordinates for Yee time offsets."""

        grid = self._config.resolved_grid
        if grid is None:
            return None
        local_edges = []
        for axis in range(3):
            lower, upper = self.grid_slice_tuple[axis]
            edges = grid.edges(axis)[lower : upper + 1]
            local_edges.append(edges - edges[0])
        e0, e1, e2 = local_edges
        return e0, e1, e2

    def _source_resolution(self) -> float:
        """Return a scalar spacing for the legacy uniform-grid argument."""

        if self._config.has_nonuniform_grid:
            assert self._config.resolved_grid is not None
            return self._config.resolved_grid.min_spacing
        return self._config.uniform_spacing()

    def _source_center_physical(self) -> jax.Array | None:
        """Return the source centre in its local physical coordinate frame."""

        local_edges = self._local_edge_coordinates()
        if local_edges is None:
            return None
        center = []
        for axis, edges in enumerate(local_edges):
            if axis == self.propagation_axis:
                center.append(jnp.asarray(0.0, dtype=self._config.dtype))
            else:
                center.append(0.5 * edges[-1])
        return jnp.asarray(center, dtype=self._config.dtype)

    def _plane_coordinates(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return global cell-centre meshes for the placed source plane."""

        grid = self._config.resolved_grid
        axis_centers: list[jax.Array] = []
        for axis in range(3):
            lower, upper = self.grid_slice_tuple[axis]
            if grid is not None:
                centers = jnp.asarray(grid.centers(axis))[lower:upper]
            else:
                spacing = self._config.uniform_spacing()
                centers = (jnp.arange(lower, upper) + 0.5) * spacing
            axis_centers.append(centers)
        x_coords, y_coords, z_coords = jnp.meshgrid(*axis_centers, indexing="ij")
        return x_coords, y_coords, z_coords

    def _normalization_area_weights(self) -> jax.Array | None:
        """Return physical face-area weights for a non-uniform source plane."""

        grid = self._config.resolved_grid
        if grid is None:
            return None
        return grid.face_area(axis=self.propagation_axis, slice_tuple=self.grid_slice_tuple)

    def validate_placement(self, objects) -> list[str]:
        """Reject a parameterized-device overlap unless it is explicitly acknowledged."""

        errors = list(super().validate_placement(objects))
        if not self.allow_device_overlap:
            for device in objects.devices:
                if device.check_overlap(self):
                    errors.append(
                        f"Custom mode source '{self.name}' overlaps Device '{device.name}'. "
                        "The imported mode is a static setup boundary, so its material dependence "
                        "is absent from the FDTD VJP. Move the source outside the Device or set "
                        "allow_device_overlap=True to acknowledge that limitation."
                    )
        return errors

    def _validate_field(self, field: jax.Array, *, label: str) -> None:
        """Reject shape or precision changes at the custom-mode boundary."""

        expected_shape = (3, *self.grid_shape)
        if field.shape != expected_shape:
            raise ValueError(f"custom mode {label} must have shape {expected_shape}, got {field.shape}")
        expected_real_dtype = jnp.dtype(self._config.dtype)
        if field.dtype not in (
            expected_real_dtype,
            jnp.dtype(jnp.complex64 if expected_real_dtype == jnp.float32 else jnp.complex128),
        ):
            raise ValueError(
                f"custom mode {label} dtype {field.dtype} is incompatible with simulation dtype {expected_real_dtype}"
            )

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
        electric_conductivity: jax.Array | None = None,
        dispersive_c4: jax.Array | None = None,
    ) -> Self:
        """Evaluate and bind the external mode to the existing TFSF injector."""

        del key
        if (
            self.azimuth_angle != 0
            or self.elevation_angle != 0
            or self.max_angle_random_offset != 0
            or self.max_vertical_offset != 0
            or self.max_horizontal_offset != 0
        ):
            raise NotImplementedError("custom mode fields already encode phase and cannot be tilted or randomized")

        inv_eps_inf_slice = inv_permittivities[:, *self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_permeability_slice: jax.Array | float = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        c1_slice = c2_slice = c3_slice = c4_slice = None
        inv_permittivity_slice = inv_eps_inf_slice
        if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
            c1_slice = dispersive_c1[:, :, *self.grid_slice]
            c2_slice = dispersive_c2[:, :, *self.grid_slice]
            c3_slice = dispersive_c3[:, :, *self.grid_slice]
            c4_slice = None if dispersive_c4 is None else dispersive_c4[:, :, *self.grid_slice]
            inv_permittivity_slice = effective_inv_permittivity(
                inv_eps=inv_eps_inf_slice,
                c1=c1_slice,
                c2=c2_slice,
                c3=c3_slice,
                omega=2.0 * np.pi * self.wave_character.get_frequency(),
                dt=self._config.time_step_duration,
                c4=c4_slice,
            )

        sigma_slice = None if electric_conductivity is None else electric_conductivity[:, *self.grid_slice]
        mode_inv_permittivity = inv_eps_inf_slice
        if sigma_slice is not None or c1_slice is not None:
            mode_inv_permittivity = effective_complex_inv_permittivity(
                inv_eps=inv_eps_inf_slice,
                omega=2.0 * np.pi * self.wave_character.get_frequency(),
                dt=self._config.time_step_duration,
                c1=c1_slice,
                c2=c2_slice,
                c3=c3_slice,
                c4=c4_slice,
                electric_conductivity=sigma_slice,
                conductivity_spacing=(
                    None
                    if sigma_slice is None
                    else constants.c * self._config.time_step_duration / self._config.courant_number
                ),
            )

        mode_E, mode_H = self.mode_function(
            coordinates=self._plane_coordinates(),
            frequency=self.wave_character.get_frequency(),
            propagation_axis=self.propagation_axis,
            inv_permittivity=mode_inv_permittivity,
            inv_permeability=inv_permeability_slice,
        )
        mode_E = jnp.asarray(mode_E)
        mode_H = jnp.asarray(mode_H)
        self._validate_field(mode_E, label="electric field")
        self._validate_field(mode_H, label="magnetic field")
        if mode_E.dtype != mode_H.dtype:
            raise ValueError("custom mode electric and magnetic fields must have the same dtype")
        if self.normalize:
            mode_E, mode_H = normalize_by_poynting_flux(
                mode_E,
                mode_H,
                axis=self.propagation_axis,
                area_weights=self._normalization_area_weights(),
            )

        neff_dtype = jnp.complex128 if self._config.dtype == jnp.float64 else jnp.complex64
        neff = jnp.asarray(self.effective_index, dtype=neff_dtype)
        self = self.aset("_E", mode_E, create_new_ok=True)
        self = self.aset("_H", mode_H, create_new_ok=True)
        self = self.aset("_neff", neff, create_new_ok=True)
        self = self.aset("_inv_permittivity", inv_permittivity_slice, create_new_ok=True)
        self = self.aset("_inv_permeability", inv_permeability_slice, create_new_ok=True)

        center = jnp.asarray(
            [round(self.grid_shape[self.horizontal_axis]), round(self.grid_shape[self.vertical_axis])],
            dtype=jnp.int32,
        )
        raw_wave_vector = get_wave_vector_raw(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            dtype=self._config.dtype,
        )
        time_offset_E, time_offset_H = calculate_time_offset_yee(
            center=center,
            wave_vector=raw_wave_vector,
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=jnp.ones_like(inv_permeability_slice),
            resolution=self._source_resolution(),
            time_step_duration=self._config.time_step_duration,
            effective_index=jnp.real(neff),
            coordinate_edges=self._local_edge_coordinates(),
            center_physical=self._source_center_physical(),
        )
        self = self.aset("_time_offset_E", time_offset_E, create_new_ok=True)
        self = self.aset("_time_offset_H", time_offset_H, create_new_ok=True)

        if c1_slice is not None and c2_slice is not None and c3_slice is not None:
            filtered = _build_dispersive_H_filter(
                temporal_profile=self.temporal_profile,
                wave_character=self.wave_character,
                dt=self._config.time_step_duration,
                num_time_steps=self._config.time_steps_total,
                c1_slice=c1_slice,
                c2_slice=c2_slice,
                c3_slice=c3_slice,
                inv_eps_inf_slice=inv_eps_inf_slice,
                dtype=self._config.dtype,
                c4_slice=c4_slice,
            )
            self = self.aset("_temporal_H_filter", filtered, create_new_ok=True)
        else:
            self = self.aset("_temporal_H_filter", None, create_new_ok=True)
        return self


__all__ = ["CustomModePlaneSource", "ModeFunction"]
