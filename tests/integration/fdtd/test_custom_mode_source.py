"""Integration test for the public custom-mode source path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

pytestmark = pytest.mark.integration


def test_custom_mode_source_drives_fdtd_without_an_eigensolver() -> None:
    """A supplied complex profile survives placement and drives Maxwell updates."""

    shape = (4, 4, 12)
    spacing = 125e-9
    relative_permittivity = 2.085136
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=shape,
        material=fdtdx.Material(permittivity=relative_permittivity),
    )
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=1,
            override_types={face: "periodic" for face in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")},
        ),
        volume,
    )
    callback_calls = 0

    def mode_function(**kwargs):
        nonlocal callback_calls
        callback_calls += 1
        source_shape = kwargs["coordinates"][0].shape
        electric = jnp.zeros((3, *source_shape), dtype=jnp.complex128)
        magnetic = jnp.zeros((3, *source_shape), dtype=jnp.complex128)
        electric = electric.at[0].set(1.0 + 0.25j)
        magnetic = magnetic.at[1].set(np.sqrt(relative_permittivity) * (1.0 + 0.25j))
        return electric, magnetic

    source = fdtdx.CustomModePlaneSource(
        name="external-mode",
        partial_grid_shape=(4, 4, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
        direction="+",
        mode_function=mode_function,
        effective_index=np.sqrt(relative_permittivity),
        normalize=False,
    )
    constraints = [
        *boundary_constraints,
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(2,)),
    ]
    config = fdtdx.SimulationConfig(
        time=15e-15,
        grid=fdtdx.UniformGrid(spacing=spacing),
        backend="cpu",
        dtype=jnp.float64,
    )
    key = jax.random.PRNGKey(91)
    objects, arrays, parameters, config, _info = fdtdx.place_objects(
        [volume, *boundaries.values(), source],
        config,
        constraints,
        key=key,
    )
    arrays, objects, _apply_info = fdtdx.apply_params(
        arrays=arrays,
        objects=objects,
        params=parameters,
        key=key,
    )
    _step, final_arrays = fdtdx.run_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
        show_progress=False,
    )

    assert callback_calls == 1
    assert float(jnp.linalg.norm(final_arrays.fields.E)) > 0.0
    assert float(jnp.linalg.norm(final_arrays.fields.H)) > 0.0
    assert bool(jnp.all(jnp.isfinite(final_arrays.fields.E)))
    assert bool(jnp.all(jnp.isfinite(final_arrays.fields.H)))
