"""Tests for externally supplied mode-plane sources."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid
from fdtdx.core.physics.metrics import compute_integrated_power
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.custom_mode import CustomModePlaneSource

pytestmark = pytest.mark.unit


def _config(dtype=jnp.float64) -> SimulationConfig:
    return SimulationConfig(
        time=20e-15,
        grid=RectilinearGrid(
            x_edges=jnp.asarray((0.0, 100e-9, 300e-9), dtype=dtype),
            y_edges=jnp.asarray((0.0, 120e-9, 350e-9), dtype=dtype),
            z_edges=jnp.asarray((0.0, 40e-9), dtype=dtype),
        ),
        backend="cpu",
        dtype=dtype,
    )


def _fields(dtype=jnp.complex128) -> tuple[jax.Array, jax.Array]:
    shape = (3, 2, 2, 1)
    electric = jnp.zeros(shape, dtype=dtype)
    magnetic = jnp.zeros(shape, dtype=dtype)
    electric = electric.at[0].set(jnp.asarray((1.0 + 0.5j), dtype=dtype))
    magnetic = magnetic.at[1].set(jnp.asarray((0.25 - 0.75j), dtype=dtype))
    return electric, magnetic


def _source(
    mode_function: Callable[..., tuple[jax.Array, jax.Array]],
    *,
    effective_index: complex = 2.4 + 0.0j,
    normalize: bool = False,
) -> CustomModePlaneSource:
    return CustomModePlaneSource(
        name="external-mode",
        partial_grid_shape=(2, 2, 1),
        wave_character=WaveCharacter(wavelength=1.55e-6),
        direction="+",
        mode_function=mode_function,
        effective_index=effective_index,
        normalize=normalize,
    )


def _place(source: CustomModePlaneSource, config: SimulationConfig) -> CustomModePlaneSource:
    return source.place_on_grid(
        grid_slice_tuple=((0, 2), (0, 2), (0, 1)),
        config=config,
        key=jax.random.PRNGKey(3),
    )


def test_custom_mode_source_preserves_complex_fields_and_callback_inputs() -> None:
    config = _config()
    expected_e, expected_h = _fields()
    observed: dict[str, object] = {}

    def callback(**kwargs):
        observed.update(kwargs)
        return expected_e, expected_h

    source = _place(_source(callback), config)
    inv_permittivity = jnp.ones((3, 2, 2, 1), dtype=jnp.float64)
    applied = source.apply(jax.random.PRNGKey(4), inv_permittivity, 1.0)

    np.testing.assert_array_equal(applied._E, expected_e)
    np.testing.assert_array_equal(applied._H, expected_h)
    assert applied._neff == pytest.approx(2.4 + 0.0j)
    assert applied._time_offset_E.shape == expected_e.shape
    assert applied._time_offset_H.shape == expected_h.shape
    assert observed["frequency"] == pytest.approx(source.wave_character.get_frequency())
    assert observed["propagation_axis"] == 2
    np.testing.assert_array_equal(observed["inv_permittivity"], inv_permittivity)
    assert observed["inv_permeability"] == 1.0
    x, y, z = observed["coordinates"]
    np.testing.assert_allclose(x[:, 0, 0], (50e-9, 200e-9))
    np.testing.assert_allclose(y[0, :, 0], (60e-9, 235e-9))
    np.testing.assert_allclose(z[0, 0, :], (20e-9,))


def test_custom_mode_source_uses_existing_complex_tfsf_updates() -> None:
    config = _config()
    electric, magnetic = _fields()
    source = _place(_source(lambda **_: (electric, magnetic)), config)
    inv_permittivity = jnp.ones((3, 2, 2, 1), dtype=jnp.float64)
    source = source.apply(jax.random.PRNGKey(5), inv_permittivity, 1.0)

    zeros = jnp.zeros((3, 2, 2, 1), dtype=jnp.float64)
    updated_e = source.update_E(zeros, inv_permittivity, 1.0, jnp.asarray(2), False)
    updated_h = source.update_H(zeros, inv_permittivity, 1.0, jnp.asarray(2), False)
    inverse_e = source.update_E(zeros, inv_permittivity, 1.0, jnp.asarray(2), True)
    inverse_h = source.update_H(zeros, inv_permittivity, 1.0, jnp.asarray(2), True)

    assert float(jnp.linalg.norm(updated_e)) > 0.0
    assert float(jnp.linalg.norm(updated_h)) > 0.0
    np.testing.assert_allclose(inverse_e, -updated_e)
    np.testing.assert_allclose(inverse_h, -updated_h)


def test_custom_mode_source_can_apply_explicit_fdtdx_normalization() -> None:
    config = _config()
    shape = (3, 2, 2, 1)
    electric = jnp.zeros(shape, dtype=jnp.complex128).at[0].set(2.0)
    magnetic = jnp.zeros(shape, dtype=jnp.complex128).at[1].set(0.5)
    source = _place(_source(lambda **_: (electric, magnetic), normalize=True), config)
    source = source.apply(
        jax.random.PRNGKey(6),
        jnp.ones((3, 2, 2, 1), dtype=jnp.float64),
        1.0,
    )

    weights = config.resolved_grid.face_area(axis=2, slice_tuple=source.grid_slice_tuple)
    power = compute_integrated_power(source._E, source._H, axis=2, area_weights=weights)
    assert power == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("effective_index", "message"),
    [
        (0.0 + 0.0j, "positive real part"),
        (complex(np.nan, 0.0), "finite scalar"),
        (np.asarray((2.0,)), "finite scalar"),
    ],
)
def test_custom_mode_source_rejects_invalid_effective_index(effective_index, message) -> None:
    electric, magnetic = _fields()
    with pytest.raises(ValueError, match=message):
        _place(
            _source(
                lambda **_: (electric, magnetic),
                effective_index=effective_index,
            ),
            _config(),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: (
                jnp.ones((3, 2, 1, 1), dtype=jnp.complex128),
                jnp.ones((3, 2, 2, 1), dtype=jnp.complex128),
            ),
            "electric field must have shape",
        ),
        (
            lambda: (
                jnp.ones((3, 2, 2, 1), dtype=jnp.complex128),
                jnp.ones((3, 2, 2, 1), dtype=jnp.complex64),
            ),
            "magnetic field dtype",
        ),
    ],
)
def test_custom_mode_source_rejects_shape_and_precision_changes(factory, message) -> None:
    config = _config()
    source = _place(_source(lambda **_: factory()), config)
    with pytest.raises(ValueError, match=message):
        source.apply(
            jax.random.PRNGKey(7),
            jnp.ones((3, 2, 2, 1), dtype=jnp.float64),
            1.0,
        )


def test_custom_mode_source_rejects_added_tilt_or_randomization() -> None:
    config = _config()
    electric, magnetic = _fields()
    source = _source(lambda **_: (electric, magnetic)).aset("azimuth_angle", 1.0)
    source = _place(source, config)
    with pytest.raises(NotImplementedError, match="cannot be tilted or randomized"):
        source.apply(
            jax.random.PRNGKey(8),
            jnp.ones((3, 2, 2, 1), dtype=jnp.float64),
            1.0,
        )


def test_custom_mode_source_requires_explicit_device_overlap_acknowledgement() -> None:
    electric, magnetic = _fields()
    source = _place(_source(lambda **_: (electric, magnetic)), _config())
    device = SimpleNamespace(name="design", check_overlap=lambda _: True)

    errors = source.validate_placement(SimpleNamespace(devices=[device]))
    assert len(errors) == 1
    assert "absent from the FDTD VJP" in errors[0]

    acknowledged = source.aset("allow_device_overlap", True)
    assert acknowledged.validate_placement(SimpleNamespace(devices=[device])) == []
