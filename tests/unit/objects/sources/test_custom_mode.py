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


def _config(dtype=jnp.float32) -> SimulationConfig:
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


def _fields(dtype=jnp.complex64) -> tuple[jax.Array, jax.Array]:
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
    allow_profile_updates: bool = False,
) -> CustomModePlaneSource:
    return CustomModePlaneSource(
        name="external-mode",
        partial_grid_shape=(2, 2, 1),
        wave_character=WaveCharacter(wavelength=1.55e-6),
        direction="+",
        mode_function=mode_function,
        effective_index=effective_index,
        normalize=normalize,
        allow_profile_updates=allow_profile_updates,
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
    inv_permittivity = jnp.ones((3, 2, 2, 1), dtype=jnp.float32)
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
    inv_permittivity = jnp.ones((3, 2, 2, 1), dtype=jnp.float32)
    source = source.apply(jax.random.PRNGKey(5), inv_permittivity, 1.0)

    zeros = jnp.zeros((3, 2, 2, 1), dtype=jnp.float32)
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
    electric = jnp.zeros(shape, dtype=jnp.complex64).at[0].set(2.0)
    magnetic = jnp.zeros(shape, dtype=jnp.complex64).at[1].set(0.5)
    source = _place(_source(lambda **_: (electric, magnetic), normalize=True), config)
    source = source.apply(
        jax.random.PRNGKey(6),
        jnp.ones((3, 2, 2, 1), dtype=jnp.float32),
        1.0,
    )

    weights = config.resolved_grid.face_area(axis=2, slice_tuple=source.grid_slice_tuple)
    power = compute_integrated_power(source._E, source._H, axis=2, area_weights=weights)
    assert power == pytest.approx(1.0)


def test_custom_mode_source_stops_callback_field_gradients() -> None:
    config = _config()
    electric, magnetic = _fields()

    def callback(*, inv_permittivity, **_):
        gain = jnp.mean(inv_permittivity)
        return electric * gain, magnetic * gain

    source = _place(_source(callback), config)

    def objective(scale):
        inverse_permittivity = jnp.ones((3, 2, 2, 1), dtype=jnp.float32) * scale
        applied = source.apply(jax.random.PRNGKey(61), inverse_permittivity, 1.0)
        return jnp.real(jnp.sum(applied._E) + jnp.sum(applied._H))

    assert jax.grad(objective)(jnp.asarray(0.75, dtype=jnp.float32)) == pytest.approx(0.0)


def test_custom_mode_source_explicit_profile_update_preserves_jax_gradient() -> None:
    config = _config()
    electric, magnetic = _fields()
    source = _place(
        _source(
            lambda **_: (electric, magnetic),
            allow_profile_updates=True,
        ),
        config,
    )
    source = source.apply(
        jax.random.PRNGKey(62),
        jnp.ones((3, 2, 2, 1), dtype=jnp.float32),
        1.0,
    )

    @jax.jit
    def objective(scale):
        updated = source.with_mode_profile(
            mode_E=electric * (1.0 + scale),
            mode_H=magnetic * (1.0 - 0.25 * scale),
            effective_index=jnp.asarray(2.4 + 0.1 * scale, dtype=jnp.complex64),
        )
        field_value = jnp.real(jnp.sum(updated._E) + 0.5 * jnp.sum(updated._H))
        time_offset_value = 1e15 * jnp.sum(updated._time_offset_E + updated._time_offset_H)
        return field_value + time_offset_value

    value = jnp.asarray(0.2, dtype=jnp.float32)
    gradient = jax.grad(objective)(value)
    step = jnp.asarray(1e-3, dtype=jnp.float32)
    finite_difference = (objective(value + step) - objective(value - step)) / (2.0 * step)

    assert bool(jnp.isfinite(gradient))
    assert float(jnp.abs(gradient)) > 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-3, atol=2e-4)


def test_custom_mode_source_profile_gradient_reaches_tfsf_injection() -> None:
    config = _config()
    electric, magnetic = _fields()
    source = _place(
        _source(
            lambda **_: (electric, magnetic),
            allow_profile_updates=True,
        ),
        config,
    )
    inverse_permittivity = jnp.ones((3, 2, 2, 1), dtype=jnp.float32)
    source = source.apply(jax.random.PRNGKey(65), inverse_permittivity, 1.0)
    zeros = jnp.zeros((3, 2, 2, 1), dtype=jnp.float32)

    def objective(scale):
        updated = source.with_mode_profile(
            mode_E=electric * (1.0 + 0.2 * scale),
            mode_H=magnetic * (1.0 - 0.1 * scale),
            effective_index=jnp.asarray(2.4 + 0.05 * scale, dtype=jnp.complex64),
        )
        injected_e = updated.update_E(
            zeros,
            inverse_permittivity,
            1.0,
            jnp.asarray(2),
            False,
        )
        injected_h = updated.update_H(
            zeros,
            inverse_permittivity,
            1.0,
            jnp.asarray(2),
            False,
        )
        return jnp.sum(injected_e**2) + jnp.sum(injected_h**2)

    value = jnp.asarray(0.1, dtype=jnp.float32)
    gradient = jax.grad(objective)(value)
    step = jnp.asarray(1e-3, dtype=jnp.float32)
    finite_difference = (objective(value + step) - objective(value - step)) / (2.0 * step)

    assert bool(jnp.isfinite(gradient))
    assert float(jnp.abs(gradient)) > 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-3, atol=2e-4)


def test_dynamic_profile_does_not_add_material_to_mode_derivative() -> None:
    config = _config()
    electric, magnetic = _fields()
    source = _place(
        _source(
            lambda **_: (electric, magnetic),
            allow_profile_updates=True,
        ),
        config,
    )
    baseline_inverse_permittivity = jnp.ones((3, 2, 2, 1), dtype=jnp.float32)
    source = source.apply(
        jax.random.PRNGKey(66),
        baseline_inverse_permittivity,
        1.0,
    )
    source = source.with_mode_profile(
        mode_E=electric,
        mode_H=magnetic,
        effective_index=jnp.asarray(2.4 + 0.0j, dtype=jnp.complex64),
    )
    zeros = jnp.zeros((3, 2, 2, 1), dtype=jnp.float32)

    def source_injection(scale):
        inverse_permittivity = baseline_inverse_permittivity * scale
        injected_e = source.update_E(
            zeros,
            inverse_permittivity,
            1.0,
            jnp.asarray(2),
            False,
        )
        injected_h = source.update_H(
            zeros,
            inverse_permittivity,
            scale,
            jnp.asarray(2),
            False,
        )
        return jnp.sum(injected_e**2) + jnp.sum(injected_h**2)

    assert jax.grad(source_injection)(jnp.asarray(0.75, dtype=jnp.float32)) == pytest.approx(0.0)


def test_custom_mode_source_rejects_profile_update_without_explicit_opt_in() -> None:
    config = _config()
    electric, magnetic = _fields()
    source = _place(_source(lambda **_: (electric, magnetic)), config)
    source = source.apply(
        jax.random.PRNGKey(63),
        jnp.ones((3, 2, 2, 1), dtype=jnp.float32),
        1.0,
    )

    with pytest.raises(ValueError, match="not configured for profile updates"):
        source.with_mode_profile(
            mode_E=electric,
            mode_H=magnetic,
            effective_index=jnp.asarray(2.4 + 0.0j, dtype=jnp.complex64),
        )


def test_custom_mode_source_profile_update_fails_closed_for_invalid_values() -> None:
    config = _config()
    electric, magnetic = _fields()
    source = _place(
        _source(
            lambda **_: (electric, magnetic),
            allow_profile_updates=True,
        ),
        config,
    )
    source = source.apply(
        jax.random.PRNGKey(64),
        jnp.ones((3, 2, 2, 1), dtype=jnp.float32),
        1.0,
    )
    updated = source.with_mode_profile(
        mode_E=electric,
        mode_H=magnetic,
        effective_index=jnp.asarray(-1.0 + 0.0j, dtype=jnp.complex64),
    )

    assert bool(jnp.isnan(updated._E).all())
    assert bool(jnp.isnan(updated._H).all())
    assert bool(jnp.isnan(updated._neff))


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
                jnp.ones((3, 2, 1, 1), dtype=jnp.complex64),
                jnp.ones((3, 2, 2, 1), dtype=jnp.complex64),
            ),
            "electric field must have shape",
        ),
        (
            lambda: (
                jnp.ones((3, 2, 2, 1), dtype=jnp.complex64),
                jnp.ones((3, 2, 2, 1), dtype=jnp.int32),
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
            jnp.ones((3, 2, 2, 1), dtype=jnp.float32),
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
            jnp.ones((3, 2, 2, 1), dtype=jnp.float32),
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
