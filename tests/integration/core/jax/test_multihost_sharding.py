"""Multi-process regression coverage for named-sharded array construction.

This test skips in ordinary single-process CI. On a TPU VM slice, use
``checks/check_multihost_sharding.py`` to initialize JAX distributed and invoke
pytest in the same Python process on all hosts simultaneously.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.constants import SHARD_STR
from fdtdx.core.grid import UniformGrid
from fdtdx.core.jax.sharding import create_named_sharded_matrix
from fdtdx.fdtd.initialization import apply_params, capture_material_array_shardings, place_objects
from fdtdx.materials import Material
from fdtdx.objects.device.device import Device
from fdtdx.objects.static_material.static import SimulationVolume


@pytest.mark.skipif(jax.process_count() == 1, reason="requires a multi-process JAX cluster")
def test_create_named_sharded_matrix_uses_addressable_devices():
    """Construct the global array from only this process's addressable devices."""
    assert jax.process_count() > 1
    assert jax.default_backend() == "tpu"

    global_device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    assert global_device_count > local_device_count

    shape = (3, 2 * global_device_count, 8, 8)
    array = create_named_sharded_matrix(
        shape=shape,
        value=2.0,
        sharding_axis=1,
        dtype=jnp.float32,
        backend="tpu",
    )
    addressable_map = array.sharding.addressable_devices_indices_map(shape)

    assert array.shape == shape
    assert len(array.devices()) == global_device_count
    assert len(array.addressable_shards) == local_device_count
    assert len(array.addressable_shards) == len(addressable_map)

    for shard in array.addressable_shards:
        assert shard.data.shape == (3, 2, 8, 8)
        assert bool(jnp.all(shard.data == 2.0).block_until_ready())


@pytest.mark.skipif(jax.process_count() == 1, reason="requires a multi-process JAX cluster")
def test_full_domain_material_placement_preserves_global_sharding():
    """Preserve global sharding during full-domain material placement."""
    assert jax.process_count() > 1
    assert jax.default_backend() == "tpu"

    global_device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(2 * global_device_count, 8, 8),
        material=Material(permittivity=2.0),
    )
    config = SimulationConfig(
        grid=UniformGrid(spacing=1e-6),
        time=1e-15,
        backend="tpu",
        dtype=jnp.float32,
    )

    _objects, arrays, _params, _config, _info = place_objects(
        [volume],
        config,
        [],
        jax.random.PRNGKey(0),
    )
    inv_permittivities = arrays.inv_permittivities

    assert inv_permittivities.shape == (1, 2 * global_device_count, 8, 8)
    assert not inv_permittivities.is_fully_addressable
    assert len(inv_permittivities.devices()) == global_device_count
    assert isinstance(inv_permittivities.sharding, jax.sharding.NamedSharding)
    assert inv_permittivities.sharding.spec == jax.sharding.PartitionSpec(None, SHARD_STR, None, None)
    assert len(inv_permittivities.addressable_shards) == local_device_count
    for shard in inv_permittivities.addressable_shards:
        assert shard.data.shape == (1, 2, 8, 8)
        assert shard.data.nbytes == 512
        assert bool(jnp.all(shard.data == 0.5).block_until_ready())


@pytest.mark.skipif(jax.device_count() == 1, reason="requires a multi-device JAX runtime")
def test_traced_apply_params_preserves_material_sharding_and_gradient():
    """Keep the material layout through two traced Device updates and reverse mode."""
    global_device_count = jax.device_count()
    x_cells = 2 * global_device_count
    half_x_cells = global_device_count
    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(x_cells, 2, 4),
        material=Material(permittivity=2.0),
    )
    device_kwargs = {
        "partial_grid_shape": (half_x_cells, 2, 4),
        "materials": {
            "low": Material(permittivity=2.0),
            "high": Material(permittivity=4.0),
        },
        "param_transforms": [],
        "partial_voxel_grid_shape": (1, 1, 1),
    }
    left = Device(name="left", **device_kwargs)
    right = Device(name="right", **device_kwargs)
    constraints = [
        left.place_at_center(
            volume,
            axes=(0, 1, 2),
            own_positions=(-1, 0, 0),
            other_positions=(-1, 0, 0),
        ),
        right.place_at_center(
            volume,
            axes=(0, 1, 2),
            own_positions=(1, 0, 0),
            other_positions=(1, 0, 0),
        ),
    ]
    config = SimulationConfig(
        grid=UniformGrid(spacing=1e-6),
        time=1e-15,
        backend=jax.default_backend(),
        dtype=jnp.float32,
    )
    objects, arrays, base_params, _config, _info = place_objects(
        [volume, left, right],
        config,
        constraints,
        jax.random.PRNGKey(0),
    )
    shardings = capture_material_array_shardings(arrays)
    parameter_shape = (half_x_cells, 2, 4)

    def material_update(inv_permittivities, left_value, right_value):
        traced_arrays = arrays.at["inv_permittivities"].set(inv_permittivities)
        params = dict(base_params)
        params["left"] = jnp.full(parameter_shape, left_value, dtype=jnp.float32)
        params["right"] = jnp.full(parameter_shape, right_value, dtype=jnp.float32)
        updated, _objects, _application = apply_params(
            traced_arrays,
            objects,
            params,
            key=jax.random.PRNGKey(1),
            material_array_shardings=shardings,
        )
        return updated.inv_permittivities

    compiled_update = jax.jit(
        material_update,
        in_shardings=(shardings.inv_permittivities, None, None),
        out_shardings=shardings.inv_permittivities,
    )
    left_value = jnp.asarray(0.25, dtype=jnp.float32)
    right_value = jnp.asarray(0.75, dtype=jnp.float32)
    lowered = compiled_update.lower(arrays.inv_permittivities, left_value, right_value)
    stablehlo = str(lowered.compiler_ir("stablehlo")).lower()
    updated = compiled_update(arrays.inv_permittivities, left_value, right_value)

    assert updated.sharding == shardings.inv_permittivities
    assert updated.sharding.spec == jax.sharding.PartitionSpec(None, SHARD_STR, None, None)
    assert "all_gather" not in stablehlo
    assert bool(jnp.allclose(1.0 / updated[:, :half_x_cells], 2.5).block_until_ready())
    assert bool(jnp.allclose(1.0 / updated[:, half_x_cells:], 3.5).block_until_ready())

    def objective(left_value, right_value):
        inv_permittivities = compiled_update(
            arrays.inv_permittivities,
            left_value,
            right_value,
        )
        return jnp.mean(1.0 / inv_permittivities)

    value, gradient = jax.value_and_grad(objective, argnums=(0, 1))(left_value, right_value)
    assert bool(jnp.allclose(value, 3.0).block_until_ready())
    assert bool(jnp.allclose(jnp.stack(gradient), jnp.ones((2,), dtype=jnp.float32)).block_until_ready())
