"""
Benchmarks for ``ordito.boolean``: union, intersection and difference of closed meshes.

Axis: **scale**, the closed icospheres from ``sphere_small`` to ``sphere_large``. Each operand is
the sphere and a rotated, shifted copy of it, so the crossing curve runs through about the
square root of the face count's faces: the cost is the edge-face crossings' broad phase over every
edge, then the cut faces' triangulation, which is what the size sweep reads.

References
----------
**trimesh**'s ``boolean.union`` / ``intersection`` / ``difference`` with the ``manifold`` engine
is Manifold itself (``manifold3d``, TBB-parallel), the library whose Boolean ordito ports; the
row passes ``check_volume=False`` so its per-call watertightness check is not timed. Its answers
equal ordito's (``tests/test_boolean.py``, Class A).

Measured (2026-11-21, CUDA, two icospheres rotated and shifted, union): ordito 1.4x faster than
Manifold at 5 k faces and 3.8x at 328 k. The serial parts -- the inputs' coplanar groups and the
result's edge collapses, both serial in Manifold too -- run as one GPU thread, so a mesh made of
large planar faces cut along long lines (a finely subdivided box overlapping a copy in a plane)
is 5-17x slower than Manifold; the CPU device runs one thread against Manifold's pool.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from conftest import BenchCase

# The second operand: the sphere rotated about its centre and shifted by a fraction of its size.
_ROTATION_SEED = 3
_SHIFT = np.array([0.31, 0.17, 0.11])

Operation = Callable[
    [wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.vec3], wp.array[wp.int32]],
    tuple[wp.array[wp.vec3], wp.array[wp.int32]],
]

_moved_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], np.ndarray]] = {}


def _moved_vertices(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], np.ndarray]:
    """Return the moved copy's vertices on the case's device and on the host, cached."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _moved_cache:
        vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
        centre = vertices.mean(axis=0)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        rotation = tm.transformations.random_rotation_matrix(
            np.random.default_rng(_ROTATION_SEED).random(3)
        )[:3, :3]
        moved = ((vertices - centre) @ rotation.T + centre + extent * _SHIFT).astype(np.float32)
        device = bench_case.device if bench_case.kind == "ordito" else "cpu"
        _moved_cache[key] = (wp.array(moved, dtype=wp.vec3, device=device), moved)
    return _moved_cache[key]


def _run(bench_case: BenchCase, name: str, operation: Operation) -> None:
    """Time one Boolean of the sphere and its moved copy on the case's library."""
    moved_wp, moved_np = _moved_vertices(bench_case)
    if bench_case.kind == "trimesh":
        a_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        b_tm = tm.Trimesh(moved_np, bench_case.faces_np, process=False)
        operation_tm = getattr(tm.boolean, name)
        result_tm = bench_case.run(
            lambda: operation_tm([a_tm, b_tm], engine="manifold", check_volume=False)
        )
        assert len(result_tm.faces) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _vertices, result_faces = bench_case.run(lambda: operation(vertices, faces, moved_wp, faces))
    assert result_faces.size > 0


@pytest.mark.benchmark(group="boolean_union")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.benchaxis("scale")
def test_union(bench_case: BenchCase) -> None:
    """``od.boolean.union`` of a sphere and its rotated, shifted copy."""
    _run(bench_case, "union", od.boolean.union)


@pytest.mark.benchmark(group="boolean_intersection")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.benchaxis("scale")
def test_intersection(bench_case: BenchCase) -> None:
    """``od.boolean.intersection`` of a sphere and its rotated, shifted copy."""
    _run(bench_case, "intersection", od.boolean.intersection)


@pytest.mark.benchmark(group="boolean_difference")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.benchaxis("scale")
def test_difference(bench_case: BenchCase) -> None:
    """``od.boolean.difference`` of a sphere and its rotated, shifted copy."""
    _run(bench_case, "difference", od.boolean.difference)
