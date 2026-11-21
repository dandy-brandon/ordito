"""
Benchmarks for ``ordito.boolean``: union, intersection and difference of closed meshes.

Axis: **scale**, the closed icospheres from ``sphere_small`` to ``sphere_large``. Each operand is
the sphere and a rotated, shifted copy of it, so the crossing curve runs through about the
square root of the face count's faces: the cost is the edge-face crossings' broad phase over every
edge, then the cut faces' triangulation, which is what the size sweep reads.

References
----------
**manifold3d** is Manifold itself, TBB-parallel, timed on its own ``Manifold`` objects (built
outside the timed call); its CSG is lazy, so each row reads the vertex count to force the result.
Its answers equal ordito's array for array (``tests/test_boolean.py``, Class A).

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

import manifold3d as m3
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from conftest import BenchCase, BenchLibrary

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


def _manifolds_m3(bench_case: BenchCase) -> tuple[m3.Manifold, m3.Manifold]:
    """Return the sphere and its moved copy as Manifolds."""
    _, moved_np = _moved_vertices(bench_case)
    faces_np = np.asarray(bench_case.faces_np, dtype=np.uint32)
    sphere_np = np.asarray(bench_case.vertices_np, dtype=np.float32)
    a_m3 = m3.Manifold(m3.Mesh(vert_properties=sphere_np, tri_verts=faces_np))
    b_m3 = m3.Manifold(m3.Mesh(vert_properties=moved_np, tri_verts=faces_np))
    return a_m3, b_m3


def _forced(result_m3: m3.Manifold) -> m3.Manifold:
    """Evaluate Manifold's lazy CSG result."""
    assert result_m3.num_vert() >= 0
    return result_m3


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
    if bench_case.kind == "manifold3d":
        a_m3, b_m3 = _manifolds_m3(bench_case)
        operations_m3 = {
            "union": lambda: a_m3 + b_m3,
            "intersection": lambda: a_m3 ^ b_m3,
            "difference": lambda: a_m3 - b_m3,
        }
        result_m3 = bench_case.run(lambda: _forced(operations_m3[name]()))
        assert result_m3.num_tri() > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _vertices, result_faces = bench_case.run(lambda: operation(vertices, faces, moved_wp, faces))
    assert result_faces.size > 0


@pytest.mark.benchmark(group="boolean_union")
@pytest.mark.benchlibs("ordito", "trimesh", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_union(bench_case: BenchCase) -> None:
    """``od.boolean.union`` of a sphere and its rotated, shifted copy."""
    _run(bench_case, "union", od.boolean.union)


@pytest.mark.benchmark(group="boolean_intersection")
@pytest.mark.benchlibs("ordito", "trimesh", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_intersection(bench_case: BenchCase) -> None:
    """``od.boolean.intersection`` of a sphere and its rotated, shifted copy."""
    _run(bench_case, "intersection", od.boolean.intersection)


@pytest.mark.benchmark(group="boolean_difference")
@pytest.mark.benchlibs("ordito", "trimesh", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_difference(bench_case: BenchCase) -> None:
    """``od.boolean.difference`` of a sphere and its rotated, shifted copy."""
    _run(bench_case, "difference", od.boolean.difference)


@pytest.mark.benchmark(group="split_by_mesh")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_split_by_mesh(bench_case: BenchCase) -> None:
    """``od.boolean.split_by_mesh``: both parts from one pass over the crossings."""
    if bench_case.kind == "manifold3d":
        a_m3, b_m3 = _manifolds_m3(bench_case)

        def split_m3() -> tuple[m3.Manifold, m3.Manifold]:
            inside_m3, outside_m3 = a_m3.split(b_m3)
            return _forced(inside_m3), _forced(outside_m3)

        inside_m3, _outside_m3 = bench_case.run(split_m3)
        assert inside_m3.num_tri() > 0
        return
    moved_wp, _ = _moved_vertices(bench_case)
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    (_, inside_faces), _outside = bench_case.run(
        lambda: od.boolean.split_by_mesh(vertices, faces, moved_wp, faces)
    )
    assert inside_faces.size > 0


# An oblique plane through the sphere's middle.
_PLANE_NORMAL = (0.3, -0.5, 0.8)
_PLANE_OFFSET = 0.1


def _plane_offset(bench_case: BenchCase) -> float:
    """Return the plane's offset, through the mesh's centre."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
    normal = np.asarray(_PLANE_NORMAL) / np.linalg.norm(_PLANE_NORMAL)
    return float(vertices.mean(axis=0) @ normal) + _PLANE_OFFSET


@pytest.mark.benchmark(group="split_by_plane")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_split_by_plane(bench_case: BenchCase) -> None:
    """``od.boolean.split_by_plane`` by an oblique plane through the middle."""
    offset = _plane_offset(bench_case)
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)

        def split_m3() -> tuple[m3.Manifold, m3.Manifold]:
            above_m3, below_m3 = a_m3.split_by_plane(_PLANE_NORMAL, offset)
            return _forced(above_m3), _forced(below_m3)

        above_m3, _below_m3 = bench_case.run(split_m3)
        assert above_m3.num_tri() > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    (_, above_faces), _below = bench_case.run(
        lambda: od.boolean.split_by_plane(vertices, faces, _PLANE_NORMAL, offset)
    )
    assert above_faces.size > 0


@pytest.mark.benchmark(group="trim_by_plane")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_trim_by_plane(bench_case: BenchCase) -> None:
    """``od.boolean.trim_by_plane`` by an oblique plane through the middle."""
    offset = _plane_offset(bench_case)
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        kept_m3 = bench_case.run(lambda: _forced(a_m3.trim_by_plane(_PLANE_NORMAL, offset)))
        assert kept_m3.num_tri() > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _, kept_faces = bench_case.run(
        lambda: od.boolean.trim_by_plane(vertices, faces, _PLANE_NORMAL, offset)
    )
    assert kept_faces.size > 0


# Batches: the sphere and copies turned and moved around it, overlapping in a ring.
_BATCH_COPIES = 4


def _batch_inputs(bench_case: BenchCase) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the batch's meshes as ``float32`` NumPy pairs."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
    faces = np.asarray(bench_case.faces_np)
    centre = vertices.mean(axis=0)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    rng = np.random.default_rng(_ROTATION_SEED)
    meshes = [(vertices.astype(np.float32), faces)]
    for i in range(1, _BATCH_COPIES):
        rotation = tm.transformations.random_rotation_matrix(rng.random(3))[:3, :3]
        angle = 2.0 * np.pi * i / _BATCH_COPIES
        shift = 0.3 * extent * np.array([np.cos(angle), np.sin(angle), 0.1 * i])
        moved = (vertices - centre) @ rotation.T + centre + shift
        meshes.append((moved.astype(np.float32), faces))
    return meshes


def _run_batch(
    bench_case: BenchCase,
    function: Callable[[list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]], object],
    op_m3: m3.OpType,
) -> None:
    """Time one batch Boolean of the sphere and its copies."""
    meshes = _batch_inputs(bench_case)
    if bench_case.kind == "manifold3d":
        manifolds_m3 = [
            m3.Manifold(m3.Mesh(vert_properties=v, tri_verts=f.astype(np.uint32)))
            for v, f in meshes
        ]
        result_m3 = bench_case.run(lambda: _forced(m3.Manifold.batch_boolean(manifolds_m3, op_m3)))
        assert result_m3.num_tri() > 0
        return
    device = bench_case.device
    meshes_wp = [
        (
            wp.array(v, dtype=wp.vec3, device=device),
            wp.array(f.reshape(-1).astype(np.int32), dtype=wp.int32, device=device),
        )
        for v, f in meshes
    ]
    bench_case.run(lambda: function(meshes_wp))


@pytest.mark.benchmark(group="union_all")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_union_all(bench_case: BenchCase) -> None:
    """``od.boolean.union_all`` of the sphere and three copies around it."""
    _run_batch(bench_case, od.boolean.union_all, m3.OpType.Add)


@pytest.mark.benchmark(group="intersection_all")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_intersection_all(bench_case: BenchCase) -> None:
    """``od.boolean.intersection_all`` of the sphere and three copies around it."""
    _run_batch(bench_case, od.boolean.intersection_all, m3.OpType.Intersect)


@pytest.mark.benchmark(group="difference_all")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_difference_all(bench_case: BenchCase) -> None:
    """``od.boolean.difference_all``: the sphere less three copies around it."""
    _run_batch(bench_case, od.boolean.difference_all, m3.OpType.Subtract)


# Segments across the sphere's box, random, half of them crossing.
_N_SEGMENTS = 4096


def _segments(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """Return random segments across the mesh's box as ``float32`` ends."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    rng = np.random.default_rng(_ROTATION_SEED)
    origins = rng.uniform(lower, upper, (_N_SEGMENTS, 3)).astype(np.float32)
    endpoints = rng.uniform(lower, upper, (_N_SEGMENTS, 3)).astype(np.float32)
    return origins, endpoints


@pytest.mark.benchmark(group="segment_hits")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_segment_hits(bench_case: BenchCase) -> None:
    """
    ``od.boolean.segment_hits`` of 4 096 random segments: one launch walks them all.

    The manifold3d row casts them one ``ray_cast`` at a time, as its API takes them.
    """
    origins, endpoints = _segments(bench_case)
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        pairs = [
            (tuple(o.tolist()), tuple(e.tolist())) for o, e in zip(origins, endpoints, strict=True)
        ]
        hits_m3 = bench_case.run(lambda: [a_m3.ray_cast(o, e) for o, e in pairs])
        assert len(hits_m3) == _N_SEGMENTS
        return
    device = bench_case.device
    origins_wp = wp.array(origins, dtype=wp.vec3, device=device)
    endpoints_wp = wp.array(endpoints, dtype=wp.vec3, device=device)
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    *_, offsets = bench_case.run(
        lambda: od.boolean.segment_hits(vertices, faces, origins_wp, endpoints_wp)
    )
    assert offsets.size == _N_SEGMENTS + 1


@pytest.mark.benchmark(group="min_gap")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_min_gap(bench_case: BenchCase) -> None:
    """``od.boolean.min_gap`` of the sphere and a copy moved clear of it."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
    extent = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    moved = (vertices + np.array([1.1 * extent, 0.0, 0.0])).astype(np.float32)
    faces = np.asarray(bench_case.faces_np)
    search = 0.5 * extent
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        b_m3 = m3.Manifold(m3.Mesh(vert_properties=moved, tri_verts=faces.astype(np.uint32)))
        gap_m3 = bench_case.run(lambda: a_m3.min_gap(b_m3, search))
        assert gap_m3 > 0.0
        return
    moved_wp = wp.array(moved, dtype=wp.vec3, device=bench_case.device)
    vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
    gap = bench_case.run(
        lambda: od.boolean.min_gap(vertices_wp, faces_wp, moved_wp, faces_wp, search)
    )
    assert gap > 0.0


@pytest.mark.benchmark(group="simplify")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_simplify(bench_case: BenchCase) -> None:
    """``od.boolean.simplify`` at a tolerance of a thousandth of the sphere's size."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
    tolerance = 1e-3 * float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        result_m3 = bench_case.run(lambda: _forced(a_m3.simplify(tolerance)))
        assert result_m3.num_tri() > 0
        return
    vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
    _, faces = bench_case.run(lambda: od.boolean.simplify(vertices_wp, faces_wp, tolerance))
    assert faces.size > 0


@pytest.mark.benchmark(group="refine")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_refine(bench_case: BenchCase) -> None:
    """``od.boolean.refine`` of the sphere into three pieces an edge."""
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        result_m3 = bench_case.run(lambda: _forced(a_m3.refine(3)))
        assert result_m3.num_tri() > 0
        return
    vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
    _, faces = bench_case.run(lambda: od.boolean.refine(vertices_wp, faces_wp, 3))
    assert faces.size == 9 * faces_wp.size


@pytest.mark.benchmark(group="refine_to_length")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_refine_to_length(bench_case: BenchCase) -> None:
    """``od.boolean.refine_to_length`` at a third of the sphere's median edge."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
    faces = np.asarray(bench_case.faces_np).reshape(-1, 3)
    length = float(np.median(np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 1]], axis=1)))
    length /= 3.0
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        result_m3 = bench_case.run(lambda: _forced(a_m3.refine_to_length(length)))
        assert result_m3.num_tri() > 0
        return
    vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
    _, faces = bench_case.run(lambda: od.boolean.refine_to_length(vertices_wp, faces_wp, length))
    assert faces.size > faces_wp.size


@pytest.mark.benchmark(group="smooth_out")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_smooth_out(bench_case: BenchCase) -> None:
    """``od.boolean.smooth_out`` of the sphere into two pieces an edge."""
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        result_m3 = bench_case.run(lambda: _forced(a_m3.smooth_out().refine(2)))
        assert result_m3.num_tri() > 0
        return
    vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
    _, faces = bench_case.run(lambda: od.boolean.smooth_out(vertices_wp, faces_wp, 2))
    assert faces.size == 4 * faces_wp.size


@pytest.mark.benchmark(group="hull")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_hull(bench_case: BenchCase) -> None:
    """``od.boolean.hull`` of the sphere's vertices (manifold3d: ``hull_points``)."""
    if bench_case.kind == "manifold3d":
        points_np = np.asarray(bench_case.vertices_np, dtype=np.float64)
        result_m3 = bench_case.run(lambda: _forced(m3.Manifold.hull_points(points_np)))
        assert result_m3.num_tri() > 0
        return
    vertices_wp = bench_case.vertices_wp
    _, faces = bench_case.run(lambda: od.boolean.hull(vertices_wp))
    assert faces.size > 0


@pytest.mark.benchmark(group="cross_section")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_cross_section(bench_case: BenchCase) -> None:
    """``od.boolean.cross_section`` through the sphere's middle (manifold3d: ``slice``)."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
    height = float(vertices[:, 2].mean()) + 0.01
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        loops_m3 = bench_case.run(lambda: a_m3.slice(height).to_polygons())
        assert len(loops_m3) > 0
        return
    vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
    loops = bench_case.run(lambda: od.boolean.cross_section(vertices_wp, faces_wp, height))
    assert len(loops) > 0


@pytest.mark.benchmark(group="cross_section_with_offsets")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_cross_section_with_offsets(bench_case: BenchCase) -> None:
    """``od.boolean.cross_section_with_offsets``, the packed loops."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float64)
    height = float(vertices[:, 2].mean()) + 0.01
    if bench_case.kind == "manifold3d":
        a_m3, _ = _manifolds_m3(bench_case)
        loops_m3 = bench_case.run(lambda: a_m3.slice(height).to_polygons())
        assert len(loops_m3) > 0
        return
    vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
    _, offsets = bench_case.run(
        lambda: od.boolean.cross_section_with_offsets(vertices_wp, faces_wp, height)
    )
    assert offsets.size > 1


@pytest.mark.benchmark(group="decompose")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.benchaxis("scale")
def test_decompose(bench_case: BenchCase) -> None:
    """``od.boolean.decompose`` of the sphere beside a disjoint moved copy of itself."""
    vertices = np.asarray(bench_case.vertices_np, dtype=np.float32)
    faces = np.asarray(bench_case.faces_np, dtype=np.int32)
    shift = 2.0 * (vertices.max(axis=0) - vertices.min(axis=0))
    both_np = np.concatenate([vertices, vertices + shift.astype(np.float32)])
    both_faces = np.concatenate([faces, faces + vertices.shape[0]])
    if bench_case.kind == "manifold3d":
        whole_m3 = m3.Manifold(
            m3.Mesh(vert_properties=both_np, tri_verts=both_faces.astype(np.uint32))
        )
        pieces_m3 = bench_case.run(whole_m3.decompose)
        assert len(pieces_m3) == 2
        return
    device = bench_case.vertices_wp.device
    vertices_wp = wp.array(both_np, dtype=wp.vec3, device=device)
    faces_wp = wp.array(both_faces.reshape(-1), dtype=wp.int32, device=device)
    pieces = bench_case.run(lambda: od.boolean.decompose(vertices_wp, faces_wp))
    assert len(pieces) == 2


# Points around the swept star ring, and the slices or steps a sweep takes.
_SWEEP_RINGS = [64, 1024]


def _ring_loop_np(n: int, shift: float) -> np.ndarray:
    """Return a star ring of ``n`` points shifted ``shift`` along x, as one loop."""
    angle_np = 2.0 * np.pi * np.arange(n) / n
    radius_np = np.where(np.arange(n) % 2 == 0, 1.0, 0.6)
    return np.column_stack((radius_np * np.cos(angle_np) + shift, radius_np * np.sin(angle_np)))


@pytest.mark.benchmark(group="extrude_region")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.parametrize("ring_size", _SWEEP_RINGS)
def test_extrude_region(bench_lib: BenchLibrary, ring_size: int) -> None:
    """``od.boolean.extrude_region`` of a star ring in 16 twisted, tapering slices."""
    loop_np = _ring_loop_np(ring_size, 0.0)
    if bench_lib.kind == "manifold3d":
        section_m3 = m3.CrossSection([loop_np])
        result_m3 = bench_lib.run(
            lambda: _forced(m3.Manifold.extrude(section_m3, 2.0, 15, 45.0, (0.5, 0.5)))
        )
        assert result_m3.num_tri() > 0
        return
    device = str(bench_lib.device)
    points = wp.array(loop_np, dtype=wp.vec2d, device=device)
    offsets = wp.array([0, ring_size], dtype=wp.int32, device=device)
    _, faces = bench_lib.run(
        lambda: od.boolean.extrude_region(points, offsets, 2.0, 15, 45.0, (0.5, 0.5))
    )
    assert faces.size > 0


@pytest.mark.benchmark(group="revolve_region")
@pytest.mark.benchlibs("ordito", "manifold3d")
@pytest.mark.parametrize("ring_size", _SWEEP_RINGS)
def test_revolve_region(bench_lib: BenchLibrary, ring_size: int) -> None:
    """``od.boolean.revolve_region`` of a star ring off the axis, in 64 steps round."""
    loop_np = _ring_loop_np(ring_size, 2.0)
    if bench_lib.kind == "manifold3d":
        section_m3 = m3.CrossSection([loop_np])
        result_m3 = bench_lib.run(lambda: _forced(m3.Manifold.revolve(section_m3, 64)))
        assert result_m3.num_tri() > 0
        return
    device = str(bench_lib.device)
    points = wp.array(loop_np, dtype=wp.vec2d, device=device)
    offsets = wp.array([0, ring_size], dtype=wp.int32, device=device)
    _, faces = bench_lib.run(lambda: od.boolean.revolve_region(points, offsets, 64))
    assert faces.size > 0
