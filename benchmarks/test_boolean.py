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

**pymeshlab**'s ``generate_boolean_*`` is libigl's exact CGAL Boolean (single-threaded), and
**meshlib**'s ``boolean`` is MeshLib's (multi-threaded). Neither simplifies its result, so their
answers equal ordito's ``collapse_collinear=False`` up to the triangulation of each cut face
(``tests/test_boolean.py``, Class B); on these curved operands that is the default's answer too.
The pymeshlab row deletes the result layer it adds inside the timed call, so the set does not grow
across rounds. The meshlib row builds its two operands inside the timed call: MeshLib caches each
mesh's AABB tree on the mesh, so shared operands would skip on every round after the first the
tree builds ordito pays on each call (3-6x faster rows, 5 k to 328 k faces).

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
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, mesh_ml_from_numpy

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
    if bench_case.kind == "pymeshlab":
        meshset = ml.MeshSet()
        for vertices_np in (bench_case.vertices_np, moved_np):
            meshset.add_mesh(
                ml.Mesh(
                    np.asarray(vertices_np, dtype=np.float64),
                    np.asarray(bench_case.faces_np, dtype=np.int32),
                )
            )
        filter_pml = getattr(meshset, f"generate_boolean_{name}")

        def run_pml() -> int:
            filter_pml(first_mesh=0, second_mesh=1)
            n_faces = meshset.current_mesh().face_number()
            meshset.delete_current_mesh()
            return n_faces

        assert bench_case.run(run_pml) > 0
        return
    if bench_case.kind == "meshlib":
        faces_np = bench_case.faces_np
        operation_ml = {
            "union": mm.BooleanOperation.Union,
            "intersection": mm.BooleanOperation.Intersection,
            "difference": mm.BooleanOperation.DifferenceAB,
        }[name]

        def run_ml() -> mm.BooleanResult:
            # Fresh operands each call: MeshLib caches a mesh's AABB tree on the mesh, which a
            # shared operand would build once and then skip, where ordito builds its trees per call.
            a_ml = mesh_ml_from_numpy(bench_case.vertices_np, faces_np)
            b_ml = mesh_ml_from_numpy(moved_np, faces_np)
            return mm.boolean(a_ml, b_ml, operation_ml)

        result_ml = bench_case.run(run_ml)
        assert result_ml.valid()
        assert result_ml.mesh.topology.numValidFaces() > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _vertices, result_faces = bench_case.run(lambda: operation(vertices, faces, moved_wp, faces))
    assert result_faces.size > 0


@pytest.mark.benchmark(group="boolean_union")
@pytest.mark.benchlibs("ordito", "trimesh", "manifold3d", "pymeshlab", "meshlib")
@pytest.mark.benchaxis("scale")
def test_union(bench_case: BenchCase) -> None:
    """``od.boolean.union`` of a sphere and its rotated, shifted copy."""
    _run(bench_case, "union", od.boolean.union)


@pytest.mark.benchmark(group="boolean_intersection")
@pytest.mark.benchlibs("ordito", "trimesh", "manifold3d", "pymeshlab", "meshlib")
@pytest.mark.benchaxis("scale")
def test_intersection(bench_case: BenchCase) -> None:
    """``od.boolean.intersection`` of a sphere and its rotated, shifted copy."""
    _run(bench_case, "intersection", od.boolean.intersection)


@pytest.mark.benchmark(group="boolean_difference")
@pytest.mark.benchlibs("ordito", "trimesh", "manifold3d", "pymeshlab", "meshlib")
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
