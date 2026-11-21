"""Regression tests for ``ordito.boolean``."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from tests.conftest import CLOSED_MESHES
from tests.conversions import numpy_to_warp, warp_empty

MeshPair = tuple[tm.Trimesh, wp.Mesh]
Operation = Callable[
    [wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.vec3], wp.array[wp.int32]],
    tuple[wp.array[wp.vec3], wp.array[wp.int32]],
]

OPERATIONS: list[tuple[str, Operation]] = [
    ("union", od.boolean.union),
    ("intersection", od.boolean.intersection),
    ("difference", od.boolean.difference),
]

# The closed fixtures the generic comparison sweeps: genus 0 and genus 1, both with needles.
_GENERIC_MESHES = [*CLOSED_MESHES, "torus_irregular"]


def _moved_copy(mesh_tm: tm.Trimesh, seed: int) -> tm.Trimesh:
    """Return ``mesh_tm`` rotated about its centroid and shifted by a third of its extent."""
    rng = np.random.default_rng(seed)
    moved = mesh_tm.copy()
    centre = mesh_tm.vertices.mean(axis=0)
    moved.apply_translation(-centre)
    moved.apply_transform(tm.transformations.random_rotation_matrix(rng.random(3)))
    moved.apply_translation(centre + mesh_tm.extents * np.array([0.31, 0.17, 0.11]))
    # Both operands carry float32 coordinates, as ordito and Manifold both read them.
    moved.vertices = moved.vertices.astype(np.float32).astype(np.float64)
    return moved


def _reference(name: str, a_tm: tm.Trimesh, b_tm: tm.Trimesh) -> tm.Trimesh:
    """Return trimesh's Boolean of the two, through its Manifold engine."""
    operation = getattr(tm.boolean, name)
    return operation([a_tm, b_tm], engine="manifold")


@pytest.mark.parity("boolean_union", "trimesh")
@pytest.mark.parity("boolean_intersection", "trimesh")
@pytest.mark.parity("boolean_difference", "trimesh")
@pytest.mark.parametrize("mesh_name", _GENERIC_MESHES)
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=[n for n, _ in OPERATIONS])
def test_boolean_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str, name: str, operation: Operation
) -> None:
    """
    The Boolean of a mesh and a rotated, shifted copy equals trimesh's (its Manifold engine).

    Class A: equal up to vertex numbering, face order and rotation. ordito ports Manifold's
    exact predicates and its topology cleanup, so on a generic configuration the two results are
    the same mesh, crossing vertices included, to rounding. The reference is non-vacuous: its
    volume is neither operand's nor zero, so the crossing curve is really there.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    a_tm = tm.Trimesh(mesh_tm.vertices.astype(np.float32), mesh_tm.faces, process=False)
    b_tm = _moved_copy(a_tm, seed=7)
    device = mesh_wp.device
    reference_tm = _reference(name, a_tm, b_tm)
    assert len(reference_tm.faces) > 0
    assert not np.isclose(reference_tm.volume, a_tm.volume, rtol=1e-3)
    vertices_a, faces_a = numpy_to_warp(a_tm.vertices, a_tm.faces, device)
    vertices_b, faces_b = numpy_to_warp(b_tm.vertices, b_tm.faces, device)
    vertices_wp, faces_wp = operation(vertices_a, faces_a, vertices_b, faces_b)
    reference_wp = numpy_to_warp(reference_tm.vertices, reference_tm.faces, device)
    assert od.equality.meshes_allclose(vertices_wp, faces_wp, *reference_wp, atol=1e-6)


def _box(lower: list[float], upper: list[float]) -> tm.Trimesh:
    return tm.creation.box(bounds=[lower, upper])


# Configurations where the operands touch, overlap in a plane or coincide: every tie Manifold's
# symbolic perturbation breaks, and the cleanup it then runs. The corner contact is left out:
# its union keeps two vertices at one point, which ``meshes_allclose`` cannot pair up.
_DEGENERATE_CASES = {
    "identical": (_box([0, 0, 0], [1, 1, 1]), _box([0, 0, 0], [1, 1, 1])),
    "face_contact": (_box([0, 0, 0], [1, 1, 1]), _box([1, 0, 0], [2, 1, 1])),
    "coplanar_overlap": (_box([0, 0, 0], [1, 1, 1]), _box([0.5, 0, 0], [1.5, 1, 1])),
    "partial_face_contact": (
        _box([-0.25, -0.5, -0.5], [0.25, 0.5, 0.5]),
        _box([0.25, 0.25, -0.75], [0.75, 1.75, -0.25]),
    ),
    "nested": (_box([0, 0, 0], [1, 1, 1]), _box([0.2, 0.2, 0.2], [0.8, 0.8, 0.8])),
    "disjoint": (_box([0, 0, 0], [1, 1, 1]), _box([2, 2, 2], [3, 3, 3])),
}


@pytest.mark.parity("boolean_union", "trimesh")
@pytest.mark.parity("boolean_intersection", "trimesh")
@pytest.mark.parity("boolean_difference", "trimesh")
@pytest.mark.parametrize("case", list(_DEGENERATE_CASES))
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=[n for n, _ in OPERATIONS])
def test_boolean_degenerate_contacts_match_trimesh(
    device: str, case: str, name: str, operation: Operation
) -> None:
    """
    Touching, coplanar, coincident, nested and disjoint solids give trimesh's (Manifold's) mesh.

    Class A, the empty result included. These are the ties the symbolic perturbation decides and
    the slivers and duplicate edges the topology cleanup removes. ``partial_face_contact`` is the
    case whose ties the vertex normals' last bit decides (two boxes' matching corners): with the
    platform's arc cosine, CUDA kept two vertices Manifold collapses.
    """
    a_tm, b_tm = _DEGENERATE_CASES[case]
    reference_tm = _reference(name, a_tm, b_tm)
    vertices_a, faces_a = numpy_to_warp(a_tm.vertices, a_tm.faces, device)
    vertices_b, faces_b = numpy_to_warp(b_tm.vertices, b_tm.faces, device)
    vertices_wp, faces_wp = operation(vertices_a, faces_a, vertices_b, faces_b)
    if len(reference_tm.faces) == 0:
        assert faces_wp.size == 0
        return
    result_tm = tm.Trimesh(vertices_wp.numpy(), faces_wp.numpy().reshape(-1, 3), process=False)
    assert result_tm.is_watertight
    assert result_tm.is_winding_consistent
    reference_wp = numpy_to_warp(reference_tm.vertices, reference_tm.faces, device)
    assert od.equality.meshes_allclose(vertices_wp, faces_wp, *reference_wp, atol=1e-6)


@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=[n for n, _ in OPERATIONS])
def test_boolean_with_an_empty_operand(device: str, name: str, operation: Operation) -> None:
    """
    An empty operand leaves the other mesh, or nothing, as a solid's set algebra says.

    Not a library comparison: trimesh's engine rejects an empty mesh, so the cases are checked
    against their definitions: the union is the other operand, the intersection empty, the
    difference the first operand (or empty when that is the empty one).
    """
    box_tm = _box([0, 0, 0], [1, 1, 1])
    vertices, faces = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    empty_vertices = warp_empty(0, wp.vec3, device)
    empty_faces = warp_empty(0, wp.int32, device)
    _, faces_left = operation(empty_vertices, empty_faces, vertices, faces)
    _, faces_right = operation(vertices, faces, empty_vertices, empty_faces)
    expected_left = {"union": 12, "intersection": 0, "difference": 0}[name]
    expected_right = {"union": 12, "intersection": 0, "difference": 12}[name]
    assert faces_left.size // 3 == expected_left
    assert faces_right.size // 3 == expected_right


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "boy_surface"])
def test_boolean_rejects_open_or_non_orientable_meshes(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    A mesh with a rim, or one that cannot be wound consistently, raises ``ValueError``.

    Not a library comparison: the contract is the error. ``saddle_graded`` has a rim and
    ``boy_surface`` no consistent winding; both operand positions are checked.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    box_tm = _box([0, 0, 0], [1, 1, 1])
    vertices, faces = numpy_to_warp(box_tm.vertices, box_tm.faces, mesh_wp.device)
    with pytest.raises(ValueError, match="closed, edge-manifold and consistently wound"):
        od.boolean.union(mesh_wp.points, mesh_wp.indices, vertices, faces)
    with pytest.raises(ValueError, match="closed, edge-manifold and consistently wound"):
        od.boolean.difference(vertices, faces, mesh_wp.points, mesh_wp.indices)


def test_boolean_requires_one_device() -> None:
    """Operands on two devices raise ``RuntimeError``. Not a library comparison: the guard."""
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device beside the CPU")
    box_tm = _box([0, 0, 0], [1, 1, 1])
    vertices_cpu, faces_cpu = numpy_to_warp(box_tm.vertices, box_tm.faces, "cpu")
    vertices_cuda, faces_cuda = numpy_to_warp(box_tm.vertices, box_tm.faces, "cuda:0")
    with pytest.raises(RuntimeError):
        od.boolean.union(vertices_cpu, faces_cpu, vertices_cuda, faces_cuda)
