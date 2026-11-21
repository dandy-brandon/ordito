"""Regression tests for ``ordito.boolean``."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import manifold3d as m3
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from tests.comparisons import assert_same_as_manifold3d
from tests.conftest import CLOSED_MESHES
from tests.conversions import float32_trimesh, numpy_to_warp, trimesh_to_manifold3d, warp_empty

MeshPair = tuple[tm.Trimesh, wp.Mesh]
WarpMesh = tuple[wp.array[wp.vec3], wp.array[wp.int32]]
Operation = Callable[
    [wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.vec3], wp.array[wp.int32]], WarpMesh
]

OPERATIONS: list[tuple[str, Operation]] = [
    ("union", od.boolean.union),
    ("intersection", od.boolean.intersection),
    ("difference", od.boolean.difference),
]
_OPERATION_IDS = [name for name, _ in OPERATIONS]

# The closed fixtures the generic comparisons sweep: genus 0 and genus 1, both with needles.
_GENERIC_MESHES = [*CLOSED_MESHES, "torus_irregular"]


def _moved_copy(mesh_tm: tm.Trimesh, seed: int) -> tm.Trimesh:
    """Return ``mesh_tm`` rotated about its centroid and shifted by a third of its extent."""
    rng = np.random.default_rng(seed)
    moved = mesh_tm.copy()
    centre = mesh_tm.vertices.mean(axis=0)
    moved.apply_translation(-centre)
    moved.apply_transform(tm.transformations.random_rotation_matrix(rng.random(3)))
    moved.apply_translation(centre + mesh_tm.extents * np.array([0.31, 0.17, 0.11]))
    return float32_trimesh(moved)


def _box(lower: list[float], upper: list[float]) -> tm.Trimesh:
    return tm.creation.box(bounds=[lower, upper])


# Configurations where the operands touch, overlap in a plane or coincide: every tie Manifold's
# symbolic perturbation breaks, and the cleanup it then runs.
_DEGENERATE_CASES = {
    "identical": (_box([0, 0, 0], [1, 1, 1]), _box([0, 0, 0], [1, 1, 1])),
    "face_contact": (_box([0, 0, 0], [1, 1, 1]), _box([1, 0, 0], [2, 1, 1])),
    "corner_contact": (_box([0, 0, 0], [1, 1, 1]), _box([1, 1, 1], [2, 2, 2])),
    "coplanar_overlap": (_box([0, 0, 0], [1, 1, 1]), _box([0.5, 0, 0], [1.5, 1, 1])),
    "partial_face_contact": (
        _box([-0.25, -0.5, -0.5], [0.25, 0.5, 0.5]),
        _box([0.25, 0.25, -0.75], [0.75, 1.75, -0.25]),
    ),
    "nested": (_box([0, 0, 0], [1, 1, 1]), _box([0.2, 0.2, 0.2], [0.8, 0.8, 0.8])),
    "disjoint": (_box([0, 0, 0], [1, 1, 1]), _box([2, 2, 2], [3, 3, 3])),
}


def _m3_operation(name: str, a_m3: m3.Manifold, b_m3: m3.Manifold) -> m3.Manifold:
    if name == "union":
        return a_m3 + b_m3
    if name == "intersection":
        return a_m3 ^ b_m3
    return a_m3 - b_m3


@pytest.mark.parity("boolean_union", "manifold3d")
@pytest.mark.parity("boolean_intersection", "manifold3d")
@pytest.mark.parity("boolean_difference", "manifold3d")
@pytest.mark.parametrize("mesh_name", _GENERIC_MESHES)
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_matches_manifold3d(
    request: pytest.FixtureRequest, mesh_name: str, name: str, operation: Operation
) -> None:
    """
    The Boolean of a mesh and a rotated, shifted copy is Manifold's, array for array.

    Class A, the strictest form: the same vertices in the same order, the same triangles in the
    same order and rotation, as ``manifold3d``'s ``to_mesh()``. The reference is non-vacuous: its
    volume is neither operand's, so the crossing curve is really there.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    a_tm = float32_trimesh(mesh_tm)
    b_tm = _moved_copy(a_tm, seed=7)
    result_m3 = _m3_operation(name, trimesh_to_manifold3d(a_tm), trimesh_to_manifold3d(b_tm))
    assert result_m3.num_tri() > 0
    assert not np.isclose(result_m3.volume(), a_tm.volume, rtol=1e-3)
    device = mesh_wp.device
    assert_same_as_manifold3d(
        operation(
            *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
            *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
        ),
        result_m3,
    )


@pytest.mark.parity("boolean_union", "manifold3d")
@pytest.mark.parity("boolean_intersection", "manifold3d")
@pytest.mark.parity("boolean_difference", "manifold3d")
@pytest.mark.parametrize("case", list(_DEGENERATE_CASES))
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_degenerate_contacts_match_manifold3d(
    device: str, case: str, name: str, operation: Operation
) -> None:
    """
    Touching, coplanar, coincident, nested and disjoint solids give Manifold's mesh exactly.

    Class A, array for array, the empty result included. These are the ties the symbolic
    perturbation decides (``partial_face_contact``'s turn on the last bit of the vertex normals'
    angle weights), the slivers and duplicate edges the topology cleanup removes, and the union of
    apart meshes that Manifold composes rather than intersects (``disjoint``; ``corner_contact``
    touches, so it is intersected).
    """
    a_tm, b_tm = _DEGENERATE_CASES[case]
    result_m3 = _m3_operation(name, trimesh_to_manifold3d(a_tm), trimesh_to_manifold3d(b_tm))
    result_wp = operation(
        *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
        *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
    )
    assert_same_as_manifold3d(result_wp, result_m3)


@pytest.mark.parity("boolean_union", "trimesh")
@pytest.mark.parity("boolean_intersection", "trimesh")
@pytest.mark.parity("boolean_difference", "trimesh")
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_matches_trimesh(device: str, name: str, operation: Operation) -> None:
    """
    The Boolean equals ``trimesh.boolean``'s with its default engine (Manifold).

    Class A, up to vertex numbering, face order and rotation (trimesh merges and reorders the
    engine's output).
    """
    a_tm = float32_trimesh(tm.creation.icosphere(subdivisions=3))
    b_tm = _moved_copy(a_tm, seed=11)
    reference_tm = getattr(tm.boolean, name)([a_tm, b_tm], engine="manifold")
    assert len(reference_tm.faces) > 0
    vertices_wp, faces_wp = operation(
        *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
        *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
    )
    reference_wp = numpy_to_warp(reference_tm.vertices, reference_tm.faces, device)
    assert od.equality.meshes_allclose(vertices_wp, faces_wp, *reference_wp, atol=1e-6)


@pytest.mark.parity("split_by_mesh", "manifold3d")
@pytest.mark.parametrize("case", ["spheres", "coplanar_overlap", "nested", "disjoint"])
def test_split_by_mesh_matches_manifold3d(device: str, case: str) -> None:
    """
    Both parts of a split are Manifold's ``Manifold.split``, array for array.

    Class A. The generic pair is two icospheres crossing; the boxes are the coplanar, nested and
    disjoint ties, where one part is the whole mesh or empty.
    """
    if case == "spheres":
        a_tm = float32_trimesh(tm.creation.icosphere(subdivisions=3))
        b_tm = _moved_copy(a_tm, seed=3)
    else:
        a_tm, b_tm = _DEGENERATE_CASES[case]
    inside_m3, outside_m3 = trimesh_to_manifold3d(a_tm).split(trimesh_to_manifold3d(b_tm))
    inside_wp, outside_wp = od.boolean.split_by_mesh(
        *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
        *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
    )
    assert_same_as_manifold3d(inside_wp, inside_m3)
    assert_same_as_manifold3d(outside_wp, outside_m3)


# Planes through the box and the sphere: axis-aligned through vertices and edges (every tie),
# diagonal, oblique, and clear of the mesh.
_PLANES = [
    ((0.0, 0.0, 1.0), 0.25),
    ((1.0, 0.0, 0.0), 0.0),
    ((0.0, 1.0, 0.0), 0.5),
    ((1.0, 1.0, 0.0), 0.0),
    ((0.3, -0.5, 0.8), 0.1),
    ((-1.0, 2.0, 3.0), -0.2),
    ((0.0, 0.0, 1.0), 5.0),
]


@pytest.mark.parity("split_by_plane", "manifold3d")
@pytest.mark.parity("trim_by_plane", "manifold3d")
@pytest.mark.parametrize("shape", ["box", "sphere", "torus"])
@pytest.mark.parametrize(("normal", "offset"), _PLANES)
def test_plane_cuts_match_manifold3d(
    device: str, shape: str, normal: tuple[float, float, float], offset: float
) -> None:
    """
    ``split_by_plane`` and ``trim_by_plane`` are Manifold's, array for array.

    Class A. Manifold cuts with a cube it scales, moves and turns onto the plane in ``float64``,
    its rotation from its own ``sin`` / ``cos`` of degrees; ordito builds the same cube the same
    way, so planes through the box's vertices and edges tie exactly as Manifold's do. The last
    plane misses the mesh: everything on one side.
    """
    mesh_tm = {
        "box": _box([0, 0, 0], [1, 1, 1]),
        "sphere": float32_trimesh(tm.creation.icosphere(subdivisions=3)),
        "torus": float32_trimesh(tm.creation.torus(1.0, 0.3, 24, 12)),
    }[shape]
    mesh_m3 = trimesh_to_manifold3d(mesh_tm)
    above_m3, below_m3 = mesh_m3.split_by_plane(normal, offset)
    trimmed_m3 = mesh_m3.trim_by_plane(normal, offset)
    mesh_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    above_wp, below_wp = od.boolean.split_by_plane(*mesh_wp, normal, offset)
    assert_same_as_manifold3d(above_wp, above_m3)
    assert_same_as_manifold3d(below_wp, below_m3)
    assert_same_as_manifold3d(od.boolean.trim_by_plane(*mesh_wp, normal, offset), trimmed_m3)


def test_plane_cut_rejects_a_zero_normal(device: str) -> None:
    """A zero normal raises ``ValueError``. Not a library comparison: the guard."""
    box_tm = _box([0, 0, 0], [1, 1, 1])
    mesh_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    with pytest.raises(ValueError, match="normal"):
        od.boolean.split_by_plane(*mesh_wp, (0.0, 0.0, 0.0))


def _batch_meshes(case: str) -> list[tm.Trimesh]:
    """Return a batch case's meshes: touching in a row, scattered, or apart beside a bar."""
    if case == "row":
        return [_box([0.75 * i, 0, 0], [0.75 * i + 1, 1, 1]) for i in range(3)]
    rng = np.random.default_rng(3 if case == "scattered" else 5)
    meshes: list[tm.Trimesh] = []
    for i in range(6):
        sphere = tm.creation.icosphere(subdivisions=2, radius=float(rng.uniform(0.4, 0.8)))
        offset = rng.uniform(-1.0, 1.0, 3) if case == "scattered" else np.array([3.0 * i, 0, 0])
        sphere.apply_translation(offset)
        meshes.append(float32_trimesh(sphere))
    if case == "apart":
        meshes.append(_box([0.5, 0, 0], [4, 0.5, 0.5]))
    return meshes


@pytest.mark.parity("union_all", "manifold3d")
@pytest.mark.parity("intersection_all", "manifold3d")
@pytest.mark.parity("difference_all", "manifold3d")
@pytest.mark.parametrize("case", ["row", "scattered", "apart"])
@pytest.mark.parametrize(
    ("function", "op_m3"),
    [
        (od.boolean.union_all, m3.OpType.Add),
        (od.boolean.intersection_all, m3.OpType.Intersect),
        (od.boolean.difference_all, m3.OpType.Subtract),
    ],
    ids=["union", "intersection", "difference"],
)
def test_batch_booleans_match_manifold3d(
    device: str, case: str, function: Callable[[Sequence[WarpMesh]], WarpMesh], op_m3: m3.OpType
) -> None:
    """
    The batch Booleans are Manifold's ``batch_boolean``, array for array.

    Class A. ``apart`` holds spheres whose boxes do not touch, which Manifold composes rather
    than intersects, then joins to the bar across them; ``scattered`` overlaps in a chain the
    pairing order decides; ``row`` touches face to face.
    """
    meshes_tm = _batch_meshes(case)
    result_m3 = m3.Manifold.batch_boolean(
        [trimesh_to_manifold3d(mesh) for mesh in meshes_tm], op_m3
    )
    result_wp = function([numpy_to_warp(mesh.vertices, mesh.faces, device) for mesh in meshes_tm])
    assert_same_as_manifold3d(result_wp, result_m3)


def test_batch_booleans_reject_an_empty_list() -> None:
    """No mesh at all raises ``ValueError``. Not a library comparison: the guard."""
    with pytest.raises(ValueError, match="at least one mesh"):
        od.boolean.union_all([])


@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_with_an_empty_operand(device: str, name: str, operation: Operation) -> None:
    """
    An empty operand leaves the other mesh, or nothing, as a solid's set algebra says.

    Not a library comparison: Manifold builds no solid from an empty mesh to compare with, so the
    cases are checked against their definitions: the union is the other operand, the
    intersection empty, the difference the first operand (or empty when that is the empty one).
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
    box_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, mesh_wp.device)
    with pytest.raises(ValueError, match="closed, edge-manifold and consistently wound"):
        od.boolean.union(mesh_wp.points, mesh_wp.indices, *box_wp)
    with pytest.raises(ValueError, match="closed, edge-manifold and consistently wound"):
        od.boolean.difference(*box_wp, mesh_wp.points, mesh_wp.indices)


def test_boolean_requires_one_device() -> None:
    """Operands on two devices raise ``RuntimeError``. Not a library comparison: the guard."""
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device beside the CPU")
    box_tm = _box([0, 0, 0], [1, 1, 1])
    with pytest.raises(RuntimeError):
        od.boolean.union(
            *numpy_to_warp(box_tm.vertices, box_tm.faces, "cpu"),
            *numpy_to_warp(box_tm.vertices, box_tm.faces, "cuda:0"),
        )
