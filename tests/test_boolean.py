"""Regression tests for ``ordito.boolean``."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import manifold3d as m3
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from tests.conftest import CLOSED_MESHES
from tests.conversions import numpy_to_warp, warp_empty

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


def _float32_mesh(mesh_tm: tm.Trimesh) -> tm.Trimesh:
    """Return ``mesh_tm`` with ``float32`` coordinates, as ordito and Manifold both read them."""
    return tm.Trimesh(mesh_tm.vertices.astype(np.float32), mesh_tm.faces, process=False)


def _moved_copy(mesh_tm: tm.Trimesh, seed: int) -> tm.Trimesh:
    """Return ``mesh_tm`` rotated about its centroid and shifted by a third of its extent."""
    rng = np.random.default_rng(seed)
    moved = mesh_tm.copy()
    centre = mesh_tm.vertices.mean(axis=0)
    moved.apply_translation(-centre)
    moved.apply_transform(tm.transformations.random_rotation_matrix(rng.random(3)))
    moved.apply_translation(centre + mesh_tm.extents * np.array([0.31, 0.17, 0.11]))
    return _float32_mesh(moved)


def _box(lower: list[float], upper: list[float]) -> tm.Trimesh:
    return tm.creation.box(bounds=[lower, upper])


def _manifold_m3(mesh_tm: tm.Trimesh) -> m3.Manifold:
    """Return ``mesh_tm`` as a Manifold, its ``float32`` coordinates as ordito reads them."""
    mesh_m3 = m3.Mesh(
        vert_properties=np.asarray(mesh_tm.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh_tm.faces, dtype=np.uint32),
    )
    return m3.Manifold(mesh_m3)


def _upload(mesh_tm: tm.Trimesh, device: str) -> WarpMesh:
    return numpy_to_warp(np.asarray(mesh_tm.vertices, dtype=np.float32), mesh_tm.faces, device)


def _assert_same_as_m3(result_wp: WarpMesh, result_m3: m3.Manifold) -> None:
    """Assert ordito's mesh is Manifold's ``to_mesh()``, array for array, order included."""
    mesh_m3 = result_m3.to_mesh()
    vertices_wp, faces_wp = result_wp
    assert np.array_equal(vertices_wp.numpy(), np.asarray(mesh_m3.vert_properties)[:, :3])
    assert np.array_equal(faces_wp.numpy().reshape(-1, 3), np.asarray(mesh_m3.tri_verts))


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
    a_tm = _float32_mesh(mesh_tm)
    b_tm = _moved_copy(a_tm, seed=7)
    result_m3 = _m3_operation(name, _manifold_m3(a_tm), _manifold_m3(b_tm))
    assert result_m3.num_tri() > 0
    assert not np.isclose(result_m3.volume(), a_tm.volume, rtol=1e-3)
    device = mesh_wp.device
    _assert_same_as_m3(operation(*_upload(a_tm, device), *_upload(b_tm, device)), result_m3)


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
    result_m3 = _m3_operation(name, _manifold_m3(a_tm), _manifold_m3(b_tm))
    result_wp = operation(*_upload(a_tm, device), *_upload(b_tm, device))
    _assert_same_as_m3(result_wp, result_m3)


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
    a_tm = _float32_mesh(tm.creation.icosphere(subdivisions=3))
    b_tm = _moved_copy(a_tm, seed=11)
    reference_tm = getattr(tm.boolean, name)([a_tm, b_tm], engine="manifold")
    assert len(reference_tm.faces) > 0
    vertices_wp, faces_wp = operation(*_upload(a_tm, device), *_upload(b_tm, device))
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
        a_tm = _float32_mesh(tm.creation.icosphere(subdivisions=3))
        b_tm = _moved_copy(a_tm, seed=3)
    else:
        a_tm, b_tm = _DEGENERATE_CASES[case]
    inside_m3, outside_m3 = _manifold_m3(a_tm).split(_manifold_m3(b_tm))
    inside_wp, outside_wp = od.boolean.split_by_mesh(*_upload(a_tm, device), *_upload(b_tm, device))
    _assert_same_as_m3(inside_wp, inside_m3)
    _assert_same_as_m3(outside_wp, outside_m3)


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
        "sphere": _float32_mesh(tm.creation.icosphere(subdivisions=3)),
        "torus": _float32_mesh(tm.creation.torus(1.0, 0.3, 24, 12)),
    }[shape]
    mesh_m3 = _manifold_m3(mesh_tm)
    above_m3, below_m3 = mesh_m3.split_by_plane(normal, offset)
    trimmed_m3 = mesh_m3.trim_by_plane(normal, offset)
    mesh_wp = _upload(mesh_tm, device)
    above_wp, below_wp = od.boolean.split_by_plane(*mesh_wp, normal, offset)
    _assert_same_as_m3(above_wp, above_m3)
    _assert_same_as_m3(below_wp, below_m3)
    _assert_same_as_m3(od.boolean.trim_by_plane(*mesh_wp, normal, offset), trimmed_m3)


def test_plane_cut_rejects_a_zero_normal(device: str) -> None:
    """A zero normal raises ``ValueError``. Not a library comparison: the guard."""
    mesh_wp = _upload(_box([0, 0, 0], [1, 1, 1]), device)
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
        meshes.append(_float32_mesh(sphere))
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
    result_m3 = m3.Manifold.batch_boolean([_manifold_m3(mesh) for mesh in meshes_tm], op_m3)
    result_wp = function([_upload(mesh, device) for mesh in meshes_tm])
    _assert_same_as_m3(result_wp, result_m3)


def test_batch_booleans_reject_an_empty_list() -> None:
    """No mesh at all raises ``ValueError``. Not a library comparison: the guard."""
    with pytest.raises(ValueError, match="at least one mesh"):
        od.boolean.union_all([])


@pytest.mark.parity("segment_hits", "manifold3d")
@pytest.mark.parametrize("shape", ["box", "sphere_irregular", "torus"])
def test_segment_hits_match_manifold3d(
    request: pytest.FixtureRequest, device: str, shape: str
) -> None:
    """
    Each segment's crossings are Manifold's ``ray_cast``: faces, parameters and points.

    Class A, after one named transform: Manifold numbers faces in its internal order, ordito in
    the caller's, so each hit's face is compared as the triangle it names. Segments through the
    box's vertices, edges and face centres are the ties; the rest are random, crossing and
    missing. The reference is non-vacuous: some segments cross, some miss.
    """
    if shape == "sphere_irregular":
        mesh_tm = _float32_mesh(request.getfixturevalue("sphere_irregular")[0])
    elif shape == "box":
        mesh_tm = _box([0, 0, 0], [1, 1, 1])
    else:
        mesh_tm = _float32_mesh(tm.creation.torus(1.0, 0.3, 24, 12))
    rng = np.random.default_rng(13)
    centre = mesh_tm.vertices.mean(axis=0)
    span = float(np.max(mesh_tm.extents))
    origins_np = (centre + rng.uniform(-span, span, (64, 3))).astype(np.float32)
    endpoints_np = (centre + rng.uniform(-span, span, (64, 3))).astype(np.float32)
    origins_np[:4] = [[0.5, 0.5, -1.0], [0.0, 0.0, -1.0], [0.5, 0.0, -1.0], [1.0, 1.0, 2.0]]
    endpoints_np[:4] = [[0.5, 0.5, 2.0], [0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [1.0, 1.0, -1.0]]
    mesh_m3 = _manifold_m3(mesh_tm)
    mesh_out_m3 = mesh_m3.to_mesh()
    vertices_m3 = np.asarray(mesh_out_m3.vert_properties)[:, :3]
    faces_m3 = np.asarray(mesh_out_m3.tri_verts)
    hit_faces, parameters, points, _, offsets = od.boolean.segment_hits(
        *_upload(mesh_tm, device),
        wp.array(origins_np, dtype=wp.vec3, device=device),
        wp.array(endpoints_np, dtype=wp.vec3, device=device),
    )
    offsets_np = offsets.numpy()
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh_tm.faces)
    n_crossing = 0
    for s in range(len(origins_np)):
        hits_m3 = mesh_m3.ray_cast(tuple(origins_np[s].tolist()), tuple(endpoints_np[s].tolist()))
        hits_m3 = sorted(hits_m3, key=lambda hit: hit.distance)
        ours = range(int(offsets_np[s]), int(offsets_np[s + 1]))
        assert len(ours) == len(hits_m3)
        n_crossing += len(hits_m3) > 0
        for hit_m3, k in zip(hits_m3, ours, strict=True):
            triangle_m3 = sorted(map(tuple, vertices_m3[faces_m3[hit_m3.face_id]].tolist()))
            triangle_np = sorted(map(tuple, vertices_np[faces_np[hit_faces.numpy()[k]]].tolist()))
            assert triangle_np == triangle_m3
            assert parameters.numpy()[k] == hit_m3.distance
            assert np.array_equal(points.numpy()[k], np.asarray(hit_m3.position, dtype=np.float32))
    assert 0 < n_crossing < len(origins_np)


# Offsets along z for the second sphere: overlapping, near, nearer the search length, out of reach.
_GAP_OFFSETS = [0.0, 2.0, 2.5, 4.0]


@pytest.mark.parity("min_gap", "manifold3d")
@pytest.mark.parametrize("seed", range(len(_GAP_OFFSETS)))
def test_min_gap_matches_manifold3d(device: str, seed: int) -> None:
    """
    The gap between two closed meshes is Manifold's ``min_gap``, to the last bit.

    Class A on the scalar. Seeds place the second sphere overlapping (gap 0), near (inside the
    search length) and far (the search length itself), so every branch is compared.
    """
    a_tm = _float32_mesh(tm.creation.icosphere(subdivisions=2))
    b_tm = _moved_copy(a_tm, seed=seed)
    b_tm.apply_translation([0.0, 0.0, _GAP_OFFSETS[seed]])
    b_tm = _float32_mesh(b_tm)
    search = 1.0
    gap_m3 = _manifold_m3(a_tm).min_gap(_manifold_m3(b_tm), search)
    gap = od.boolean.min_gap(*_upload(a_tm, device), *_upload(b_tm, device), search)
    assert gap == gap_m3


def test_min_gap_spans_its_branches() -> None:
    """The seeds above reach all three answers. Not a library comparison: their coverage."""
    a_tm = _float32_mesh(tm.creation.icosphere(subdivisions=2))
    gaps_m3 = []
    for seed, offset in enumerate(_GAP_OFFSETS):
        b_tm = _moved_copy(a_tm, seed=seed)
        b_tm.apply_translation([0.0, 0.0, offset])
        gaps_m3.append(_manifold_m3(a_tm).min_gap(_manifold_m3(_float32_mesh(b_tm)), 1.0))
    assert 0.0 in gaps_m3
    assert 1.0 in gaps_m3
    assert any(0.0 < gap < 1.0 for gap in gaps_m3)


@pytest.mark.parity("simplify", "manifold3d")
@pytest.mark.parametrize(
    ("shape", "tolerance"),
    [("subdivided_box", 0.0), ("sphere", 0.01), ("sphere", 0.05), ("sphere_irregular", 0.0)],
)
def test_simplify_matches_manifold3d(
    request: pytest.FixtureRequest, device: str, shape: str, tolerance: float
) -> None:
    """
    ``simplify`` is Manifold's ``Manifold.simplify``, array for array.

    Class A. A subdivided box collapses to its twelve triangles at the mesh's own precision;
    an icosphere loses the vertices within the tolerance of a plane through their neighbours;
    ``sphere_irregular``'s needles stay where nothing is collinear.
    """
    if shape == "subdivided_box":
        mesh_tm = _box([0, 0, 0], [1, 1, 1]).subdivide().subdivide()
    elif shape == "sphere":
        mesh_tm = _float32_mesh(tm.creation.icosphere(subdivisions=3))
    else:
        mesh_tm = _float32_mesh(request.getfixturevalue("sphere_irregular")[0])
    result_m3 = _manifold_m3(mesh_tm).simplify(tolerance)
    if shape != "sphere_irregular":
        assert result_m3.num_vert() < len(mesh_tm.vertices)
    _assert_same_as_m3(od.boolean.simplify(*_upload(mesh_tm, device), tolerance), result_m3)


def _without_repeats(loop: np.ndarray, tolerance: float = 1e-7) -> np.ndarray:
    """Return a loop with its consecutive coincident points (and a repeated closing one) dropped."""
    kept: list[np.ndarray] = []
    for point in loop:
        if kept and np.linalg.norm(point - kept[-1]) <= tolerance:
            continue
        kept.append(point)
    while len(kept) > 1 and np.linalg.norm(kept[0] - kept[-1]) <= tolerance:
        kept.pop()
    return np.asarray(kept)


def _same_loops(loops: list[np.ndarray], loops_m3: list[np.ndarray]) -> bool:
    """Whether two sets of loops match one to one, each up to its starting point."""
    if len(loops) != len(loops_m3):
        return False
    used = [False] * len(loops_m3)
    for loop in loops:
        for j, loop_m3 in enumerate(loops_m3):
            if used[j] or len(loop) != len(loop_m3):
                continue
            if any(
                np.allclose(np.roll(loop_m3, -shift, axis=0), loop, atol=1e-7, rtol=0)
                for shift in range(len(loop_m3))
            ):
                used[j] = True
                break
        else:
            return False
    return True


@pytest.mark.parity("cross_section", "manifold3d")
@pytest.mark.parity("cross_section_with_offsets", "manifold3d")
@pytest.mark.parametrize("shape", ["sphere", "box", "torus", "holed_box"])
@pytest.mark.parametrize("height", [-0.5, 0.0, 0.1, 0.25, 0.5, 0.999, 1.0, 2.0])
def test_cross_section_matches_manifold3d(device: str, shape: str, height: float) -> None:
    """
    The section's loops are Manifold's ``slice``, each up to where it starts.

    Class B, after one named transform: ``manifold3d`` hands the loops through a ``CrossSection``,
    whose cleanup drops a point repeated at a vertex the plane passes through (ordito keeps the C++
    ``Slice``'s loops, a point a face); so consecutive coincident points are dropped before the
    loops are matched, one to one, up to rotation, at ``1e-7``. Heights below, through vertices
    (``0``, ``1``, the bottom face and the top), between, and above; the holed box has two loops.
    """
    mesh_tm = {
        "sphere": _float32_mesh(tm.creation.icosphere(subdivisions=3)),
        "box": _box([0, 0, 0], [1, 1, 1]),
        "torus": _float32_mesh(tm.creation.torus(1.0, 0.3, 24, 12)),
        "holed_box": _manifold_mesh_tm(
            _manifold_m3(_box([0, 0, 0], [1, 1, 1]))
            - _manifold_m3(_box([0.25, 0.25, -1], [0.75, 0.75, 2]))
        ),
    }[shape]
    loops_m3 = [np.asarray(loop) for loop in _manifold_m3(mesh_tm).slice(height).to_polygons()]
    loops = [loop.numpy() for loop in od.boolean.cross_section(*_upload(mesh_tm, device), height)]
    assert _same_loops([_without_repeats(loop) for loop in loops], loops_m3)
    points, offsets = od.boolean.cross_section_with_offsets(*_upload(mesh_tm, device), height)
    assert np.array_equal(points.numpy(), np.concatenate(loops) if loops else np.zeros((0, 2)))
    assert offsets.numpy()[-1] == sum(len(loop) for loop in loops)


def _manifold_mesh_tm(result_m3: m3.Manifold) -> tm.Trimesh:
    """Return a Manifold as a ``float32`` trimesh mesh."""
    mesh_m3 = result_m3.to_mesh()
    return tm.Trimesh(
        np.asarray(mesh_m3.vert_properties)[:, :3], np.asarray(mesh_m3.tri_verts), process=False
    )


@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_with_an_empty_operand(device: str, name: str, operation: Operation) -> None:
    """
    An empty operand leaves the other mesh, or nothing, as a solid's set algebra says.

    Not a library comparison: Manifold builds no solid from an empty mesh to compare with, so the
    cases are checked against their definitions: the union is the other operand, the
    intersection empty, the difference the first operand (or empty when that is the empty one).
    """
    box_tm = _box([0, 0, 0], [1, 1, 1])
    vertices, faces = _upload(box_tm, device)
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
    box_wp = _upload(_box([0, 0, 0], [1, 1, 1]), mesh_wp.device)
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
        od.boolean.union(*_upload(box_tm, "cpu"), *_upload(box_tm, "cuda:0"))
