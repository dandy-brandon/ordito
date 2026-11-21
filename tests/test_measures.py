"""
Regression tests for ``ordito.measures``: the whole-mesh reductions.

Against ``trimesh``'s cached mass properties, ``igl.moments`` and pymeshlab's
``get_geometric_measures``, which answers six of these questions in one call. Mirrors the module's
source order -- volume, surface centroid, moments, Euler characteristic.
"""

from __future__ import annotations

import igl
import manifold3d as m3
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from tests.conftest import MESHES
from tests.conversions import (
    faces_igl,
    numpy_to_warp,
    trimesh_to_manifold3d,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    trimesh_to_warp,
)


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "sphere_irregular_hollow"])
@pytest.mark.parity("moments", "meshlib", "pyvista")
def test_volume(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: the divergence-theorem volume against trimesh, MeshLib and pyvista.

    The three references are ``Trimesh.volume``, MeshLib's ``volume`` and ``PolyData.volume``.

    A closed mesh is required for the integral to mean anything, which is why every fixture is
    closed and the open-mesh behaviour is not asserted here. ``cave_cube`` is the fixture that makes
    the *sign* convention matter: it is a shell whose inner box is wound inwards, so a volume taken
    without signs would report the outer box alone.

    The parity markers name the ``moments`` group rather than a group of their own, because that is
    where the integral family is timed. VTK computes the volume as this module does -- a signed sum
    of per-face tetrahedra -- so no transform applies and the agreement is float32-limited
    (measured 2.5e-07 on ``icosahedron``, where ordito reads 2.536150455 against 2.536150710 and
    trimesh's float64 answer is 2.536150710). MeshLib's ``volume`` sits in the group exactly as
    pyvista's does -- the volume alone, no centre of mass and no inertia tensor. Its ``region``
    argument defaults to the whole mesh, which is the comparison; ordito has no per-region form, so
    nothing is left untested by passing ``None``.

    Reversing every face must negate the volume, with trimesh agreeing on the sign: the *signed*
    result is the claim -- an implementation taking the absolute value would pass the first asserts
    and fail that one, and ``validation.is_volume`` depends on the sign.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    volume_pv = float(trimesh_to_pyvista(mesh_tm).volume)
    assert volume_pv > 0.0

    volume_wp = od.measures.volume(mesh_wp.points, mesh_wp.indices)
    assert np.isclose(volume_wp, mesh_tm.volume, rtol=1e-5, atol=1e-5)
    assert np.isclose(volume_wp, mm.volume(mesh_ml.topology, mesh_ml.points), rtol=1e-5, atol=1e-5)
    assert np.isclose(volume_wp, volume_pv, rtol=1e-5)
    assert np.isclose(volume_pv, mesh_tm.volume, rtol=1e-5)

    faces_flipped_np = np.ascontiguousarray(mesh_tm.faces[:, ::-1].reshape(-1), dtype=np.int32)
    faces_flipped_wp = wp.array(faces_flipped_np, dtype=wp.int32, device=mesh_wp.device)
    volume_flipped_wp = od.measures.volume(mesh_wp.points, faces_flipped_wp)
    assert np.isclose(volume_flipped_wp, -mesh_tm.volume, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "sphere_irregular_hollow"])
@pytest.mark.parity(
    "volume",
    "manifold3d",
    benchmarked=False,
    reason=(
        "the volume group sweeps the open scan meshes, every one of which Manifold refuses as "
        "not closed, so a manifold3d row there would skip on every mesh it is given"
    ),
)
def test_volume_matches_manifold3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: Manifold's ``volume()`` of the same solid.

    Manifold reads the same ``float32`` coordinates, so the two divergence-theorem sums differ
    by rounding alone: measured 2.7e-8 and 2.2e-8 relative, bounded at ``1e-6``. The hollow
    fixture is two shells, the inner one wound inward, which Manifold accepts as one solid: its
    volume is the outer body less the cavity, so a volume taken without signs fails.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    solid_m3 = trimesh_to_manifold3d(mesh_tm)
    assert solid_m3.status() == m3.Error.NoError
    volume_m3 = solid_m3.volume()
    assert volume_m3 > 0.0  # non-vacuity
    assert np.isclose(od.measures.volume(mesh_wp.points, mesh_wp.indices), volume_m3, rtol=1e-6)


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
def test_volume_and_moments_on_a_large_mesh(device: str, deterministic: bool) -> None:
    """
    Class A: volume and moments of a 1.3M-face sphere against trimesh, on both commits.

    The fixtures above fit in a few blocks; this mesh spans 1280 of them, where one atomic commit
    per block (determinism off) added the block sums in arrival order and moved the last
    bits from call to call on CUDA. Both arms must match trimesh's float64 integrals, and the
    fixed-order arm must return the same bits on a second call.
    """
    vertices_wp, faces_wp = od.creation.icosphere(8, device=device)
    mesh_tm = tm.Trimesh(vertices_wp.numpy(), faces_wp.numpy().reshape(-1, 3), process=False)
    volume_wp = od.measures.volume(vertices_wp, faces_wp)
    volume_m, center_wp, inertia_wp = od.measures.moments(vertices_wp, faces_wp)
    assert np.isclose(volume_wp, mesh_tm.volume, rtol=1e-5)
    assert np.isclose(volume_m, mesh_tm.volume, rtol=1e-5)
    assert np.allclose(odt.vec3_floats(center_wp), mesh_tm.center_mass, rtol=1e-5, atol=1e-5)
    scale = float(np.abs(mesh_tm.moment_inertia).max())
    assert (
        np.abs(np.asarray(inertia_wp).reshape(3, 3) - mesh_tm.moment_inertia).max() < 1e-5 * scale
    )
    if deterministic:
        assert od.measures.volume(vertices_wp, faces_wp) == volume_wp
        assert od.measures.moments(vertices_wp, faces_wp)[0] == volume_m


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("surface_centroid", "trimesh", "meshlib", "pymeshlab")
def test_surface_centroid(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A against trimesh and MeshLib, Class B against MeshLab: the area-weighted centroid.

    The references are ``Trimesh.centroid``, MeshLib's ``findCenterFromFaces`` and MeshLab's
    ``shell_barycenter``.

    An open fixture, because on a closed symmetric mesh the centroid sits at the origin and a
    wrongly-weighted sum lands there too -- and this fixture is translated as well, so a function
    returning the origin fails rather than passing by luck.

    MeshLib carries **two** centres and picking the wrong one would still pass a symmetric fixture:
    ``findCenterFromFaces`` is the area-weighted surface centroid ordito computes, while
    ``findCenterFromPoints`` is the plain vertex mean that [`centroid`][ordito.points.centroid]
    computes. The final assert holds them apart -- measured 2.63 apart on ``half_torus`` -- so a
    future rebinding to the other one cannot pass quietly.

    MeshLab's ``get_geometric_measures`` answers six questions in one call -- which is why
    ``benchmarks/test_triangles.py`` reads its row as an *upper* bound on the centroid alone -- but
    "upper bound" is a statement about cost, not about the value. Indexing the dict is the whole
    transform. The same call also returns ``barycenter``, the plain *vertex* mean; asserting against
    ``shell_barycenter`` while checking that ``barycenter`` is measurably different is what makes
    the comparison a test of the area weighting: on ``hemisphere`` they sit about 0.02 apart, some
    400x the 1e-5 tolerance.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_geometric_measures()

    centroid_tm = mesh_tm.centroid
    centroid_ml = np.array([*mm.findCenterFromFaces(mesh_ml.topology, mesh_ml.points)])

    centroid_wp = od.measures.surface_centroid(mesh_wp.points, mesh_wp.indices)
    centroid_wp = np.array(odt.vec3_floats(centroid_wp))
    assert np.allclose(centroid_wp, centroid_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(centroid_wp, centroid_ml, rtol=1e-5, atol=1e-5)
    assert np.allclose(centroid_wp, measures_pml["shell_barycenter"], rtol=1e-5, atol=1e-5)
    assert not np.allclose(measures_pml["barycenter"], measures_pml["shell_barycenter"], atol=1e-3)

    # The vertex mean is a different point, and is ordito's points.centroid rather than this.
    vertex_mean_ml = np.array([*mm.findCenterFromPoints(mesh_ml.topology, mesh_ml.points)])
    assert not np.allclose(centroid_wp, vertex_mean_ml, atol=1e-3)
    assert np.allclose(
        od.points.centroid(mesh_wp.points).numpy()[0], vertex_mean_ml, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("kernel_device", ["cpu", "cuda:0"])
def test_surface_centroid_matches_trimesh_on_a_skewed_mesh_on_both_devices(kernel_device: str):
    """
    Class A on both devices: the area-weighted centroid sum on an asymmetric mesh.

    Three things this guards. ``wp.launch_tiled`` runs exactly one lane per block on Warp 1.19's CPU
    backend, so the block-wide ``wp.tile_sum`` this reduction used to perform accumulated one face
    per 64-face tile there. A *symmetric* mesh hides that completely -- the centroid of every 64th
    face of a sphere is still the sphere's centre -- which is why the mesh is stretched and sheared
    first. Measured: the sub-sampled sum was off by 1.1e-2 on this shape and by 4e-8 on the
    unmodified sphere. The reduction is one kernel on both devices now (``surface_centroid_totals``,
    lanes striding by ``wp.block_dim()``), and the parametrization covers both, the single-lane CPU
    block being the case the old bug lived in.
    """
    if kernel_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    mesh_tm = tm.creation.icosphere(subdivisions=3)
    mesh_tm.vertices[:, 0] *= 6.0
    mesh_tm.vertices[:, 2] += 0.4 * mesh_tm.vertices[:, 1] ** 3
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)

    centroid_wp = od.measures.surface_centroid(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(
        np.array(odt.vec3_floats(centroid_wp)), mesh_tm.centroid, rtol=1e-5, atol=1e-5
    )


def test_surface_centroid_is_reproducible(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    """
    Ordito against ordito: the same mesh gives the same centroid bits on every call.

    The oracle for the value is ``test_surface_centroid``; this pins repeatability. One float
    atomic per block summed the blocks in arrival order, and on CUDA the result moved in its last
    bits on every call: 10 distinct centroids in 10 calls on every scan mesh. The per-block partials
    are folded in a fixed order now (the deterministic setting, on for every test).
    """
    _, mesh_wp = sphere_irregular
    centroids = {
        tuple(odt.vec3_floats(od.measures.surface_centroid(mesh_wp.points, mesh_wp.indices)))
        for _ in range(10)
    }
    assert len(centroids) == 1


def test_surface_centroid_all_degenerate(device: str):
    """
    Not a library comparison: the other input whose area weights sum to zero.

    An area-weighted mean of nothing is the empty mesh's answer, and a mesh of collinear faces is
    the same statement with faces present -- it used to be a bare ``ZeroDivisionError``, which the
    docstring documented nowhere and which no other reduction here raises (``moments`` already
    guards its zero volume the same way).
    """
    vertices_wp, faces_wp = numpy_to_warp(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        device,
    )

    # Non-vacuity: every face really is degenerate, so the weights really do all vanish.
    _, areas_wp = od.triangles.face_normals_and_areas(vertices_wp, faces_wp)
    assert float(areas_wp.numpy().max()) == 0.0
    centroid_wp = od.measures.surface_centroid(vertices_wp, faces_wp)
    assert np.isnan(odt.vec3_floats(centroid_wp)).all()


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "sphere_irregular_hollow"])
@pytest.mark.parity("moments", "igl", "trimesh")
def test_moments(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B on igl (its first moment is un-normalised), Class A on trimesh.

    ``igl.moments`` returns ``(m0, m1, m2)`` where ``m1`` is the centre of mass **times the mass**,
    so the named transform is ``m1 / m0``. ``m2`` needs no transform: it is already referred to the
    centre of mass rather than the origin, which was verified against a *translated* mesh rather
    than assumed -- on a mesh centred at the origin the two references coincide and the check would
    be vacuous.

    trimesh answers all three too (``volume`` / ``center_mass`` / ``moment_inertia``), so this is a
    three-way comparison of the same integrals.

    The inertia tolerance is relative to the tensor's own scale: ordito integrates ``float32``
    positions in ``float64``, so the deviation tracks the positions' precision (measured 3.5e-8
    relative on ``icosahedron``, and 3.6e-15 on an axis-aligned box whose coordinates are exact in
    ``float32``).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)

    volume_igl, first_moment_igl, inertia_igl = igl.moments(vertices_np, faces_igl(mesh_tm))
    volume_wp, center_wp, inertia_wp = od.measures.moments(mesh_wp.points, mesh_wp.indices)

    assert np.isclose(volume_wp, volume_igl, rtol=1e-5)
    assert np.isclose(volume_wp, mesh_tm.volume, rtol=1e-5)
    assert np.allclose(
        odt.vec3_floats(center_wp), np.asarray(first_moment_igl) / volume_igl, rtol=1e-5, atol=1e-5
    )
    assert np.allclose(odt.vec3_floats(center_wp), mesh_tm.center_mass, rtol=1e-5, atol=1e-5)
    scale = float(np.abs(np.asarray(inertia_igl)).max())
    assert (
        np.abs(np.asarray(inertia_wp).reshape(3, 3) - np.asarray(inertia_igl)).max() < 1e-5 * scale
    )
    assert (
        np.abs(np.asarray(inertia_wp).reshape(3, 3) - mesh_tm.moment_inertia).max() < 1e-5 * scale
    )


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "sphere_irregular_hollow"])
@pytest.mark.parity(
    "moments",
    "pymeshlab",
    benchmarked=False,
    reason=(
        "get_geometric_measures omits mesh_volume, center_of_mass and inertia_tensor on a mesh "
        "that is not watertight, and every scan mesh the moments group sweeps is open, so a row "
        "would time a call that integrates nothing"
    ),
)
def test_moments_match_pymeshlab(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: MeshLab's ``mesh_volume``, ``center_of_mass`` and ``inertia_tensor``.

    ``get_geometric_measures`` returns the three integrals among its keys on a watertight mesh
    (it omits them on an open one), indexed directly: the inertia tensor is already referred to
    the centre of mass, ordito's convention. The fixtures sit off the origin and are rotated, so
    an inertia about the origin or a centre at it fails, and the hollow fixture's inward shell
    makes the signed integrals matter. The tolerances follow
    ``test_moments``: the inertia relative to the tensor's own scale.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_geometric_measures()
    inertia_pml = np.asarray(measures_pml["inertia_tensor"])
    center_pml = np.asarray(measures_pml["center_of_mass"])
    assert np.abs(center_pml).max() > 1e-2  # non-vacuity: the centre is not the origin

    volume_wp, center_wp, inertia_wp = od.measures.moments(mesh_wp.points, mesh_wp.indices)
    assert np.isclose(volume_wp, measures_pml["mesh_volume"], rtol=1e-5)
    assert np.allclose(odt.vec3_floats(center_wp), center_pml, rtol=1e-5, atol=1e-5)
    scale = float(np.abs(inertia_pml).max())
    assert np.abs(np.asarray(inertia_wp).reshape(3, 3) - inertia_pml).max() < 1e-5 * scale


def test_empty_mesh(device: str):
    """
    Not a library comparison: every reduction's answer on a mesh with no faces.

    The volume of nothing is zero; the two centres are area- or volume-weighted means of nothing,
    so ``NaN``; the inertia tensor is the zero matrix.
    """
    faces_wp = wp.array([], dtype=wp.int32, device=device)
    one_vertex_wp = wp.zeros(1, dtype=wp.vec3, device=device)
    no_vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)

    assert od.measures.volume(one_vertex_wp, faces_wp) == 0.0
    centroid_wp = od.measures.surface_centroid(one_vertex_wp, faces_wp)
    assert np.isnan(odt.vec3_floats(centroid_wp)).all()
    volume, center, inertia = od.measures.moments(no_vertices_wp, faces_wp)
    assert volume == 0.0
    assert np.isnan(odt.vec3_floats(center)).all()
    assert np.array_equal(np.asarray(inertia).reshape(3, 3), np.zeros((3, 3)))


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("euler_characteristic", "trimesh")
def test_euler_characteristic(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on an integer, against ``Trimesh.euler_number`` across the fixture set.

    Non-vacuous by construction: the parametrisation spans chi = 2 and 1, so a constant answer
    cannot pass (the igl and Open3D comparison below spans five values). A given ``n_vertices``
    (the readback skipped) gives the same count.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    euler_wp = od.measures.euler_characteristic(mesh_wp.indices)
    assert euler_wp == int(mesh_tm.euler_number)
    euler_bound_wp = od.measures.euler_characteristic(
        mesh_wp.indices, n_vertices=mesh_wp.points.size
    )
    assert euler_bound_wp == int(mesh_tm.euler_number)


_TOPOLOGY_MESHES = [
    "sphere_irregular",
    "saddle_graded",
    "torus_irregular",
    "sphere_irregular_hollow",
    "torus_irregular_holes",
    "boy_surface",
    "mobius",
]
"""Fixtures spanning chi = 2, 1, 0, 4 and -3: closed, open, genus 1, two shells, three holes."""


@pytest.mark.parametrize("mesh_name", _TOPOLOGY_MESHES)
@pytest.mark.parity("euler_characteristic", "igl", "open3d")
def test_euler_characteristic_matches_igl_and_open3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on an integer: igl's ``euler_characteristic`` and Open3D's Euler-Poincare count.

    igl counts ``F.max() + 1`` vertices and Open3D every stored vertex, where ordito counts the
    referenced ones; on these fixtures, which reference every vertex (asserted), the three counts
    are one number. The parametrisation spans five values of chi, so a constant answer cannot pass.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.asarray(mesh_tm.faces)
    assert np.unique(faces_np).shape[0] == faces_np.max() + 1 == len(mesh_tm.vertices)

    euler_igl = igl.euler_characteristic(faces_igl(mesh_tm))
    euler_o3d = trimesh_to_open3d(mesh_tm).euler_poincare_characteristic()
    euler_wp = od.measures.euler_characteristic(mesh_wp.indices)
    assert euler_wp == euler_igl
    assert euler_wp == euler_o3d


@pytest.mark.parametrize("mesh_name", [*_TOPOLOGY_MESHES, "non_manifold_fin"])
@pytest.mark.parity("euler_characteristic", "pymeshlab")
def test_euler_characteristic_matches_pymeshlab(
    request: pytest.FixtureRequest, device: str, mesh_name: str
) -> None:
    """
    Class B (``V - E + F`` from MeshLab's counts): ``get_topological_measures``.

    MeshLab reports ``vertices_number``, ``edges_number`` and ``faces_number`` rather than chi, so
    the named transform is the alternating sum. The last case is not edge-manifold: a closed
    tetrahedron with a fin, a fourth face on one of its edges, so that edge counts once though
    three faces share it. MeshLab's ``is_mesh_two_manifold`` must read ``False`` there and
    ``validation.is_edge_manifold`` agree, which keeps the edge count honest on input where a
    halfedge-pairing count and a unique-edge count part.
    """
    if mesh_name == "non_manifold_fin":
        vertices_np = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, -1.0]]
        )
        faces_np = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2], [0, 1, 4]])
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        faces_wp = wp.array(faces_np.reshape(-1).astype(np.int32), device=device)
    else:
        mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
        faces_wp = mesh_wp.indices

    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_topological_measures()
    euler_pml = (
        int(measures_pml["vertices_number"])
        - int(measures_pml["edges_number"])
        + int(measures_pml["faces_number"])
    )
    euler_wp = od.measures.euler_characteristic(faces_wp)
    assert euler_wp == euler_pml
    assert euler_wp == int(mesh_tm.euler_number)
    manifold_pml = bool(measures_pml["is_mesh_two_manifold"])
    assert manifold_pml == (mesh_name != "non_manifold_fin")
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True) == manifold_pml


@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "torus_irregular", "sphere_irregular_hollow"]
)
@pytest.mark.parity(
    "euler_characteristic",
    "manifold3d",
    benchmarked=False,
    reason=(
        "Manifold.genus() reads vertex, edge and triangle counts the Manifold constructor already "
        "holds, so a timed row would price a cached lookup and not the count"
    ),
)
def test_euler_characteristic_matches_manifold3d_genus(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (``chi = 2 - 2 * genus``): Manifold's ``genus()`` on closed manifold solids.

    Manifold reports ``genus = 1 - chi / 2`` over the whole solid, components summed, so the named
    transform is ``2 - 2 * genus``: 2 for the sphere, 0 for the torus and 4 for the hollow sphere,
    whose genus reads -1. Manifold accepts only closed, oriented, manifold input, so the open and
    non-orientable fixtures stay with the other references.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    solid_m3 = trimesh_to_manifold3d(mesh_tm)
    assert solid_m3.status() == m3.Error.NoError
    euler_m3 = 2 - 2 * solid_m3.genus()
    assert od.measures.euler_characteristic(mesh_wp.indices) == euler_m3


def test_euler_characteristic_on_tiny_random_face_sets(device: str) -> None:
    """
    Class A on an integer, against ``Trimesh.euler_number`` on tiny random face sets.

    Faces over eight vertices, repeated and degenerate edges and unreferenced vertices included,
    keep the edge count's open-addressing set nearly as full as its sizing allows, where every
    probe sequence must still reach an empty slot (``tests/test_edges.py`` has the same guard for
    ``unique_edge_count``). The answers spread over several values, so a constant cannot pass.
    """
    rng = np.random.default_rng(11)
    answers: set[int] = set()
    for n_faces in range(1, 13):
        for _ in range(20):
            faces_np = rng.integers(0, 8, size=(n_faces, 3))
            mesh_tm = tm.Trimesh(np.zeros((8, 3)), faces_np, process=False)
            faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
            euler_wp = od.measures.euler_characteristic(faces_wp, n_vertices=8)
            assert euler_wp == int(mesh_tm.euler_number)
            answers.add(euler_wp)
    assert len(answers) > 3


def test_euler_characteristic_rejects_a_non_positive_vertex_count(device: str) -> None:
    """Not a library comparison: a vertex count of zero cannot hold a face's corners."""
    faces_wp = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="n_vertices must be positive, got 0"):
        od.measures.euler_characteristic(faces_wp, n_vertices=0)
