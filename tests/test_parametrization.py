from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from typing import Any, cast

import igl
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from tests.comparisons import sparse_allclose
from tests.conftest import saddle_graded_arrays
from tests.conversions import (
    mesh_igl,
    numpy_to_warp,
    numpy_to_warp_uv,
    points_to_warp_uv,
    trimesh_to_pymeshlab,
    warp_empty,
)


def _face_flipped_indices_np(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """NumPy reference for libigl ``flipped_triangles``: 2D signed area strictly negative."""
    tri = vertices_np[faces_np]  # (n_faces, 3, 2)
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    signed_area2 = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    return np.flatnonzero(signed_area2 < 0.0).astype(np.int64)


def _random_2d_mesh(rng: np.random.Generator, n_faces: int):
    """Random 2D triangle soup with mixed orientations as (vertices_2d, flat_faces)."""
    vertices_np = rng.standard_normal((n_faces * 3, 2)).astype(np.float64)
    faces_np = np.arange(n_faces * 3, dtype=np.int64).reshape(n_faces, 3)
    return vertices_np, faces_np


def _circle_boundary(mesh_wp: wp.Mesh) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    """Return the longest boundary loop and its arc-length map onto the unit circle."""
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    return boundary_wp, od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)


def _harmonic_warm_start(
    mesh_wp: wp.Mesh,
) -> tuple[wp.array[wp.int32], wp.array[wp.vec2], wp.array[wp.vec2]]:
    """Return the circle boundary plus the harmonic map it pins, ARAP's warm start."""
    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    uv_init_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    return boundary_wp, boundary_uv_wp, uv_init_wp


def _two_rim_pins(mesh_wp: wp.Mesh) -> tuple[np.ndarray, wp.array[wp.int32]]:
    """Two pins half a loop apart on the longest rim, as ``int32`` host and device indices."""
    loop_np = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices).numpy()
    pins_np = np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int32)
    return pins_np, wp.array(pins_np, dtype=wp.int32, device=mesh_wp.device)


def _unit_quad(device: str) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Return the two-triangle unit square in the ``z = 0`` plane."""
    vertices = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        dtype=wp.vec3,
        device=device,
    )
    faces = wp.array(np.array([0, 1, 2, 1, 3, 2], dtype=np.int32), dtype=wp.int32, device=device)
    return vertices, faces


def test_flipped_faces_random_mixed(device: str):
    rng = np.random.default_rng(0)
    vertices_np, faces_np = _random_2d_mesh(rng, n_faces=64)
    vertices_wp, faces_wp = numpy_to_warp_uv(vertices_np, faces_np, device)

    tri = vertices_np[faces_np]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    mask_np = (e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) < 0.0

    assert np.array_equal(
        od.parametrization.face_flipped_mask(vertices_wp, faces_wp).numpy(), mask_np
    )
    assert np.array_equal(
        od.parametrization.face_flipped_indices(vertices_wp, faces_wp).numpy(),
        _face_flipped_indices_np(vertices_np, faces_np),
    )


_SINGLE_TRIANGLE_NP = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)


@pytest.mark.parametrize(
    ("vertices_np", "faces_np", "expected_np"),
    [
        (_SINGLE_TRIANGLE_NP, np.array([[0, 1, 2]]), np.array([False])),
        (_SINGLE_TRIANGLE_NP, np.array([[2, 1, 0]]), np.array([True])),
        # Zero area: the strict ``< 0`` means a collinear triangle is not flagged.
        (np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), np.array([[0, 1, 2]]), np.array([False])),
        (np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=bool)),
    ],
    ids=["ccw", "cw", "collinear", "empty"],
)
def test_flipped_faces_on_known_windings(
    device: str, vertices_np: np.ndarray, faces_np: np.ndarray, expected_np: np.ndarray
):
    """Not a library comparison: the winding of each triangle is known by construction."""
    vertices_wp, faces_wp = numpy_to_warp_uv(vertices_np, faces_np, device)
    assert np.array_equal(
        od.parametrization.face_flipped_mask(vertices_wp, faces_wp).numpy(), expected_np
    )
    assert np.array_equal(
        od.parametrization.face_flipped_indices(vertices_wp, faces_wp).numpy(),
        np.flatnonzero(expected_np),
    )


@pytest.mark.usefixtures("deterministic")
@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("map_vertices_to_circle", "igl")
def test_map_vertices_to_circle_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """Class A: the arc-length circle map against ``igl.map_vertices_to_circle``, no transform."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, _ = mesh_igl(mesh_tm)

    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_np = boundary_wp.numpy().astype(np.int64)

    circle_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    circle_igl = igl.map_vertices_to_circle(vertices_np, boundary_np)

    assert np.allclose(circle_wp.numpy(), circle_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
@pytest.mark.parametrize("n_loop", [4_096, 4_097, 100_000])
def test_map_vertices_to_circle_on_a_long_loop(device: str, deterministic: bool, n_loop: int):
    """
    Class A: long loops against ``igl.map_vertices_to_circle``, no transform.

    The fixtures' loops take the one-block form; on CUDA a loop of 4 097 vertices or more takes the
    device-wide scan instead (4 096 is the last on the block), the 100k-vertex loop spanning dozens
    of chunks of both the deterministic scan and CUB's (``False``), and the deterministic map must
    repeat bit for bit. The CPU takes the block at every length. The tolerance is ``1e-4``: the
    float32 arc length over 100k edges carries ~1e-5 relative rounding into the angle, against
    libigl's float64 sum. Re-measured: 3.0e-05 on CPU (both settings, so ``1e-5`` fails there),
    1.1e-06 / 2.3e-06 on CUDA (deterministic / CUB).
    """
    t = np.sort(np.random.default_rng(6).random(n_loop)) * 2.0 * np.pi
    vertices_np = np.stack([np.cos(t) * (1.0 + 0.3 * np.sin(5.0 * t)), np.sin(t), 0.0 * t], 1)
    boundary_np = np.arange(t.shape[0], dtype=np.int64)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    boundary_wp = wp.array(boundary_np.astype(np.int32), dtype=wp.int32, device=device)
    circle_np = od.parametrization.map_vertices_to_circle(vertices_wp, boundary_wp).numpy()
    circle_igl = igl.map_vertices_to_circle(
        vertices_np.astype(np.float32).astype(np.float64), boundary_np
    )
    assert np.allclose(circle_np, circle_igl, rtol=1e-4, atol=1e-4)
    if deterministic:
        assert np.array_equal(
            od.parametrization.map_vertices_to_circle(vertices_wp, boundary_wp).numpy(), circle_np
        )


@pytest.mark.parametrize("n_loop", [0, 1])
def test_map_vertices_to_circle_degenerate_loop(device: str, n_loop: int):
    """
    Not a library comparison: an empty or single-vertex loop maps without a raise or a NaN.

    An empty loop maps to an empty ``wp.vec2`` buffer; a single-vertex loop has zero perimeter,
    and the arc-length map must not divide 0/0 into NaN.
    """
    vertices_wp = wp.array(np.eye(3, dtype=np.float32), dtype=wp.vec3, device=device)
    boundary_wp = wp.array(np.arange(n_loop, dtype=np.int32), dtype=wp.int32, device=device)
    circle_wp = od.parametrization.map_vertices_to_circle(vertices_wp, boundary_wp)
    assert circle_wp.dtype == wp.vec2
    assert circle_wp.shape == (n_loop,)
    assert np.isfinite(circle_wp.numpy()).all()


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity(
    "graph_laplacian",
    "igl",
    benchmarked=False,
    reason="libigl builds A - diag(rowsum(A)) inline inside igl::harmonic rather than binding "
    "it, so the reference here is igl.adjacency_matrix plus a NumPy diagonal -- timing that "
    "would price a hand-rolled composition under a library's name. scipy binds the operation "
    "outright (csgraph.laplacian, measured an exact sign-flipped match at 0.0 on icosphere(2)) "
    "but takes an adjacency matrix, which ordito's face buffer is not, so its row would time "
    "the same composition one step earlier. benchmarks/test_laplacian.py carries the decline.",
)
def test_graph_laplacian_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B: igl has no ``graph_laplacian``, so the reference is assembled from its adjacency.

    The named transform is ``A - diag(rowsum(A))`` over ``igl.adjacency_matrix``: the definition of
    the umbrella operator, built on the reference side. Note igl's adjacency is sized by ``F.max() +
    1`` rather than ``len(V)``, which is why it runs on fixtures where every
    vertex is referenced.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, faces_np = mesh_igl(mesh_tm)
    n_vertices = mesh_wp.points.size

    operator_wp = od.laplacian.graph_laplacian(mesh_wp.points, mesh_wp.indices)
    operator_csr = scipy.sparse.csr_matrix(
        (operator_wp.values.numpy(), operator_wp.columns.numpy(), operator_wp.offsets.numpy()),
        shape=(n_vertices, n_vertices),
    )

    adjacency_igl = igl.adjacency_matrix(faces_np).astype(np.float64)
    laplacian_igl = adjacency_igl - scipy.sparse.diags(
        np.asarray(adjacency_igl.sum(axis=1)).ravel()
    )

    assert sparse_allclose(operator_csr, laplacian_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("harmonic", "igl")
@pytest.mark.parity("harmonic_conditioning", "igl")
def test_harmonic_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: the same boundary constraints in, the same interior UVs out.

    Both sides receive the identical circle map, so nothing about the boundary is under test here --
    the comparison isolates the interior solve. igl factorizes where ordito iterates and verifies;
    measured 1.3e-07 at worst on both fixtures and devices, so the standard tolerance holds.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)

    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    uv_igl = igl.harmonic(vertices_np, faces_np, boundary_np, boundary_uv_np, 1)

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_cap"])
def test_harmonic_with_nonnegative_weights_solves_igls_clamped_laplacian(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: the Dirichlet solve of igl's cotangent Laplacian with negative weights set to zero.

    The reference clamps ``igl.cotmatrix_entries`` (the named transform), assembles the operator
    in SciPy and solves its interior rows with SciPy's direct solver, with the same circle map
    pinned. Each free vertex of that map is a convex combination of its neighbors, so it lies
    strictly inside the disk and no face folds; on these obtuse fixtures it differs from the
    signed harmonic map (by 0.026 and 0.34 measured), so the comparison is not the plain map's.
    A `Trimesh` keeps the two systems apart: a signed call after a clamped direct one still
    answers the signed map.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)
    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    boundary_np = boundary_wp.numpy()

    clamped_igl = np.maximum(igl.cotmatrix_entries(vertices_np, faces_np), 0.0)
    n_vertices = vertices_np.shape[0]
    rows = np.concatenate([faces_np[:, (c + 1) % 3] for c in range(3)])
    cols = np.concatenate([faces_np[:, (c + 2) % 3] for c in range(3)])
    weights = np.concatenate([clamped_igl[:, c] for c in range(3)])
    off_diagonal = scipy.sparse.coo_matrix((weights, (rows, cols)), shape=(n_vertices,) * 2)
    off_diagonal = (off_diagonal + off_diagonal.T).tocsr()
    laplacian = (
        off_diagonal - scipy.sparse.diags(np.asarray(off_diagonal.sum(axis=1)).ravel())
    ).tocsr()
    free = np.setdiff1d(np.arange(n_vertices), boundary_np)
    uv_np = np.zeros((n_vertices, 2))
    uv_np[boundary_np] = boundary_uv_wp.numpy()
    rhs = -laplacian[free][:, boundary_np] @ uv_np[boundary_np]
    uv_np[free] = scipy.sparse.linalg.spsolve(laplacian[free][:, free].tocsc(), rhs)

    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    uv_wp = od.parametrization.harmonic(
        mesh, boundary_wp, boundary_uv_wp, nonnegative_weights=True, solver="direct"
    )
    assert np.allclose(uv_wp.numpy(), uv_np, rtol=1e-5, atol=1e-5)
    assert np.linalg.norm(uv_wp.numpy()[free], axis=1).max() < 1.0
    assert od.parametrization.face_flipped_indices(uv_wp, mesh_wp.indices).size == 0

    signed_wp = od.parametrization.harmonic(mesh, boundary_wp, boundary_uv_wp)
    signed_np = np.asarray(
        igl.harmonic(vertices_np, faces_np, boundary_np.astype(np.int64), uv_np[boundary_np], 1)
    )
    assert np.allclose(signed_wp.numpy(), signed_np, rtol=1e-5, atol=1e-5)
    assert np.abs(signed_np - uv_np).max() > 1e-2


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("harmonic", "pymeshlab")
@pytest.mark.parity("harmonic_conditioning", "pymeshlab")
def test_harmonic_matches_pymeshlab(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: MeshLab's harmonic parametrization is ordito's circle-pinned harmonic map.

    ``compute_texcoord_parametrization_harmonic`` wraps libigl (MeshLab says so), so this re-checks
    igl through a second wrapper rather than adding an independent implementation; it pins what
    MeshLab chooses for the caller. It takes the longest boundary loop from its smallest vertex,
    maps it onto the unit circle by arc length and solves the Laplacian (``harm_function`` is a
    no-op in pymeshlab 2025.7), which are ``longest_boundary_loop`` and ``map_vertices_to_circle``:
    measured 3e-8 on ``saddle_graded``. The per-vertex texture coordinates are compared directly.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_texcoord_parametrization_harmonic()
    uv_pml = meshset_pml.current_mesh().vertex_tex_coord_matrix()
    assert np.ptp(uv_pml[:, 1]) > 1.0

    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    uv_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )

    assert np.allclose(uv_wp.numpy(), uv_pml, rtol=1e-5, atol=1e-5)


def _polyharmonic_reference(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    boundary_np: np.ndarray,
    boundary_uv_np: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    ``harmonic(k)`` solved directly in SciPy over igl's cotangent and barycentric mass matrices.

    ``Q = -L (M^-1 (-L))^(k - 1)``, the interior block factored by ``spsolve``. igl.harmonic's own
    default mass is Voronoi, where ordito uses the barycentric lumped mass, so the reference is
    assembled here rather than taken from ``igl.harmonic`` at ``k > 1``.
    """
    laplacian = igl.cotmatrix(vertices_np, faces_np)
    mass_inv = scipy.sparse.diags(
        1.0
        / np.asarray(
            igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
        )
    )
    product = -laplacian
    for _ in range(k - 1):
        product = product @ mass_inv @ (-laplacian)
    # scipy-stubs type the product as ``_spbase | ArrayND``, neither of which it gives ``tocsc``.
    operator = scipy.sparse.csr_matrix(product)
    n_vertices = vertices_np.shape[0]
    interior = np.setdiff1d(np.arange(n_vertices), boundary_np)
    solution = scipy.sparse.linalg.spsolve(
        operator[interior][:, interior].tocsc(),
        -(operator[interior][:, boundary_np] @ boundary_uv_np),
    )
    uv_ref = np.zeros((n_vertices, 2))
    uv_ref[boundary_np] = boundary_uv_np
    uv_ref[interior] = solution
    return uv_ref


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("mesh_name", ["sphere_irregular_cap", "sphere_irregular_band"])
def test_polyharmonic_matches_reference(request: pytest.FixtureRequest, mesh_name: str, k: int):
    """
    Class B: igl's operators with the barycentric mass, solved directly.

    The named transform is the mass: igl.harmonic's default is Voronoi. ``k == 2`` agrees to
    7.0e-08 and holds the standard ``1e-5``; ``k == 3`` reaches 6.8e-05 (re-measured 2026-10-26:
    CPU 4.4e-05 ``cap`` / 6.8e-05 ``band``, CUDA 9.5e-06 / 8.9e-06; the 1.3e-04 once recorded on
    CUDA predates its direct solve) because its system squares the biharmonic one's conditioning,
    which igl flags as not numerically robust, so it keeps ``1e-4``. ``k == 3`` is solved by a
    factorization; its conjugate gradient used to stop at its tolerance 1e-5 to 1e-2 of the range
    away from the answer on these fixtures.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)

    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=k
    )
    uv_ref = _polyharmonic_reference(vertices_np, faces_np, boundary_np, boundary_uv_np, k)

    tolerance = 1e-5 if k == 2 else 1e-4
    assert np.allclose(uv_wp.numpy(), uv_ref, rtol=tolerance, atol=tolerance)


def test_biharmonic_on_a_hub_matches_a_direct_solve(device: str):
    """
    Class B against ``_polyharmonic_reference`` on a planar disk fan whose centre is a hub.

    The hub's 1 100-entry Laplacian row is long (``kernels/array.CSR_LONG_ROW_ENTRIES``), so on
    CUDA the squared-Laplacian polynomial steps it by pieces, and the sandwich assembling the
    biharmonic system writes a 1.2 M-triplet row; 1 101 free rows keep the solve off the
    one-block path. The named transform is the mass, as in ``test_polyharmonic_matches_reference``.
    """
    spokes, rings = 1100, 2
    points = [np.zeros(3)]
    for ring in range(1, rings + 1):
        angles = 2.0 * np.pi * (np.arange(spokes) + 0.5 * (ring % 2)) / spokes
        points += list(
            np.stack([ring * np.cos(angles), ring * np.sin(angles), np.zeros(spokes)], axis=1)
        )
    vertices_np = np.array(points, dtype=np.float32).astype(np.float64)
    spoke = np.arange(spokes)
    following = (spoke + 1) % spokes
    inner, outer = 1 + spoke, 1 + spokes + spoke
    faces_np = np.concatenate(
        [
            np.stack([np.zeros(spokes, dtype=np.int64), 1 + spoke, 1 + following], axis=1),
            np.stack([inner, outer, 1 + spokes + following], axis=1),
            np.stack([inner, 1 + spokes + following, 1 + following], axis=1),
        ]
    )
    boundary_np = 1 + spokes + spoke
    boundary_uv_np = vertices_np[boundary_np, :2] / rings
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    uv_wp = od.parametrization.harmonic(
        vertices_wp,
        faces_wp,
        wp.array(boundary_np.astype(np.int32), dtype=wp.int32, device=device),
        wp.array(boundary_uv_np.astype(np.float32), dtype=wp.vec2, device=device),
        k=2,
    )
    uv_ref = _polyharmonic_reference(vertices_np, faces_np, boundary_np, boundary_uv_np, 2)
    assert np.ptp(uv_ref[: 1 + spokes]) > 1.0
    assert np.allclose(uv_wp.numpy(), uv_ref, rtol=1e-5, atol=1e-5)


def test_biharmonic_on_a_graded_patch_matches_a_direct_solve(device: str):
    """
    Class B against ``_polyharmonic_reference`` on the graded saddle, at 2e-4 of the UV range.

    The graded patch's biharmonic system is past what conjugate gradient reaches in ``float64``: it
    stalled for its whole iteration cap and returned a map 0.76 of the range away on this fixture
    (0.29 on the 68 x 68 one, 0.54 on ``benchmarks``' 133 x 133), with no warning. The solve is
    verified now and falls back to a factorization: measured 3.5e-05 (CUDA) / 5.8e-05 (CPU) of the
    range, so the bound is 3x the worse device and 3 800x under the defect. On the benchmark mesh
    the answer is itself ambiguous at the percent level (direct solves under four SuperLU orderings
    disagree by 0.6 to 2.2 %), which is why this runs on the smaller fixture.
    """
    vertices_np, faces_np = saddle_graded_arrays()
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(vertices_np), np.asarray(faces_np, dtype=np.int32).ravel(), device
    )
    boundary_wp = od.boundary.longest_boundary_loop(vertices_wp, faces_wp)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(vertices_wp, boundary_wp)

    uv_wp = od.parametrization.harmonic(vertices_wp, faces_wp, boundary_wp, boundary_uv_wp, k=2)
    uv_ref = _polyharmonic_reference(
        vertices_wp.numpy().astype(np.float64),
        np.asarray(faces_np, dtype=np.int64),
        boundary_wp.numpy().astype(np.int64),
        boundary_uv_wp.numpy().astype(np.float64),
        2,
    )

    assert np.ptp(uv_ref) > 1.0
    assert np.abs(uv_wp.numpy() - uv_ref).max() < 2e-4 * np.ptp(uv_ref)


def test_biharmonic_is_deterministic(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # Regression guard for operator-assembly nondeterminism: the float64-native biharmonic operator
    # must give the same result across repeated calls. The original defect sized a rebuild's triplet
    # buffers by ``BsrMatrix.nnz`` (the capacity, not the entry count) and so fed the uninitialized
    # tail to ``bsr_from_triplets``, manifesting as ~1e22 / NaN corruption; a tight tolerance (well
    # above conjugate-gradient's ~1e-8 atomic last-ULP jitter) reliably catches a regression. Two
    # repeats: the corruption is gross, and each repeat is a full graded-mesh biharmonic solve.
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    first = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
    ).numpy()
    for _ in range(2):
        again = od.parametrization.harmonic(
            mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
        ).numpy()
        assert np.allclose(again, first, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("tutte", "igl")
def test_tutte_matches_igl_reference(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B: igl has no ``tutte``, so the reference is its fixed-value minimizer on ``D - A``.

    Two named transforms, both on the reference side: assemble the uniform Laplacian from
    ``igl.adjacency_matrix`` and hand it to ``igl.min_quad_with_fixed`` under ordito's own boundary
    constraints. That isolates the interior solve, which is the only part not already covered by
    [`test_graph_laplacian_matches_igl`] and [`test_map_vertices_to_circle_matches_igl`].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, faces_np = mesh_igl(mesh_tm)
    n_vertices = mesh_wp.points.size

    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.tutte(mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp)

    # Reference: uniform Laplacian D - A solved with igl's fixed-value quadratic minimizer, given
    # the identical boundary constraints, so the comparison isolates the interior solve.
    # igl's binding types the matrix ``csc_matrix[int]``, which scipy-stubs' ``sum`` rejects.
    adjacency = cast("scipy.sparse.csc_matrix[np.int64]", igl.adjacency_matrix(faces_np))
    laplacian = (
        scipy.sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)) - adjacency
    ).tocsc()
    if laplacian.shape[0] != n_vertices:
        laplacian.resize((n_vertices, n_vertices))  # pyright: ignore[reportArgumentType]  # stubs: Never
    uv_igl = np.asarray(
        igl.min_quad_with_fixed(
            laplacian, np.zeros((n_vertices, 2), np.float64), boundary_np, boundary_uv_np
        )
    )

    # Measured 3.8e-08 at worst, both devices.
    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-5, atol=1e-5)


def test_tutte_disk_is_fold_free(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # A disk-topology mesh with a convex (circle) boundary yields a bijective, fold-free Tutte map.
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    uv_wp = od.parametrization.tutte(mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp)
    assert od.parametrization.face_flipped_indices(uv_wp, mesh_wp.indices).numpy().size == 0


def _arap_igl(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    fixed_np: np.ndarray,
    fixed_uv_np: np.ndarray,
    uv_init_np: np.ndarray,
    max_iterations: int,
):
    """Libigl ARAP reference (dim=2, elements energy); returns the (n, 2) UV."""
    data = igl.ARAPData()
    data.max_iter = max_iterations
    igl.arap_precomputation(vertices_np, faces_np, 2, fixed_np.astype(np.int32), data)
    return igl.arap_solve(
        fixed_uv_np.astype(np.float64), data, np.ascontiguousarray(uv_init_np.astype(np.float64))
    )


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("arap", "igl")
def test_arap_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A at a fixed iteration count, with the *same warm start* fed to both sides.

    ARAP is a local-global iteration, so its answer depends on where it started and how many rounds
    it ran: both are pinned here (ordito's own harmonic solve, 10 iterations), or the comparison
    would be measuring two different points along two different trajectories.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)

    # Full boundary loop pinned to the unit circle; identical harmonic warm start fed to both sides.
    boundary_wp, boundary_uv_wp, uv_init_wp = _harmonic_warm_start(mesh_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)
    uv_init_np = uv_init_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp, max_iterations=10
    )
    uv_igl = _arap_igl(vertices_np, faces_np, boundary_np, boundary_uv_np, uv_init_np, 10)

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-5, atol=1e-5)


def test_arap_free_boundary_matches_igl(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the free-boundary branch: only two vertices pinned, the rest of the rim moving.

    The fully-pinned case above cannot see a bug in the boundary rows of the system, since there are
    none to solve; this is the same comparison with 4 iterations and almost the whole rim free.
    """
    # Pin only two boundary vertices to their harmonic UV; the rest of the boundary is free.
    mesh_tm, mesh_wp = saddle_graded
    vertices_np, faces_np = mesh_igl(mesh_tm)

    _, _, uv_init_wp = _harmonic_warm_start(mesh_wp)
    uv_init_np = uv_init_wp.numpy().astype(np.float64)

    fixed_np, fixed_wp = _two_rim_pins(mesh_wp)
    fixed_uv_np = uv_init_np[fixed_np]
    fixed_uv_wp = points_to_warp_uv(fixed_uv_np, mesh_wp.device)

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, fixed_wp, fixed_uv_wp, uv_init_wp, max_iterations=4
    )
    uv_igl = _arap_igl(vertices_np, faces_np, fixed_np, fixed_uv_np, uv_init_np, 4)

    # float32-UV drift vs igl's float64 grows slowly; four iterations stay inside the standard 1e-5
    # on both devices.
    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-5, atol=1e-5)


def test_arap_default_tolerance_tracks_a_tight_solve(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # ``arap`` defaults its inner CG to 1e-7 rather than the 1e-8 the other solvers use: its global
    # solves are inner steps of a truncated outer iteration. Guard that the looser default still
    # tracks a tight solve two orders below it, at the 1e-5 gate the igl oracles use. The
    # pinned rows must equal the prescribed UV exactly (they are re-enforced every iteration).
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp, uv_init_wp = _harmonic_warm_start(mesh_wp)

    uv_default_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp, max_iterations=10
    )
    uv_tight_wp = od.parametrization.arap(
        mesh_wp.points,
        mesh_wp.indices,
        boundary_wp,
        boundary_uv_wp,
        uv_init_wp,
        max_iterations=10,
        tolerance=1e-9,
    )
    assert np.allclose(uv_default_wp.numpy(), uv_tight_wp.numpy(), rtol=1e-5, atol=1e-5)
    assert np.array_equal(uv_default_wp.numpy()[boundary_wp.numpy()], boundary_uv_wp.numpy())


def test_arap_factored_global_step_tracks_the_iteration(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
):
    """
    Ordito against ordito: the two routes of ``arap``'s global step, factored and iterated.

    ``arap`` factors its operator once its second solve proves long; only the iterated route has
    the igl oracle above for every fixture, so the factored one is pinned to it here, with the gate
    forced each way and the factored route checked to have run.
    """
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp, uv_init_wp = _harmonic_warm_start(mesh_wp)
    factored: list[bool] = []
    build = od.parametrization._arap_factorization  # pyright: ignore[reportPrivateUsage]

    def counted(*args: Any) -> object:
        held = build(*args)
        factored.append(held is not None)
        return held

    monkeypatch.setattr(od.parametrization, "_arap_factorization", counted)
    uv_np = {}
    for threshold in (2**62, 0):
        monkeypatch.setattr(od.parametrization, "_ARAP_FACTOR_FROM_ROUNDS_CUDA", threshold)
        monkeypatch.setattr(od.parametrization, "_ARAP_FACTOR_FROM_ROUNDS_CPU", threshold)
        uv_np[threshold] = od.parametrization.arap(
            mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp
        ).numpy()
    assert factored == [True]
    assert np.allclose(uv_np[0], uv_np[2**62], rtol=1e-5, atol=1e-5)
    assert np.array_equal(uv_np[0][boundary_wp.numpy()], boundary_uv_wp.numpy())


def test_arap_all_vertices_fixed(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # Every vertex pinned: the prescribed UV is returned with no solve, so this runs on CPU too.
    _, mesh_wp = saddle_graded
    n_vertices = mesh_wp.points.size
    rng = np.random.default_rng(7)
    fixed_uv_np = rng.standard_normal((n_vertices, 2)).astype(np.float32)
    all_indices_np = np.arange(n_vertices, dtype=np.int32)
    fixed_wp = wp.array(all_indices_np, dtype=wp.int32, device=mesh_wp.device)
    fixed_uv_wp = points_to_warp_uv(fixed_uv_np, mesh_wp.device)
    uv_init_wp = wp.zeros(n_vertices, dtype=wp.vec2, device=mesh_wp.device)

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, fixed_wp, fixed_uv_wp, uv_init_wp, max_iterations=10
    )
    assert np.array_equal(uv_wp.numpy(), fixed_uv_np)


def test_arap_is_reproducible_when_deterministic(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    """
    Not a parity assert: under the deterministic setting repeated calls agree bit for bit.

    Each local step scatters ``float64`` rotation terms onto the edge endpoints, which atomics add
    in arrival order on CUDA; the deterministic path adds each vertex's terms in ascending face and
    edge order. Without it repeated CUDA calls on this fixture differed in the last bits of the UV.
    The off mode is checked against the on mode to round-off, not to the bit.
    """
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp, uv_init_wp = _harmonic_warm_start(mesh_wp)
    arguments = (mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp)
    uv_np = od.parametrization.arap(*arguments).numpy()
    for _ in range(4):
        assert np.array_equal(od.parametrization.arap(*arguments).numpy(), uv_np)
    with od.config.deterministic(False):
        arrival_wp = od.parametrization.arap(*arguments)
    assert np.allclose(arrival_wp.numpy(), uv_np, rtol=1e-5, atol=1e-5)


def test_arap_pattern_changes_no_answer(saddle_graded: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Ordito against ordito: ``pattern=`` only spares the cotangent assembly its sort.

    The faces' ``"cotmatrix"`` pattern must give the plain call's UV bit for bit.
    """
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp, uv_init_wp = _harmonic_warm_start(mesh_wp)
    arguments = (mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp)
    pattern = od.laplacian.mesh_operator_pattern(mesh_wp.indices, mesh_wp.points.size)

    plain_np = od.parametrization.arap(*arguments).numpy()
    assert np.array_equal(od.parametrization.arap(*arguments, pattern=pattern).numpy(), plain_np)


def _signed_chart_areas(uv_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """Every face's signed area in the chart."""
    a = uv_np[faces_np[:, 1]] - uv_np[faces_np[:, 0]]
    b = uv_np[faces_np[:, 2]] - uv_np[faces_np[:, 0]]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _area_scaled_tutte(mesh_wp: wp.Mesh) -> wp.array[wp.vec2]:
    """Return a Tutte disk on the unit circle scaled to the surface's area: a fold-free start."""
    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    disk_np = (
        od.parametrization.tutte(mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp)
        .numpy()
        .astype(np.float64)
    )
    faces_np = mesh_wp.indices.numpy().reshape(-1, 3)
    surface = float(
        od.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)[1].numpy().sum()
    )
    scale = np.sqrt(surface / np.abs(_signed_chart_areas(disk_np, faces_np)).sum())
    return points_to_warp_uv(disk_np * scale, mesh_wp.device)


def _symmetric_dirichlet(
    vertices_np: np.ndarray, faces_np: np.ndarray, uv_np: np.ndarray
) -> np.ndarray:
    """Every face's ``s1^2 + s2^2 + s1^-2 + s2^-2`` (``inf`` once inverted), times its area."""
    first = vertices_np[faces_np[:, 1]] - vertices_np[faces_np[:, 0]]
    second = vertices_np[faces_np[:, 2]] - vertices_np[faces_np[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(first, second), axis=1)
    length = np.linalg.norm(first, axis=1)
    along, height = (second * first).sum(1) / length, 2 * areas / length
    a = uv_np[faces_np[:, 1]] - uv_np[faces_np[:, 0]]
    b = uv_np[faces_np[:, 2]] - uv_np[faces_np[:, 0]]
    jacobian = np.stack(
        (a / length[:, None], (b - a * (along / length)[:, None]) / height[:, None]), axis=2
    )
    sigma = np.linalg.svd(jacobian, compute_uv=False)
    energy = (sigma**2).sum(1) + (sigma**-2.0).sum(1)
    return np.where(np.linalg.det(jacobian) > 0, areas * energy, np.inf)


def _slim_igl(
    vertices_np: np.ndarray, faces_np: np.ndarray, uv_init_np: np.ndarray, iterations: int
) -> np.ndarray:
    """``igl::slim`` with ``SYMMETRIC_DIRICHLET`` and no constraints, from ``uv_init_np``."""
    data = igl.slim_precompute(
        np.asfortranarray(vertices_np, dtype=np.float64),
        np.asfortranarray(faces_np, dtype=np.int32),
        np.asfortranarray(uv_init_np, dtype=np.float64),
        igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
        np.zeros(0, dtype=np.int32),
        np.zeros((0, 2), dtype=np.float64, order="F"),
        0.0,
    )
    return np.asarray(igl.slim_solve(data, iterations))[:, :2]


def _rigidly_aligned(source_np: np.ndarray, target_np: np.ndarray) -> np.ndarray:
    """Return ``source_np`` moved by the rigid motion that best fits it to ``target_np``."""
    source_mean, target_mean = source_np.mean(0), target_np.mean(0)
    u, _, vt = np.linalg.svd((source_np - source_mean).T @ (target_np - target_mean))
    rotation = u @ np.diag([1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return (source_np - source_mean) @ rotation + target_mean


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_cap"])
@pytest.mark.parity("slim", "igl")
def test_slim_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B: both converge from the same fold-free start to one chart, up to a rigid motion.

    The start is a Tutte disk scaled to the surface's area, the same ``float32`` values on both
    sides. The energy does not see the chart's rotation and position, which nothing pins, so the
    two trajectories (libigl refactors every iteration, ordito every eight) leave the converged
    chart in slightly different places (a rotation of ~3e-3 on ``saddle_graded``): ordito's chart
    is rigidly aligned to libigl's, the gauge fix, and compared at the standard tolerance. Both
    run to a standstill. The start is far from the answer (the comparison is not vacuous), and
    neither chart inverts a face.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)
    uv_init_wp = _area_scaled_tutte(mesh_wp)
    uv_init_np = uv_init_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.slim(
        mesh_wp.points, mesh_wp.indices, uv_init_wp, max_iterations=400, tolerance=0.0
    )
    uv_igl = _slim_igl(vertices_np, faces_np, uv_init_np, 300)

    assert np.abs(uv_igl - uv_init_np).max() > 1e-2
    assert (_signed_chart_areas(uv_igl, faces_np) > 0).all()
    assert (_signed_chart_areas(uv_wp.numpy().astype(np.float64), faces_np) > 0).all()
    aligned_np = _rigidly_aligned(uv_wp.numpy().astype(np.float64), uv_igl)
    assert np.allclose(aligned_np, uv_igl, rtol=1e-5, atol=1e-5)


def test_slim_lays_a_stretched_flat_sheet_out_isometrically(device: str):
    """
    Not a library comparison: a flat sheet's least symmetric Dirichlet chart is an isometry.

    Started from the sheet sheared and stretched (every face ``|log s|`` up to ~0.45), the chart
    must come back with every side its surface length: the energy's unique minimum up to a rigid
    motion. A chart left anywhere short of it keeps sides off by the shear's few percent.
    """
    vertices_np, faces_np = _sheet(24, 3.0, 0.0)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    stretched_np = vertices_np[:, :2] @ np.array([[1.5, 0.4], [0.0, 0.7]])
    start_wp = points_to_warp_uv(stretched_np, device)

    uv_np = (
        od.parametrization.slim(vertices_wp, faces_wp, start_wp, max_iterations=200, tolerance=0.0)
        .numpy()
        .astype(np.float64)
    )

    lengths_np = _edge_lengths(vertices_np, faces_np)
    assert np.abs(_edge_lengths(stretched_np, faces_np) / lengths_np - 1).max() > 0.3
    assert np.allclose(_edge_lengths(uv_np, faces_np), lengths_np, rtol=1e-5, atol=1e-5)


def test_slim_keeps_every_face_the_start_had_the_right_way_round(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
):
    """
    Not a library comparison: the rule for faces inverted in the start is ordito's own.

    Six interior vertices of a fold-free start are pushed past a neighbour, inverting the faces
    around them. ``slim`` must invert none of the others, may right the folded ones (they take the
    unbarriered ARAP term), and must lower the energy of the faces it barriers by a tenth.
    """
    mesh_tm, mesh_wp = saddle_graded
    vertices_np, faces_np = mesh_igl(mesh_tm)
    start_np = _area_scaled_tutte(mesh_wp).numpy().astype(np.float64)
    boundary_np = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices).numpy()
    interior_np = np.setdiff1d(np.arange(len(start_np)), boundary_np)
    rng = np.random.default_rng(3)
    for v in rng.choice(interior_np, size=6, replace=False):
        neighbour = faces_np[np.flatnonzero((faces_np == v).any(1))[0]]
        other = neighbour[neighbour != v][0]
        start_np[v] += 2.5 * (start_np[other] - start_np[v])
    start_np = start_np.astype(np.float32).astype(np.float64)
    inverted_before = _signed_chart_areas(start_np, faces_np) <= 0
    assert 6 <= inverted_before.sum()

    uv_np = (
        od.parametrization.slim(
            mesh_wp.points, mesh_wp.indices, points_to_warp_uv(start_np, mesh_wp.device)
        )
        .numpy()
        .astype(np.float64)
    )

    inverted_after = _signed_chart_areas(uv_np, faces_np) <= 0
    assert not (inverted_after & ~inverted_before).any()
    upright = ~inverted_before
    assert (
        _symmetric_dirichlet(vertices_np, faces_np[upright], uv_np).sum()
        < 0.9 * _symmetric_dirichlet(vertices_np, faces_np[upright], start_np).sum()
    )


def test_slim_moves_a_start_with_faces_squashed_nearly_flat(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
):
    """
    Not a library comparison: a start whose faces are crushed but not inverted still descends.

    Six interior vertices of a fold-free start are moved 99.9999 % of the way to a neighbour, so
    the faces around them keep their orientation but are squashed a millionfold, as a Tutte map
    of a surface with a handle squashes some. Barriered with the energy's own gradient against a
    capped stiffness, such a face's step bound held every step to ~1e-16 and the chart stood
    still; either the collapsed-face rule or the capped gradient keeps it moving, and the chart's
    energy must fall by orders of magnitude.
    """
    mesh_tm, mesh_wp = saddle_graded
    vertices_np, faces_np = mesh_igl(mesh_tm)
    start_np = _area_scaled_tutte(mesh_wp).numpy().astype(np.float64)
    boundary_np = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices).numpy()
    interior_np = np.setdiff1d(np.arange(len(start_np)), boundary_np)
    for v in np.random.default_rng(5).choice(interior_np, size=6, replace=False):
        neighbour = faces_np[np.flatnonzero((faces_np == v).any(1))[0]]
        other = neighbour[neighbour != v][0]
        start_np[v] += 0.999999 * (start_np[other] - start_np[v])
    assert (_signed_chart_areas(start_np.astype(np.float32).astype(np.float64), faces_np) > 0).all()
    start_wp = points_to_warp_uv(start_np, mesh_wp.device)

    uv_np = (
        od.parametrization.slim(mesh_wp.points, mesh_wp.indices, start_wp)
        .numpy()
        .astype(np.float64)
    )

    start_energy = _symmetric_dirichlet(vertices_np, faces_np, start_wp.numpy().astype(np.float64))
    assert np.abs(uv_np - start_np).max() > 1e-2
    assert _symmetric_dirichlet(vertices_np, faces_np, uv_np).sum() < 1e-3 * start_energy.sum()


def test_slim_opens_a_band_the_start_squashes_flat(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh],
):
    """
    Not a library comparison: faces the start squashes past ``SLIM_COLLAPSED`` are unbarriered.

    A fold-free Tutte start whose middle band is compressed a hundred-thousandfold across its
    width, the rest shifted to stay attached: every face keeps its orientation, as when a Tutte
    map of a jaw scan crushes a whole arm of the arch. Barriered, those faces' capped stiffness
    lets each step crush them while their bound cuts the step to nothing: the chart moved 0.025
    and its energy stayed ~5e9 (CPU and CUDA). Held like inverted faces they open up, and the
    chart reaches the energy a clean Tutte start reaches (95.4; 95.8 here).
    """
    mesh_tm, mesh_wp = sphere_irregular_cap
    vertices_np, faces_np = mesh_igl(mesh_tm)
    start_np = _area_scaled_tutte(mesh_wp).numpy().astype(np.float64)
    x = start_np[:, 0]
    centre = np.median(x)
    band = np.abs(x - centre) < 0.15 * np.ptp(x)
    low, high = x[band].min(), x[band].max()
    squash = 1e-5
    start_np[:, 0] = np.where(
        band,
        centre + (x - centre) * squash,
        np.where(x > centre, x - (high - centre) * (1 - squash), x + (centre - low) * (1 - squash)),
    )
    start_wp = points_to_warp_uv(start_np, mesh_wp.device)
    start_used = start_wp.numpy().astype(np.float64)
    assert (_signed_chart_areas(start_used, faces_np) > 0).all()

    uv_np = (
        od.parametrization.slim(mesh_wp.points, mesh_wp.indices, start_wp)
        .numpy()
        .astype(np.float64)
    )

    energy = _symmetric_dirichlet(vertices_np, faces_np, uv_np)
    assert _symmetric_dirichlet(vertices_np, faces_np, start_used).sum() > 1e11
    assert energy[np.isfinite(energy)].sum() < 200.0


def test_slim_returns_a_clockwise_start_with_its_winding(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    """
    Ordito against ordito: a mirrored start is refined as the start and mirrored back.

    The barrier holds every face's orientation, so a clockwise start is refined mirrored; the
    mirror is exact, so the two charts are each other's reflections bit for bit.
    """
    _, mesh_wp = saddle_graded
    start_np = _area_scaled_tutte(mesh_wp).numpy()
    mirror = np.array([1.0, -1.0], dtype=np.float32)
    arguments = (mesh_wp.points, mesh_wp.indices)

    uv_np = od.parametrization.slim(*arguments, points_to_warp_uv(start_np, mesh_wp.device)).numpy()
    mirrored_np = od.parametrization.slim(
        *arguments, points_to_warp_uv(start_np * mirror, mesh_wp.device)
    ).numpy()

    assert not np.array_equal(uv_np, start_np)
    assert np.array_equal(mirrored_np * mirror, uv_np)


def test_slim_stops_once_eight_iterations_lower_the_energy_by_less_than_the_tolerance(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
):
    """
    Ordito against ordito: the stop test reads the energy every eight iterations.

    A tolerance no decrease meets stops after the first eight, which is the chart of a budget of
    eight; with no tolerance the iterations run on and move the chart further.
    """
    _, mesh_wp = saddle_graded
    arguments = (mesh_wp.points, mesh_wp.indices, _area_scaled_tutte(mesh_wp))

    stopped_np = od.parametrization.slim(*arguments, tolerance=1e9).numpy()
    eight_np = od.parametrization.slim(*arguments, max_iterations=8, tolerance=0.0).numpy()
    sixteen_np = od.parametrization.slim(*arguments, max_iterations=16, tolerance=0.0).numpy()

    assert np.array_equal(stopped_np, eight_np)
    assert not np.array_equal(sixteen_np, eight_np)


def test_slim_is_reproducible_when_deterministic(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    """
    Not a parity assert: under the deterministic setting repeated calls agree bit for bit.

    The trial energies are block sums, whose arrival order decides the line search's comparisons
    on CUDA with the setting off; the chart then agrees with the deterministic one to round-off.
    """
    _, mesh_wp = saddle_graded
    arguments = (mesh_wp.points, mesh_wp.indices, _area_scaled_tutte(mesh_wp))

    uv_np = od.parametrization.slim(*arguments).numpy()
    for _ in range(3):
        assert np.array_equal(od.parametrization.slim(*arguments).numpy(), uv_np)
    with od.config.deterministic(False):
        arrival_wp = od.parametrization.slim(*arguments)
    assert np.allclose(arrival_wp.numpy(), uv_np, rtol=1e-5, atol=1e-5)


def test_slim_cpu_matches_cuda():
    """
    Class A: the CPU chart is the CUDA one.

    They differ only in floating-point contraction and summation order, below the ``float32``
    chart's resolution.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")
    vertices_np, faces_np = _sheet(30, 3.0, 1.5)
    stretched_np = vertices_np[:, :2] @ np.array([[1.2, 0.3], [0.0, 0.9]])

    uv = {}
    for device in ("cpu", "cuda:0"):
        vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
        uv[device] = od.parametrization.slim(
            vertices_wp, faces_wp, points_to_warp_uv(stretched_np, device)
        ).numpy()

    assert np.isfinite(uv["cpu"]).all()
    assert np.allclose(uv["cpu"], uv["cuda:0"], rtol=0, atol=1e-5)


def test_slim_with_no_iterations_returns_its_start(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    """Not a library comparison: a budget of zero iterations returns the start unchanged."""
    _, mesh_wp = saddle_graded
    start_wp = _area_scaled_tutte(mesh_wp)

    uv_wp = od.parametrization.slim(mesh_wp.points, mesh_wp.indices, start_wp, max_iterations=0)

    assert uv_wp.ptr != start_wp.ptr
    assert np.array_equal(uv_wp.numpy(), start_wp.numpy())


def test_slim_on_an_empty_mesh(device: str):
    """Not a library comparison: no vertices give an empty chart rather than a raise."""
    uv_wp = od.parametrization.slim(
        warp_empty(0, wp.vec3, device),
        warp_empty(0, wp.int32, device),
        warp_empty(0, wp.vec2, device),
    )

    assert uv_wp.shape == (0,)
    assert uv_wp.device == wp.get_device(device)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("plane_vertices", "wp.vec3 vertices"),
        ("float_faces", "wp.vec3 vertices"),
        ("ragged_faces", "wp.vec3 vertices"),
        ("space_uv", "wp.vec3 vertices"),
        ("short_uv", "wp.vec3 vertices"),
        ("negative_iterations", "max_iterations"),
        ("negative_tolerance", "tolerance"),
        ("nan_tolerance", "tolerance"),
    ],
)
def test_slim_rejects_malformed_inputs(device: str, case: str, message: str):
    """Not a library comparison: each malformed input raises a ``ValueError`` naming it."""
    vertices = wp.array(np.eye(3, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    uv_init = wp.array(np.eye(3, 2, dtype=np.float32), dtype=wp.vec2, device=device)
    iterations, tolerance = 100, 1e-3
    if case == "plane_vertices":
        vertices = wp.zeros(3, dtype=wp.vec2, device=device)
    elif case == "float_faces":
        faces = wp.array([0.0, 1.0, 2.0], dtype=wp.float32, device=device)
    elif case == "ragged_faces":
        faces = wp.array([0, 1, 2, 0], dtype=wp.int32, device=device)
    elif case == "space_uv":
        uv_init = wp.zeros(3, dtype=wp.vec3, device=device)
    elif case == "short_uv":
        uv_init = wp.zeros(2, dtype=wp.vec2, device=device)
    elif case == "negative_iterations":
        iterations = -1
    elif case == "negative_tolerance":
        tolerance = -1e-3
    elif case == "nan_tolerance":
        tolerance = float("nan")

    with pytest.raises(ValueError, match=message):
        od.parametrization.slim(
            vertices, faces, uv_init, max_iterations=iterations, tolerance=tolerance
        )


def test_slim_rejects_arrays_on_two_devices():
    """Not a library comparison: inputs on two devices raise rather than launch."""
    if not wp.is_cuda_available():
        pytest.skip("needs two devices")
    vertices = wp.array(np.eye(3, dtype=np.float32), dtype=wp.vec3, device="cpu")
    faces = wp.array([0, 1, 2], dtype=wp.int32, device="cuda:0")
    uv_init = wp.zeros(3, dtype=wp.vec2, device="cpu")

    with pytest.raises(RuntimeError):
        od.parametrization.slim(vertices, faces, uv_init)


def _sheet(steps: int, half_size: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the patch ``z = height * exp(-r^2 / 30)`` over ``[-half_size, half_size]^2``."""
    axis = np.linspace(-half_size, half_size, steps + 1)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    radius_sq = x.ravel() ** 2 + y.ravel() ** 2
    vertices_np = np.stack((x.ravel(), y.ravel(), height * np.exp(-radius_sq / 30.0)), axis=1)
    index = np.arange((steps + 1) ** 2).reshape(steps + 1, steps + 1)
    corner = index[:-1, :-1].ravel()
    right, up, diagonal = corner + 1, corner + steps + 1, corner + steps + 2
    faces_np = np.concatenate(
        (np.stack((corner, right, diagonal), 1), np.stack((corner, diagonal, up), 1))
    )
    return vertices_np, faces_np.astype(np.int64)


def _random_tilt() -> np.ndarray:
    """Return a fixed random rotation (or rotoreflection) of space."""
    return np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]


def _strains(
    vertices_np: np.ndarray, faces_np: np.ndarray, uv_np: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Area-weighted mean ``|ln(s1 s2)|`` and ``ln(s1 / s2)``, and each face's signed UV area."""
    first = vertices_np[faces_np[:, 1]] - vertices_np[faces_np[:, 0]]
    second = vertices_np[faces_np[:, 2]] - vertices_np[faces_np[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(first, second), axis=1)
    length = np.linalg.norm(first, axis=1)
    along, height = (second * first).sum(1) / length, 2 * areas / length
    a = uv_np[faces_np[:, 1]] - uv_np[faces_np[:, 0]]
    b = uv_np[faces_np[:, 2]] - uv_np[faces_np[:, 0]]
    column1 = a / length[:, None]
    column2 = (b - a * (along / length)[:, None]) / height[:, None]
    sigma = np.linalg.svd(np.stack((column1, column2), axis=2), compute_uv=False)
    weights = areas / areas.sum()
    signed = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return (
        float((weights * np.abs(np.log(sigma[:, 0] * sigma[:, 1]))).sum()),
        float((weights * np.log(sigma[:, 0] / sigma[:, 1])).sum()),
        signed,
    )


def _edge_lengths(points_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """Every face's three side lengths."""
    corners = points_np[faces_np]
    return np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2)


def _low_distortion_np(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    device: str,
    coarse_cell: float,
    *,
    iterations: int = 150,
    single_level_vertices: int = 6_000,
) -> np.ndarray:
    """``low_distortion_parametrization`` on a NumPy mesh, read back as ``float64``."""
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    return (
        od.parametrization.low_distortion_parametrization(
            vertices_wp,
            faces_wp,
            coarse_cell=coarse_cell,
            iterations=iterations,
            single_level_vertices=single_level_vertices,
        )
        .numpy()
        .astype(np.float64)
    )


def _coarse_mesh_np(
    vertices_np: np.ndarray, faces_np: np.ndarray, coarse_cell: float, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the coarse mesh as ``(cluster, centroids, coarse_faces)`` NumPy arrays."""
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    points_np = vertices_wp.numpy().astype(np.float64)
    coarse = od.parametrization._coarse_mesh(  # pyright: ignore[reportPrivateUsage]
        vertices_wp,
        faces_wp,
        tuple(points_np.min(axis=0).tolist()),
        tuple(points_np.max(axis=0).tolist()),
        coarse_cell,
    )
    assert coarse is not None
    cluster_wp, centroids_wp, _, coarse_faces_wp = coarse
    return cluster_wp.numpy(), centroids_wp.numpy(), coarse_faces_wp.numpy().reshape(-1, 3)


@pytest.mark.parametrize("steps", [20, 100], ids=["single_level", "two_level"])
def test_low_distortion_parametrization_lays_a_tilted_flat_sheet_out_undeformed(
    device: str, steps: int
):
    """
    Not a library comparison: a plane's pin-free element ARAP chart is an exact isometry.

    No library computes a pin-free, free-boundary element ARAP. 441 vertices are iterated
    directly, 10 201 through the coarse level. A wrong coarse winding, a wrong prolongation or a
    wrong sign in the stiffness leaves edges stretched or faces folded.
    The tolerance is the ``float32`` positions' and chart's resolution at the sheet's size.
    """
    vertices_np, faces_np = _sheet(steps, 8.0, 0.0)
    vertices_np = vertices_np @ _random_tilt().T

    uv_np = _low_distortion_np(vertices_np, faces_np, device, 0.6)

    assert np.allclose(
        _edge_lengths(uv_np, faces_np), _edge_lengths(vertices_np, faces_np), rtol=0, atol=1e-5
    )
    assert (_strains(vertices_np, faces_np, uv_np)[2] > 0).all()


def test_low_distortion_parametrization_is_close_to_a_converged_single_level_solve(device: str):
    """
    Not a library comparison: the two-level chart's distortion is a converged direct solve's.

    No library computes a pin-free element ARAP. The oracle is the same function iterated directly
    on the full mesh to convergence (a coarse cell of ``1e-6`` sets the stop at ``1e-9``).

    On a curved bump of 12 321 vertices the area and shear distortions agree within 0.002, no face
    folds, and the area distortion is well below the plane projection's the iterations start
    from, so the comparison is not met by doing nothing.
    """
    vertices_np, faces_np = _sheet(110, 10.0, 6.0)
    n_vertices = len(vertices_np)

    uv_np = _low_distortion_np(vertices_np, faces_np, device, 0.7)
    converged_np = _low_distortion_np(
        vertices_np, faces_np, device, 1e-6, iterations=1000, single_level_vertices=n_vertices
    )
    projection_np = _low_distortion_np(
        vertices_np, faces_np, device, 0.7, iterations=0, single_level_vertices=n_vertices
    )

    area, shear, signed = _strains(vertices_np, faces_np, uv_np)
    converged_area, converged_shear, _ = _strains(vertices_np, faces_np, converged_np)
    assert area == pytest.approx(converged_area, abs=0.002)
    assert shear == pytest.approx(converged_shear, abs=0.002)
    assert (signed > 0).all()
    assert area < 0.7 * _strains(vertices_np, faces_np, projection_np)[0]


def test_low_distortion_parametrization_takes_its_layout_from_the_coarse_level(device: str):
    """
    Not a library comparison: without the coarse iterations most area distortion stays.

    The smoothing passes alone, from the same projection (``iterations=0``), leave most of
    the area distortion in.
    """
    vertices_np, faces_np = _sheet(110, 10.0, 6.0)

    two_level_np = _low_distortion_np(vertices_np, faces_np, device, 0.7)
    without_coarse_np = _low_distortion_np(vertices_np, faces_np, device, 0.7, iterations=0)

    two_level_area = _strains(vertices_np, faces_np, two_level_np)[0]
    assert two_level_area < 0.7 * _strains(vertices_np, faces_np, without_coarse_np)[0]


# ``low_distortion_parametrization`` on a fixture: iterated directly (the fixtures are under the
# 6 000-vertex default), or forced through a coarse level of cells five mean edges wide.
_TWO_LEVEL_PATHS = [pytest.param(6_000, id="direct"), pytest.param(0, id="two_level")]
# The fixtures the two-level chart is compared with harmonic + ARAP on: a graded disk, a curved
# strip, a non-orientable strip and a genus-1 surface with three holes. ``hemisphere`` is left
# out: five ARAP iterations already converge on its 97 vertices, so the two charts tie.
_HARMONIC_ARAP_MESHES = ["saddle_graded", "half_torus", "mobius", "torus_irregular_holes"]


def _mean_edge_length(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    """Return the mean side length of the faces."""
    return float(_edge_lengths(vertices_np, faces_np).mean())


def _mesh_np(mesh_wp: wp.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixture's ``float64`` vertices and ``(n_faces, 3)`` faces as uploaded."""
    return mesh_wp.points.numpy().astype(np.float64), mesh_wp.indices.numpy().reshape(-1, 3)


def _harmonic_arap_np(mesh_wp: wp.Mesh, iterations: int) -> np.ndarray:
    """
    Return the harmonic + ARAP chart: the usual free-boundary refinement two-level ARAP replaces.

    The harmonic map onto the unit circle, scaled to the mesh's area, then ``iterations`` ARAP
    iterations with one rim vertex pinned, which leaves the boundary, the scale and the rotation
    free.
    """
    boundary_wp, _, uv_init_wp = _harmonic_warm_start(mesh_wp)
    vertices_np, faces_np = _mesh_np(mesh_wp)
    uv_np = uv_init_wp.numpy().astype(np.float64)
    first = uv_np[faces_np[:, 1]] - uv_np[faces_np[:, 0]]
    second = uv_np[faces_np[:, 2]] - uv_np[faces_np[:, 0]]
    uv_area = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]).sum()
    surface_area = (
        0.5
        * np.linalg.norm(
            np.cross(
                vertices_np[faces_np[:, 1]] - vertices_np[faces_np[:, 0]],
                vertices_np[faces_np[:, 2]] - vertices_np[faces_np[:, 0]],
            ),
            axis=1,
        ).sum()
    )
    uv_np *= np.sqrt(surface_area / uv_area)
    device = mesh_wp.device
    pin_np = boundary_wp.numpy()[:1]
    return (
        od.parametrization.arap(
            mesh_wp.points,
            mesh_wp.indices,
            wp.array(pin_np, dtype=wp.int32, device=device),
            points_to_warp_uv(uv_np[pin_np], device),
            points_to_warp_uv(uv_np, device),
            max_iterations=iterations,
        )
        .numpy()
        .astype(np.float64)
    )


@pytest.mark.parametrize("mesh_name", _HARMONIC_ARAP_MESHES)
def test_low_distortion_parametrization_distorts_less_than_harmonic_and_arap(
    request: pytest.FixtureRequest, mesh_name: str
):
    """
    Ordito against ordito: the chart distorts less than harmonic + 5 ARAP iterations.

    No library computes a pin-free element ARAP. The baseline is the refinement this method
    replaces, from ordito's own ``harmonic`` and ``arap`` (``_harmonic_arap_np``). Measured
    (area / shear, ``exp(.) - 1`` of the area-weighted mean log): ``saddle_graded`` 0.018 / 0.019
    against 0.019 / 0.021, ``half_torus`` 0.43 / 0.43 against 0.62 / 0.59 (3 flips against 176),
    ``mobius`` 0.118 / 0.195 against 0.121 / 0.254, ``torus_irregular_holes`` 1.34 / 1.24 against
    1.44 / 1.33. The plane projection the iterations start from is no control here: on the strips
    and the torus it folds hundreds of faces, which a mean of absolute log area does not see.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = _mesh_np(mesh_wp)
    cell = 5.0 * _mean_edge_length(vertices_np, faces_np)

    uv_np = (
        od.parametrization.low_distortion_parametrization(
            mesh_wp.points, mesh_wp.indices, coarse_cell=cell
        )
        .numpy()
        .astype(np.float64)
    )
    baseline_np = _harmonic_arap_np(mesh_wp, 5)

    area, shear, signed = _strains(vertices_np, faces_np, uv_np)
    baseline_area, baseline_shear, baseline_signed = _strains(vertices_np, faces_np, baseline_np)
    assert area < baseline_area
    assert shear < baseline_shear
    assert (signed <= 0).sum() <= (baseline_signed <= 0).sum()


def test_low_distortion_parametrization_distorts_less_than_harmonic_and_arap_when_coarsened(
    device: str,
):
    """
    Ordito against ordito: on the coarse-level path too, less distortion than harmonic + 5 ARAP.

    A 90 x 90 graded saddle (8 100 vertices, over the 6 000-vertex default) built here. Measured:
    0.0173 / 0.0191 against 0.0189 / 0.0204 (area / shear), the converged chart 0.0174 / 0.0187.
    Without its coarse iterations the chart is 0.055 / 0.056, worse than the baseline: the coarse
    level is what wins.
    """
    vertices_np, faces_np = saddle_graded_arrays(90, 3.0)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    vertices_np, faces_np = _mesh_np(mesh_wp)
    cell = 5.0 * _mean_edge_length(vertices_np, faces_np)

    uv_np = _low_distortion_np(vertices_np, faces_np, device, cell)
    without_coarse_np = _low_distortion_np(vertices_np, faces_np, device, cell, iterations=0)
    baseline_np = _harmonic_arap_np(mesh_wp, 5)

    area, shear, signed = _strains(vertices_np, faces_np, uv_np)
    baseline_area, baseline_shear, _ = _strains(vertices_np, faces_np, baseline_np)
    assert area < baseline_area
    assert shear < baseline_shear
    assert (signed > 0).all()
    assert _strains(vertices_np, faces_np, without_coarse_np)[0] > baseline_area


@pytest.mark.parametrize("reverse", [False, True], ids=["as_given", "reversed"])
@pytest.mark.parametrize("single_level_vertices", _TWO_LEVEL_PATHS)
def test_low_distortion_parametrization_follows_the_face_winding(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], reverse: bool, single_level_vertices: int
):
    """
    Not a library comparison: the chart winds counter-clockwise whichever way the faces are given.

    The start is mirrored to the faces' winding, and ARAP never fits a reflection.
    """
    _, mesh_wp = saddle_graded
    vertices_np, faces_np = _mesh_np(mesh_wp)
    if reverse:
        faces_np = np.ascontiguousarray(faces_np[:, ::-1])

    uv_np = _low_distortion_np(
        vertices_np,
        faces_np,
        str(mesh_wp.device),
        5.0 * _mean_edge_length(vertices_np, faces_np),
        single_level_vertices=single_level_vertices,
    )

    assert (_strains(vertices_np, faces_np, uv_np)[2] > 0).all()


@pytest.mark.parametrize("single_level_vertices", _TWO_LEVEL_PATHS)
def test_low_distortion_parametrization_ignores_faces_of_no_area(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], single_level_vertices: int
):
    """
    Ordito against ordito: faces of no area change nothing.

    Appending a face with a repeated corner and one with all three gives the chart of the mesh
    without them, on either path. The projection's normal is given: the default, the plane of the
    longest boundary loop, follows the boundary walk, which the added faces change.
    """
    _, mesh_wp = saddle_graded
    vertices_np, faces_np = _mesh_np(mesh_wp)
    degenerate_np = np.array([[0, 1, 1], [5, 5, 5]], dtype=np.int64)
    cell = 5.0 * _mean_edge_length(vertices_np, faces_np)

    charts = []
    for faces_case_np in (np.concatenate((faces_np, degenerate_np)), faces_np):
        vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_case_np, mesh_wp.device)
        charts.append(
            od.parametrization.low_distortion_parametrization(
                vertices_wp,
                faces_wp,
                coarse_cell=cell,
                normal=wp.vec3(0.0, 0.0, 1.0),
                single_level_vertices=single_level_vertices,
            ).numpy()
        )

    assert np.allclose(charts[0], charts[1], rtol=0, atol=1e-9)


_MAP_STARTS = ["harmonic", "tutte", "lscm"]


def _projected_edge_lengths(
    vertices_np: np.ndarray, faces_np: np.ndarray, normal_np: np.ndarray
) -> np.ndarray:
    """Return every face's three side lengths after projecting along the unit normal."""
    flattened = vertices_np - np.outer(vertices_np @ normal_np, normal_np)
    return _edge_lengths(flattened, faces_np)


@pytest.mark.parametrize("given", [False, True], ids=["longest_loop", "given"])
def test_low_distortion_parametrization_projects_the_start_onto_the_normal_plane(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh], given: bool
):
    """
    Not a library comparison: the start is the projection along ``normal``, or the rim's plane.

    With no iterations and no smoothing the chart is the projection onto the plane of ``normal``
    (a direction off every axis), so its edges are the edges with their normal component removed.
    Without ``normal`` it is the plane fitted to the longest boundary loop, here the cap's planar
    rim, so the same holds with the rim's own normal.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices_np, faces_np = _mesh_np(mesh_wp)
    if given:
        normal_np = np.array([0.3, -0.5, 0.8])
        normal_np /= np.linalg.norm(normal_np)
    else:
        loop_np = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices).numpy()
        rim_np = vertices_np[loop_np] - vertices_np[loop_np].mean(axis=0)
        normal_np = np.linalg.svd(rim_np)[2][2]

    uv_np = (
        od.parametrization.low_distortion_parametrization(
            mesh_wp.points,
            mesh_wp.indices,
            normal=wp.vec3(*normal_np.tolist()) if given else None,
            iterations=0,
            smoothing_passes=0,
        )
        .numpy()
        .astype(np.float64)
    )

    assert np.allclose(
        _edge_lengths(uv_np, faces_np),
        _projected_edge_lengths(vertices_np, faces_np, normal_np),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("reverse", [False, True], ids=["as_given", "reversed"])
@pytest.mark.parametrize("init", _MAP_STARTS)
def test_arap_parametrization_scales_its_start_to_the_surface(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], init: str, reverse: bool
):
    """
    Not a library comparison: the start comes out in the mesh's units and counter-clockwise.

    With no iterations the chart is the named map scaled by ``sqrt(surface / UV area)`` and
    mirrored if it winds clockwise: its total UV area is the surface's and its signed area is
    positive, whichever way the faces are given.
    """
    _, mesh_wp = saddle_graded
    vertices_np, faces_np = _mesh_np(mesh_wp)
    if reverse:
        faces_np = np.ascontiguousarray(faces_np[:, ::-1])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, mesh_wp.device)

    uv_np = (
        od.parametrization.arap_parametrization(
            vertices_wp,
            faces_wp,
            init=init,  # pyright: ignore[reportArgumentType]
            iterations=0,
        )
        .numpy()
        .astype(np.float64)
    )

    first = uv_np[faces_np[:, 1]] - uv_np[faces_np[:, 0]]
    second = uv_np[faces_np[:, 2]] - uv_np[faces_np[:, 0]]
    signed = 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    surface = 0.5 * np.linalg.norm(
        np.cross(
            vertices_np[faces_np[:, 1]] - vertices_np[faces_np[:, 0]],
            vertices_np[faces_np[:, 2]] - vertices_np[faces_np[:, 0]],
        ),
        axis=1,
    )
    assert np.abs(signed).sum() == pytest.approx(surface.sum(), rel=1e-4)
    assert signed.sum() > 0


@pytest.mark.parametrize("init", _MAP_STARTS)
@pytest.mark.parametrize(
    "mesh_name", ["saddle_graded", "sphere_irregular_cap", "convex_irregular_cap"]
)
def test_arap_parametrization_reaches_the_coarse_chart_from_every_start(
    request: pytest.FixtureRequest, mesh_name: str, init: str
):
    """
    Ordito against ordito: every disk start is iterated to the coarse chart's distortion.

    The starts are other charts (circle-bounded, or conformal and pinned), and the factored
    iterations take each to the minimum ``low_distortion_parametrization`` reaches, within 0.002
    in area and shear and with no more inverted faces. Measured spread: 0.0177-0.0179 on
    ``saddle_graded``, 0.2480-0.2481 on ``sphere_irregular_cap``, 0.1122-0.1128 on
    ``convex_irregular_cap``.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = _mesh_np(mesh_wp)

    disk_np = (
        od.parametrization.arap_parametrization(
            mesh_wp.points,
            mesh_wp.indices,
            init=init,  # pyright: ignore[reportArgumentType]
        )
        .numpy()
        .astype(np.float64)
    )
    coarse_np = (
        od.parametrization.low_distortion_parametrization(mesh_wp.points, mesh_wp.indices)
        .numpy()
        .astype(np.float64)
    )

    coarse_area, coarse_shear, coarse_signed = _strains(vertices_np, faces_np, coarse_np)
    area, shear, signed = _strains(vertices_np, faces_np, disk_np)
    assert area == pytest.approx(coarse_area, abs=0.002)
    assert shear == pytest.approx(coarse_shear, abs=0.002)
    assert (signed <= 0).sum() <= (coarse_signed <= 0).sum()


@pytest.mark.parametrize("init", _MAP_STARTS)
@pytest.mark.parametrize(
    ("mesh_name", "gap"),
    [("torus_irregular_holes", 0.15), ("mobius", 0.01)],
    ids=["handle_and_holes", "mobius"],
)
def test_arap_parametrization_charts_a_surface_that_is_not_a_disk(
    request: pytest.FixtureRequest, mesh_name: str, gap: float, init: str
):
    """
    Ordito against ordito: off a disk, every start is iterated to about the coarse chart's strain.

    ``torus_irregular_holes`` has a handle and three boundary loops: the start inverts 225-266
    faces and its strains are 1.3-1.9x the coarse chart's; the iterated chart reaches the coarse
    chart's area and shape strains within 15 % (measured 1.03-1.08x on CUDA) with at most a tenth
    more inverted faces (209-215 against 206). The Mobius strip (one loop, Euler characteristic 0)
    lands within 1 % (0.5 % measured) from starts 3.5-10x off.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = _mesh_np(mesh_wp)
    arguments = (mesh_wp.points, mesh_wp.indices)

    start_np = (
        od.parametrization.arap_parametrization(
            *arguments,
            init=init,  # pyright: ignore[reportArgumentType]
            iterations=0,
        )
        .numpy()
        .astype(np.float64)
    )
    uv_np = (
        od.parametrization.arap_parametrization(
            *arguments,
            init=init,  # pyright: ignore[reportArgumentType]
        )
        .numpy()
        .astype(np.float64)
    )
    coarse_np = (
        od.parametrization.low_distortion_parametrization(*arguments).numpy().astype(np.float64)
    )

    coarse_area, coarse_shear, coarse_signed = _strains(vertices_np, faces_np, coarse_np)
    start_area, start_shear, _ = _strains(vertices_np, faces_np, start_np)
    area, shear, signed = _strains(vertices_np, faces_np, uv_np)
    assert max(start_area / coarse_area, start_shear / coarse_shear) > 1 + 2 * gap
    assert abs(area / coarse_area - 1) < gap
    assert abs(shear / coarse_shear - 1) < gap
    assert (signed <= 0).sum() <= 1.1 * (coarse_signed <= 0).sum() + 1


@pytest.mark.parametrize(
    "chart",
    [
        pytest.param(od.parametrization.arap_parametrization, id="arap"),
        pytest.param(od.parametrization.slim_parametrization, id="slim"),
    ],
)
@pytest.mark.parametrize(
    ("case", "message"),
    [("closed", "needs a boundary loop"), ("disk_and_torus", "needs one connected component")],
)
def test_rim_starts_refuse_a_closed_or_disconnected_mesh(
    request: pytest.FixtureRequest,
    device: str,
    chart: Callable[..., wp.array[wp.vec2]],
    case: str,
    message: str,
):
    """
    Not a library comparison: a rim start needs a boundary loop that reaches the whole mesh.

    A closed sphere has no loop. A disk beside a closed torus has one, but the map placing it
    leaves the torus collapsed to a point (strain 11 after the iterations, against 1.4 for the
    coarse chart), so both refuse it, in either chart taking these starts.
    """
    if case == "disk_and_torus":
        disk_np, disk_faces_np = saddle_graded_arrays()
        torus_tm = tm.creation.torus(major_radius=1.0, minor_radius=0.4)
        vertices_np = np.concatenate((disk_np, torus_tm.vertices + np.array([5.0, 0.0, 0.0])))
        faces_np = np.concatenate((disk_faces_np, torus_tm.faces + len(disk_np)))
        vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    else:
        _, mesh_wp = request.getfixturevalue("sphere_irregular")
        vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    with pytest.raises(ValueError, match=message):
        chart(vertices_wp, faces_wp, init="tutte")


def test_arap_parametrization_rejects_an_unknown_start(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    """Not a library comparison: ``init`` outside the three starts raises, naming them."""
    _, mesh_wp = saddle_graded

    with pytest.raises(ValueError, match="init must be one of"):
        od.parametrization.arap_parametrization(
            mesh_wp.points,
            mesh_wp.indices,
            init="spectral",  # pyright: ignore[reportArgumentType]
        )


def test_low_distortion_parametrization_accepts_a_handle_and_a_separate_piece(device: str):
    """
    Not a library comparison: a handle, a hole and a separate piece are all charted.

    A torus with one quad cut out sits next to a separate flat sheet. The sheet, which no face
    ties to the torus, is held in place by the coarse solve's proximal anchor and comes out
    isometric.
    """
    rings, sides = 96, 64
    ring, side = np.meshgrid(np.arange(rings), np.arange(sides), indexing="ij")
    major, minor = 2 * np.pi * ring / rings, 2 * np.pi * side / sides
    torus_np = np.stack(
        (
            (6 + 2 * np.cos(minor)) * np.cos(major),
            (6 + 2 * np.cos(minor)) * np.sin(major),
            2 * np.sin(minor),
        ),
        axis=2,
    ).reshape(-1, 3)
    corner = (ring * sides + side).ravel()
    right = (((ring + 1) % rings) * sides + side).ravel()
    up = (ring * sides + (side + 1) % sides).ravel()
    diagonal = (((ring + 1) % rings) * sides + (side + 1) % sides).ravel()
    torus_faces_np = np.concatenate(
        (np.stack((corner, right, diagonal), 1), np.stack((corner, diagonal, up), 1))
    )
    torus_faces_np = torus_faces_np[torus_faces_np[:, 0] != 0]
    sheet_np, sheet_faces_np = _sheet(30, 4.0, 0.0)
    vertices_np = np.concatenate((torus_np, sheet_np + np.array([20.0, 0.0, 0.0])))
    sheet_faces_np = sheet_faces_np + len(torus_np)
    faces_np = np.concatenate((torus_faces_np, sheet_faces_np)).astype(np.int64)

    uv_np = _low_distortion_np(vertices_np, faces_np, device, 0.8)

    assert np.isfinite(uv_np).all()
    assert np.allclose(
        _edge_lengths(uv_np, sheet_faces_np),
        _edge_lengths(vertices_np, sheet_faces_np),
        rtol=0,
        atol=1e-4,
    )


def test_low_distortion_parametrization_coarsening_joins_clusters_once_per_triple_and_orientation(
    device: str,
):
    """
    Not a library comparison: a grid's coarse mesh has the expected clusters and windings.

    A 41 x 41 grid over ``[-4, 4]^2`` at a cell of 1 has a 9 x 9 grid of clusters, and its coarse
    faces repeat no triple and all wind as the surface does.
    """
    vertices_np, faces_np = _sheet(40, 4.0, 0.0)

    cluster_np, centroids_np, coarse_faces_np = _coarse_mesh_np(vertices_np, faces_np, 1.0, device)

    assert cluster_np.shape == (len(vertices_np),)
    assert len(centroids_np) == int(cluster_np.max()) + 1 == 81
    assert len(coarse_faces_np) > 0
    assert len(np.unique(np.sort(coarse_faces_np, axis=1), axis=0)) == len(coarse_faces_np)
    first = centroids_np[coarse_faces_np[:, 1]] - centroids_np[coarse_faces_np[:, 0]]
    second = centroids_np[coarse_faces_np[:, 2]] - centroids_np[coarse_faces_np[:, 0]]
    assert (np.cross(first, second)[:, 2] > 0).all()


def test_low_distortion_parametrization_coarse_faces_wind_as_the_surface_whatever_the_grid_cuts(
    device: str,
):
    """
    Not a library comparison: every coarse face of a tilted plane winds with its normal.

    Three centroids often turn the other way from the face joining them; the coarse face must not.
    """
    vertices_np, faces_np = _sheet(100, 8.0, 0.0)
    tilt = _random_tilt()
    normal = tilt[:, 2] * np.sign(np.linalg.det(tilt))

    _, centroids_np, coarse_faces_np = _coarse_mesh_np(vertices_np @ tilt.T, faces_np, 0.6, device)

    first = centroids_np[coarse_faces_np[:, 1]] - centroids_np[coarse_faces_np[:, 0]]
    second = centroids_np[coarse_faces_np[:, 2]] - centroids_np[coarse_faces_np[:, 0]]
    assert len(coarse_faces_np) > 0
    assert (np.cross(first, second) @ normal > 0).all()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("plane_vertices", "wp.vec3 vertices"),
        ("float_faces", "wp.vec3 vertices"),
        ("ragged_faces", "wp.vec3 vertices"),
        ("zero_cell", "coarse_cell"),
        ("infinite_cell", "coarse_cell"),
        ("negative_iterations", "iterations"),
        ("zero_normal", "normal"),
        ("nan_vertex", "finite"),
        ("no_positive_face", "positive area"),
    ],
)
def test_low_distortion_parametrization_rejects_malformed_inputs(
    device: str, case: str, message: str
):
    """Not a library comparison: each malformed input raises a ``ValueError`` naming it."""
    vertices = wp.array(np.eye(3, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    cell, iterations, normal = 1.0, 150, None
    if case == "plane_vertices":
        vertices = wp.zeros(4, dtype=wp.vec2, device=device)
    elif case == "float_faces":
        faces = wp.array([0.0, 1.0, 2.0], dtype=wp.float32, device=device)
    elif case == "ragged_faces":
        faces = wp.array([0, 1, 2, 0], dtype=wp.int32, device=device)
    elif case == "zero_cell":
        cell = 0.0
    elif case == "infinite_cell":
        cell = float("inf")
    elif case == "negative_iterations":
        iterations = -1
    elif case == "zero_normal":
        normal = wp.vec3(0.0, 0.0, 0.0)
    elif case == "nan_vertex":
        vertices = wp.array(np.full((3, 3), np.nan, dtype=np.float32), dtype=wp.vec3, device=device)
    elif case == "no_positive_face":
        vertices = wp.zeros(3, dtype=wp.vec3, device=device)

    with pytest.raises(ValueError, match=message):
        od.parametrization.low_distortion_parametrization(
            vertices, faces, coarse_cell=cell, normal=normal, iterations=iterations
        )


@pytest.mark.parametrize(
    "chart",
    [
        pytest.param(od.parametrization.low_distortion_parametrization, id="any_topology"),
        pytest.param(od.parametrization.arap_parametrization, id="disk"),
    ],
)
def test_low_distortion_parametrization_on_an_empty_mesh(
    device: str, chart: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], wp.array[wp.vec2]]
):
    """Not a library comparison: no vertices give an empty chart rather than a raise."""
    uv_wp = chart(warp_empty(0, wp.vec3, device), warp_empty(0, wp.int32, device))

    assert uv_wp.shape == (0,)
    assert uv_wp.device == wp.get_device(device)


def test_low_distortion_parametrization_is_reproducible_when_deterministic(device: str):
    """
    Not a parity assert: under the deterministic setting repeated calls agree bit for bit.

    On the two-level path. Off it, the adjacency rows the gathers walk come in arrival order on
    CUDA; the chart then agrees with the deterministic one to round-off.
    """
    vertices_np, faces_np = _sheet(100, 8.0, 2.0)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    uv_np = od.parametrization.low_distortion_parametrization(
        vertices_wp, faces_wp, coarse_cell=0.6
    ).numpy()
    for _ in range(3):
        assert np.array_equal(
            od.parametrization.low_distortion_parametrization(
                vertices_wp, faces_wp, coarse_cell=0.6
            ).numpy(),
            uv_np,
        )
    with od.config.deterministic(False):
        arrival_wp = od.parametrization.low_distortion_parametrization(
            vertices_wp, faces_wp, coarse_cell=0.6
        )
    assert np.allclose(arrival_wp.numpy(), uv_np, rtol=0, atol=1e-5)


def test_low_distortion_parametrization_cpu_matches_cuda():
    """
    Class A: the CPU chart is the CUDA one, on the two-level path.

    They differ only in floating-point contraction and summation order, below the ``float32``
    chart's resolution.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")
    vertices_np, faces_np = _sheet(110, 10.0, 6.0)

    uv = {
        device: _low_distortion_np(vertices_np, faces_np, device, 0.7)
        for device in ("cpu", "cuda:0")
    }

    assert np.isfinite(uv["cpu"]).all()
    assert np.allclose(uv["cpu"], uv["cuda:0"], rtol=0, atol=1e-5)


_SLIM_STARTS = ["free", "harmonic", "tutte", "lscm"]


def _slim_start(mesh_wp: wp.Mesh, init: str) -> wp.array[wp.vec2]:
    """Return the start ``slim_parametrization`` documents for ``init``, on a disk."""
    if init == "free":
        return od.parametrization.low_distortion_parametrization(mesh_wp.points, mesh_wp.indices)
    return od.parametrization.arap_parametrization(
        mesh_wp.points,
        mesh_wp.indices,
        init=init,  # pyright: ignore[reportArgumentType]
        iterations=0,
    )


@pytest.mark.parametrize("init", _SLIM_STARTS)
def test_slim_parametrization_is_slim_from_its_start(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], init: str
):
    """
    Ordito against ordito: each ``init`` is ``slim`` from the start its docstring names.

    The free start is ``low_distortion_parametrization``'s chart and the disk starts are
    ``arap_parametrization``'s scaled starts (``iterations=0``); ``slim``'s own
    oracle is libigl (``test_slim_matches_igl``). Bit for bit under the deterministic setting, and
    the chart moved off its start.
    """
    _, mesh_wp = saddle_graded
    arguments = (mesh_wp.points, mesh_wp.indices)
    start_wp = _slim_start(mesh_wp, init)

    uv_np = od.parametrization.slim_parametrization(
        *arguments,
        init=init,  # pyright: ignore[reportArgumentType]
    ).numpy()

    assert np.array_equal(uv_np, od.parametrization.slim(*arguments, start_wp).numpy())
    assert np.abs(uv_np - start_wp.numpy()).max() > 1e-3


@pytest.mark.parametrize(
    ("mesh_name", "init"),
    [("saddle_graded", "tutte"), ("torus_irregular_holes", "free")],
    ids=["disk", "handle_and_holes"],
)
def test_slim_parametrization_starts_from_tutte_on_a_disk_only(
    request: pytest.FixtureRequest, mesh_name: str, init: str
):
    """
    Ordito against ordito: the default start is Tutte on a disk and the free chart otherwise.

    ``saddle_graded`` is a disk; ``torus_irregular_holes`` has a handle and three boundary loops.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    arguments = (mesh_wp.points, mesh_wp.indices)

    default_np = od.parametrization.slim_parametrization(*arguments).numpy()

    assert np.array_equal(
        default_np,
        od.parametrization.slim_parametrization(
            *arguments,
            init=init,  # pyright: ignore[reportArgumentType]
        ).numpy(),
    )


@pytest.mark.parametrize(
    ("mesh_name", "gain"),
    [("saddle_graded", 1.0), ("sphere_irregular_cap", 0.995)],
    ids=["needles", "bumpy"],
)
def test_slim_parametrization_unwraps_a_disk_with_no_fold(
    request: pytest.FixtureRequest, mesh_name: str, gain: float
):
    """
    Not a library comparison: from its default Tutte start a disk's chart folds no face.

    The Tutte start inverts no face, so neither does the chart, on ``saddle_graded``'s needles too;
    and its symmetric Dirichlet energy is below the ARAP chart's from the same start (the energy it
    minimizes): 1.5 % on the bumpy cap, a few parts in a million on the nearly developable saddle,
    where only no rise is asserted.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)
    arguments = (mesh_wp.points, mesh_wp.indices)

    uv_np = od.parametrization.slim_parametrization(*arguments).numpy().astype(np.float64)
    start_np = _slim_start(mesh_wp, "tutte").numpy().astype(np.float64)
    arap_np = (
        od.parametrization.arap_parametrization(*arguments, init="tutte").numpy().astype(np.float64)
    )

    assert (_signed_chart_areas(start_np, faces_np) > 0).all()
    assert (_signed_chart_areas(uv_np, faces_np) > 0).all()
    assert (
        _symmetric_dirichlet(vertices_np, faces_np, uv_np).sum()
        < gain * _symmetric_dirichlet(vertices_np, faces_np, arap_np).sum()
    )


@pytest.mark.parametrize("init", _SLIM_STARTS)
def test_slim_parametrization_unwraps_a_surface_with_a_handle_from_every_start(
    torus_irregular_holes: tuple[tm.Trimesh, wp.Mesh], init: str
):
    """
    Not a library comparison: every start of a surface that is not a disk reaches the same chart.

    ``torus_irregular_holes`` has a handle and three boundary loops, which no chart lays flat
    without folding: the free start keeps 81 % of its faces upright, the rim starts 75-79 % (the
    Tutte start alone inverts 266). Every start must keep 70 % upright and reach the free start's
    distortion within a tenth, in both the area and the shape strain (measured 0.62-0.70 and
    0.68-0.72, both devices); the Tutte start itself is far off it.
    """
    mesh_tm, mesh_wp = torus_irregular_holes
    vertices_np, faces_np = mesh_igl(mesh_tm)
    arguments = (mesh_wp.points, mesh_wp.indices)

    uv_np = (
        od.parametrization.slim_parametrization(
            *arguments,
            init=init,  # pyright: ignore[reportArgumentType]
        )
        .numpy()
        .astype(np.float64)
    )
    free_np = (
        od.parametrization.slim_parametrization(*arguments, init="free").numpy().astype(np.float64)
    )

    area_strain, shape_strain, signed = _strains(vertices_np, faces_np, uv_np)
    free_area_strain, free_shape_strain, _ = _strains(vertices_np, faces_np, free_np)
    assert (signed > 0).mean() > 0.7
    assert abs(area_strain / free_area_strain - 1) < 0.1
    assert abs(shape_strain / free_shape_strain - 1) < 0.1


@pytest.mark.parametrize("init", ["harmonic", "tutte", "lscm"])
def test_slim_parametrization_needs_a_boundary_for_a_rim_start(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], init: str
):
    """Not a library comparison: a closed mesh has no rim, so only the free start runs on it."""
    _, mesh_wp = sphere_irregular
    arguments = (mesh_wp.points, mesh_wp.indices)

    with pytest.raises(ValueError, match="needs a boundary loop"):
        od.parametrization.slim_parametrization(
            *arguments,
            init=init,  # pyright: ignore[reportArgumentType]
        )
    assert od.parametrization.slim_parametrization(*arguments).size == mesh_wp.points.size


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown_init", "init must be one of"),
        ("negative_iterations", "max_iterations"),
        ("nan_tolerance", "tolerance"),
        ("float_faces", "wp.vec3 vertices"),
    ],
)
def test_slim_parametrization_rejects_malformed_inputs(device: str, case: str, message: str):
    """Not a library comparison: each malformed input raises a ``ValueError`` naming it."""
    vertices = wp.array(np.eye(3, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    options: dict[str, Any] = {}
    if case == "unknown_init":
        options["init"] = "spectral"
    elif case == "negative_iterations":
        options["max_iterations"] = -1
    elif case == "nan_tolerance":
        options["tolerance"] = float("nan")
    elif case == "float_faces":
        faces = wp.array([0.0, 1.0, 2.0], dtype=wp.float32, device=device)

    with pytest.raises(ValueError, match=message):
        od.parametrization.slim_parametrization(vertices, faces, **options)


def test_slim_parametrization_on_an_empty_mesh(device: str):
    """Not a library comparison: no vertices give an empty chart rather than a raise."""
    uv_wp = od.parametrization.slim_parametrization(
        warp_empty(0, wp.vec3, device), warp_empty(0, wp.int32, device)
    )

    assert uv_wp.shape == (0,)
    assert uv_wp.device == wp.get_device(device)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("lscm", "igl")
def test_lscm_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A against ``igl.lscm`` with the same two pins, the libigl tutorial-502 convention.

    LSCM is defined only up to the pins, so pinning both sides identically is what makes an
    elementwise comparison meaningful at all -- with a free gauge there would be nothing to compare.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)

    # Pin two boundary vertices to (0, 0) and (1, 0), the libigl tutorial-502 convention.
    pins_np, pins_wp = _two_rim_pins(mesh_wp)
    pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    pins_uv_wp = points_to_warp_uv(pins_uv_np, mesh_wp.device)

    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)
    uv_igl, _ = igl.lscm(
        vertices_np, faces_np, pins_np.astype(np.int64), pins_uv_np.astype(np.float64)
    )

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("lscm", "pymeshlab")
def test_lscm_matches_pymeshlab(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: MeshLab's least-squares conformal map is ordito's with the tutorial-502 pins.

    ``compute_texcoord_parametrization_least_squares_conformal_maps`` wraps libigl and chooses its
    own pins: the longest boundary loop's first vertex at ``(0, 0)`` and the vertex half a loop
    along at ``(1, 0)``, the convention of
    [`test_lscm_matches_igl`][tests.test_parametrization.test_lscm_matches_igl] (measured 3e-8
    against ``igl.lscm`` on ``saddle_graded``). Not an independent
    implementation: it pins what MeshLab picks, through the same pins ordito is handed.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_texcoord_parametrization_least_squares_conformal_maps()
    uv_pml = meshset_pml.current_mesh().vertex_tex_coord_matrix()
    assert np.ptp(uv_pml[:, 1]) > 0.1

    _, pins_wp = _two_rim_pins(mesh_wp)
    pins_uv_wp = points_to_warp_uv(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32), mesh_wp.device
    )
    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)

    assert np.allclose(uv_wp.numpy(), uv_pml, rtol=1e-5, atol=1e-5)


def test_lscm_closed_mesh_matches_igl(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the degenerate closed-mesh branch, where the area term vanishes.

    With no boundary, ``A = 0`` and the system reduces to ``-repdiag(L, 2)``; igl accepts that
    input, so the branch has a real oracle rather than only an invariant. It is its own test
    because the boundary fixtures never exercise it.
    """
    # Closed mesh: A = 0, Q = -repdiag(L, 2). igl.lscm accepts closed input.
    mesh_tm, mesh_wp = sphere_irregular
    vertices_np, faces_np = mesh_igl(mesh_tm)

    pins_np = np.array([0, 7], dtype=np.int32)
    pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    pins_wp = wp.array(pins_np, dtype=wp.int32, device=mesh_wp.device)
    pins_uv_wp = points_to_warp_uv(pins_uv_np, mesh_wp.device)

    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)
    uv_igl, _ = igl.lscm(
        vertices_np, faces_np, pins_np.astype(np.int64), pins_uv_np.astype(np.float64)
    )

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-5, atol=1e-5)


def test_lscm_is_fold_free(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # LSCM of a disk-topology open surface with two pins is conformal and fold-free.
    _, mesh_wp = saddle_graded
    _, pins_wp = _two_rim_pins(mesh_wp)
    pins_uv_wp = wp.array(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)
    assert od.parametrization.face_flipped_indices(uv_wp, mesh_wp.indices).numpy().size == 0


def test_lscm_waives_pin_count_below_two_vertices(device: str):
    # The docstring's "unless the mesh has fewer than two vertices" exemption: a single-vertex,
    # zero-face mesh with no pins at all must not raise.
    vertices_wp = wp.array(np.array([[0.0, 0.0, 0.0]]), dtype=wp.vec3, device=device)
    faces_wp = warp_empty(0, wp.int32, device)
    pins_wp = warp_empty(0, wp.int32, device)
    pins_uv_wp = warp_empty(0, wp.vec2, device)
    uv_wp = od.parametrization.lscm(vertices_wp, faces_wp, pins_wp, pins_uv_wp)
    assert np.isfinite(uv_wp.numpy()).all()


def _harmonic_on_quad(device: str) -> wp.array[wp.vec2]:
    vertices, faces = _unit_quad(device)
    boundary = wp.array(np.array([0, 1, 3], dtype=np.int32), dtype=wp.int32, device=device)
    boundary_uv = points_to_warp_uv(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), device)
    return od.parametrization.harmonic(vertices, faces, boundary, boundary_uv)


def _arap_on_quad(device: str) -> wp.array[wp.vec2]:
    vertices, faces = _unit_quad(device)
    fixed = wp.array(np.array([0, 1, 3], dtype=np.int32), dtype=wp.int32, device=device)
    fixed_uv = points_to_warp_uv(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), device)
    uv_init = wp.zeros(4, dtype=wp.vec2, device=device)
    return od.parametrization.arap(vertices, faces, fixed, fixed_uv, uv_init)


def _lscm_on_quad(device: str) -> wp.array[wp.vec2]:
    vertices, faces = _unit_quad(device)
    pins = wp.array(np.array([0, 3], dtype=np.int32), dtype=wp.int32, device=device)
    pins_uv = points_to_warp_uv(np.array([[0.0, 0.0], [1.0, 1.0]]), device)
    return od.parametrization.lscm(vertices, faces, pins, pins_uv)


@pytest.mark.parametrize(
    "solve",
    [_harmonic_on_quad, _arap_on_quad, _lscm_on_quad],
    ids=["harmonic_three_pins", "arap_three_pins", "lscm_two_pins"],
)
def test_solvers_cpu_match_cuda(solve: Callable[[str], wp.array[wp.vec2]]):
    """
    Class A: each CPU solve is the CUDA one, on the two-triangle unit quad.

    Covers the harmonic interior solve, the ARAP local/global alternation and the LSCM free-vertex
    solve, each under the pins its arm names.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")

    uv = {device: solve(device).numpy() for device in ("cpu", "cuda:0")}

    assert np.isfinite(uv["cpu"]).all()
    assert np.allclose(uv["cpu"], uv["cuda:0"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["harmonic", "tutte", "arap", "lscm"])
def test_solvers_on_an_empty_mesh(device: str, method: str):
    """
    Not a library comparison: no vertices, no pins, an empty UV buffer rather than a raise.

    For the fixed-boundary maps a power below one raises too: ``igl.harmonic`` rejects ``k=0`` with
    an assertion rather than a message; ordito's ``ValueError`` names the method and the power, and
    is raised before the empty-mesh return.
    """
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    pins_wp = warp_empty(0, wp.int32, device)
    pins_uv_wp = warp_empty(0, wp.vec2, device)
    function = getattr(od.parametrization, method)
    if method == "arap":
        uv_init_wp = warp_empty(0, wp.vec2, device)
        uv_wp = function(vertices_wp, faces_wp, pins_wp, pins_uv_wp, uv_init_wp)
    else:
        uv_wp = function(vertices_wp, faces_wp, pins_wp, pins_uv_wp)
    assert uv_wp.dtype == wp.vec2
    assert uv_wp.shape == (0,)
    if method in ("harmonic", "tutte"):
        with pytest.raises(ValueError, match=rf"{method} power k must be >= 1, got 0"):
            function(vertices_wp, faces_wp, pins_wp, pins_uv_wp, k=0)


def _short_boundary_uv(mesh_wp: wp.Mesh) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    return boundary_wp, wp.zeros(boundary_wp.size - 1, dtype=wp.vec2, device=mesh_wp.device)


def _uv_init(mesh_wp: wp.Mesh) -> wp.array[wp.vec2]:
    return wp.zeros(mesh_wp.points.size, dtype=wp.vec2, device=mesh_wp.device)


def _pins_and_zero_uv(
    mesh_wp: wp.Mesh, n_pins: int
) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    pins_wp = wp.array(np.arange(n_pins, dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    return pins_wp, wp.zeros(n_pins, dtype=wp.vec2, device=mesh_wp.device)


_GUARD_CASES: list[tuple[str, Callable[[wp.Mesh], object], str]] = [
    # The index/UV pairs feed a scatter kernel indexed by the indices' own length; a shorter UV
    # buffer would otherwise be an out-of-bounds read.
    (
        "harmonic_short_boundary_uv",
        lambda m: od.parametrization.harmonic(m.points, m.indices, *_short_boundary_uv(m)),
        "same length",
    ),
    # Interior vertices but no pins: the ARAP global system is singular; raised pre-solve.
    (
        "arap_no_fixed",
        lambda m: od.parametrization.arap(
            m.points, m.indices, *_pins_and_zero_uv(m, 0), _uv_init(m)
        ),
        "at least one fixed vertex",
    ),
    (
        "arap_short_fixed_uv",
        lambda m: od.parametrization.arap(m.points, m.indices, *_short_boundary_uv(m), _uv_init(m)),
        "same length",
    ),
    (
        "arap_zero_iterations",
        lambda m: od.parametrization.arap(
            m.points, m.indices, *_circle_boundary(m), _uv_init(m), max_iterations=0
        ),
        "max_iterations",
    ),
    (
        "arap_zero_tolerance",
        lambda m: od.parametrization.arap(
            m.points, m.indices, *_circle_boundary(m), _uv_init(m), tolerance=0.0
        ),
        "tolerance",
    ),
    # Fewer than two pins leaves the similarity-transform null space; raised pre-solve.
    (
        "lscm_no_pins",
        lambda m: od.parametrization.lscm(m.points, m.indices, *_pins_and_zero_uv(m, 0)),
        "at least two pinned vertices",
    ),
    (
        "lscm_one_pin",
        lambda m: od.parametrization.lscm(m.points, m.indices, *_pins_and_zero_uv(m, 1)),
        "at least two pinned vertices",
    ),
    (
        "lscm_short_pinned_uv",
        lambda m: od.parametrization.lscm(
            m.points, m.indices, _two_rim_pins(m)[1], wp.zeros(1, dtype=wp.vec2, device=m.device)
        ),
        "same length",
    ),
]


@pytest.mark.parametrize(
    ("call", "message"),
    [(call, message) for _, call, message in _GUARD_CASES],
    ids=[name for name, _, _ in _GUARD_CASES],
)
def test_solvers_reject_malformed_constraints(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], call: Callable[[wp.Mesh], object], message: str
):
    """Not a parity assert: every guard raises before any solve, so each runs on the CPU too."""
    _, mesh_wp = saddle_graded
    with pytest.raises(ValueError, match=message):
        call(mesh_wp)


def _count_factorizations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the sparse Cholesky factorizations built from here on through ``ordito.linalg``."""
    built: list[int] = []
    factor = od.linalg.sparse_cholesky

    def counted(*args: Any, **kwargs: Any) -> od.cholesky.SparseCholesky:
        result = factor(*args, **kwargs)
        built.append(result.n)
        return result

    monkeypatch.setattr(od.linalg, "sparse_cholesky", counted)
    return built


def _fixed_vertex_call(
    method: str, mesh_wp: wp.Mesh, scale: float
) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    """Return the fixed set and values a ``method`` call takes: the loop, or two of its vertices."""
    vertices, faces = mesh_wp.points, mesh_wp.indices
    loop = od.boundary.longest_boundary_loop(vertices, faces)
    if method != "lscm":
        circle = od.parametrization.map_vertices_to_circle(vertices, loop).numpy()
        return loop, wp.array(scale * circle, dtype=wp.vec2, device=vertices.device)
    loop_np = loop.numpy()
    pins = wp.array(
        np.array([loop_np[0], loop_np[loop_np.size // 2]], dtype=np.int32),
        dtype=wp.int32,
        device=vertices.device,
    )
    pins_uv = np.array([[0.0, 0.0], [scale, 0.0]], dtype=np.float32)
    return pins, wp.array(pins_uv, dtype=wp.vec2, device=vertices.device)


@pytest.mark.parametrize(
    ("method", "k"), [("harmonic", 1), ("harmonic", 2), ("harmonic", 3), ("tutte", 1), ("lscm", 1)]
)
def test_fixed_vertex_solver_factors_once_for_every_call_fixing_the_same_vertices(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh],
    method: str,
    k: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ordito against ordito: a prepared factorization serves later calls with no further build.

    ``FixedVertexSolver.factor`` builds one factorization; two calls on the mesh with different
    fixed *values* build none of their own and agree with the ``vertices, faces`` form (which
    iterates below ``k == 3``, verified) to the solves' tolerance. The oracles are the igl tests
    above (``test_harmonic_matches_igl``, ``test_tutte_matches_igl_reference``,
    ``test_lscm_matches_igl``, ``test_polyharmonic_matches_reference``); this pins that the
    prepared path solves the same system.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices, faces = mesh_wp.points, mesh_wp.indices
    function = getattr(od.parametrization, method)
    keywords = {} if method == "lscm" else {"k": k}
    built = _count_factorizations(monkeypatch)
    mesh = od.Trimesh(vertices, faces)
    fixed, _ = _fixed_vertex_call(method, mesh_wp, 1.0)
    assert mesh.fixed_vertex_solver().factor(method, fixed, k=k)
    assert len(built) == 1
    assert mesh.fixed_vertex_solver().nbytes > 0
    for scale in (1.0, 2.5):
        fixed, values = _fixed_vertex_call(method, mesh_wp, scale)
        before = len(built)
        prepared = function(mesh, fixed, values, **keywords).numpy()
        assert len(built) == before
        # The ``vertices, faces`` form factors on its own at ``k >= 3``; that build is not counted.
        direct = function(vertices, faces, fixed, values, **keywords).numpy()
        assert np.allclose(prepared, direct, rtol=1e-5, atol=1e-5 * np.ptp(direct))
    mesh.release_factorizations()
    assert mesh.fixed_vertex_solver().nbytes == 0


def test_fixed_vertex_solver_ignores_a_factorization_of_another_fixed_set(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: a call fixing other vertices than the kept factorization's solves anew.

    The kept system's reduced operator is that of its own fixed set; a call fixing the loop less
    one vertex must not be answered by it (it would solve a different system), so it iterates --
    builds nothing at ``k == 1`` -- and agrees with the ``vertices, faces`` form. The kept
    factorization stays and still serves its own set.
    """
    _, mesh_wp = saddle_graded
    vertices, faces = mesh_wp.points, mesh_wp.indices
    mesh = od.Trimesh(vertices, faces)
    loop, circle = _fixed_vertex_call("harmonic", mesh_wp, 1.0)
    mesh.fixed_vertex_solver().factor("harmonic", loop, k=1)
    built = _count_factorizations(monkeypatch)
    fewer = wp.array(loop.numpy()[1:], dtype=wp.int32, device=vertices.device)
    fewer_uv = wp.array(circle.numpy()[1:], dtype=wp.vec2, device=vertices.device)
    other = od.parametrization.harmonic(mesh, fewer, fewer_uv).numpy()
    direct = od.parametrization.harmonic(vertices, faces, fewer, fewer_uv).numpy()
    assert built == []
    assert np.allclose(other, direct, rtol=1e-5, atol=1e-5 * np.ptp(direct))
    own = od.parametrization.harmonic(mesh, loop, circle).numpy()
    assert built == []
    assert np.allclose(
        own, od.parametrization.harmonic(vertices, faces, loop, circle).numpy(), atol=1e-5
    )


def test_fixed_vertex_solver_keeps_one_factorization_per_method_and_outlives_nothing(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the solver's ownership rules, which no reference has.

    Factoring ``(method, k)`` again for another fixed set releases the first factorization, so the
    solver then holds exactly what a fresh solver factoring only the second set holds. An empty
    mesh factors trivially. A solver whose ``Trimesh`` was collected raises rather than reading a
    dead mesh: the solver holds its mesh weakly.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices, faces = mesh_wp.points, mesh_wp.indices
    loop = od.boundary.longest_boundary_loop(vertices, faces)
    shorter = wp.array(loop.numpy()[:-1], dtype=wp.int32, device=loop.device)
    mesh = od.Trimesh(vertices, faces)
    solver = mesh.fixed_vertex_solver()
    assert solver.factor("harmonic", loop)
    first_nbytes = solver.nbytes
    assert first_nbytes > 0
    assert solver.factor("harmonic", shorter)
    fresh_mesh = od.Trimesh(vertices, faces)
    assert fresh_mesh.fixed_vertex_solver().factor("harmonic", shorter)
    assert solver.nbytes == fresh_mesh.fixed_vertex_solver().nbytes

    empty = od.Trimesh(
        warp_empty(0, wp.vec3, vertices.device), warp_empty(0, wp.int32, vertices.device)
    )
    assert empty.fixed_vertex_solver().factor("tutte", warp_empty(0, wp.int32, vertices.device))
    assert empty.fixed_vertex_solver().nbytes == 0

    del mesh
    gc.collect()
    with pytest.raises(RuntimeError, match="its Trimesh no longer exists"):
        solver.factor("harmonic", loop)


@pytest.mark.parametrize(("k", "discards"), [(1, False), (3, True)])
def test_fixed_vertex_calls_say_when_they_discard_a_factorization(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh],
    caplog: pytest.LogCaptureFixture,
    k: int,
    discards: bool,
) -> None:
    """
    Not a parity assert: the ``vertices, faces`` form logs at ``INFO`` exactly when it drops one.

    ``k == 3`` is factored from the start, so the call builds a factorization it cannot keep and
    says so; ``k == 1`` iterates on this mesh and builds none. Given a ``Trimesh``, the same
    ``k == 3`` call keeps it on the mesh and logs nothing.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices, faces = mesh_wp.points, mesh_wp.indices
    loop, circle = _fixed_vertex_call("harmonic", mesh_wp, 1.0)
    with caplog.at_level(logging.INFO, logger="ordito.parametrization"):
        od.parametrization.harmonic(vertices, faces, loop, circle, k=k)
    infos = [r for r in caplog.records if r.levelno == logging.INFO and "Trimesh" in r.message]
    assert len(infos) == (1 if discards else 0)
    caplog.clear()
    mesh = od.Trimesh(vertices, faces)
    with caplog.at_level(logging.INFO, logger="ordito.parametrization"):
        od.parametrization.harmonic(mesh, loop, circle, k=k)
    assert [r for r in caplog.records if r.levelno == logging.INFO] == []
    assert (mesh.fixed_vertex_solver().nbytes > 0) == discards


@pytest.mark.parametrize(("method", "k"), [("arap", 1), ("harmonic", 0), ("lscm", 2)])
def test_fixed_vertex_solver_rejects_an_off_menu_method_or_power(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], method: str, k: int
) -> None:
    """Not a parity assert: ``factor`` and ``solve`` raise ``ValueError`` naming the argument."""
    _, mesh_wp = saddle_graded
    solver = od.Trimesh(mesh_wp.points, mesh_wp.indices).fixed_vertex_solver()
    loop, _ = _fixed_vertex_call("harmonic", mesh_wp, 1.0)
    with pytest.raises(ValueError, match=r"method|k="):
        solver.factor(method, loop, k=k)
    mask = wp.zeros(mesh_wp.points.size, dtype=wp.bool, device=mesh_wp.points.device)
    values = odt.as_array2d(
        wp.zeros((2, mesh_wp.points.size), dtype=wp.float64, device=mesh_wp.points.device),
        wp.float64,
    )
    with pytest.raises(ValueError, match=r"method|k="):
        solver.solve(method, mask, values, k=k)


@pytest.mark.parametrize(
    ("method", "k"), [("harmonic", 1), ("harmonic", 2), ("tutte", 1), ("lscm", 1)]
)
def test_solver_direct_factors_at_once_and_keeps_it_on_the_mesh(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
    method: str,
    k: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ordito against ordito: ``solver="direct"`` is ``factor`` folded into the first call.

    On a ``Trimesh`` the first call factors (one build) and keeps it, so later default-``solver``
    calls fixing the same vertices build nothing; each agrees with the ``vertices, faces`` form's
    verified iteration to the solves' tolerance (the igl oracles are the tests above). The
    ``vertices, faces`` form under ``"direct"`` factors too, and says at ``INFO`` that it drops it.
    """
    _, mesh_wp = saddle_graded
    vertices, faces = mesh_wp.points, mesh_wp.indices
    function = getattr(od.parametrization, method)
    keywords = {} if method == "lscm" else {"k": k}
    fixed, values = _fixed_vertex_call(method, mesh_wp, 1.0)
    iterated = function(vertices, faces, fixed, values, **keywords).numpy()
    built = _count_factorizations(monkeypatch)
    mesh = od.Trimesh(vertices, faces)
    direct = function(mesh, fixed, values, solver="direct", **keywords).numpy()
    assert len(built) == 1
    again = function(mesh, fixed, values, **keywords).numpy()
    assert len(built) == 1
    for result in (direct, again):
        assert np.allclose(result, iterated, rtol=1e-5, atol=1e-5 * np.ptp(iterated))
    with caplog.at_level(logging.INFO, logger="ordito.parametrization"):
        function(vertices, faces, fixed, values, solver="direct", **keywords)
    assert len(built) == 2
    assert [r for r in caplog.records if r.levelno == logging.INFO and "Trimesh" in r.message]


@pytest.mark.parametrize("method", ["harmonic", "tutte", "lscm"])
def test_fixed_vertex_maps_reject_an_off_menu_solver(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], method: str
) -> None:
    """Not a parity assert: ``solver`` outside the menu raises ``ValueError`` naming it."""
    _, mesh_wp = saddle_graded
    fixed, values = _fixed_vertex_call(method, mesh_wp, 1.0)
    function = getattr(od.parametrization, method)
    with pytest.raises(ValueError, match="solver"):
        function(mesh_wp.points, mesh_wp.indices, fixed, values, solver="cholesky")
    with pytest.raises(ValueError, match="solver"):
        function(od.Trimesh(mesh_wp.points, mesh_wp.indices), fixed, values, solver="cholesky")


def _malformed_fixed_set(
    method: str, mesh_wp: wp.Mesh, case: str
) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    """
    Return a ``method`` fixed set that repeats a vertex or leaves the mesh, and matching UVs.

    ``lscm``'s repeat is ``[p, p]``: two pins that fix one vertex, a null space left unpinned.
    """
    fixed, _ = _fixed_vertex_call(method, mesh_wp, 1.0)
    fixed_np = fixed.numpy()
    n_vertices = mesh_wp.points.size
    if case == "repeated":
        if method == "lscm":
            fixed_np = np.array([fixed_np[0], fixed_np[0]], dtype=np.int32)
        else:
            fixed_np = np.append(fixed_np, fixed_np[0]).astype(np.int32)
    elif case == "out_of_range":
        fixed_np = fixed_np.copy()
        fixed_np[-1] = n_vertices
    else:  # every index out of range: nothing at all would be fixed
        fixed_np = (n_vertices + np.arange(fixed_np.size)).astype(np.int32)
    device = mesh_wp.points.device
    uv = wp.zeros(fixed_np.size, dtype=wp.vec2, device=device)
    return wp.array(fixed_np, dtype=wp.int32, device=device), uv


@pytest.mark.parametrize(
    ("method", "k"), [("harmonic", 1), ("harmonic", 2), ("harmonic", 3), ("tutte", 1), ("lscm", 1)]
)
def test_fixed_vertex_maps_reject_repeated_or_out_of_range_indices(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh], method: str, k: int
) -> None:
    """
    Not a parity assert: a fixed set that repeats a vertex or leaves the mesh raises ``ValueError``.

    The scatter skips an out-of-range index and marks a repeated one once, so either fixes fewer
    vertices than given: unchecked, ``lscm`` pinned at ``[p, p]`` returned an all-zero map and
    ``harmonic`` with every index out of range returned zeros. Both forms raise, and a `Trimesh`
    holding a factorization of the valid set raises too (the repeated set's mask equals it, so
    the kept system would otherwise answer it). ``FixedVertexSolver.factor`` raises as well.
    The three malformed sets share one factorization of the valid set.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices, faces = mesh_wp.points, mesh_wp.indices
    function = getattr(od.parametrization, method)
    keywords = {} if method == "lscm" else {"k": k}
    mesh = od.Trimesh(vertices, faces)
    valid, _ = _fixed_vertex_call(method, mesh_wp, 1.0)
    assert mesh.fixed_vertex_solver().factor(method, valid, k=k)
    for case in ("repeated", "out_of_range", "all_out_of_range"):
        fixed, uv = _malformed_fixed_set(method, mesh_wp, case)
        with pytest.raises(ValueError, match="distinct"):
            function(vertices, faces, fixed, uv, **keywords)
        with pytest.raises(ValueError, match="distinct"):
            function(mesh, fixed, uv, **keywords)
        with pytest.raises(ValueError, match="distinct"):
            mesh.fixed_vertex_solver().factor(method, fixed, k=k)


def test_arap_rejects_repeated_or_out_of_range_indices(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Not a parity assert: ``arap``'s fixed set is checked as the fixed-vertex maps' is."""
    _, mesh_wp = sphere_irregular_cap
    for case in ("repeated", "out_of_range"):
        fixed, uv = _malformed_fixed_set("harmonic", mesh_wp, case)
        with pytest.raises(ValueError, match="distinct"):
            od.parametrization.arap(mesh_wp.points, mesh_wp.indices, fixed, uv, _uv_init(mesh_wp))


def test_a_refused_factorization_is_not_attempted_again_on_a_trimesh(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a parity assert: ``"auto"``'s factor-at-once refusal is remembered by the mesh's solver.

    With the factor-at-once size lowered to this mesh and every factorization refused (the
    analysis raising as it does over the memory budget), each ``lscm`` call on a `Trimesh` would
    otherwise pay a refused analysis before iterating. The first call attempts it once; later calls
    attempt none; ``release_factorizations`` forgets the refusal; the ``vertices, faces`` form,
    holding no solver, attempts it on every call. The iterated answers agree with each other.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices, faces = mesh_wp.points, mesh_wp.indices
    attempts: list[int] = []

    def refused(*_args: Any, **_kwargs: Any) -> od.cholesky.SparseCholesky:
        attempts.append(1)
        raise ValueError("over the memory budget")

    monkeypatch.setattr(od.parametrization, "_FACTOR_FROM_VERTICES", 0)
    monkeypatch.setattr(od.linalg, "sparse_cholesky", refused)
    pins, pins_uv = _fixed_vertex_call("lscm", mesh_wp, 1.0)
    mesh = od.Trimesh(vertices, faces)
    first = od.parametrization.lscm(mesh, pins, pins_uv).numpy()
    assert len(attempts) == 1
    again = od.parametrization.lscm(mesh, pins, pins_uv).numpy()
    assert len(attempts) == 1
    assert np.allclose(again, first, rtol=1e-5, atol=1e-5 * np.ptp(first))
    mesh.release_factorizations()
    od.parametrization.lscm(mesh, pins, pins_uv)
    assert len(attempts) == 2
    for expected in (3, 4):
        od.parametrization.lscm(vertices, faces, pins, pins_uv)
        assert len(attempts) == expected


@pytest.mark.parametrize("method", ["harmonic", "tutte"])
def test_a_k1_system_factors_at_once_on_the_cpu_from_its_size(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """
    Ordito against ordito: the CPU's ``k == 1`` factor-at-once gate, at its boundary.

    With the gate set to this mesh's vertex count the CPU device factors before iterating and
    CUDA does not; one vertex above it neither does. The factored and iterated maps agree; the
    iterated route has the igl oracle above.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices, faces = mesh_wp.points, mesh_wp.indices
    boundary, boundary_uv = _fixed_vertex_call(method, mesh_wp, 1.0)
    built: list[int] = []
    analyse = od.linalg.sparse_cholesky

    def counted(*args: Any, **kwargs: Any) -> od.cholesky.SparseCholesky:
        built.append(1)
        return analyse(*args, **kwargs)

    monkeypatch.setattr(od.linalg, "sparse_cholesky", counted)
    call = getattr(od.parametrization, method)
    uv_np = {}
    for gate in (vertices.size + 1, vertices.size):
        monkeypatch.setattr(od.parametrization, "_FACTOR_K1_FROM_VERTICES_CPU", gate)
        built.clear()
        uv_np[gate] = call(vertices, faces, boundary, boundary_uv).numpy()
        assert len(built) == int(gate == vertices.size and wp.get_device(vertices.device).is_cpu)
    assert np.allclose(uv_np[vertices.size], uv_np[vertices.size + 1], rtol=1e-5, atol=1e-5)
