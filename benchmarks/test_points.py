"""
Benchmarks for ``ordito.points``.

The point cloud is a registry mesh's own vertices — deterministic, and it scales with the mesh. The
functions here fall into two groups:

* **Tiled reductions** — ``fit_line`` and ``fit_plane`` accumulate a 3x3 scatter matrix and finish
  with a single-thread ``wp.svd3``. Both end in a host readback, so the numbers include one
  synchronisation by construction.
* **Element-wise maps and sorts** — ``point_plane_distance`` and ``vector_angle`` are one
  ``wp.map`` each (the cheapest thing here, so the most sensitive to launch overhead), and
  ``radial_sort`` is a key kernel plus a radix sort.

Where a neighbour table is an *input* of ordito's function, the group is timed twice — the kernel
alone on a cached table, and the table plus the kernel. Only the second is a fair cross-library
comparison, since every reference builds its own k-d tree per call. That covers
``estimate_normals`` / ``estimate_normals_knn`` and both outlier groups.

``farthest_point_sample`` is the one to read carefully: its greedy loop is ``Theta(count)``
*dependent* rounds, so as one launch each it is thousands of launches and essentially no kernel
time. It runs instead as one persistent block, which wins against open3d at the small end where the
launch-bound form loses. The count is still the axis — it is the number of rounds — but a round
costs block time rather than a launch.

References
----------
**trimesh** covers everything except normal estimation: ``point_plane_distance``, ``major_axis``,
``plane_fit``, ``radial_sort`` and ``geometry.vector_angle`` are the functions ordito's are ports
of. **open3d** is the reference for ``estimate_normals`` (``KDTreeSearchParamKNN``, the same
k-nearest PCA estimator) and for both outlier filters — those additionally *copy* the survivors
into a new cloud where ordito returns a mask, so those rows are upper bounds.

**pymeshlab** covers three. ``compute_normal_for_point_clouds(k=)`` is the same estimator and
exists for exactly this case, so its input is a *face-less* MeshSet.
``compute_matrix_by_fitting_to_plane`` is the ``fit_plane`` counterpart; it raises ``Cannot compute
rotation: there is no selection`` unless something is selected, so ``set_selection_all`` runs
first, untimed — that is how the filter is told "fit all the points", not part of the fit. It also
builds a rotation onto a target plane, which ordito does not, so its row is an upper bound. Third,
``compute_selection_point_cloud_outliers`` is the LoOP score behind ``outlier_probability``.
MeshLab has nothing for the array primitives (``fit_line``, ``point_plane_distance``,
``vector_angle``, ``radial_sort``). **libigl** covers ``fit_plane`` alone: ``igl.fit_plane``
returns the centred covariance's smallest eigenvector and the centroid, ordito's answer exactly.

Caps
----
Both ``estimate_normals`` groups stop at ``bunny``: the neighbour table is what costs at ``k = 30``
(that kernel is timed on its own in [`test_neighbors.py`](test_neighbors.py)) and open3d's serial
``KDTreeFlann`` scales the same way.

``fit_line``'s **trimesh** case stops there for an unrelated reason: ``points.major_axis`` calls
``numpy.linalg.svd`` on the ``(n, 3)`` matrix with the default ``full_matrices=True``, so it
materializes the full ``(n, n)`` left-singular matrix and raises ``_ArrayMemoryError`` on a scan
mesh. Ordito accumulates a 3x3 Gram matrix instead and runs the full registry. ``plane_fit`` does
not take that path and needs no cap.

Convex hull
-----------
``convex_hull`` is the exact hull, timed against qhull itself (``scipy.spatial.ConvexHull``) and
the three qhull-backed mesh wrappers, all of which compute the same surface. A registry mesh's
vertices are a scan cloud whose hull holds a few thousand points, so both sides spend most of their
time discarding the rest; ordito's growth rounds do that in parallel passes over the points still
outside the surface.

``convex_subset_mask`` / ``convex_subset`` / ``convex_superset_mask`` are the compute-bound
approximations: cost is ``n_points * n_directions`` for the support sweep, and
``convex_superset_mask`` adds a pass testing every point against the tetrahedra spanned by the
resulting shell, where every thread reads the same plane table and the arithmetic dominates.
``convex_subset`` is the mask plus a ``flatnonzero`` and a gather, so its delta over the mask is the
compaction cost.

**The baselines of the two subset groups compute a different and stronger result**, which is the
point of the comparison rather than a flaw in it. trimesh, open3d and pymeshlab all run **qhull**,
producing the exact hull as a mesh; ``convex_subset`` produces an approximate vertex *subset*. So
this is not parity -- it quantifies what the approximation buys, which is why the function exists.
Three qhull rows rather than one because the argument is about the operation, and because a second
and third wrapper around the same computation is the only way to tell whether trimesh's number is
qhull or trimesh.

**scipy** is registered for ``convex_superset_mask``, and there it is a genuine parity row: that
filter exists to run *before* an exact hull, so ``scipy.spatial.ConvexHull`` on the same cloud is
exactly the cost it must be cheap against, and its output provably contains that hull's vertex set.
The ratio is what decides whether the prefilter is worth running, and it widens with the point
count because the filter is linear where qhull is not. **libigl** has no hull binding.

``n_directions`` and ``subdivisions`` are each swept over two values spanning their useful range;
both costs are close to linear in the direction count, so two points fix the line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import igl
import manifold3d as m3
import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import pymeshlab as ml
import pytest
import pytorch3d.ops as p3d_ops
import pyvista as pv
import scipy.spatial
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from conftest import (
    BenchCase,
    BenchLibrary,
    points_torch_from_numpy,
    skip_larger_than,
    std_points_ml,
)

if TYPE_CHECKING:
    from typing_extensions import Buffer

# Neighbour count for the PCA normal estimate (open3d's own default for KDTreeSearchParamKNN).
_KNN = 30

# Neighbourhood widths to sweep: the per-point 3x3 PCA is linear in k, and the k-NN table build
# that feeds it is not, so the pair separates the estimator's cost from its input's. The table is
# cached (``_neighbor_table``), so this sweep times the estimator *only* — the query's own k axis is
# ``query_nearest_bvh_k1`` / ``_k7`` / ``_k64`` in ``test_neighbors.py``.
_KNN_SWEEP = [8, 64]

# The trimesh references here are single-threaded host passes and one of them is far worse than
# single-threaded: ``tm.points.fit_line`` takes tens of seconds a call on a scan mesh, and that one
# reference was most of this module's wall clock. So every host branch is capped at the smallest
# scan mesh -- the ratio
# against ordito is four orders of magnitude and needs no larger input to establish.
_HOST_CAP_REASON = "host reference is a single-threaded pass; capped at bunny_decimated"

# Radius sweep for ``radius_outlier_mask``, in mean edge lengths. The ball count is linear in how
# many points each ball holds, so a 2x radius is ~8x the point tests -- the two points bracket where
# the hash grid stops paying for the extra cells.
_RADIUS_SCALES = [2.0, 4.0]

# Density floor for the same group, held fixed so the sweep is the radius alone. Open3D's own
# ``nb_points`` semantics: a point survives when it has strictly more than this many neighbours,
# itself included.
_MIN_NEIGHBORS = 6

# Sample counts for ``farthest_point_sample``. The greedy sweep is Theta(count) dependent rounds of
# one persistent block, so this axis is the round count and the step between the two is 16x.
_SAMPLE_COUNTS = [64, 1024]

# Fixed plane / sort axis, deliberately not axis-aligned so no branch is skipped.
_PLANE_NORMAL = np.array([0.3, -0.6, 0.74])

_neighbors_cache: dict[tuple[str, str, int], odt.Array2dInt32] = {}
_normals_wp_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}
_pcd_cache: dict[str, o3d.geometry.PointCloud] = {}
_cloud_pml_cache: dict[str, ml.MeshSet] = {}


def _neighbor_table(bench_case: BenchCase, k: int = _KNN) -> odt.Array2dInt32:
    """``(n, k)`` k-nearest table over the cloud itself — an *input* of ``estimate_normals``."""
    key = (bench_case.mesh_name, str(bench_case.device), k)
    if key not in _neighbors_cache:
        points = bench_case.vertices_wp
        _neighbors_cache[key] = od.neighbors.query_nearest(points, points, k=k)[0]
    return _neighbors_cache[key]


def _unit_normals_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Area-weighted unit vertex normals: the left operand of the ``vector_angle`` pairs."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _normals_wp_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _normals_wp_cache[key] = od.normals.vertex_normals(vertices, faces)
    return _normals_wp_cache[key]


def _unit_directions_np(bench_case: BenchCase) -> np.ndarray:
    """Build unit vectors from the cloud centroid to each point (right operand of the pairs)."""
    vertices = bench_case.vertices_np
    directions = vertices - vertices.mean(axis=0)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _pcd(bench_case: BenchCase) -> o3d.geometry.PointCloud:
    """Open3D point cloud over the same vertices, built once per mesh (setup, not timed)."""
    name = bench_case.mesh_name
    if name not in _pcd_cache:
        _pcd_cache[name] = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(bench_case.vertices_np)
        )
    return _pcd_cache[name]


@pytest.mark.benchmark(group="point_plane_distance")
@pytest.mark.benchlibs("ordito", "trimesh", "pyvista")
def test_point_plane_distance(bench_case: BenchCase) -> None:
    """
    Signed point-to-plane distance of every point: a single ``wp.map`` over the cloud.

    The thinnest kernel in the module, so it is the module's floor row: below roughly ``10 ** 3``
    points it reports the wrapper floor rather than the map (see ``test_creation::test_box``), and
    being a tens-of-microseconds GPU row it also has the widest run-to-run spread in the suite --
    more than an order of magnitude on unchanged code across two processes. Only read it as part of
    its axis.

    pyvista's ``compute_implicit_distance`` evaluates VTK's plane implicit function over the cloud,
    which is the same signed dot product -- both sides land at their own storage precision
    (``tests/test_points.py``). It needs
    a ``pv.Plane`` **large enough to span the cloud**: the implicit function is unbounded but the
    plane object carries an extent, and it is also the reason this row builds the plane outside the
    timed callable -- it is the query's parameter, not its input.
    """
    n_points = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        cloud_pv = pv.PolyData(bench_case.vertices_np)
        extent = 4.0 * float(
            np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
        )
        plane_pv = pv.Plane(
            center=(0.0, 0.0, 0.0), direction=_PLANE_NORMAL.tolist(), i_size=extent, j_size=extent
        )
        distances_pv = bench_case.run(lambda: cloud_pv.compute_implicit_distance(plane_pv))
        assert distances_pv.point_data["implicit_distance"].shape[0] == n_points
        return
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        distances = bench_case.run(lambda: od.points.point_plane_distance(points, normal))
        assert distances.shape == (n_points,)
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        distances_tm = bench_case.run(
            lambda: tm.points.point_plane_distance(points_np, _PLANE_NORMAL)
        )
        assert distances_tm.shape == (n_points,)


@pytest.mark.noparity(
    "pyvista",
    oracle="trimesh",
    reason="D2 a different estimator with a measured disagreement: pv.fit_line_to_points returns "
    "the first *principal* axis of the centred covariance, while fit_line is a faithful port of "
    "trimesh.points.major_axis -- normalize(S @ V) over the SVD of the *uncentered* point matrix, "
    "a singular-value-weighted sum of all three right singular vectors. Measured |dot| between the "
    "two of 0.802 on an aspect-3:1:0.05 cloud offset from the origin, where pyvista agrees with "
    "the leading eigenvector to 1.0000000 and ordito does not; on a 1000:1 needle the two "
    "definitions coincide to 3e-6, which is why the difference is easy to miss. The principal "
    "frame is a *separate* ordito function -- points.principal_axes, whose own group carries the "
    "pyvista row -- so this is a naming coincidence rather than two implementations of one "
    "quantity. trimesh is the oracle here, in tests/test_points.py::test_fit_line, and the "
    "distinction is pinned in test_principal_axes_is_not_fit_line.",
)
@pytest.mark.benchmark(group="fit_line")
@pytest.mark.benchlibs("ordito", "trimesh", "pyvista")
def test_fit_line(bench_case: BenchCase) -> None:
    """
    Major axis from the uncentred Gram matrix: tiled outer-product sum plus one ``svd3``.

    pyvista's row computes a *different* axis -- see the exemption above -- and additionally returns
    the fitted segment as a ``PolyData`` rather than a direction, so read it as the cost of "fit a
    line to this cloud" in VTK rather than as the same arithmetic.
    """
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        line_pv = bench_case.run(lambda: pv.fit_line_to_points(points_np))
        assert np.asarray(line_pv.points).shape[1] == 3
        return
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        axis = bench_case.run(lambda: od.points.fit_line(points))
        assert len(axis) == 3
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        # major_axis SVDs with full_matrices=True -> an (n, n) allocation; see the module docstring.
        skip_larger_than(bench_case, "bunny", "trimesh's major_axis allocates an (n, n) SVD matrix")
        points_np = bench_case.vertices_np
        axis_tm = bench_case.run(lambda: tm.points.major_axis(points_np))
        assert axis_tm.shape == (3,)


@pytest.mark.benchmark(group="centroid")
@pytest.mark.benchlibs("ordito", "numpy")
def test_centroid(bench_case: BenchCase) -> None:
    """
    The mean position of the cloud as a ``(1,)`` device array: one block-strided ``vec3`` sum.

    The scan sweep, the point count being the work. ``numpy.mean(axis=0)`` is the reference
    ``tests/test_points.py`` compares against; ordito's total stays on the device, so unlike the
    readback reductions in ``test_reduce.py`` its row pays no host synchronisation beyond the
    harness's own.
    """
    if bench_case.kind == "numpy":
        points_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        assert bench_case.run(lambda: points_np.mean(axis=0)).shape == (3,)
        return
    points = bench_case.vertices_wp
    assert bench_case.run(lambda: od.points.centroid(points)).shape == (1,)


@pytest.mark.benchmark(group="centered_covariance")
@pytest.mark.benchlibs("ordito")
def test_centered_covariance(bench_case: BenchCase) -> None:
    """
    The scatter matrix about the mean: ``centroid``, then ``gram_matrix``'s fold on centred points.

    At ``gram_matrix``'s operating point, so the pair prices the centring pass. ordito-only: its
    test checks it against the NumPy definition, not a library function.
    """
    points = bench_case.vertices_wp
    assert bench_case.run(lambda: od.points.centered_covariance(points)).shape == (1,)


@pytest.mark.benchmark(group="covariance")
@pytest.mark.benchlibs("ordito", "numpy")
def test_covariance(bench_case: BenchCase) -> None:
    """
    The sample covariance: ``centered_covariance`` scaled by ``1 / (n - 1)`` on the device.

    At ``centered_covariance``'s operating point. ``numpy.cov`` is the reference
    ``tests/test_points.py`` compares against.
    """
    if bench_case.kind == "numpy":
        points_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        assert bench_case.run(lambda: np.cov(points_np.T, ddof=1)).shape == (3, 3)
        return
    points = bench_case.vertices_wp
    assert bench_case.run(lambda: od.points.covariance(points)).shape == (1,)


@pytest.mark.benchmark(group="gram_matrix")
@pytest.mark.benchlibs("ordito")
def test_gram_matrix(bench_case: BenchCase) -> None:
    """
    The uncentred scatter matrix as a device array: one lane-strided block fold, no readback.

    ordito-only: NumPy's ``X.T @ X`` is the oracle in ``tests/test_points.py`` and a host matmul,
    not this reduction.
    """
    points = bench_case.vertices_wp
    gram = bench_case.run(lambda: od.points.gram_matrix(points))
    assert gram.shape == (1,)


_cloud_ml_cache: dict[str, mm.PointCloud] = {}
_cloud_normals_ml_cache: dict[str, mm.PointCloud] = {}
_points_ml_cache: dict[str, mm.std_vector_Vector3_float] = {}


def _cloud_ml(bench_case: BenchCase) -> mm.PointCloud:
    """
    Wrap the vertices in a ``meshlib.PointCloud``, cached per mesh.

    Cached because it is the *input*, and because MeshLib's projector-style objects keep a raw
    pointer to the cloud they are given -- a temporary is a segfault, not an exception (see
    ``test_proximity.py``'s ``closest_point_on_mesh`` row).
    """
    if bench_case.mesh_name not in _cloud_ml_cache:
        from meshlib import mrmeshnumpy as mn

        _cloud_ml_cache[bench_case.mesh_name] = mn.pointCloudFromPoints(
            cast("Buffer", np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64))
        )
    return _cloud_ml_cache[bench_case.mesh_name]


def _points_ml(bench_case: BenchCase) -> mm.std_vector_Vector3_float:
    """Build the vertices as a MeshLib vector, cached: it is the fit's input."""
    if bench_case.mesh_name not in _points_ml_cache:
        _points_ml_cache[bench_case.mesh_name] = std_points_ml(bench_case.vertices_np)
    return _points_ml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="half_space_mask")
@pytest.mark.benchlibs("ordito", "meshlib", "pyvista")
def test_half_space_mask(bench_case: BenchCase) -> None:
    """
    Select every point on one side of a plane: the same ``wp.map`` shape as the distance row.

    Read next to ``point_plane_distance``, which is the same pass writing a ``float32`` instead of a
    ``bool`` -- so the pair prices the output width and nothing else, and both sit on this module's
    wrapper floor rather than on any real arithmetic.

    meshlib's ``findHalfSpacePoints`` is the one query in family H that needs no callback: it takes
    the whole cloud and returns a ``VertBitSet``, so this row compares two batched calls rather than
    pricing an interpreter loop. Its plane is ``dot(n, x) = d`` where ordito takes a normal and a
    point on the plane, and the packed bitset is a 64x narrower write than a ``wp.bool`` array --
    both worth knowing before reading the ratio.

    On a small cloud the GPU row is several times *behind* meshlib -- which is this module's wrapper
    floor rather than the map, exactly as ``point_plane_distance`` warns. On a cloud two orders of
    magnitude larger ordito costs the same, so the floor is the whole story below about 10^5 points
    and the axis is the only honest way to read either row.

    pyvista reaches the mask through the same ``compute_implicit_distance`` the
    ``point_plane_distance`` row times, plus one host threshold -- so read the two pyvista rows as
    that threshold's cost, which is the same thing this pair measures on ordito's side. The
    threshold is inside the timed callable for that reason.
    """
    n_points = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        cloud_pv = pv.PolyData(bench_case.vertices_np)
        extent = 4.0 * float(
            np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
        )
        plane_pv = pv.Plane(
            center=(0.0, 0.0, 0.0), direction=_PLANE_NORMAL.tolist(), i_size=extent, j_size=extent
        )
        mask_pv = bench_case.run(
            lambda: (
                np.asarray(cloud_pv.compute_implicit_distance(plane_pv)["implicit_distance"]) > 0.0
            )
        )
        assert mask_pv.shape[0] == n_points
        return
    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        cloud_ml = _cloud_ml(bench_case)  # held in a name, per the helper's docstring
        plane_ml = mm.Plane3f(mm.Vector3f(*_PLANE_NORMAL.tolist()), 0.0)
        mask_ml = bench_case.run(lambda: mm.findHalfSpacePoints(cloud_ml, plane_ml))
        assert mask_ml.size() <= n_points
        return
    points = bench_case.vertices_wp
    normal = wp.vec3(*_PLANE_NORMAL.tolist())
    mask = bench_case.run(lambda: od.points.half_space_mask(points, normal))
    assert mask.shape == (n_points,)


@pytest.mark.benchmark(group="principal_axes")
@pytest.mark.benchlibs("ordito", "pyvista", "meshlib", "pymeshlab")
def test_principal_axes(bench_case: BenchCase) -> None:
    """
    Principal frame: centroid reduction, centred scatter, then one 3x3 SVD.

    meshlib splits the same work across two calls -- ``accumulatePoints`` builds the scatter matrix
    and ``getCenteredCovarianceEigen`` decomposes it -- and only the pair is comparable, so both are
    inside the timed callable. The ``std_vector_Vector3_float`` the accumulator consumes is *not*:
    filling it is a per-point Python loop, which is the input rather than the fit.

    pymeshlab's ``get_geometric_measures`` on a *face-less* MeshSet returns the vertex barycenter,
    the bounding box and the ``pca`` axes, so beyond ordito's answer it adds only a box. It is
    read-only, so the cloud MeshSet is shared across rounds.
    """
    if bench_case.kind == "pymeshlab":
        meshset_pml = _cloud_meshset_pml(bench_case)
        measures_pml = bench_case.run(meshset_pml.get_geometric_measures)
        assert np.asarray(measures_pml["pca"]).shape == (3, 3)
        return
    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_ml = _points_ml(bench_case)

        def principal_axes_ml() -> mm.Vector3d:
            accumulator_ml = mm.PointAccumulator()
            mm.accumulatePoints(accumulator_ml, points_ml)
            centroid_ml, eigenvectors_ml = mm.Vector3d(), mm.Matrix3d()
            eigenvalues_ml = mm.Vector3d()
            accumulator_ml.getCenteredCovarianceEigen(centroid_ml, eigenvectors_ml, eigenvalues_ml)
            return eigenvalues_ml

        assert bench_case.run(principal_axes_ml).z > 0.0
        return
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        rotation, eigenvalues, centroid = bench_case.run(lambda: od.points.principal_axes(points))
        assert len(rotation) == 3
        assert len(eigenvalues) == 3
        assert len(centroid) == 3
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        axes_pv = bench_case.run(lambda: pv.principal_axes(points_np))
        assert axes_pv.shape == (3, 3)


@pytest.mark.benchmark(group="fit_plane")
@pytest.mark.benchlibs("ordito", "trimesh", "igl", "pymeshlab", "pyvista", "meshlib")
def test_fit_plane(bench_case: BenchCase) -> None:
    """
    Least-squares plane: centroid reduction, centred covariance, then the smallest-sigma axis.

    pyvista's ``fit_plane_to_points(return_meta=True)`` returns the normal, a centre and a whole
    ``PolyData`` of the plane itself, so its row carries that geometry as well as the fit -- and its
    centre is *not* the centroid, which is why only the normal is compared
    (``tests/test_points.py``).
    """
    if bench_case.kind == "meshlib":
        # The same accumulator as the principal_axes row, read through getBestPlanef instead: it is
        # the cheaper half, since the plane needs no eigen-decomposition of its own.
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_ml = _points_ml(bench_case)

        def fit_plane_ml() -> mm.Plane3f:
            accumulator_ml = mm.PointAccumulator()
            mm.accumulatePoints(accumulator_ml, points_ml)
            return accumulator_ml.getBestPlanef()

        assert abs(bench_case.run(fit_plane_ml).n.length() - 1.0) < 1e-5
        return
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        _plane_pv, _centre_pv, normal_pv = bench_case.run(
            lambda: pv.fit_plane_to_points(points_np, return_meta=True)
        )
        assert np.asarray(normal_pv).shape == (3,)
        return
    if bench_case.kind == "pymeshlab":
        # ``compute_matrix_by_fitting_to_plane`` raises ``Cannot compute rotation: there is no
        # selection`` unless something is selected, so ``set_selection_all`` runs first (untimed --
        # it is how the filter is told "fit all the points", not part of the fit). It returns the
        # fitted normal and the average fitting error, which is ordito's answer plus a residual;
        # the rotation matrix it also builds is the part ordito does not do.
        meshset_pml = bench_case.meshset_pml
        meshset_pml.set_selection_all()
        result_pml = bench_case.run(
            lambda: meshset_pml.compute_matrix_by_fitting_to_plane(targetplane="XY plane")
        )
        assert result_pml["fitting_plane_normal"].shape == (3,)
        return
    if bench_case.kind == "igl":
        # ``igl.fit_plane`` returns the normal and the centroid, ordito's answer and nothing else:
        # two linear passes over the points and a 3x3 eigensolve, so the row runs uncapped.
        points_np = bench_case.vertices_np
        normal_igl, centroid_igl = bench_case.run(lambda: igl.fit_plane(points_np))
        assert normal_igl.shape == (3,)
        assert centroid_igl.shape == (3,)
        return
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        normal, centroid = bench_case.run(lambda: od.points.fit_plane(points))
        assert len(centroid) == 3
        assert len(normal) == 3
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        centroid_tm, normal_tm = bench_case.run(lambda: tm.points.plane_fit(points_np))
        assert centroid_tm.shape == (3,)
        assert normal_tm.shape == (3,)


@pytest.mark.benchmark(group="plane_basis")
@pytest.mark.benchlibs("ordito")
def test_plane_basis(bench_lib: BenchLibrary) -> None:
    """
    An orthonormal ``(u, v)`` for a plane normal: two cross products on host ``wp.vec3`` values.

    Host arithmetic on one vector with no array and no launch, so it is mesh-free and has no axis;
    the row prices the per-call cost of the ``wp.vec3`` operands. ordito-only: its test checks the
    basis's own properties.
    """
    normal = wp.vec3(*_PLANE_NORMAL.tolist())
    u, v = bench_lib.run(lambda: od.points.plane_basis(normal))
    assert abs(wp.dot(u, v)) < 1e-6


@pytest.mark.benchmark(group="vector_angle")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_vector_angle(bench_case: BenchCase) -> None:
    """Unsigned angle between paired unit vectors: the ``acos``-of-dot map."""
    n_points = bench_case.n_vertices
    if bench_case.kind == "ordito":
        normals = _unit_normals_wp(bench_case)
        directions = wp.array(
            np.ascontiguousarray(_unit_directions_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
        angles = bench_case.run(lambda: od.points.vector_angle(normals, directions))
        assert angles.shape == (n_points,)
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        pairs_np = np.stack(
            (np.array(mesh_tm.vertex_normals), _unit_directions_np(bench_case)), axis=1
        )
        angles_tm = bench_case.run(lambda: tm.geometry.vector_angle(pairs_np))
        assert angles_tm.shape == (n_points,)


@pytest.mark.benchmark(group="radial_sort")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_radial_sort(bench_case: BenchCase) -> None:
    """Order points by angle about an axis: the key kernel plus a radix sort."""
    n_points = bench_case.n_vertices
    origin_np = bench_case.vertices_np.mean(axis=0)
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        origin = wp.vec3(*origin_np.tolist())
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        ordered = bench_case.run(lambda: od.points.radial_sort(points, origin, normal))
        assert ordered.shape == (n_points,)
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        ordered_tm = bench_case.run(
            lambda: tm.points.radial_sort(points_np, origin=origin_np, normal=_PLANE_NORMAL)
        )
        assert ordered_tm.shape == (n_points, 3)


@pytest.mark.benchmark(group="space_filling_order")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("curve", ["hilbert", "morton"])
def test_space_filling_order(bench_case: BenchCase, curve: Literal["hilbert", "morton"]) -> None:
    """
    The vertices along a curve: a box reduction, the key kernel and a 30-bit radix sort.

    No reference row: no reference library orders a cloud along a space-filling curve.
    """
    points = bench_case.vertices_wp
    order = bench_case.run(lambda: od.points.space_filling_order(points, curve=curve))
    assert order.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="estimate_normals")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("k", _KNN_SWEEP)
def test_estimate_normals(bench_case: BenchCase, k: int) -> None:
    """
    PCA normals from a cached neighbour table: the covariance accumulation and ``svd3``.

    The table is an *input*, so this group is linear in ``k`` and nothing else. Comparing its slope
    against ``estimate_normals_knn`` below separates the estimator from the search that feeds it --
    which matters, because the search is where the time actually goes.
    """
    points = bench_case.vertices_wp
    neighbors = _neighbor_table(bench_case, k)
    normals = bench_case.run(lambda: od.points.estimate_normals(points, neighbors))
    assert normals.shape == (bench_case.n_vertices,)


def _cloud_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """
    Build the same vertices as a *face-less* pymeshlab mesh, once per mesh.

    ``compute_normal_for_point_clouds`` exists precisely for datasets with no faces, and writing the
    vertex normals leaves the positions alone, so this is cached rather than rebuilt per round.
    """
    if bench_case.mesh_name not in _cloud_pml_cache:
        meshset_pml = ml.MeshSet()
        meshset_pml.add_mesh(
            ml.Mesh(vertex_matrix=np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64))
        )
        _cloud_pml_cache[bench_case.mesh_name] = meshset_pml
    return _cloud_pml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="estimate_normals_knn")
@pytest.mark.benchaxis("scale")
@pytest.mark.censuskeep(
    "sphere_med",
    reason="a k > 1 search defers rows from _KNN_DEFER_MIN_POINTS points; sphere_small is below",
)
@pytest.mark.benchlibs("ordito", "open3d", "pymeshlab", "meshlib", "pytorch3d")
def test_estimate_normals_knn(bench_case: BenchCase) -> None:
    """
    Neighbour search plus PCA -- what open3d's ``estimate_normals`` does in one call.

    meshlib's ``makeUnorientedNormals`` searches by **radius** where the other three take a
    neighbour count, so its row is given the radius that holds ``_KNN`` points at this cloud's
    density (2 mean spacings) rather than a count; the two see the same neighbourhood only where the
    cloud is uniform, which is what the agreement in ``tests/test_points.py`` is checked on. It is
    also the only row here that returns a *new* array rather than writing into the cloud, so nothing
    is mutated and one cloud serves every round.

    **pytorch3d**'s ``estimate_pointcloud_normals`` is the same search-plus-PCA in one call and the
    only other row with GPU kernels, but its neighbour search is the brute-force pairwise loop, so
    at this cloud size it prices the search rather than the eigen-solve. It runs at its default
    ``disambiguate_directions=True``, the SHOT sign rule ordito's default orientation is, so the
    two rows compute the same normals (the parity test compares them sign included).
    """
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d searches one point at a time; capped at sphere_med")
    if bench_case.kind == "pytorch3d":
        points_p3d = points_torch_from_numpy(bench_case.vertices_np, bench_case.torch_device)
        normals_p3d = bench_case.run(
            lambda: p3d_ops.estimate_pointcloud_normals(points_p3d, neighborhood_size=_KNN)
        )
        assert normals_p3d.shape == (1, bench_case.n_vertices, 3)
        return
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)
        radius = 2.0 * bench_case.mean_edge
        normals_ml = bench_case.run(lambda: mm.makeUnorientedNormals(cloud_ml, radius))
        assert normals_ml.size() == bench_case.n_vertices
        return
    if bench_case.kind == "pymeshlab":
        # The same search-plus-PCA in one call, at the same ``k``; ``smoothiter=0`` keeps it to that
        # and leaves out the orientation propagation ordito does not do either. A *point-cloud*
        # MeshSet: the filter is for datasets with no faces.
        cloud_pml = _cloud_meshset_pml(bench_case)
        bench_case.run(lambda: cloud_pml.compute_normal_for_point_clouds(k=_KNN, smoothiter=0))
        assert cloud_pml.current_mesh().vertex_normal_matrix().shape == (bench_case.n_vertices, 3)
        return
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        normals = bench_case.run(
            lambda: od.points.estimate_normals(
                points, od.neighbors.query_nearest(points, points, k=_KNN)[0]
            )
        )
        assert normals.shape == (bench_case.n_vertices,)
    else:
        cloud = _pcd(bench_case)
        search = o3d.geometry.KDTreeSearchParamKNN(knn=_KNN)
        bench_case.run(lambda: cloud.estimate_normals(search_param=search))
        assert np.asarray(cloud.normals).shape == (bench_case.n_vertices, 3)


def _vertex_normals_np(bench_case: BenchCase) -> np.ndarray:
    """
    Area-weighted unit vertex normals on the host, for the references' clouds (setup, not timed).

    The same normals as ``_unit_normals_wp`` up to rounding; a reference case has no device to
    build those on.
    """
    vertices, faces = bench_case.vertices_np.astype(np.float64), bench_case.faces_np
    corners = vertices[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)


@pytest.mark.benchmark(group="orient_normals")
@pytest.mark.benchlibs("ordito", "open3d")
def test_orient_normals(bench_case: BenchCase) -> None:
    """
    Minimum-spanning-tree propagation, neighbour search included, on the vertex normals.

    open3d's ``orient_normals_consistent_tangent_plane`` builds its own k-d tree and a Delaunay
    tetrahedralization (for the Euclidean spanning tree it adds) every call, then runs Kruskal and
    a breadth-first walk on one thread; ordito sorts the graph's weights, runs Boruvka rounds and a
    parity union-find. The open3d call orients the cloud's normals in place, so later rounds start
    from oriented normals, which changes no work. The tetrahedralization caps it at ``bunny``.
    """
    knn = 10
    if bench_case.kind == "open3d":
        skip_larger_than(bench_case, "bunny", "open3d tetrahedralizes the cloud on one thread")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bench_case.vertices_np))
        pcd.normals = o3d.utility.Vector3dVector(_vertex_normals_np(bench_case))
        bench_case.run(lambda: pcd.orient_normals_consistent_tangent_plane(knn))
        return
    points = bench_case.vertices_wp
    normals = _unit_normals_wp(bench_case)
    oriented = bench_case.run(
        lambda: od.points.orient_normals(
            points, normals, od.neighbors.query_nearest(points, points, k=knn)[0]
        )
    )
    assert oriented.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="smooth_normals")
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_smooth_normals(bench_case: BenchCase) -> None:
    """
    One pass of sign-aligned normal averaging, neighbour search included.

    pymeshlab's ``apply_normal_point_cloud_smoothing`` builds its k-d tree and searches every call;
    its MeshSet's normals are overwritten in place, which changes no work. ``k = 10``.
    """
    knn = 10
    if bench_case.kind == "pymeshlab":
        meshset_pml = ml.MeshSet()
        meshset_pml.add_mesh(
            ml.Mesh(
                vertex_matrix=np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64),
                v_normals_matrix=_vertex_normals_np(bench_case),
            )
        )
        bench_case.run(lambda: meshset_pml.apply_normal_point_cloud_smoothing(k=knn))
        return
    points = bench_case.vertices_wp
    normals = _unit_normals_wp(bench_case)
    smoothed = bench_case.run(
        lambda: od.points.smooth_normals(
            normals, od.neighbors.query_nearest(points, points, k=knn)[0]
        )
    )
    assert smoothed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="smooth_points")
@pytest.mark.benchlibs("ordito", "open3d")
def test_smooth_points(bench_case: BenchCase) -> None:
    """
    Ten Taubin iterations over a fixed ``k = 10`` table, the table's search included.

    open3d's ``smooth_taubin(use_fixed_neighborhoods=True)`` searches once and steps twenty times,
    as ordito does; it returns a new cloud.
    """
    knn, iterations = 10, 10
    if bench_case.kind == "open3d":
        pcd = _pcd(bench_case)
        smoothed_o3d = bench_case.run(lambda: pcd.smooth_taubin(iterations, 0.5, -0.53, knn, True))
        assert np.asarray(smoothed_o3d.points).shape == (bench_case.n_vertices, 3)
        return
    points = bench_case.vertices_wp
    smoothed = bench_case.run(
        lambda: od.points.smooth_points(
            points,
            od.neighbors.query_nearest(points, points, k=knn + 1)[0],
            iterations,
            0.5,
            mu=-0.53,
        )
    )
    assert smoothed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="smooth_points_mls")
@pytest.mark.benchlibs("ordito", "open3d")
def test_smooth_points_mls(bench_case: BenchCase) -> None:
    """
    One moving-least-squares projection over a hybrid search, the search included.

    Three mean edges, at most 30 neighbours, Gaussian weights at that radius. open3d's
    ``smooth_mls`` returns a new cloud; this cloud has no normals, so it skips its orientation
    pass, and ordito's normals are its PCA normals, unoriented, as open3d's would be before it.
    """
    radius = 3.0 * bench_case.mean_edge
    if bench_case.kind == "open3d":
        pcd = _pcd(bench_case)
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=_KNN)
        smoothed_o3d = bench_case.run(lambda: pcd.smooth_mls(search))
        assert np.asarray(smoothed_o3d.points).shape == (bench_case.n_vertices, 3)
        return
    points = bench_case.vertices_wp
    smoothed, _normals = bench_case.run(
        lambda: od.points.smooth_points_mls(
            points, od.neighbors.query_nearest(points, points, k=_KNN, max_radius=radius)[0], radius
        )
    )
    assert smoothed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="smooth_points_bilateral")
@pytest.mark.benchlibs("ordito", "open3d")
def test_smooth_points_bilateral(bench_case: BenchCase) -> None:
    """
    One bilateral pass over ``k = 30`` nearest on the vertex normals, the search included.

    Spatial scale two mean edges, range scale half a mean edge.
    """
    sigma_s, sigma_r = 2.0 * bench_case.mean_edge, 0.5 * bench_case.mean_edge
    if bench_case.kind == "open3d":
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bench_case.vertices_np))
        pcd.normals = o3d.utility.Vector3dVector(_vertex_normals_np(bench_case))
        search = o3d.geometry.KDTreeSearchParamKNN(knn=_KNN)
        smoothed_o3d = bench_case.run(lambda: pcd.smooth_bilateral(search, sigma_s, sigma_r))
        assert np.asarray(smoothed_o3d.points).shape == (bench_case.n_vertices, 3)
        return
    points = bench_case.vertices_wp
    normals = _unit_normals_wp(bench_case)
    smoothed = bench_case.run(
        lambda: od.points.smooth_points_bilateral(
            points, normals, od.neighbors.query_nearest(points, points, k=_KNN)[0], sigma_s, sigma_r
        )
    )
    assert smoothed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="point_covariances")
@pytest.mark.benchlibs("ordito", "open3d")
def test_point_covariances(bench_case: BenchCase) -> None:
    """Per-point covariances over ``k = 30`` nearest, search included (open3d's default ``knn``)."""
    if bench_case.kind == "open3d":
        pcd = _pcd(bench_case)
        search = o3d.geometry.KDTreeSearchParamKNN(knn=_KNN)
        bench_case.run(lambda: pcd.estimate_covariances(search))
        assert np.asarray(pcd.covariances).shape == (bench_case.n_vertices, 3, 3)
        return
    points = bench_case.vertices_wp
    covariances = bench_case.run(
        lambda: od.points.point_covariances(
            points, od.neighbors.query_nearest(points, points, k=_KNN)[0]
        )
    )
    assert covariances.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="point_frames")
@pytest.mark.benchlibs("ordito", "pytorch3d")
def test_point_frames(bench_case: BenchCase) -> None:
    """
    Per-point principal frames over ``k = 30`` nearest, search included.

    pytorch3d's ``estimate_pointcloud_local_coord_frames`` runs its brute-force k-nearest search
    and a batched ``eigh``, so it prices the search, as in ``estimate_normals_knn``.
    """
    if bench_case.kind == "pytorch3d":
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("pytorch3d's brute-force search; capped at sphere_med")
        points_p3d = points_torch_from_numpy(bench_case.vertices_np, bench_case.torch_device)
        _curvatures, frames_p3d = bench_case.run(
            lambda: p3d_ops.estimate_pointcloud_local_coord_frames(
                points_p3d, neighborhood_size=_KNN
            )
        )
        assert frames_p3d.shape == (1, bench_case.n_vertices, 3, 3)
        return
    points = bench_case.vertices_wp
    frames, _eigenvalues = bench_case.run(
        lambda: od.points.point_frames(
            points, od.neighbors.query_nearest(points, points, k=_KNN)[0]
        )
    )
    assert frames.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="local_triangulation")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_local_triangulation(bench_case: BenchCase) -> None:
    """
    Every point's local Delaunay star over ``k = 30`` nearest, search and normals included.

    potpourri3d's ``PointCloudLocalTriangulation`` runs its kd-tree search, an SVD a point for the
    normals and the stars one point at a time, and pads its answer to the widest star; ordito packs
    the stars into one triangle buffer.
    """
    if bench_case.kind == "potpourri3d":
        positions_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64)
        triangulation_pp = bench_case.run(
            lambda: pp3d.PointCloudLocalTriangulation(positions_np).get_local_triangulation()
        )
        assert triangulation_pp.shape[0] == bench_case.n_vertices
        return
    points = bench_case.vertices_wp
    faces = bench_case.run(
        lambda: od.points.local_triangulation(
            points, od.neighbors.query_nearest(points, points, k=_KNN + 1)[0]
        )
    )
    assert faces.size >= 3 * bench_case.n_vertices


@pytest.mark.benchmark(group="point_boundary_mask")
@pytest.mark.benchlibs("ordito", "open3d")
@pytest.mark.censusmesh("saddle_small")
def test_point_boundary_mask(bench_case: BenchCase) -> None:
    """
    Angular-gap boundary test on the vertex normals, the hybrid search included.

    open3d's tensor ``compute_boundary_points`` searches by radius with a ``max_nn`` cap and
    heap-sorts each row's angles; ordito's table is the same search
    (``query_nearest(max_radius=)``). Three mean edges, at most 30 neighbours.
    """
    radius = 3.0 * bench_case.mean_edge
    if bench_case.kind == "open3d":
        # The stub drops the dtype and device defaults.
        positions_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        cloud_o3d = o3d.t.geometry.PointCloud(o3d.core.Tensor(positions_np))  # pyright: ignore[reportCallIssue]
        normals_np = _vertex_normals_np(bench_case).astype(np.float32)
        cloud_o3d.point.normals = o3d.core.Tensor(normals_np)  # pyright: ignore[reportCallIssue]
        _boundary_o3d, mask_o3d = bench_case.run(
            lambda: cloud_o3d.compute_boundary_points(radius, _KNN)
        )
        assert mask_o3d.shape[0] == bench_case.n_vertices
        return
    points = bench_case.vertices_wp
    normals = _unit_normals_wp(bench_case)
    mask = bench_case.run(
        lambda: od.points.point_boundary_mask(
            points,
            normals,
            od.neighbors.query_nearest(points, points, k=_KNN, max_radius=radius)[0],
        )
    )
    assert mask.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="outlier_probability")
@pytest.mark.benchaxis("scale")
@pytest.mark.censuskeep(
    "sphere_med",
    reason="a k > 1 search defers rows from _KNN_DEFER_MIN_POINTS points; sphere_small is below",
)
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_outlier_probability(bench_case: BenchCase) -> None:
    """LoOP scores: the k-NN table plus four passes over it, against the filter they came from."""
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("MeshLab's k-d tree searches one point at a time; capped at sphere_med")
    if bench_case.kind == "pymeshlab":
        # Selection-only, so the geometry is untouched and the MeshSet is shared; the filter still
        # rebuilds its k-d tree every call, which is why ordito's row builds its table in-callable.
        cloud_pml = _cloud_meshset_pml(bench_case)
        bench_case.run(
            lambda: cloud_pml.compute_selection_point_cloud_outliers(
                propthreshold=0.8, knearest=_KNN
            )
        )
        assert cloud_pml.current_mesh().vertex_selection_array().shape == (bench_case.n_vertices,)
        return
    points = bench_case.vertices_wp
    probability = bench_case.run(
        lambda: od.points.outlier_probability(*od.neighbors.query_nearest(points, points, k=_KNN))
    )
    assert probability.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="statistical_outlier_mask")
@pytest.mark.benchaxis("scale")
@pytest.mark.censuskeep(
    "sphere_med",
    reason="a k > 1 search defers rows from _KNN_DEFER_MIN_POINTS points; sphere_small is below",
)
@pytest.mark.benchlibs("ordito", "open3d")
def test_statistical_outlier_mask(bench_case: BenchCase) -> None:
    """One global threshold on the mean neighbour distance — open3d's own outlier criterion."""
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d searches one point at a time; capped at sphere_med")
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        mask = bench_case.run(
            lambda: od.points.statistical_outlier_mask(
                od.neighbors.query_nearest(points, points, k=_KNN)[1]
            )
        )
        assert mask.shape == (bench_case.n_vertices,)
    else:
        # ``remove_statistical_outlier`` also materializes the kept subset, which ordito does not.
        cloud = _pcd(bench_case)
        _kept, keep_indices = bench_case.run(
            lambda: cloud.remove_statistical_outlier(nb_neighbors=_KNN, std_ratio=2.0)
        )
        assert len(keep_indices) <= bench_case.n_vertices


@pytest.mark.benchmark(group="radius_outlier_mask")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "open3d")
@pytest.mark.parametrize("radius_scale", _RADIUS_SCALES)
def test_radius_outlier_mask(bench_case: BenchCase, radius_scale: float) -> None:
    """
    A ball count plus one threshold map, swept over the radius rather than the mesh.

    The radius is the whole cost model: the count is linear in how many points each ball holds, so a
    2x radius is ~8x the point tests, and where the hash grid stops paying for itself is what the
    two points bracket. Nothing after the count scales -- the threshold is one ``wp.map``.

    open3d's ``remove_radius_outlier`` builds its own ``KDTreeFlann`` per call and *materializes*
    the kept subset, so its row is an upper bound on the same question; it is also
    **nondeterministic** (a shared tree across an OpenMP loop), which is why the comparison in
    ``tests/test_points.py`` goes through that tree serially instead.
    """
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d searches one point at a time; capped at sphere_med")
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        mask = bench_case.run(lambda: od.points.radius_outlier_mask(points, radius, _MIN_NEIGHBORS))
        assert mask.shape == (bench_case.n_vertices,)
    else:
        cloud = _pcd(bench_case)
        _kept, keep_indices = bench_case.run(
            lambda: cloud.remove_radius_outlier(nb_points=_MIN_NEIGHBORS, radius=radius)
        )
        assert len(keep_indices) <= bench_case.n_vertices


def _find_outliers_ml(
    bench_case: BenchCase, mask: mm.OutlierTypeMask, radius: float, cloud_ml: mm.PointCloud
) -> None:
    """Time ``findOutliers`` for one criterion, its point tree dropped each round as ordito's is."""

    def uncached_cloud_ml() -> mm.PointCloud:
        """Drop the lazily built point tree, which the search would otherwise reuse."""
        cloud_ml.invalidateCaches()
        return cloud_ml

    params_ml = mm.FindOutliersParams()
    params_ml.radius = radius
    params_ml.mask = mask
    outliers_ml = bench_case.run(
        lambda cloud: mm.findOutliers(cloud, params_ml), setup=uncached_cloud_ml
    )
    assert outliers_ml.size() <= bench_case.n_vertices


@pytest.mark.benchmark(group="cluster_outlier_mask")
@pytest.mark.benchlibs("ordito", "meshlib", "open3d")
def test_cluster_outlier_mask(bench_case: BenchCase) -> None:
    """
    Clusters of the radius graph: a ball walk hooking a union-find as it goes, a count.

    meshlib's ``findOutliers(SmallComponents)`` is the same union of balls (equal sets in
    ``tests/test_points.py``), unioned on its threads as it searches. open3d's
    ``cluster_dbscan(eps=radius, min_points=1)`` makes every point a core point, so its labels are
    the same components; the row times the labelling plus the label-size count that turns it into
    the mask, and the k-d tree the call builds for itself. Two mean edges, the default cluster
    size of 20.
    """
    radius = 2.0 * bench_case.mean_edge
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)  # held in a name: MeshLib's trees point into it
        _find_outliers_ml(bench_case, mm.OutlierTypeMask.SmallComponents, radius, cloud_ml)
        return
    if bench_case.kind == "open3d":
        pcd = _pcd(bench_case)

        def cluster_outlier_mask_o3d() -> np.ndarray:
            labels_o3d = np.asarray(pcd.cluster_dbscan(eps=radius, min_points=1))
            return np.bincount(labels_o3d)[labels_o3d] <= 20

        assert bench_case.run(cluster_outlier_mask_o3d).shape == (bench_case.n_vertices,)
        return
    points = bench_case.vertices_wp
    mask = bench_case.run(lambda: od.points.cluster_outlier_mask(points, radius))
    assert mask.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="cluster_labels")
@pytest.mark.benchlibs("ordito", "open3d")
def test_cluster_labels(bench_case: BenchCase) -> None:
    """
    DBSCAN at two mean edges and ``min_points = 6``: a count walk, a hook walk, a label walk.

    open3d's ``cluster_dbscan`` stores every point's radius search and grows its clusters on one
    thread; the labels are equal (``tests/test_points.py``).
    """
    radius = 2.0 * bench_case.mean_edge
    if bench_case.kind == "open3d":
        pcd = _pcd(bench_case)
        labels_o3d = bench_case.run(lambda: pcd.cluster_dbscan(eps=radius, min_points=6))
        assert len(labels_o3d) == bench_case.n_vertices
        return
    points = bench_case.vertices_wp
    labels = bench_case.run(lambda: od.points.cluster_labels(points, radius, 6))
    assert labels.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="plane_outlier_mask")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_plane_outlier_mask(bench_case: BenchCase) -> None:
    """
    A ``float64`` plane fit over each point's ball, one launch: the moments and the eigenvector.

    meshlib's ``findOutliers(FarSurface)`` fits the same plane per point; the ordito row runs its
    ``method="meshlib"`` side rule (equal sets in ``tests/test_points.py``). Two mean edges, a
    height of 0.3 of the radius (meshlib's default ``minHeight``).
    """
    radius = 2.0 * bench_case.mean_edge
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)  # held in a name: MeshLib's trees point into it
        _find_outliers_ml(bench_case, mm.OutlierTypeMask.FarSurface, radius, cloud_ml)
        return
    points = bench_case.vertices_wp
    mask = bench_case.run(
        lambda: od.points.plane_outlier_mask(points, radius, 0.3 * radius, method="meshlib")
    )
    assert mask.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="normal_outlier_mask")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_normal_outlier_mask(bench_case: BenchCase) -> None:
    """
    A normal sum over each point's ball and one angle, one launch.

    meshlib's ``findOutliers(AwayNormal)`` on the same cloud and area-weighted vertex normals
    (formed on the host for its row; equal sets in ``tests/test_points.py``). Two mean edges,
    meshlib's default angle of 60°.
    """
    radius = 2.0 * bench_case.mean_edge
    if bench_case.kind == "meshlib":
        if bench_case.mesh_name not in _cloud_normals_ml_cache:
            from meshlib import mrmeshnumpy as mn

            vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np.reshape(-1, 3)
            corners_np = vertices_np[faces_np]
            areas_np = np.cross(
                corners_np[:, 1] - corners_np[:, 0], corners_np[:, 2] - corners_np[:, 0]
            )
            normals_np = np.zeros_like(vertices_np)
            for k in range(3):
                np.add.at(normals_np, faces_np[:, k], areas_np)
            normals_np /= np.maximum(np.linalg.norm(normals_np, axis=1, keepdims=True), 1e-30)
            _cloud_normals_ml_cache[bench_case.mesh_name] = mn.pointCloudFromPoints(
                cast("Buffer", np.ascontiguousarray(vertices_np, dtype=np.float64)),
                cast("Buffer", np.ascontiguousarray(normals_np, dtype=np.float64)),
            )
        cloud_ml = _cloud_normals_ml_cache[bench_case.mesh_name]
        _find_outliers_ml(bench_case, mm.OutlierTypeMask.AwayNormal, radius, cloud_ml)
        return
    points, normals = bench_case.vertices_wp, _unit_normals_wp(bench_case)
    mask = bench_case.run(lambda: od.points.normal_outlier_mask(points, normals, radius))
    assert mask.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="point_finite_mask")
@pytest.mark.benchlibs("ordito")
def test_point_finite_mask(bench_case: BenchCase) -> None:
    """
    One ``isfinite`` flag per point: a single ``wp.map`` over the cloud.

    The scan sweep, the point count being the work; expect the launch floor until the largest
    clouds. ordito-only: open3d's ``remove_non_finite_points``, its test's reference, copies the
    surviving cloud out, so its row would time the copy (``tests/test_points.py`` records why).
    """
    points = bench_case.vertices_wp
    mask = bench_case.run(lambda: od.points.point_finite_mask(points))
    assert mask.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="point_duplicate_mask")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "open3d", "meshlib")
def test_point_duplicate_mask(bench_case: BenchCase) -> None:
    """
    Exact positional dedup: one hashed pass over a table of point indices, then a mask launch.

    Two launches and two allocations with no readback: each point probes an open-addressing table,
    compares a candidate's bits against its own, and lowers its class slot to the smallest index.

    open3d's ``remove_duplicated_points`` answers the same question with an
    ``unordered_map<Vector3d>`` on one core and copies the survivors out; the mask here is compared
    against its survivor list in ``tests/test_points.py``. A registry mesh has *no* duplicated
    vertices, so all three rows do their full work and none takes an early exit.

    meshlib's ``findSmallestCloseVertices(cloud, 0.0)`` is the third implementation and the only
    multi-threaded one: it returns each point's smallest-indexed coincident neighbour, which is
    this mask under ``map != index`` (``tests/test_points.py``). It goes through the cloud's point
    tree rather than a hash, so unlike the other two rows its cost is a *search* -- which is why
    the tree is dropped per round the way the ``nearest_neighbor_distance`` row drops it, rather
    than being reused across rounds and pricing the query alone.
    """
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)  # held in a name: MeshLib's trees point into it

        def uncached_cloud_ml() -> mm.PointCloud:
            """Drop the lazily built point tree, which the search below would otherwise reuse."""
            cloud_ml.invalidateCaches()
            return cloud_ml

        representative_ml = bench_case.run(
            lambda cloud: mm.findSmallestCloseVertices(cloud, 0.0), setup=uncached_cloud_ml
        )
        assert representative_ml.size() == bench_case.n_vertices
        return
    if bench_case.kind == "open3d":
        # The dedup mutates nothing -- it returns a fresh cloud -- so one cached input cloud is
        # valid for every round, unlike the ``remove_*`` methods that rewrite in place.
        cloud = _pcd(bench_case)
        deduplicated_o3d = bench_case.run(cloud.remove_duplicated_points)
        assert len(deduplicated_o3d.points) <= bench_case.n_vertices
        return
    points = bench_case.vertices_wp
    mask = bench_case.run(lambda: od.points.point_duplicate_mask(points))
    assert mask.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="farthest_point_sample")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "open3d", "pytorch3d")
@pytest.mark.parametrize("count", _SAMPLE_COUNTS)
def test_farthest_point_sample(bench_case: BenchCase, count: int) -> None:
    """
    The greedy maximin subsample, swept over the sample count -- which is the round count.

    Inherently sequential in ``count``: each round folds one selected point into the running
    distances and takes a global arg-max, so the work is ``Theta(count)`` rounds over the whole
    cloud however small the sample is -- run as one persistent block with a ``wp.tile_max`` per
    round, not as one launch per round. That makes this the one group in the module whose slope is
    set by a *parameter* rather than by the mesh, and the two counts here are a 16x step in it.

    open3d's ``FarthestPointDownSample`` runs the identical loop serially in C++ on one core, so
    the comparison is device parallelism against a tighter inner loop; it also copies the selected
    points out where ordito returns indices.

    **pytorch3d**'s ``sample_farthest_points`` runs the identical greedy loop with CUDA kernels of
    its own, so it is the one row here that is not a serial C++ baseline -- and the ratio is the
    narrowest and the flattest of its four, holding at roughly 3x across the point-count axis where
    its brute-force k-NN swings by two orders of magnitude because it has no structure to build.
    ``random_start_point=False`` pins its start to index 0, which is ordito's default, and it
    returns both the points and the indices where ordito returns indices alone.
    """
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d's greedy loop is serial over the whole cloud; capped at sphere_med")
    if bench_case.kind == "pytorch3d":
        points_p3d = points_torch_from_numpy(bench_case.vertices_np, bench_case.torch_device)
        _, indices_p3d = bench_case.run(
            lambda: p3d_ops.sample_farthest_points(points_p3d, K=count, random_start_point=False)
        )
        assert indices_p3d.shape == (1, count)
        return
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        indices = bench_case.run(lambda: od.points.farthest_point_sample(points, count))
        assert indices.shape == (count,)
    else:
        cloud = _pcd(bench_case)
        sampled_o3d = bench_case.run(lambda: cloud.farthest_point_down_sample(count))
        assert len(sampled_o3d.points) == count


# Direction counts for the support sweep. Cost is exactly ``points x n_directions`` -- the only
# knob in the module, and the accuracy/speed trade against exact qhull.
_N_DIRECTIONS = [32, 256]

# Icosphere refinement levels for the conservative filter: 42 directions / 80 tetrahedra at 1, and
# 642 / 1280 at 3. Both halves of its cost scale with this, so it spans the useful range.
_SUBDIVISIONS = [1, 3]


@pytest.mark.benchmark(group="convex_hull")
@pytest.mark.benchlibs(
    "ordito", "scipy", "trimesh", "open3d", "pymeshlab", "manifold3d", "meshlib", "pyvista"
)
def test_convex_hull(bench_case: BenchCase) -> None:
    """
    The exact hull against qhull, directly and through the three mesh wrappers, and three others.

    The same result on both sides: the hull's triangles, ordito's over the input points and the
    wrappers' as a compacted mesh (``scipy`` returns ``simplices`` over the input, like ordito).
    qhull is single-threaded on the host, which is the comparison: a registry mesh's vertices are a
    scan cloud, whose hull holds a few thousand of its points, and both sides discard the rest.
    manifold3d, MeshLib (``makeConvexHull`` of a ``PointCloud``) and pyvista (VTK 9.7's
    ``vtkConvexHull``) each build the hull without qhull; their input cloud is built outside the
    timed callable.
    """
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        faces = bench_case.run(lambda: od.points.convex_hull(points))
        assert faces.shape[0] >= 12
        return
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)
        hull_ml = bench_case.run(lambda: mm.makeConvexHull(cloud_ml))
        assert hull_ml.topology.numValidFaces() >= 4
        return
    if bench_case.kind == "pyvista":
        cloud_pv = pv.PolyData(np.array(bench_case.vertices_np, dtype=np.float64))
        hull_pv = bench_case.run(cloud_pv.convex_hull)
        assert hull_pv.n_cells >= 4
        return
    if bench_case.kind == "manifold3d":
        # QuickHull on the host, multi-threaded, returning the hull as a solid.
        points_np = np.asarray(bench_case.vertices_np, dtype=np.float64)
        hull_m3 = bench_case.run(lambda: m3.Manifold.hull_points(points_np))
        assert hull_m3.num_tri() >= 4
        return
    skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
    points_np = bench_case.vertices_np
    if bench_case.kind == "scipy":
        hull_np = bench_case.run(lambda: scipy.spatial.ConvexHull(points_np))
        assert hull_np.simplices.shape[0] >= 4
    elif bench_case.kind == "pymeshlab":
        bench_case.run(lambda: bench_case.new_meshset_pml().generate_convex_hull())
    elif bench_case.kind == "trimesh":
        hull_tm = bench_case.run(lambda: tm.points.PointCloud(points_np).convex_hull)
        assert hull_tm.faces.shape[0] >= 4
    else:
        mesh_o3d = bench_case.mesh_o3d
        hull_o3d, _indices = bench_case.run(lambda: mesh_o3d.compute_convex_hull())
        assert np.asarray(hull_o3d.triangles).shape[0] >= 4


@pytest.mark.benchmark(group="convex_subset_mask")
@pytest.mark.benchlibs("ordito", "trimesh", "open3d", "pymeshlab")
@pytest.mark.parametrize("n_directions", _N_DIRECTIONS)
def test_convex_subset_mask(bench_case: BenchCase, n_directions: int) -> None:
    """
    Tiled support sweep vs exact qhull (see the module docstring).

    Cost is ``points x n_directions`` with no topology involved, so the direction count is the
    axis. The references are exact and take no such parameter, so their two rows are identical by
    construction -- they are there as the fixed bar the approximation is trading accuracy against.
    """
    if bench_case.kind == "pymeshlab":  # qhull again, through MeshLab's own wrapper
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        bench_case.run(lambda: bench_case.new_meshset_pml().generate_convex_hull())
        return
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        mask = bench_case.run(
            lambda: od.points.convex_subset_mask(points, n_directions=n_directions)
        )
        assert mask.shape[0] == bench_case.n_vertices
    elif bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        points_np = bench_case.vertices_np
        hull_tm = bench_case.run(lambda: tm.points.PointCloud(points_np).convex_hull)
        assert hull_tm.vertices.shape[1] == 3
    else:
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        mesh_o3d = bench_case.mesh_o3d
        hull_o3d, _indices = bench_case.run(lambda: mesh_o3d.compute_convex_hull())
        assert np.asarray(hull_o3d.vertices).shape[1] == 3


@pytest.mark.benchmark(group="convex_subset")
@pytest.mark.benchlibs("ordito", "trimesh", "open3d", "pymeshlab")
def test_convex_subset(bench_case: BenchCase) -> None:
    """
    The mask plus ``flatnonzero`` and a gather: isolates the compaction cost.

    The same three qhull wrappers ``convex_subset_mask`` times, because the module docstring's
    argument -- that the exact hull is the accuracy bar the approximation trades against -- is about
    the *operation*, not about which of its two entry points is called. Both return the hull's
    vertices; only the container differs, and a qhull wrapper returns positions rather than a mask,
    so if anything this is the closer shape of the two. The rows carried only on the mask group for
    as long as they did because that group was written first.

    No ``n_directions`` axis here (unlike the mask group): the compaction this group isolates does
    not scale with the direction count, so one row per library is the whole comparison.
    """
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        selected = bench_case.run(lambda: od.points.convex_subset(points))
        assert selected.shape[0] <= bench_case.n_vertices
        return
    skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
    if bench_case.kind == "pymeshlab":  # qhull again, through MeshLab's own wrapper
        bench_case.run(lambda: bench_case.new_meshset_pml().generate_convex_hull())
    elif bench_case.kind == "trimesh":
        points_np = bench_case.vertices_np
        hull_tm = bench_case.run(lambda: tm.points.PointCloud(points_np).convex_hull)
        assert hull_tm.vertices.shape[1] == 3
    else:
        mesh_o3d = bench_case.mesh_o3d
        hull_o3d, _indices = bench_case.run(lambda: mesh_o3d.compute_convex_hull())
        assert np.asarray(hull_o3d.vertices).shape[1] == 3


@pytest.mark.benchmark(group="convex_superset_mask")
@pytest.mark.benchlibs("ordito", "scipy")
@pytest.mark.parametrize("subdivisions", _SUBDIVISIONS)
def test_convex_superset_mask(bench_case: BenchCase, subdivisions: int) -> None:
    """
    The conservative prefilter against the exact hull it prefilters for.

    ``scipy`` is the right row here, unlike in the two groups above where the qhull wrappers are
    only a fixed accuracy bar: this filter's *purpose* is to run before an exact hull, so the
    question the benchmark has to answer is whether it is cheap relative to the hull it feeds. Both
    rows are timed on the same cloud and produce comparable results (the filter's output contains
    the hull's vertex set, which ``tests/test_points.py`` asserts), so this one *is* a parity
    comparison.

    ``subdivisions`` is the axis because it drives both halves of the cost -- the direction count of
    the support sweep and the tetrahedron count of the interior test -- and is the knob that trades
    selectivity for time.
    """
    if bench_case.kind == "ordito":
        points = bench_case.vertices_wp
        mask = bench_case.run(
            lambda: od.points.convex_superset_mask(points, subdivisions=subdivisions)
        )
        assert mask.shape[0] == bench_case.n_vertices
    else:
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        points_np = bench_case.vertices_np
        hull_np = bench_case.run(lambda: scipy.spatial.ConvexHull(points_np))
        assert hull_np.vertices.shape[0] >= 4


@pytest.mark.benchmark(group="point_visible_mask")
@pytest.mark.benchlibs("ordito", "open3d")
def test_point_visible_mask(bench_case: BenchCase) -> None:
    """
    Hidden point removal from a viewpoint two extents above the centre, radius 100 extents.

    Both are a spherical flip plus a convex hull: open3d's through qhull on the host, which also
    builds the hull's mesh, ordito's through ``convex_hull`` on the flipped cloud. About half of a
    closed scan is visible, so the hull holds far more of the cloud than a plain hull does.
    """
    vertices_np = bench_case.vertices_np
    extent = float(np.ptp(vertices_np, axis=0).max())
    camera_np = vertices_np.mean(axis=0) + np.array([0.0, 0.0, 2.0 * extent])
    radius = 100.0 * extent
    if bench_case.kind == "open3d":
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        pcd = _pcd(bench_case)
        _mesh_o3d, visible_o3d = bench_case.run(
            lambda: pcd.hidden_point_removal(camera_np.astype(np.float64), radius)
        )
        assert 0 < len(visible_o3d) < bench_case.n_vertices
        return
    points = bench_case.vertices_wp
    camera = wp.vec3(*camera_np.tolist())
    visible = bench_case.run(lambda: od.points.point_visible_mask(points, camera, radius))
    assert visible.shape == (bench_case.n_vertices,)
