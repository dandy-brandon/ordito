"""
Benchmarks for ``ordito.metrics``: the differentiable Chamfer loss and the plain metrics.

Cloud/mesh B is the same mesh translated by 5% of its bbox diagonal (untimed setup), so both
directions of every symmetric metric do real work.

Differentiable case
-------------------
``chamfer_mesh_to_mesh_loss`` carries ``requires_grad=True`` on both vertex buffers; each timed
round records a fresh tape, runs backward and zeroes the gradients, which is the real
optimization-loop cost. It is ordito-only: open3d has no autodiff, so timing forward-only against
forward-plus-backward would be a misleading ratio rather than a useful baseline.

Non-differentiable cases
------------------------
``chamfer_points_to_points`` and ``hausdorff_points_to_points`` are each **two**
[`query_nearest`][ordito.neighbors.query_nearest] calls at ``k=1`` plus a
reduction, so they are the direct measurement for a change to the k-NN kernel — the same kernel
that sits under all twelve of ``distance.py``'s nearest-neighbour call sites.

**open3d** is the reference: ``PointCloud.compute_point_cloud_distance`` returns exactly the forward
nearest-neighbour Euclidean distances (a serial ``KDTreeFlann`` search), from which both metrics
follow — Chamfer as the mean of the squares in each direction (the pytorch3d convention ordito
uses), Hausdorff as the overall maximum. Both baselines therefore run the same two searches ordito
does, and the host-side ``numpy`` reduction over the returned vector is inside the timed region
because open3d has no device-side equivalent to hide it behind.

**pymeshlab**'s ``get_hausdorff_distance`` is the second reference for the Hausdorff group, and it
returns ``{min, max, mean, RMS, n_samples, ...}`` in one call. Two parameters decide whether the row
means anything: it samples **one** layer and searches the other, so it is one-directional and the
symmetric answer needs both calls (both are timed); and ``samplenum`` defaults to **8**, so
``samplevert=True`` with ``samplenum`` set to the full cloud size is what makes it sample every
point instead of measuring almost nothing. It has no Chamfer entry point -- the mean of the squares
is not among the statistics it returns -- so it appears in the Hausdorff group alone.

**meshlib**'s ``findMaxDistanceSq`` over two ``PointCloud`` objects is the Hausdorff group's third
reference (multi-threaded); each cloud caches its point tree, so the row drops both caches in an
untimed setup and builds them inside the call, as ordito does.

trimesh and libigl have no point-cloud Chamfer/Hausdorff entry point (``igl.hausdorff`` is
mesh-to-mesh only and is timed in the ``hausdorff_mesh_to_mesh`` group), so neither appears in the
point-cloud groups.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import igl
import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pytorch3d.loss as p3d_loss
import pytorch3d.structures as p3d_structures
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from pytorch3d.loss import point_mesh_distance as p3d_point_mesh_distance

import ordito as od
from conftest import (
    BenchCase,
    mesh_ml_from_numpy,
    points_torch_from_numpy,
    skip_larger_than,
    vert_coords_ml,
)

if TYPE_CHECKING:
    from typing_extensions import Buffer

_TRANSLATION_FRACTION = 0.05

_grad_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.vec3]]] = {}
_cloud_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_cloud_wp_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.vec3]]] = {}
_cloud_o3d_cache: dict[str, tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]] = {}
_cloud_pml_cache: dict[str, ml.MeshSet] = {}


# MeshLib's own float upper bound: ``findProjections`` segfaults on ``math.inf`` rather than
# raising, and a ``0.0`` in that slot silently returns all-zero distances.
_MESHLIB_FLT_MAX = 3.4028234663852886e38


# pytorch3d's ``_C`` carries no spatial structure on either device, so both cloud rows here are
# Theta(N^2) and the CPU one is unaffordable past a feature mesh: it quadruples per doubling, so a
# scan mesh extrapolates to minutes *per round*, and ``--benchmark-json`` is written at session end,
# which would cost the whole file's rows.


def _clouds_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(a, b)`` float64 clouds: the mesh vertices and a rigidly translated copy."""
    name = bench_case.mesh_name
    if name not in _cloud_np_cache:
        vertices = bench_case.vertices_np
        offset = _TRANSLATION_FRACTION * np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        _cloud_np_cache[name] = (vertices, np.ascontiguousarray(vertices + offset))
    return _cloud_np_cache[name]


def _clouds_wp(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _cloud_wp_cache:
        cloud_a, cloud_b = _clouds_np(bench_case)
        cloud_a_wp, cloud_b_wp = (
            wp.array(
                np.ascontiguousarray(c, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
            )
            for c in (cloud_a, cloud_b)
        )
        _cloud_wp_cache[key] = (cloud_a_wp, cloud_b_wp)
    return _cloud_wp_cache[key]


def _clouds_o3d(bench_case: BenchCase) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    name = bench_case.mesh_name
    if name not in _cloud_o3d_cache:
        cloud_a_o3d, cloud_b_o3d = (
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c)) for c in _clouds_np(bench_case)
        )
        _cloud_o3d_cache[name] = (cloud_a_o3d, cloud_b_o3d)
    return _cloud_o3d_cache[name]


def _grad_inputs(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _grad_cache:
        vertices = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        offset = 0.05 * np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        vertices_a = wp.array(vertices, dtype=wp.vec3, device=bench_case.device, requires_grad=True)
        vertices_b = wp.array(
            vertices + np.float32(offset),
            dtype=wp.vec3,
            device=bench_case.device,
            requires_grad=True,
        )
        _grad_cache[key] = (vertices_a, vertices_b)
    return _grad_cache[key]


@pytest.mark.benchmark(group="chamfer_points_to_points_loss")
@pytest.mark.benchlibs("ordito", "pytorch3d")
@pytest.mark.parametrize("single_directional", [True, False], ids=["oneway", "symmetric"])
def test_chamfer_points_to_points_loss(bench_case: BenchCase, single_directional: bool) -> None:
    """
    The differentiable cloud Chamfer, forward and backward: one optimization step's loss cost.

    Read against ``chamfer_points_to_points`` at the same direction: the gap is the tape and the
    backward pass. Each ordito round records a fresh tape, runs backward and zeroes the gradients,
    as ``chamfer_mesh_to_mesh_loss`` does.

    **pytorch3d** is the gradient oracle in ``tests/test_metrics.py`` and the one other autodiff
    here: its row runs ``chamfer_distance`` and ``backward`` on leaf tensors and drops their
    gradients, the same step. It has no spatial structure, so its crossover against ordito is the
    one ``chamfer_points_to_points`` records.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pytorch3d":
        cloud_a_p3d, cloud_b_p3d = (
            points_torch_from_numpy(cloud_np, bench_case.torch_device).requires_grad_(True)
            for cloud_np in _clouds_np(bench_case)
        )

        def step_p3d() -> float:
            import torch

            loss_p3d, _ = cast(
                "tuple[torch.Tensor, object]",
                p3d_loss.chamfer_distance(
                    cloud_a_p3d, cloud_b_p3d, single_directional=single_directional
                ),
            )
            loss_p3d.backward()
            cloud_a_p3d.grad, cloud_b_p3d.grad = None, None
            return float(loss_p3d.detach())

        assert bench_case.run(step_p3d) > 0.0
        return
    cloud_a, cloud_b = _grad_inputs(bench_case)

    def step() -> wp.array[wp.float32]:
        tape = wp.Tape()
        loss = od.metrics.chamfer_points_to_points_loss(
            cloud_a, cloud_b, tape=tape, single_directional=single_directional, squared=True
        )
        tape.backward(loss=loss)
        tape.zero()
        return loss

    assert bench_case.run(step).shape == (1,)


@pytest.mark.benchmark(group="chamfer_points_to_mesh_loss")
@pytest.mark.benchlibs("ordito", "pytorch3d")
@pytest.mark.parametrize("single_directional", [True, False], ids=["oneway", "symmetric"])
def test_chamfer_points_to_mesh_loss(bench_case: BenchCase, single_directional: bool) -> None:
    """
    The differentiable cloud-to-surface Chamfer, forward and backward, gradients on both inputs.

    Read against ``chamfer_points_to_mesh``: the gap is the tape and the backward pass through the
    point-to-triangle distances. The cloud is the translated vertex copy and the mesh the original,
    both carrying ``requires_grad``.

    **pytorch3d's row is ``oneway`` only**, for the reason its gradient comparison in
    ``tests/test_metrics.py`` is: its two-sided loss adds a face-to-point half ordito does not
    compute, so the comparable step is its ``point_face_distance`` primitive, forward and backward,
    over triangles gathered from a leaf vertex tensor inside the step. Three rounds: it is two
    exhaustive point-to-triangle sweeps with no spatial structure.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pytorch3d":
        if not single_directional:
            pytest.skip("pytorch3d's two-sided loss adds a face-to-point half: oneway only")
        import torch

        device = bench_case.torch_device
        points_p3d = points_torch_from_numpy(_clouds_np(bench_case)[1], device)[0]
        points_p3d.requires_grad_(True)
        vertices_p3d = points_torch_from_numpy(bench_case.vertices_np, device)[0]
        vertices_p3d.requires_grad_(True)
        faces_p3d = torch.as_tensor(bench_case.faces_np, dtype=torch.int64, device=device)
        first_p3d = torch.zeros(1, dtype=torch.int64, device=device)
        n_points = int(points_p3d.shape[0])

        def step_p3d() -> float:
            point_to_face_p3d = p3d_point_mesh_distance.point_face_distance(
                points_p3d, first_p3d, vertices_p3d[faces_p3d], first_p3d, n_points
            )
            loss_p3d = point_to_face_p3d.mean()
            loss_p3d.backward()
            points_p3d.grad, vertices_p3d.grad = None, None
            return float(loss_p3d.detach())

        assert bench_case.run(step_p3d, rounds=3) > 0.0
        return
    vertices, cloud = _grad_inputs(bench_case)
    faces = bench_case.faces_wp

    def step() -> wp.array[wp.float32]:
        tape = wp.Tape()
        loss = od.metrics.chamfer_points_to_mesh_loss(
            cloud, vertices, faces, tape=tape, single_directional=single_directional, squared=True
        )
        tape.backward(loss=loss)
        tape.zero()
        return loss

    assert bench_case.run(step).shape == (1,)


@pytest.mark.benchmark(group="chamfer_mesh_to_mesh_loss")
@pytest.mark.benchlibs("ordito")
def test_chamfer_mesh_to_mesh_loss(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "dragon")
    vertices_a, vertices_b = _grad_inputs(bench_case)
    faces = bench_case.faces_wp

    def run() -> wp.array[wp.float32]:
        tape = wp.Tape()
        loss = od.metrics.chamfer_mesh_to_mesh_loss(
            vertices_a, faces, vertices_b, faces, tape=tape, squared=True
        )
        tape.backward(loss=loss)
        tape.zero()
        return loss

    loss = bench_case.run(run)
    assert loss.shape == (1,)


@pytest.mark.benchmark(group="hausdorff_points_to_mesh")
@pytest.mark.benchlibs("ordito", "igl", "open3d")
def test_hausdorff_points_to_mesh(bench_case: BenchCase) -> None:
    """
    Cloud-to-surface Hausdorff: ``chamfer_points_to_mesh``'s two halves, reduced with ``max``.

    **libigl's row is the forward half only**, the lower bound ``chamfer_points_to_mesh``'s igl row
    is: ``igl.point_mesh_squared_distance`` is the oracle for this half in
    ``tests/test_metrics.py`` and has no cloud-to-cloud counterpart for the other.

    **open3d's row is both halves**, the composition ``_points_to_mesh_o3d`` describes.
    """
    skip_larger_than(bench_case, "dragon")
    cloud_np = _clouds_np(bench_case)[1]
    if bench_case.kind == "open3d":
        forward_o3d, backward_o3d = bench_case.run(_points_to_mesh_o3d(bench_case, cloud_np))
        assert max(float(forward_o3d.max()), float(backward_o3d.max())) > 0.0
        return
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference builds its AABB tree per call")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        squared_igl, _face_igl, _closest_igl = bench_case.run(
            lambda: igl.point_mesh_squared_distance(cloud_np, vertices_np, faces_np), rounds=3
        )
        assert squared_igl.shape[0] == cloud_np.shape[0]
        return
    cloud = _clouds_wp(bench_case)[1]
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    hausdorff = bench_case.run(lambda: od.metrics.hausdorff_points_to_mesh(cloud, vertices, faces))
    assert hausdorff > 0.0


@pytest.mark.benchmark(group="hausdorff_mesh_to_mesh")
@pytest.mark.benchlibs("ordito", "igl", "meshlib")
def test_hausdorff_mesh_to_mesh(bench_case: BenchCase) -> None:
    """
    Symmetric vertex-to-surface Hausdorff: both meshes' trees built, both directions queried.

    **libigl**'s ``hausdorff`` is the same two vertex-to-surface queries, each building its AABB
    tree per call as ordito builds its two BVHs. **meshlib**'s ``findMaxDistanceSq`` is the
    multi-threaded one; its trees are cached on the two ``Mesh`` objects, so an untimed setup drops
    them and the call builds them, matching the other rows. Both are compared with ordito in
    ``tests/test_metrics.py``.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference builds its AABB tree per call")
        cloud_a_np, cloud_b_np = _clouds_np(bench_case)
        faces_np = bench_case.faces_np
        hausdorff_igl = bench_case.run(
            lambda: igl.hausdorff(cloud_a_np, faces_np, cloud_b_np, faces_np), rounds=3
        )
        assert hausdorff_igl > 0.0
        return
    if bench_case.kind == "meshlib":
        mesh_a_ml = bench_case.new_mesh_ml()
        mesh_b_ml = mesh_ml_from_numpy(_clouds_np(bench_case)[1], bench_case.faces_np)
        part_a_ml, part_b_ml = mm.MeshPart(mesh_a_ml), mm.MeshPart(mesh_b_ml)

        def drop_trees() -> None:
            mesh_a_ml.invalidateCaches()
            mesh_b_ml.invalidateCaches()

        hausdorff_ml = bench_case.run(
            lambda _: mm.findMaxDistanceSq(part_a_ml, part_b_ml), setup=drop_trees
        )
        assert hausdorff_ml > 0.0
        return
    cloud_a, cloud_b = _clouds_wp(bench_case)
    faces = bench_case.faces_wp
    hausdorff = bench_case.run(
        lambda: od.metrics.hausdorff_mesh_to_mesh(cloud_a, faces, cloud_b, faces)
    )
    assert hausdorff > 0.0


@pytest.mark.benchmark(group="chamfer_points_to_points")
@pytest.mark.benchlibs("ordito", "open3d", "pytorch3d")
@pytest.mark.parametrize("single_directional", [True, False], ids=["oneway", "symmetric"])
def test_chamfer_points_to_points(bench_case: BenchCase, single_directional: bool) -> None:
    """
    Point-cloud Chamfer: ``k=1`` searches plus a mean-of-squares reduction.

    ``single_directional=False`` does not merely double the work -- it builds a *second*
    acceleration structure over the other cloud. So the symmetric row should be more than 2x the
    one-way row, and how much more is the build cost, which is otherwise invisible.

    **pytorch3d** is the only reference here with GPU kernels, and its ``chamfer_distance`` is the
    convention this module documents -- same squared distances, same mean-of-two-directions, pinned
    at 7.02e-08 in ``tests/test_metrics.py::test_chamfer_points_to_points_matches_pytorch3d``. It
    has no spatial structure on either device, which makes this the group where its crossover is
    sharpest: it wins at 20 000 points and loses by nearly two orders of magnitude at 200 000. Read
    the two rows as one curve; see the
    ``LIBRARIES`` block in [`conftest.py`](conftest.py) for why half of that swing is ordito's own
    search-radius heuristic. ``single_directional=True`` maps onto its ``single_directional=True``
    exactly, so the parametrize axis is shared rather than emulated.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pytorch3d":
        cloud_a_p3d, cloud_b_p3d = (
            points_torch_from_numpy(cloud_np, bench_case.torch_device)
            for cloud_np in _clouds_np(bench_case)
        )
        chamfer_p3d, _ = bench_case.run(
            lambda: p3d_loss.chamfer_distance(
                cloud_a_p3d, cloud_b_p3d, single_directional=single_directional
            )
        )
        assert float(chamfer_p3d) > 0.0
        return
    if bench_case.kind == "ordito":
        cloud_a, cloud_b = _clouds_wp(bench_case)
        chamfer = bench_case.run(
            lambda: od.metrics.chamfer_points_to_points(
                cloud_a, cloud_b, single_directional=single_directional, squared=True
            )
        )
    elif single_directional:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        chamfer = bench_case.run(
            lambda: float(
                np.square(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b))).mean()
            )
        )
    else:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        chamfer = bench_case.run(
            lambda: float(
                np.square(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b))).mean()
                + np.square(np.asarray(cloud_b.compute_point_cloud_distance(cloud_a))).mean()
            )
        )
    assert chamfer > 0.0


_jittered_wp_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.vec3]]] = {}


def _jittered_clouds_wp(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    """Build a *coincident* pair: the vertices and a copy jittered by ``1e-4`` of the diagonal."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _jittered_wp_cache:
        vertices = bench_case.vertices_np
        diagonal = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        jitter = np.random.default_rng(0).normal(scale=1e-4 * diagonal, size=vertices.shape)
        points_wp, jittered_wp = (
            wp.array(
                np.ascontiguousarray(c, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
            )
            for c in (vertices, vertices + jitter)
        )
        _jittered_wp_cache[key] = (points_wp, jittered_wp)
    return _jittered_wp_cache[key]


@pytest.mark.benchmark(group="chamfer_points_to_points_coincident")
@pytest.mark.benchlibs("ordito")
def test_chamfer_points_to_points_coincident(bench_case: BenchCase) -> None:
    """
    Symmetric point-cloud Chamfer on a coincident pair, the other end of the displacement axis.

    The shifted ``chamfer_points_to_points`` rows put every answer several spacings away; here
    every answer is a fraction of one, which is where a seeded hash grid answers the backward half
    faster than the closest-point descent on a large cloud, and where ``metrics`` chooses it. The
    choice reads the forward answer back, so the two rows together price what it wins and what it
    costs.
    """
    skip_larger_than(bench_case, "dragon")
    cloud_a, cloud_b = _jittered_clouds_wp(bench_case)
    chamfer = bench_case.run(
        lambda: od.metrics.chamfer_points_to_points(cloud_a, cloud_b, squared=True)
    )
    assert chamfer > 0.0


def _clouds_meshset_pml(bench_case: BenchCase) -> tuple[ml.MeshSet, int]:
    """
    Return the two clouds as face-less meshes 0 and 1 of one MeshSet, plus their point count.

    ``get_hausdorff_distance`` writes the per-sample distance into the sampled mesh's vertex scalar
    attribute and touches nothing else, so this is cached rather than rebuilt per round.
    """
    if bench_case.mesh_name not in _cloud_pml_cache:
        meshset_pml = ml.MeshSet()
        for cloud_np in _clouds_np(bench_case):
            meshset_pml.add_mesh(
                ml.Mesh(vertex_matrix=np.ascontiguousarray(cloud_np, dtype=np.float64))
            )
        _cloud_pml_cache[bench_case.mesh_name] = meshset_pml
    count = int(_clouds_np(bench_case)[0].shape[0])
    return _cloud_pml_cache[bench_case.mesh_name], count


@pytest.mark.benchmark(group="hausdorff_points_to_points")
@pytest.mark.benchlibs("ordito", "open3d", "pymeshlab", "meshlib")
def test_hausdorff_points_to_points(bench_case: BenchCase) -> None:
    """Symmetric point-cloud Hausdorff: the same two searches, reduced with ``max`` instead."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "meshlib":
        cloud_a_ml, cloud_b_ml = (
            mn.pointCloudFromPoints(cast("Buffer", np.ascontiguousarray(cloud_np)))
            for cloud_np in _clouds_np(bench_case)
        )

        def drop_trees() -> None:
            cloud_a_ml.invalidateCaches()
            cloud_b_ml.invalidateCaches()

        hausdorff_ml = bench_case.run(
            lambda _: mm.findMaxDistanceSq(cloud_a_ml, cloud_b_ml), setup=drop_trees
        )
        assert hausdorff_ml > 0.0
        return
    if bench_case.kind == "pymeshlab":
        # ``get_hausdorff_distance`` is *one-directional* by construction -- it samples one layer
        # and searches the other -- so the symmetric answer is both directions, and both are timed.
        # ``samplevert=True`` with ``samplenum`` at the full cloud size makes it sample every point
        # rather than its default 8, which would otherwise measure almost nothing.
        meshset_pml, count = _clouds_meshset_pml(bench_case)

        def hausdorff_pml() -> float:
            forward = meshset_pml.get_hausdorff_distance(
                sampledmesh=0, targetmesh=1, samplevert=True, samplenum=count
            )
            backward = meshset_pml.get_hausdorff_distance(
                sampledmesh=1, targetmesh=0, samplevert=True, samplenum=count
            )
            return max(float(forward["max"]), float(backward["max"]))

        assert bench_case.run(hausdorff_pml) > 0.0
        return
    if bench_case.kind == "ordito":
        cloud_a, cloud_b = _clouds_wp(bench_case)
        hausdorff = bench_case.run(lambda: od.metrics.hausdorff_points_to_points(cloud_a, cloud_b))
    else:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        hausdorff = bench_case.run(
            lambda: max(
                float(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b)).max()),
                float(np.asarray(cloud_b.compute_point_cloud_distance(cloud_a)).max()),
            )
        )
    assert hausdorff > 0.0


@pytest.mark.benchmark(group="chamfer_points_to_mesh")
@pytest.mark.benchlibs("ordito", "igl", "meshlib", "pytorch3d", "open3d")
def test_chamfer_points_to_mesh(bench_case: BenchCase) -> None:
    """
    Cloud-to-surface Chamfer: an exact mesh query forward, a ``k=1`` cloud search backward.

    **open3d's row is both halves**: the tensor API's ``RaycastingScene.compute_distance`` forward
    (Embree, its scene built inside the call as ordito builds its ``wp.Mesh``) and the legacy
    ``compute_point_cloud_distance`` from the mesh vertices back (``_points_to_mesh_o3d``), the
    composition compared with ordito in ``tests/test_metrics.py``.

    **libigl's row is the forward half only**, and is a *lower* bound rather than a race:
    ``igl.point_mesh_squared_distance`` is exactly the cloud-to-surface query -- it is already the
    oracle for this group in ``tests/test_metrics.py`` -- but it has no cloud-to-cloud counterpart,
    so the backward ``k=1`` search ordito also performs has no igl equivalent to pair it with. Read
    the row as "what the expensive half costs on one core"; the same partial-reference convention as
    ``igl.doublearea`` in [`test_triangles.py`](test_triangles.py).

    **MeshLib's row is the forward half too**, and is the multi-threaded one -- so it is the fair
    fight of the two references. Its AABB tree is built and pre-warmed outside the timed callable,
    because the tree is cached on the ``Mesh`` and a cold first query is one to two orders of
    magnitude above a warm one; that matches the ordito branch, which is handed a ``wp.Mesh`` it
    does not rebuild. Its
    ``distSq`` output is exactly ordito's ``point_reduction=None`` array, compared element-wise in
    ``tests/test_metrics.py``.

    **pytorch3d's row is the whole scalar and the only GPU one**, but it is a *different* pair of
    directions: ``point_mesh_face_distance`` sums point-to-triangle with **face**-to-point where
    ordito's backward half is mesh-vertex-to-point. So it is not an upper or lower bound on
    ordito's answer, it is a neighbouring quantity of the same cost shape -- two exact
    point-to-triangle sweeps -- which is why it is timed here and why
    ``tests/test_metrics.py::test_chamfer_points_to_mesh_matches_pytorch3d`` compares only the
    forward half, through a hand-ported per-pair table. Its ``Meshes`` and ``Pointclouds`` are
    immutable and pure, so both are built outside the timed callable, and the ``*_packed()``
    accessors the loss reads are warmed there too.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pytorch3d":
        mesh_p3d = bench_case.mesh_p3d
        _ = mesh_p3d.verts_packed(), mesh_p3d.faces_packed()  # warm the cached derivations
        cloud_p3d = p3d_structures.Pointclouds(
            points=[points_torch_from_numpy(_clouds_np(bench_case)[1], bench_case.torch_device)[0]]
        )
        distance_p3d = bench_case.run(
            lambda: p3d_loss.point_mesh_face_distance(mesh_p3d, cloud_p3d)
        )
        assert float(distance_p3d) > 0.0
        return
    if bench_case.kind == "open3d":
        forward_o3d, backward_o3d = bench_case.run(
            _points_to_mesh_o3d(bench_case, _clouds_np(bench_case)[1])
        )
        assert float(np.mean(forward_o3d**2) + np.mean(backward_o3d**2)) > 0.0
        return
    if bench_case.kind == "meshlib":
        cloud_np = _clouds_np(bench_case)[1]
        # The tree is built lazily on the first query and cached on the Mesh, so it is built and
        # pre-warmed *outside* the timed callable -- the row then prices the query, matching what
        # the ordito branch does with a wp.Mesh already in hand. Timing the build instead is one to
        # two orders of magnitude different.
        mesh_ml = bench_case.new_mesh_ml()
        projector_ml = mm.PointsToMeshProjector()
        projector_ml.updateMeshData(mesh_ml)
        points_ml = vert_coords_ml(cloud_np)
        warmup_ml = mm.std_vector_MeshProjectionResult()
        projector_ml.findProjections(
            warmup_ml, points_ml, mm.AffineXf3f(), mm.AffineXf3f(), _MESHLIB_FLT_MAX, 0.0
        )

        def project() -> mm.std_vector_MeshProjectionResult:
            results_ml = mm.std_vector_MeshProjectionResult()
            projector_ml.findProjections(
                results_ml, points_ml, mm.AffineXf3f(), mm.AffineXf3f(), _MESHLIB_FLT_MAX, 0.0
            )
            return results_ml

        assert len(bench_case.run(project, rounds=3)) == cloud_np.shape[0]
        return
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference builds its AABB tree per call")
        cloud_np = _clouds_np(bench_case)[1]
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        squared_igl, _face_igl, _closest_igl = bench_case.run(
            lambda: igl.point_mesh_squared_distance(cloud_np, vertices_np, faces_np), rounds=3
        )
        assert squared_igl.shape[0] == cloud_np.shape[0]
        return
    cloud = _clouds_wp(bench_case)[1]
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    chamfer = bench_case.run(
        lambda: od.metrics.chamfer_points_to_mesh(cloud, vertices, faces, squared=True)
    )
    assert chamfer > 0.0


def _points_to_mesh_o3d(
    bench_case: BenchCase, cloud_np: np.ndarray
) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    """
    Return a callable giving open3d's two cloud-to-mesh halves: forward distances, then backward.

    The forward half is ``RaycastingScene.compute_distance`` (the scene and its Embree BVH built
    inside the call, as ordito builds its ``wp.Mesh``); the backward half is the legacy
    ``compute_point_cloud_distance`` from the mesh vertices to the cloud. Both clouds and the
    tensors are built here, outside the timed region.
    """
    vertices_t = o3d.core.Tensor(np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32))  # pyright: ignore[reportCallIssue]
    faces_t = o3d.core.Tensor(np.ascontiguousarray(bench_case.faces_np, dtype=np.uint32))  # pyright: ignore[reportCallIssue]
    queries_t = o3d.core.Tensor(np.ascontiguousarray(cloud_np, dtype=np.float32))  # pyright: ignore[reportCallIssue]
    vertices_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bench_case.vertices_np))
    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_np))

    def distances_o3d() -> tuple[np.ndarray, np.ndarray]:
        scene_o3d = o3d.t.geometry.RaycastingScene()
        scene_o3d.add_triangles(vertices_t, faces_t)
        forward_o3d = scene_o3d.compute_distance(queries_t).numpy()
        backward_o3d = np.asarray(vertices_o3d.compute_point_cloud_distance(cloud_o3d))
        return forward_o3d, backward_o3d

    return distances_o3d


@pytest.mark.benchmark(group="chamfer_mesh_to_mesh")
@pytest.mark.benchlibs("ordito", "igl")
def test_chamfer_mesh_to_mesh(bench_case: BenchCase) -> None:
    """
    Symmetric vertex-to-surface Chamfer: both meshes' trees built, both directions queried.

    The plain (untaped) sibling of ``chamfer_mesh_to_mesh_loss`` and the mean-of-squares twin of
    ``hausdorff_mesh_to_mesh``, which runs the same two queries and reads beside it.

    **libigl** is the oracle in ``tests/test_metrics.py``: two ``point_mesh_squared_distance``
    calls, one per direction, reduced on the host -- the same composition timed here. Its AABB tree
    is built per call, as ordito's two BVHs are.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference builds its AABB tree per call")
        cloud_a_np, cloud_b_np = _clouds_np(bench_case)
        faces_np = bench_case.faces_np

        def chamfer_igl() -> float:
            forward_igl, _, _ = igl.point_mesh_squared_distance(cloud_a_np, cloud_b_np, faces_np)
            backward_igl, _, _ = igl.point_mesh_squared_distance(cloud_b_np, cloud_a_np, faces_np)
            return float(np.mean(np.asarray(forward_igl)) + np.mean(np.asarray(backward_igl)))

        assert bench_case.run(chamfer_igl, rounds=3) > 0.0
        return
    cloud_a, cloud_b = _clouds_wp(bench_case)
    faces = bench_case.faces_wp
    chamfer = bench_case.run(
        lambda: od.metrics.chamfer_mesh_to_mesh(cloud_a, faces, cloud_b, faces, squared=True)
    )
    assert chamfer > 0.0
