from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC10",
    title="Orienting and smoothing normals",
    summary="""
    [`estimate_normals`][ordito.points.estimate_normals] fits each normal's direction but
    can only guess its sign locally, so some normals point inward ([PC3](#pc3)).
    [`orient_normals`][ordito.points.orient_normals] makes the signs agree across the whole
    cloud: it builds the minimum spanning tree of the neighbour graph, weighting each edge by how
    far its two normals are from parallel, and flips every normal to agree with its parent's,
    Hoppe's propagation as Open3D's `orient_normals_consistent_tangent_plane` runs it. On a
    closed surface every normal then faces one side.
    [`smooth_normals`][ordito.points.smooth_normals]
    averages each normal over its neighbours, sign-aligned to its own, which takes out the
    noise of the plane fits. The cloud below is the bunny sample of [PC3](#pc3) jittered by
    scanner-like noise, scored against the exact normals of the surface it was sampled from.
    """,
    credits=(
        (
            "Open3D: normal orientation",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        (
            "PCL: normal estimation",
            "https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    clean = data.arrays("bunny_cloud")[0]
    truth = data.cloud_normals("bunny_cloud")  # the exact normals of the sampled surface
    jitter = np.random.default_rng(3).normal(0.0, 0.0008, clean.shape)  # a noisy scan
    points = wp.array(clean + jitter, dtype=wp.vec3, device=device)
    neighbours, _ = od.neighbors.query_nearest(points, points, k=16)

    def report(name: str, normals: np.ndarray) -> None:
        cosine = np.sum(normals * truth, axis=1)
        angle = np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))
        outward = (cosine > 0).mean()
        print(f"{name:>9}: outward {outward:6.1%}, median angle {np.median(angle):.2f} degrees")

    estimated = od.points.estimate_normals(points, neighbours)
    oriented = od.points.orient_normals(points, estimated, neighbours)
    smoothed = od.points.smooth_normals(oriented, neighbours, iterations=2)
    report("estimated", estimated.numpy())
    report("oriented", oriented.numpy())
    report("smoothed", smoothed.numpy())
    # --8<-- [end:code]
    return {
        "points": points.numpy(),
        "estimated": estimated.numpy(),
        "oriented": oriented.numpy(),
        "truth": truth,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    points, truth = result["points"], result["truth"]

    def panel(normals: np.ndarray, title: str) -> r.Panel:
        outward = np.sum(normals * truth, axis=1) > 0
        layers: list[r.Layer] = [r.Points(points[outward], color=r.GREEN, size=3)]
        if not outward.all():
            layers.append(r.Points(points[~outward], color=r.RED, size=4))
        return r.Panel(layers, title=title)

    return r.Figure(
        [
            panel(result["estimated"], "estimate_normals (red: inward)"),
            panel(result["oriented"], "orient_normals"),
        ],
        camera=data.camera("bunny_cloud"),
    )
