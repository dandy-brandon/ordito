from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC12",
    title="DBSCAN clustering",
    summary="""
    [`cluster_labels`][ordito.points.cluster_labels] labels each point with its DBSCAN cluster:
    a point with at least `min_points` others within `radius` is a core point, core points
    within reach of each other share a cluster, the points they reach join it, and the rest are
    noise (`-1`). The labels are Open3D's `cluster_dbscan` labels, cluster numbers included, not
    only the same partition. Here a scan of six separate parts, with 400 stray points in their
    bounding box. The parts come out as five clusters: the box and the torus pass within 2 mm
    of each other, closer than the radius, and DBSCAN rightly joins them, while the sphere and
    the cone, 4 cm apart, stay separate. Most strays come out as noise; the rest landed within
    the radius of a part, where they are border points of its cluster.
    """,
    credits=(
        (
            "Open3D: DBSCAN clustering",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        ("scikit-learn: DBSCAN", "https://scikit-learn.org/stable/modules/clustering.html#dbscan"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import open3d as o3d
    import warp as wp

    import ordito as od
    from examples import data

    samples, _ = data.sample_points("parts", 120_000, seed=4)
    rng = np.random.default_rng(5)
    strays = rng.uniform(samples.min(axis=0), samples.max(axis=0), (400, 3)).astype(np.float32)
    cloud = np.vstack([samples, strays])
    points = wp.array(cloud, dtype=wp.vec3, device=device)

    labels = od.points.cluster_labels(points, radius=0.025, min_points=8).numpy()
    print(f"{labels.max() + 1} clusters, {np.sum(labels == -1)} noise points")
    print(f"strays labelled noise: {np.mean(labels[len(samples) :] == -1):.0%}")

    reference = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud.astype(np.float64)))
    labels_o3d = np.asarray(reference.cluster_dbscan(eps=0.025, min_points=8))
    print(f"same labels as Open3D: {np.array_equal(labels, labels_o3d)}")
    # --8<-- [end:code]
    return {"points": cloud, "labels": labels}


def figure(result: dict[str, Any]) -> r.Figure:
    points, labels = result["points"], result["labels"]
    palette = [r.GREEN, r.BLUE, r.ORANGE, r.MAGENTA, "#17becf", "#8c564b", "#9467bd"]
    layers: list[r.Layer] = [
        r.Points(points[labels == label], color=palette[label % len(palette)], size=3)
        for label in range(labels.max() + 1)
    ]
    layers.append(r.Points(points[labels == -1], color=r.RED, size=6))
    return r.Figure(
        [
            r.Panel([r.Points(points, color=r.GREY, size=3)], title="input"),
            r.Panel(layers, title="cluster_labels (red: noise)"),
        ],
        camera=data.camera("parts"),
    )
