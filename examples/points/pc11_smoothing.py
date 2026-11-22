from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC11",
    title="Smoothing a noisy cloud",
    summary="""
    Four ways to take the noise out of a scan without a mesh, scored by each point's distance to
    the surface the clean cloud was sampled from.
    [`smooth_points`][ordito.points.smooth_points] moves each point towards the mean of its
    neighbours (Laplacian steps, which also shrink the cloud) or, given `mu`, alternates a
    step out (Taubin's filter, which does not);
    [`smooth_points_mls`][ordito.points.smooth_points_mls] projects each point onto the plane
    fitted to its neighbourhood, a moving least squares step that also returns normals; and
    [`smooth_points_bilateral`][ordito.points.smooth_points_bilateral] averages with weights that
    fall off along the normal, so it spares creases. All four are Open3D's point-cloud filters,
    run over a fixed neighbour table.
    """,
    credits=(
        (
            "Open3D: point cloud filters",
            "https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html",
        ),
        (
            "PCL: moving least squares",
            "https://pcl.readthedocs.io/projects/tutorials/en/latest/resampling.html",
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
    rng = np.random.default_rng(3)
    noisy = wp.array(clean + rng.normal(0.0, 0.0008, clean.shape), dtype=wp.vec3, device=device)
    bunny = data.load("bunny", device)
    neighbours, _ = od.neighbors.query_nearest(noisy, noisy, k=16)
    normals = od.points.estimate_normals(noisy, neighbours)

    smoothed = {
        "noisy": noisy,
        "Laplacian": od.points.smooth_points(noisy, neighbours, iterations=5),
        "Taubin": od.points.smooth_points(noisy, neighbours, iterations=5, mu=-0.53),
        "MLS": od.points.smooth_points_mls(noisy, neighbours)[0],
        "bilateral": od.points.smooth_points_bilateral(
            noisy, normals, neighbours, sigma_s=0.003, sigma_r=0.001
        ),
    }
    for name, points in smoothed.items():
        _, distance, _ = od.proximity.closest_point_on_mesh(*bunny, points)
        print(f"{name:>9}: median distance to the surface {np.median(distance.numpy()):.6f}")
    # --8<-- [end:code]
    distances = {}
    for name, points in smoothed.items():
        _, distance, _ = od.proximity.closest_point_on_mesh(*bunny, points)
        distances[name] = (points.numpy(), distance.numpy())
    return distances


def figure(result: dict[str, Any]) -> r.Figure:
    clim = (0.0, float(np.percentile(result["noisy"][1], 95)))
    panels = [
        r.Panel(
            [
                r.Points(
                    points,
                    scalars=distance,
                    clim=clim,
                    cmap="magma",
                    size=3,
                    scalar_bar="distance" if name == "bilateral" else None,
                )
            ],
            title=name,
        )
        for name, (points, distance) in result.items()
    ]
    return r.Figure(panels, camera=data.camera("bunny_cloud"), ncols=5)
