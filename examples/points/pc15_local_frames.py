from __future__ import annotations

from typing import Any

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="PC15",
    title="Local frames and surface variation",
    summary="""
    [`point_frames`][ordito.points.point_frames] gives every point the principal axes of its
    neighbourhood, widest first, as the rows of a rotation, together with the spreads along
    them: the eigen-decomposition of [`point_covariances`][ordito.points.point_covariances], with
    the axes' signs fixed by the SHOT rule as pytorch3d's
    `estimate_pointcloud_local_coord_frames` fixes them. The smallest spread over their sum, the
    *surface variation*, is near zero on flat patches and rises at creases and corners, a
    curvature proxy that needs no mesh: here it picks out the sharp edges of a machined part.
    """,
    credits=(
        (
            "pytorch3d: local coordinate frames",
            "https://pytorch3d.readthedocs.io/en/latest/modules/ops.html",
        ),
        (
            "Open3D: covariances",
            "https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    samples, _ = data.sample_points("cad_part", 50_000, seed=7)
    points = wp.array(samples, dtype=wp.vec3, device=device)
    neighbours, _ = od.neighbors.query_nearest(points, points, k=24)
    frames, spreads = od.points.point_frames(points, neighbours)

    spread = spreads.numpy()
    variation = spread[:, 2] / spread.sum(axis=1)
    normals = frames.numpy()[:, 2]  # the last row is the fitted normal
    print(f"surface variation: median {np.median(variation):.4f}, max {variation.max():.3f}")
    print(f"points on a crease (variation > 0.02): {np.mean(variation > 0.02):.1%}")
    # --8<-- [end:code]
    return {"points": samples, "variation": variation, "normals": normals, "frames": frames.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    points, variation = result["points"], result["variation"]
    every = slice(None, None, 397)
    frames = result["frames"][every]
    scale = 0.1
    camera = r.Camera(direction=(1.0, -1.2, 1.0), up=(0.0, 0.0, 1.0))
    return r.Figure(
        [
            r.Panel(
                [
                    r.Points(
                        points,
                        scalars=variation,
                        clim=(0.0, 0.05),
                        cmap="magma",
                        size=3,
                        scalar_bar="surface variation",
                    )
                ],
                title="surface variation",
            ),
            r.Panel(
                [
                    r.Points(points, color=r.GREY, size=2, opacity=0.25),
                    r.Arrows(points[every], frames[:, 0] * scale, color=r.RED),
                    r.Arrows(points[every], frames[:, 1] * scale, color=r.GREEN),
                    r.Arrows(points[every], frames[:, 2] * scale, color=r.BLUE),
                ],
                title="principal frames (red widest, blue normal)",
            ),
        ],
        camera=camera,
    )
