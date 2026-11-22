from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC14",
    title="Points visible from a viewpoint",
    summary="""
    [`point_visible_mask`][ordito.points.point_visible_mask] keeps the points a camera would
    see, from the points alone: Katz, Tal and Basri's hidden point removal flips the cloud
    about a large sphere centred on the camera, and the points whose images lie on the convex
    hull of the flipped cloud are the visible ones. The hull is
    [`convex_hull`][ordito.points.convex_hull]'s, exact. To check it, each point is also tested
    by casting a ray from the camera at the bunny mesh the cloud was sampled from: a point is
    visible when the first hit is (within a small distance) the point itself.
    """,
    credits=(
        (
            "Open3D: hidden point removal",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
    ),
    notes="""
    The flip radius sets how generous the test is: larger radii count more grazing points as
    visible. Katz et al. suggest about a hundred times the cloud's size, used here.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od

    points = data.load_points("bunny_cloud", device)
    cloud = points.numpy()
    diameter = float(np.linalg.norm(cloud.max(axis=0) - cloud.min(axis=0)))
    camera = cloud.mean(axis=0) + np.array([0.0, 0.05, 0.4])
    visible = od.points.point_visible_mask(points, wp.vec3(*camera), radius=100 * diameter).numpy()
    print(f"{visible.sum()} of {len(cloud)} points visible")

    # Reference: a ray from the camera to each point, against the bunny mesh.
    vertices, faces = data.load("bunny", device)
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array(np.repeat([camera], len(cloud), axis=0), dtype=wp.vec3, device=device)
    directions = wp.array(cloud - camera, dtype=wp.vec3, device=device)
    hit, index, _ = od.ray.intersects_location(mesh, origins, directions)
    first = np.full(len(cloud), np.inf)
    reach = np.linalg.norm(hit.numpy() - camera, axis=1)
    np.minimum.at(first, index.numpy(), reach)
    seen = first > np.linalg.norm(cloud - camera, axis=1) - 0.001
    print(f"agreement with ray casting: {np.mean(visible == seen):.1%}")
    # --8<-- [end:code]
    return {"points": cloud, "visible": visible, "camera": camera}


def figure(result: dict[str, Any]) -> r.Figure:
    points, visible, camera = result["points"], result["visible"], result["camera"]
    dx, dy, dz = (float(c) for c in camera - points.mean(axis=0))
    view = (dx, dy, dz)
    side = r.Camera(direction=(1.0, 0.3, -0.2), up=(0.0, 1.0, 0.0))
    return r.Figure(
        [
            r.Panel(
                [
                    r.Points(points[visible], color=r.GREEN, size=3),
                    r.Points(points[~visible], color=r.GREY, size=2, opacity=0.4),
                ],
                title="from the side (green: visible)",
                camera=side,
            ),
            r.Panel(
                [r.Points(points[visible], color=r.GREEN, size=3)],
                title="the visible points, from the camera",
                camera=r.Camera(direction=view, up=(0.0, 1.0, 0.0)),
            ),
        ]
    )
