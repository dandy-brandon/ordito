from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC13",
    title="Boundary points of a scan",
    summary="""
    [`point_boundary_mask`][ordito.points.point_boundary_mask] finds the rim of a sampled
    surface without meshing it: each point's neighbours are projected into its tangent plane,
    and a point with an empty wedge wider than `max_gap` around it is on the boundary, PCL's
    boundary estimation as Open3D's `compute_boundary_points` runs it. The cloud below samples
    the bunny with its holes cut open; the flagged points trace the holes' rims, and nearly all
    of them lie within a few spacings of a rim of the mesh the cloud came from. The
    neighbourhood must be wide enough that random sampling gaps do not look like an edge: with
    16 neighbours many interior points of this area-uniform sample have an empty wedge.
    """,
    credits=(
        (
            "Open3D: boundary points",
            "https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html",
        ),
        (
            "PCL: boundary estimation",
            "https://pointclouds.org/documentation/classpcl_1_1_boundary_estimation.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    samples, sample_normals = data.sample_points("holey_bunny", 60_000, seed=6)
    points = wp.array(samples, dtype=wp.vec3, device=device)
    normals = wp.array(sample_normals, dtype=wp.vec3, device=device)
    neighbours, _ = od.neighbors.query_nearest(points, points, k=40)
    on_rim = od.points.point_boundary_mask(points, normals, neighbours).numpy()
    print(f"{on_rim.sum()} of {len(samples)} points flagged as boundary")

    # Check against the rims of the mesh the cloud was sampled from.
    vertices, faces = data.load("holey_bunny", device)
    loops = od.boundary.boundary_loops(vertices, faces)
    rim = np.vstack([vertices.numpy()[loop.numpy()] for loop in loops])
    _, distance = od.neighbors.query_nearest(wp.array(rim, dtype=wp.vec3, device=device), points)
    spacing = np.median(od.neighbors.nearest_neighbor_distance(points).numpy())
    near = distance.numpy()[on_rim] < 6 * spacing
    print(
        f"{len(loops)} rims in the mesh; "
        f"{near.mean():.1%} of flagged points within 6 spacings of one"
    )
    # --8<-- [end:code]
    return {"points": samples, "on_rim": on_rim, "rim": rim}


def figure(result: dict[str, Any]) -> r.Figure:
    points, on_rim = result["points"], result["on_rim"]
    return r.Figure(
        [
            r.Panel([r.Points(points, color=r.GREY, size=2)], title="cloud with holes"),
            r.Panel(
                [
                    r.Points(points[~on_rim], color=r.LIGHT_GREEN, size=2),
                    r.Points(points[on_rim], color=r.RED, size=5),
                ],
                title="point_boundary_mask",
            ),
        ],
        camera=data.camera("holey_bunny"),
    )
