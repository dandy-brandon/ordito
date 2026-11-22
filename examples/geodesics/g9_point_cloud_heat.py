from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="G9",
    title="Geodesics on a point cloud",
    summary="""
    [`PointCloudHeatSolver`][ordito.heat.PointCloudHeatSolver] runs the heat method on a bare
    point cloud, geometry-central's point-cloud solver (`potpourri3d.PointCloudHeatSolver`).
    Each point builds its star in the Delaunay triangulation of its nearest neighbours in its
    tangent plane ([`local_triangulation`][ordito.points.local_triangulation]); the union of
    the stars is given its tufted cover, an intrinsic mesh on which the cotangent weights are
    never negative, and the mesh solvers run on that. One solver answers
    [`distance`][ordito.heat.PointCloudHeatSolver.distance],
    [`signed_distance`][ordito.heat.PointCloudHeatSolver.signed_distance] to curves,
    [`extend_scalar`][ordito.heat.PointCloudHeatSolver.extend_scalar],
    [`transport_tangent_vectors`][ordito.heat.PointCloudHeatSolver.transport_tangent_vectors]
    and [`log_map`][ordito.heat.PointCloudHeatSolver.log_map], each factoring its system once
    and reusing it for every later call. Its distances are compared here with the heat method
    on the bunny mesh the cloud was sampled from.
    """,
    credits=(
        (
            "potpourri3d: point cloud distance",
            "https://github.com/nmwsharp/potpourri3d#point-cloud-distance--vector-heat",
        ),
        (
            "geometry-central: point cloud heat solver",
            "https://geometry-central.net/pointcloud/algorithms/heat_solver/",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    points = data.load_points("bunny_cloud", device)
    normals = wp.array(data.cloud_normals("bunny_cloud"), dtype=wp.vec3, device=device)
    solver = od.heat.PointCloudHeatSolver(points, normals=normals)

    source = int(np.argmax(points.numpy()[:, 1]))  # the tip of an ear
    distance = solver.distance(wp.array([source], dtype=wp.int32, device=device)).numpy()
    coordinates = solver.log_map(source).numpy()
    print(f"farthest point: {distance.max():.4f} away")
    print(
        "log-map radius equals the distance:",
        np.allclose(np.linalg.norm(coordinates, axis=1), distance, atol=2e-3),
    )

    # The heat method on the mesh the cloud was sampled from, at each point's nearest vertex.
    vertices, faces = data.load("bunny", device)
    nearest, _ = od.neighbors.query_nearest(vertices, points)
    start = wp.array(nearest.numpy()[[source]], dtype=wp.int32, device=device)
    on_mesh = od.heat.heat_geodesic(vertices, faces, start, method="potpourri3d")
    on_mesh = on_mesh.numpy()[nearest.numpy()]
    difference = np.abs(distance - on_mesh) / on_mesh.max()
    print(
        f"against the mesh: median difference {np.median(difference):.2%} of the largest distance"
    )
    # --8<-- [end:code]
    return {"points": points.numpy(), "distance": distance, "log": coordinates, "source": source}


def figure(result: dict[str, Any]) -> r.Figure:
    points, distance, log = result["points"], result["distance"], result["log"]
    source = points[[result["source"]]]
    cell = 0.012
    checker = (np.floor(log[:, 0] / cell) + np.floor(log[:, 1] / cell)) % 2
    near = np.linalg.norm(log, axis=1) < 0.08
    return r.Figure(
        [
            r.Panel(
                [
                    r.Points(points, scalars=distance, cmap=r.striped("viridis"), size=3),
                    r.Points(source, color=r.RED, size=18),
                ],
                title="distance from the ear tip",
            ),
            r.Panel(
                [
                    r.Points(points[~near], color=r.GREY, size=2),
                    r.Points(points[near & (checker > 0)], color=r.ORANGE, size=4),
                    r.Points(points[near & (checker == 0)], color="#303030", size=4),
                    r.Points(source, color=r.RED, size=18),
                ],
                title="log map: checkerboard",
            ),
        ],
        camera=data.camera("bunny_cloud"),
    )
