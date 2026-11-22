from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="B8",
    title="Exact segment crossings",
    summary="""
    [`segment_hits`][ordito.ray.segment_hits] lists every crossing of each segment with a closed
    surface, in order along it, decided by the same exact predicates as the Booleans. A segment
    through an edge or a vertex crosses there exactly once, never zero or two times, by
    construction rather than by luck, so the parity of the count is a guaranteed inside/outside
    test. The segments below run from the centre of an icosphere straight through each of its
    vertices and each of its edge midpoints, the cases where a floating-point ray cast has to
    rely on its tie-breaking: each crosses exactly once. Here
    [`intersects_location`][ordito.ray.intersects_location], the floating-point cast, also finds
    one hit per ray; `segment_hits` adds the guarantee, every crossing rather than the first,
    and the parameter of each along its segment.
    """,
    credits=(
        ("Manifold: ray_cast", "https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html"),
        ("trimesh: ray", "https://trimesh.org/trimesh.ray.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3, device=device)
    edges = od.edges.edges_unique(faces)[0].numpy()
    corners = vertices.numpy()
    targets = np.vstack([corners, (corners[edges[:, 0]] + corners[edges[:, 1]]) / 2])
    ends = wp.array(2.0 * targets, dtype=wp.vec3, device=device)  # well beyond the surface
    starts = wp.zeros(len(targets), dtype=wp.vec3, device=device)  # the centre

    _, _, points, _, offsets = od.ray.segment_hits(vertices, faces, starts, ends)
    counts = np.diff(offsets.numpy())
    print(f"{len(targets)} segments through vertices and edge midpoints")
    print(f"segment_hits: crossings per segment {sorted(set(counts.tolist()))}")

    mesh = wp.Mesh(points=vertices, indices=faces)
    directions = wp.array(targets, dtype=wp.vec3, device=device)
    _, ray_index, _ = od.ray.intersects_location(mesh, starts, directions)
    ray_counts = np.bincount(ray_index.numpy(), minlength=len(targets))
    hits, rays = np.unique(ray_counts, return_counts=True)
    print(
        f"intersects_location: hits per ray {dict(zip(hits.tolist(), rays.tolist(), strict=True))}"
    )
    # --8<-- [end:code]
    return {"mesh": (corners, faces.numpy()), "targets": targets, "points": points.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = result["mesh"]
    targets = result["targets"]
    segments = np.stack([np.zeros_like(targets), 1.25 * targets], axis=1)
    style: r.MeshStyle = {"show_edges": True, "smooth": False, "opacity": 0.35}
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN, **style),
                    r.Segments(segments, color=r.GREY, width=1.0),
                    r.Points(result["points"], color=r.RED, size=5),
                ],
                title="segment_hits: one crossing each",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN, show_edges=True, smooth=False),
                    r.Points(result["points"], color=r.RED, size=9),
                ],
                title="crossings at vertices and edge midpoints",
                camera=r.Camera(direction=(0.2, 0.3, 1.0), up=(0.0, 1.0, 0.0), zoom=2.5),
            ),
        ],
        camera=r.Camera(direction=(1.0, 0.6, 1.4), up=(0.0, 1.0, 0.0)),
        link_bounds=False,
    )
