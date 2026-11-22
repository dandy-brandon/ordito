from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="B7",
    title="Polygons with holes, extruded and countersunk",
    summary="""
    [`triangulate_polygons`][ordito.polyline.triangulate_polygons] fills a planar region given
    as loops: outer boundaries counter-clockwise, holes clockwise, all triangulated together
    and no point added, so every triangle indexes the input points. It is Manifold's
    triangulator, robust to loops that touch or come within `epsilon` of each other.
    [`triangulate_polygons_from_offsets`][ordito.polyline.triangulate_polygons_from_offsets]
    takes the same loops packed into one array with offsets, the form a GPU pipeline produces.
    [`extrude_triangulation`][ordito.creation.extrude_triangulation] raises the region into a
    closed solid, and from there the Booleans apply: here a mounting plate whose four bolt holes
    are countersunk by subtracting four cones with
    [`difference_all`][ordito.boolean.difference_all].
    """,
    credits=(
        ("Manifold: triangulate", "https://manifoldcad.org/docs/html/namespacemanifold.html"),
        ("trimesh: extrude_polygon", "https://trimesh.org/trimesh.creation.html"),
        ("libigl 609 (CSG)", "https://libigl.github.io/tutorial/#boolean-operations-on-meshes"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od

    def circle(x: float, y: float, radius: float, n: int) -> np.ndarray:
        """Return a counter-clockwise circle of ``n`` points."""
        angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.c_[x + radius * np.cos(angle), y + radius * np.sin(angle)]

    def rounded_rectangle(half_x: float, half_y: float, radius: float, n: int) -> np.ndarray:
        """Return a counter-clockwise rectangle with quarter-circle corners of ``n`` points."""
        quarter = np.linspace(0.0, np.pi / 2, n)
        centers = [(half_x, half_y), (-half_x, half_y), (-half_x, -half_y), (half_x, -half_y)]
        arcs = [
            np.c_[
                x + radius * np.cos(quarter + k * np.pi / 2),
                y + radius * np.sin(quarter + k * np.pi / 2),
            ]
            for k, (x, y) in enumerate(centers)
        ]
        return np.vstack(arcs)

    # A rounded 4 x 2 plate (counter-clockwise), four bolt holes and a slot (clockwise: reversed).
    outline = rounded_rectangle(1.7, 0.7, 0.3, 12)
    bolts = [circle(x, y, 0.18, 32)[::-1] for x in (-1.6, 1.6) for y in (-0.6, 0.6)]
    turn = np.linspace(-np.pi / 2, np.pi / 2, 17)
    right = np.c_[0.6 + 0.25 * np.cos(turn), 0.25 * np.sin(turn)]
    slot = np.vstack([right, -right])[::-1]  # two half circles, joined by straight sides
    loops = [outline, *bolts, slot]

    polygons = [wp.array(loop, dtype=wp.vec2, device=device) for loop in loops]
    faces = od.polyline.triangulate_polygons(polygons)
    points = np.vstack(loops)
    print(f"{len(loops)} loops, {len(points)} points -> {faces.shape[0] // 3} triangles")

    # The same loops packed with offsets give the same triangles.
    offsets = wp.array(np.cumsum([0, *map(len, loops)]), dtype=wp.int32, device=device)
    packed = od.polyline.triangulate_polygons_from_offsets(
        wp.array(points, dtype=wp.vec2, device=device), offsets
    )
    print(f"packed form, same triangles: {np.array_equal(packed.numpy(), faces.numpy())}")

    plate = od.creation.extrude_triangulation(
        wp.array(points, dtype=wp.vec2, device=device), faces, height=0.25
    )
    # 90-degree countersinks: cones turned point down, widening to 0.32 at the top face.
    flip = od.transform.rotation_matrix((1.0, 0.0, 0.0), np.pi)
    cone = od.transform.transform_mesh(*od.creation.cone(0.42, 0.42, 64, device=device), flip)
    cones = [
        od.transform.transform_mesh(*cone, od.transform.translation_matrix((x, y, 0.35)))
        for x in (-1.6, 1.6)
        for y in (-0.6, 0.6)
    ]
    countersunk = od.boolean.difference_all([plate, *cones])
    removed = od.measures.volume(*plate) - od.measures.volume(*countersunk)
    print(f"plate solid: {od.validation.is_volume(*plate)}")
    print(f"countersunk: solid {od.validation.is_volume(*countersunk)}, {removed:.4f} removed")
    # --8<-- [end:code]
    return {
        "loops": loops,
        "faces": faces.numpy().reshape(-1, 3),
        "plate": (countersunk[0].numpy(), countersunk[1].numpy()),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    loops, faces = result["loops"], result["faces"]
    points = np.vstack(loops)

    def draw(ax: Any) -> None:
        ax.tripcolor(
            points[:, 0],
            points[:, 1],
            faces,
            facecolors=np.random.default_rng(0).random(len(faces)),
            cmap="YlGn",
            edgecolors="#303030",
            lw=0.4,
        )
        for loop in loops:
            closed = np.vstack([loop, loop[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=r.RED, lw=1.5)
        ax.axis("off")

    return r.Figure(
        [
            r.Plot(draw, title="triangulate_polygons"),
            r.Panel(
                [r.Mesh(*result["plate"], smooth=False)],
                title="extruded, countersunk",
                camera=r.Camera(direction=(0.5, -1.0, 1.2), up=(0.0, 0.0, 1.0)),
            ),
        ],
        link_bounds=False,
        panel_size=(720, 480),
    )
