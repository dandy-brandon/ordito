from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="B3",
    title="Capped cuts by a plane",
    summary="""
    [`split_by_plane`][ordito.boolean.split_by_plane] cuts a solid into the two solids on
    either side of a plane, each capped where the plane passes, so the two volumes add up to
    the whole; [`trim_by_plane`][ordito.boolean.trim_by_plane] keeps one side. The cut is
    exact, the same mesh as a [`difference`][ordito.boolean.difference] with a large box beyond
    the plane. Compare [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane],
    which cuts any surface, closed or not, but leaves the section open. Here the bunny, closed
    into a solid by [`make_solid`][ordito.repair.make_solid], is sliced in half and then
    trimmed by four planes into a slab.
    """,
    credits=(
        ("trimesh: slice_plane", "https://trimesh.org/trimesh.intersections.html"),
        (
            "Manifold: split_by_plane",
            "https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html",
        ),
        ("PyVista: clip with plane", "https://docs.pyvista.org/examples/01-filter/clipping"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    bunny = od.repair.make_solid(*data.load("bunny", device))  # the scan, closed into a solid
    total = od.measures.volume(*bunny)
    lower, upper = bunny[0].numpy().min(axis=0), bunny[0].numpy().max(axis=0)
    middle = float((lower[1] + upper[1]) / 2)

    above, below = od.boolean.split_by_plane(*bunny, (0.0, 1.0, 0.0), middle)
    halves = od.measures.volume(*above) + od.measures.volume(*below)
    closed = od.validation.is_volume(*above), od.validation.is_volume(*below)
    print(f"halves are solids: {closed}")
    print(f"their volumes add up to the whole within {abs(halves - total) / total:.0e}")

    # The open cut: same plane, one side kept, no cap.
    open_half = od.intersection.slice_mesh_with_plane(
        *bunny, wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, middle, 0.0)
    )
    print(f"slice_mesh_with_plane's half is a solid: {od.validation.is_volume(*open_half)}")

    # A slab through the body: four trims, each a closed solid. A plane is a normal and its
    # offset along it, and trim_by_plane keeps the side the normal points to.
    centre = float((lower[2] + upper[2]) / 2)
    slab = bunny
    for normal, offset in (
        ((0.0, 1.0, 0.0), float(lower[1] + 0.2 * (upper[1] - lower[1]))),
        ((0.0, -1.0, 0.0), -float(lower[1] + 0.8 * (upper[1] - lower[1]))),
        ((0.0, 0.0, 1.0), centre - 0.008),
        ((0.0, 0.0, -1.0), -(centre + 0.008)),
    ):
        slab = od.boolean.trim_by_plane(*slab, normal, offset)
    print(f"slab: {slab[1].shape[0] // 3} triangles, solid {od.validation.is_volume(*slab)}")
    # --8<-- [end:code]
    return {
        "above": (above[0].numpy(), above[1].numpy()),
        "below": (below[0].numpy(), below[1].numpy()),
        "slab": (slab[0].numpy(), slab[1].numpy()),
        "middle": middle,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    above, below = result["above"], result["below"]
    lift = above[0] + np.array([0.0, 0.03, 0.0])
    camera = r.Camera(direction=(0.9, 0.7, 1.0), up=(0.0, 1.0, 0.0), zoom=1.05)
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(lift, above[1], color=r.GREEN), r.Mesh(*below, color=r.LIGHT_GREEN)],
                title="split_by_plane (halves pulled apart)",
                camera=camera,
                bounds=np.array([below[0].min(axis=0), lift.max(axis=0)]),
            ),
            r.Panel(
                [r.Mesh(*below, color=r.LIGHT_GREEN)],
                title="the lower half's cap",
                camera=r.Camera(direction=(0.2, 1.0, 0.4), up=(0.0, 0.0, -1.0)),
            ),
            r.Panel(
                [r.Mesh(*result["slab"], color=r.GREEN)],
                title="trim_by_plane, four planes",
                camera=r.Camera(direction=(0.6, 0.2, 1.0), up=(0.0, 1.0, 0.0)),
            ),
        ],
        camera=data.camera("bunny"),
        link_bounds=False,
    )
