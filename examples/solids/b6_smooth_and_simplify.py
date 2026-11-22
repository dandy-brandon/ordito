from __future__ import annotations

from typing import Any

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="B6",
    title="Smooth refinement that keeps creases, and simplification",
    summary="""
    [`subdivide_smooth`][ordito.remesh.subdivide_smooth] refines a closed mesh onto a smooth
    surface that passes through its vertices: each edge becomes a curve leaving its ends along
    the tangent planes, each face a patch between them. Edges whose faces meet at more than
    `min_sharp_angle` stay creases, and regions of coplanar triangles stay flat, so a part built
    from a box and a coarse ten-sided cylinder keeps its edges while the cylinder's facets become
    round; with no creases at all (`min_sharp_angle=180`) the whole part swells into one smooth
    body. `length` splits each edge into pieces of about that length, `n` splits every edge into
    that many, and `tolerance` refines curved regions more than flat ones.
    [`simplify`][ordito.remesh.simplify] goes the other way: it removes the vertices that do not
    shape the surface, those inside flat regions or along straight edges, moving the surface by
    at most its `tolerance`. Here it takes the refined part from 8 476 triangles to 254 and leaves
    the round boss round, since a straight line runs down each of its sides.
    """,
    credits=(
        (
            "Manifold: smooth_out and refine",
            "https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html",
        ),
        ("PyVista: subdivide", "https://docs.pyvista.org/examples/01-filter/subdivide"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od

    # A plate with a coarse ten-sided boss: creases everywhere, and facets that should be round.
    plate = od.creation.box(extents=(2.0, 1.2, 0.4), device=device)
    boss = od.creation.cylinder(
        0.45, segment=[[0.4, 0.0, 0.0], [0.4, 0.0, 0.9]], sections=10, device=device
    )
    part = od.boolean.union(*plate, *boss)
    creased = od.remesh.subdivide_smooth(*part, length=0.05)
    rounded = od.remesh.subdivide_smooth(*part, length=0.05, min_sharp_angle=180.0)
    print(f"input: {part[1].shape[0] // 3} triangles")
    print(f"creases kept: {creased[1].shape[0] // 3} triangles")
    print(f"all smooth: {rounded[1].shape[0] // 3} triangles")

    # Simplify: back down, within a tolerance of the refined surface.
    coarse = od.remesh.simplify(*creased, tolerance=0.002)
    gap = od.metrics.hausdorff_mesh_to_mesh(*coarse, *creased)
    print(f"simplified: {coarse[1].shape[0] // 3} triangles, Hausdorff distance {gap:.4f}")

    # A box subdivided three times has 386 vertices, all but 8 redundant.
    box = od.creation.box(extents=(1.0, 1.0, 1.0), device=device)
    for _ in range(3):
        box = od.remesh.subdivide(*box)
    plain = od.remesh.simplify(*box)
    print(f"subdivided box: {box[0].shape[0]} vertices -> simplified: {plain[0].shape[0]}")
    # --8<-- [end:code]
    return {
        name: (mesh[0].numpy(), mesh[1].numpy())
        for name, mesh in (
            ("part", part),
            ("creased", creased),
            ("rounded", rounded),
            ("coarse", coarse),
        )
    }


def figure(result: dict[str, Any]) -> r.Figure:
    style: r.MeshStyle = {"show_edges": True, "line_width": 0.3, "smooth": False}
    camera = r.Camera(direction=(1.0, -1.2, 1.0), up=(0.0, 0.0, 1.0))
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(result["part"][0], result["part"][1], color=r.LIGHT_GREEN, **style)],
                title="input",
            ),
            r.Panel(
                [r.Mesh(result["creased"][0], result["creased"][1], **style)],
                title="subdivide_smooth",
            ),
            r.Panel(
                [r.Mesh(result["rounded"][0], result["rounded"][1], color=r.BLUE, **style)],
                title="min_sharp_angle=180: no creases",
            ),
            r.Panel(
                [r.Mesh(result["coarse"][0], result["coarse"][1], **style)],
                title="simplify(tolerance=0.002)",
            ),
        ],
        camera=camera,
        ncols=4,
    )
