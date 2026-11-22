from __future__ import annotations

from typing import Any

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="B2",
    title="A CSG tree",
    summary="""
    The textbook constructive-solid-geometry part: a cube intersected with a ball, less three
    cylinders along the axes. [`union_all`][ordito.boolean.union_all] joins the cylinders in
    one call and [`difference_all`][ordito.boolean.difference_all] subtracts a whole list from
    the first mesh, grouping the operands the way Manifold's batch Boolean does: the same solid
    as the pairwise route, its new vertices rounded along a different path. Every intermediate
    result is a closed solid, and
    the topology comes out exactly: three bores crossing at the centre leave a surface of
    genus 5, read off the Euler characteristic.
    """,
    credits=(
        (
            "Wikipedia: constructive solid geometry",
            "https://en.wikipedia.org/wiki/Constructive_solid_geometry",
        ),
        ("libigl 609 (CSG)", "https://libigl.github.io/tutorial/#boolean-operations-on-meshes"),
        ("trimesh: boolean", "https://trimesh.org/trimesh.boolean.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od

    cube = od.creation.box(extents=(2.0, 2.0, 2.0), device=device)
    ball = od.creation.icosphere(subdivisions=6, radius=1.35, device=device)
    axes = ([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]])
    axes += ([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]],)
    cylinders = [od.creation.cylinder(0.6, segment=a, sections=96, device=device) for a in axes]

    rounded = od.boolean.intersection(*cube, *ball)
    cross = od.boolean.union_all(cylinders)
    part = od.boolean.difference(*rounded, *cross)
    in_one_call = od.boolean.difference_all([rounded, *cylinders])

    vertices, faces = part
    genus = (2 - od.measures.euler_characteristic(faces)) // 2
    print(f"part: {faces.shape[0] // 3} triangles, solid {od.validation.is_volume(*part)}")
    print(f"volume {od.measures.volume(*part):.4f}, genus {genus}")
    alike = od.measures.volume(*in_one_call) / od.measures.volume(*part)
    print(f"difference_all: {in_one_call[1].shape[0] // 3} triangles, volume ratio {alike:.6f}")
    # --8<-- [end:code]
    return {
        "rounded": (rounded[0].numpy(), rounded[1].numpy()),
        "cross": (cross[0].numpy(), cross[1].numpy()),
        "part": (vertices.numpy(), faces.numpy()),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    style: r.MeshStyle = {"smooth": False}
    camera = r.Camera(direction=(1.0, 0.8, 1.2), up=(0.0, 1.0, 0.0))
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(result["rounded"][0], result["rounded"][1], color=r.LIGHT_GREEN, **style)],
                title="cube ∩ ball",
            ),
            r.Panel(
                [r.Mesh(result["cross"][0], result["cross"][1], color=r.BLUE, **style)],
                title="union_all(cylinders)",
            ),
            r.Panel(
                [r.Mesh(result["part"][0], result["part"][1], **style)],
                title="(cube ∩ ball) - cylinders",
            ),
        ],
        camera=camera,
    )
