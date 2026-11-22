# Warp's stubs do not model kernel-scope vector indexing and arithmetic, which the `@wp.func` below
# uses (the same reason ordito/kernels/ is excluded from basedpyright).
# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from typing import Any

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="B4",
    title="Solids from a signed distance function",
    summary="""
    [`function_level_set`][ordito.levelset.function_level_set] meshes the region where a
    function of position is positive, straight from a `@wp.func`, with no grid for you to
    allocate: the function is sampled on a body-centred cubic lattice and the surface is cut
    out of its tetrahedra, with grid points near the surface snapped onto it. The result is a
    closed, outward-wound solid ready for the Booleans. Here a gyroid sheet, thickened and
    clipped to a ball by taking the minimum of the two functions, gives a porous lattice; a
    smooth blend of two spheres, the second shape, checks the volume against the exact one of
    a single sphere.
    """,
    credits=(
        (
            "Manifold: level_set",
            "https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html",
        ),
        (
            "PyVista: gyroid",
            "https://docs.pyvista.org/examples/00-load/create_parametric_geometric_objects",
        ),
        ("Inigo Quilez: distance functions", "https://iquilezles.org/articles/distfunctions/"),
    ),
    notes="""
    The function need not be a true distance, nor even continuous: only its sign and, near the
    surface, its values matter. `tolerance` refines each vertex by root finding along its
    lattice edge, which helps where the function is far from linear across a cell. Each new
    `@wp.func` compiles its own kernels the first time it is used. For a field already sampled
    on a grid, use [`marching_cubes`][ordito.levelset.marching_cubes] instead ([V3](volumes.md#v3)).
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import math

    import warp as wp

    import ordito as od

    @wp.func
    def gyroid_ball(p: wp.vec3d) -> wp.float64:
        q = wp.float64(5.0) * p
        sheet = wp.sin(q[0]) * wp.cos(q[1]) + wp.sin(q[1]) * wp.cos(q[2])
        sheet += wp.sin(q[2]) * wp.cos(q[0])
        thick = wp.float64(0.45) - wp.abs(sheet)  # positive inside the thickened sheet
        return wp.min(thick, wp.float64(1.0) - wp.length(p))  # ... and inside the unit ball

    bounds = [-1.1, -1.1, -1.1, 1.1, 1.1, 1.1]
    lattice = od.levelset.function_level_set(gyroid_ball, bounds, edge_length=0.02, device=device)
    print(
        f"gyroid ball: {lattice[1].shape[0] // 3} triangles, "
        f"solid {od.validation.is_volume(*lattice)}"
    )
    print(f"it fills {od.measures.volume(*lattice) / (4 / 3 * math.pi):.0%} of the ball")

    @wp.func
    def blended_spheres(p: wp.vec3d) -> wp.float64:
        a = wp.float64(0.5) - wp.length(p - wp.vec3d(-0.3, 0.0, 0.0))
        b = wp.float64(0.5) - wp.length(p - wp.vec3d(0.3, 0.0, 0.0))
        k = wp.float64(0.2)  # blend width: a smooth maximum of the two
        h = wp.clamp(
            wp.float64(0.5) + wp.float64(0.5) * (a - b) / k, wp.float64(0.0), wp.float64(1.0)
        )
        return b + (a - b) * h + k * h * (wp.float64(1.0) - h)

    blend = od.levelset.function_level_set(
        blended_spheres,
        [-1.0, -0.7, -0.7, 1.0, 0.7, 0.7],
        edge_length=0.02,
        tolerance=1e-5,
        device=device,
    )
    sphere = 4 / 3 * math.pi * 0.5**3
    print(f"blend: {od.measures.volume(*blend) / sphere:.2f} times one sphere's volume")
    # --8<-- [end:code]
    return {
        "lattice": (lattice[0].numpy(), lattice[1].numpy()),
        "blend": (blend[0].numpy(), blend[1].numpy()),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    camera = r.Camera(direction=(1.0, 0.7, 1.2), up=(0.0, 1.0, 0.0))
    vertices, faces = result["lattice"]
    cut = vertices[faces.reshape(-1, 3)].mean(axis=1)[:, 2] < 0.15
    return r.Figure(
        [
            r.Panel([r.Mesh(vertices, faces)], title="gyroid ∩ ball", camera=camera),
            r.Panel(
                [r.Mesh(vertices, faces.reshape(-1, 3)[cut].ravel())],
                title="cut open",
                camera=r.Camera(direction=(0.3, 0.4, 1.0), up=(0.0, 1.0, 0.0)),
            ),
            r.Panel(
                [r.Mesh(*result["blend"], color=r.BLUE)],
                title="smooth union of two spheres",
                camera=r.Camera(direction=(0.2, 0.5, 1.0), up=(0.0, 1.0, 0.0)),
            ),
        ],
        link_bounds=False,
    )
