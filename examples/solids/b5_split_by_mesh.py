# Warp's stubs do not model kernel-scope vector indexing and arithmetic, which the `@wp.func` below
# uses (the same reason ordito/kernels/ is excluded from basedpyright).
# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="B5",
    title="Lattice infill: splitting a solid by another",
    summary="""
    [`split_by_mesh`][ordito.boolean.split_by_mesh] cuts one solid by another and returns
    both pieces at once: the part inside the cutter, the
    [`intersection`][ordito.boolean.intersection], and the part outside, the
    [`difference`][ordito.boolean.difference], at the cost of one Boolean. The cutter here is a
    gyroid lattice grown over the bunny's bounding box with
    [`function_level_set`][ordito.levelset.function_level_set], and the bunny is the scan closed
    into a solid by [`make_solid`][ordito.repair.make_solid]; inside the lattice is the infill a
    3D printer would lay down, outside it the voids. The two volumes add up to the bunny's.
    """,
    credits=(
        ("Manifold: split", "https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html"),
        ("trimesh: boolean", "https://trimesh.org/trimesh.boolean.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    bunny = od.repair.make_solid(*data.load("bunny", device))  # the scan, closed into a solid
    lower, upper = bunny[0].numpy().min(axis=0) - 0.002, bunny[0].numpy().max(axis=0) + 0.002

    @wp.func
    def gyroid(p: wp.vec3d) -> wp.float64:
        q = wp.float64(260.0) * p  # a cell about a sixth of the bunny's height
        sheet = wp.sin(q[0]) * wp.cos(q[1]) + wp.sin(q[1]) * wp.cos(q[2])
        sheet += wp.sin(q[2]) * wp.cos(q[0])
        return wp.float64(0.5) - wp.abs(sheet)

    lattice = od.levelset.function_level_set(
        gyroid, [*lower, *upper], edge_length=0.001, device=device
    )
    inside, outside = od.boolean.split_by_mesh(*bunny, *lattice)

    whole = od.measures.volume(*bunny)
    infill, voids = od.measures.volume(*inside), od.measures.volume(*outside)
    print(f"lattice: {lattice[1].shape[0] // 3} triangles")
    print(f"infill: {inside[1].shape[0] // 3} triangles, {infill / whole:.0%} of the volume")
    print(f"voids: {outside[1].shape[0] // 3} triangles, {voids / whole:.0%} of the volume")
    print("both solids:", od.validation.is_volume(*inside) and od.validation.is_volume(*outside))
    # --8<-- [end:code]
    return {
        "inside": (inside[0].numpy(), inside[1].numpy()),
        "outside": (outside[0].numpy(), outside[1].numpy()),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    camera = data.camera("bunny")
    close = r.Camera(direction=(0.25, 0.25, 1.0), up=(0.0, 1.0, 0.0), zoom=2.5)
    return r.Figure(
        [
            r.Panel([r.Mesh(*result["inside"])], title="inside: the infill", camera=camera),
            r.Panel([r.Mesh(*result["inside"])], title="the infill, closer", camera=close),
            r.Panel(
                [r.Mesh(*result["outside"], color=r.BLUE)],
                title="outside: the voids",
                camera=camera,
            ),
        ],
        link_bounds=False,
    )
