from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="B1",
    title="Union, intersection and difference",
    summary="""
    [`union`][ordito.boolean.union], [`intersection`][ordito.boolean.intersection] and
    [`difference`][ordito.boolean.difference] combine two closed meshes as solids: here the
    Stanford bunny, closed into a solid by [`make_solid`][ordito.repair.make_solid], and a ball
    sunk into its flank. Each result is again
    a valid volume ([`is_volume`][ordito.validation.is_volume]: every edge shared by two faces,
    consistently wound, outward), so it can go straight into the next Boolean, a volume, a ray
    test or a print. The volumes obey inclusion-exclusion to within the `float32` rounding of
    the new vertices, and each result is the very mesh `manifold3d` returns for the same
    inputs: the same vertex and triangle arrays, bit for bit.
    """,
    credits=(
        ("trimesh: boolean", "https://trimesh.org/trimesh.boolean.html"),
        ("Manifold: Boolean", "https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html"),
        ("libigl 609 (CSG)", "https://libigl.github.io/tutorial/#boolean-operations-on-meshes"),
    ),
    notes="""
    Timings against `manifold3d`, MeshLib, PyMeshLab and trimesh are on the
    [boolean benchmarks](../benchmarks/boolean.md) page.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import manifold3d as m3
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    def to_manifold(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> m3.Manifold:
        triangles = faces.numpy().reshape(-1, 3).astype(np.uint32)
        return m3.Manifold(m3.Mesh(vert_properties=vertices.numpy(), tri_verts=triangles))

    def from_manifold(solid: m3.Manifold) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
        mesh = solid.to_mesh()
        vertices = wp.array(np.asarray(mesh.vert_properties)[:, :3], dtype=wp.vec3, device=device)
        return vertices, wp.array(np.asarray(mesh.tri_verts).ravel(), dtype=wp.int32, device=device)

    # The bunny scan has holes; make_solid closes it into a valid input. Manifold also drops a
    # mesh's degenerate triangles when it builds a solid, so both libraries start from its copy.
    solid_a = to_manifold(*od.repair.make_solid(*data.load("bunny", device)))
    bunny = from_manifold(solid_a)
    print(
        f"bunny solid: {bunny[1].shape[0] // 3} triangles, valid {od.validation.is_volume(*bunny)}"
    )

    lower, upper = bunny[0].numpy().min(axis=0), bunny[0].numpy().max(axis=0)
    middle = (lower + upper) / 2
    centre = [middle[0] + 0.01, lower[1] + 0.4 * (upper[1] - lower[1]), upper[2] - 0.01]
    ball = od.creation.icosphere(subdivisions=5, radius=0.045, device=device)
    ball = od.transform.transform_mesh(*ball, od.transform.translation_matrix(centre))
    solid_b = to_manifold(*ball)

    results = {
        "union": od.boolean.union(*bunny, *ball),
        "intersection": od.boolean.intersection(*bunny, *ball),
        "difference": od.boolean.difference(*bunny, *ball),
    }
    reference = {
        "union": solid_a + solid_b,
        "intersection": solid_a ^ solid_b,
        "difference": solid_a - solid_b,
    }
    for name, (vertices, faces) in results.items():
        valid = od.validation.is_volume(vertices, faces)
        vertices_m3, faces_m3 = from_manifold(reference[name])
        same = np.array_equal(vertices.numpy(), vertices_m3.numpy())
        same &= np.array_equal(faces.numpy(), faces_m3.numpy())
        print(f"{name:>12}: {faces.shape[0] // 3:>6} triangles, solid {valid}, manifold3d's {same}")

    volume = {name: od.measures.volume(*mesh) for name, mesh in results.items()}
    excess = volume["union"] + volume["intersection"]
    excess -= od.measures.volume(*bunny) + od.measures.volume(*ball)
    print(f"inclusion-exclusion holds to {abs(excess) / volume['union']:.0e} of the volume")
    # --8<-- [end:code]
    return {
        "inputs": ((bunny[0].numpy(), bunny[1].numpy()), (ball[0].numpy(), ball[1].numpy())),
        "results": {name: (v.numpy(), f.numpy()) for name, (v, f) in results.items()},
    }


def figure(result: dict[str, Any]) -> r.Figure:
    (bunny, bunny_faces), (ball, ball_faces) = result["inputs"]
    panels = [
        r.Panel(
            [
                r.Mesh(bunny, bunny_faces, color=r.LIGHT_GREEN),
                r.Mesh(ball, ball_faces, color=r.BLUE, opacity=0.45),
            ],
            title="bunny and a ball",
        )
    ]
    colours = {"union": r.GREEN, "intersection": r.BLUE, "difference": r.GREEN}
    for name, (vertices, faces) in result["results"].items():
        panels.append(r.Panel([r.Mesh(vertices, faces, color=colours[name])], title=name))
    return r.Figure(panels, camera=data.camera("bunny"), ncols=4)
