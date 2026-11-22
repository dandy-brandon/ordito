"""Gallery page: solids and Booleans."""

TITLE = "Solids and Booleans"
HEAD_TITLE = "Mesh Booleans, CSG and plane cuts on the GPU in Python"
DESCRIPTION = (
    "Exact union, intersection and difference of closed meshes, CSG trees, capped plane cuts, "
    "solids from signed distance functions, smooth refinement and polygon triangulation on the "
    "GPU in Python."
)
ORDER = 11
INTRO = """
Exact Booleans of closed meshes, CSG trees, capped cuts, implicit solids, smooth refinement that
keeps creases, polygons with holes and exact segment crossings.

Every function on this page takes and returns *closed, consistently wound* meshes and treats them
as solids. They are a port of the [Manifold](https://github.com/elalish/manifold) library, the
engine behind [`trimesh.boolean`][] and `manifold3d`: every crossing of an edge with a face and
every inside/outside decision is an exact sign computed from the `float32` coordinates, ties
(touching or coplanar surfaces) are broken by one consistent symbolic perturbation, and so the
result is always closed and manifold, the same mesh Manifold returns. A mesh that is not closed
raises `ValueError`; [`make_solid`][ordito.repair.make_solid] repairs one first, and the
[voxel route](volumes.md#v6) combines shapes that are not solids at all.
"""
