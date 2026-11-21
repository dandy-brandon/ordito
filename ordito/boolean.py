"""
Boolean operations on closed meshes: union, intersection, difference, and cuts.

[`union`][ordito.boolean.union], [`intersection`][ordito.boolean.intersection] and
[`difference`][ordito.boolean.difference] combine two closed, consistently wound meshes as solids
and return a closed, consistently wound mesh; [`union_all`][ordito.boolean.union_all],
[`intersection_all`][ordito.boolean.intersection_all] and
[`difference_all`][ordito.boolean.difference_all] combine many.
[`split_by_mesh`][ordito.boolean.split_by_mesh], [`split_by_plane`][ordito.boolean.split_by_plane]
and [`trim_by_plane`][ordito.boolean.trim_by_plane] cut a solid into capped parts. Mirrors
``trimesh.boolean``, and gives the answers of its default engine, the Manifold library, whose
Boolean this module ports: the same meshes, vertex and triangle order included.

Every crossing of an edge of one mesh with a face of the other, and every vertex's winding number
with respect to the other mesh, is decided exactly from the ``float32`` coordinates. Ties --
surfaces that touch, overlap in a plane or coincide -- are broken by one consistent symbolic
perturbation, so the result is closed and manifold whatever the inputs' relative position, and
the slivers such contacts leave are collapsed afterwards.
"""

from __future__ import annotations

import heapq
import math
import struct
from collections.abc import Sequence
from typing import Literal, NamedTuple, cast

import numpy as np
import warp as wp
from warp._src.codegen import StructInstance

import ordito.typing as odt
from ordito import _launch
from ordito._common import offsets_buffer, read_array, read_values, require_same_device, scan_count
from ordito._solid import (
    FLOAT32_EPSILON,
    Solid,
    empty_solid,
    finish_solid,
    halfedge_mesh,
    halfedge_mesh_from_solid,
    mesh_from_solid,
    simplify_topology,
    solid_from_mesh,
    solid_from_positions,
)
from ordito.graph import connected_component_labels_from_edges
from ordito.halfedge import halfedge_twins
from ordito.kernels import boolean as kernel_boolean
from ordito.kernels import polyline as kernel_polyline
from ordito.kernels.algorithms import solid as kernel_solid

_Operation = Literal["union", "intersection", "difference"]


# ``BatchUnion``'s chunk: it groups at most this many of its last children at a time.
_MAX_UNION_SIZE = 1000


# A result face of this many polygon sides or more is triangulated and paired on the CPU, whose
# one thread runs those quadratic passes far faster than a GPU thread (2026-11-21: an 82 k-face
# sphere's plane cut, whose cap is one polygon, 433 + 136 -> 13 + 6 ms; the copies are prefixes).
# Swept over icospheres of 3 to 6 subdivisions cut by a plane, 16-32 sides best at every size
# (5 k faces 45 -> 8.6 ms against 256); 64 lost the smallest cap. Many-sided faces come only from
# an operand's few large triangles, so the CPU pass stays short.
_HEAVY_FACE_SIDES = 32


# Up to this many heavy faces travel to the CPU entry by entry, each with its own small copies;
# more take the whole arrays (``_launch_heavy_faces``).
_HEAVY_FACE_LIST = 16


# An operand with at most this many halfedges crosses the other's faces pair by pair.
_PAIR_CROSSING_HALFEDGES = 384
_PAIR_CROSSING_MAX_PAIRS = 1 << 26


def union(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    collapse_collinear: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid both closed meshes enclose together, as a closed mesh.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` first closed, consistently wound mesh.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` second closed, consistently wound mesh.
    collapse_collinear
        Whether a crossing vertex whose surrounding triangles lie in at most two planes -- one
        on a straight crease or inside a flat face -- is collapsed into a neighbour, as Manifold
        does. ``False`` keeps every vertex of the two surfaces' arrangement but those closer than
        the solids' tolerance: the vertices an exact-arithmetic Boolean such as libigl's CGAL
        one returns. Only flat regions crossed along straight lines have such vertices; on
        curved inputs both settings give the same mesh.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` result vertices: the inputs' vertices it keeps, then the crossings.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` result triangles, wound as the inputs are.

    Raises
    ------
    RuntimeError
        If the four arrays are not all on one device.
    ValueError
        If either mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`intersection`][ordito.boolean.intersection]
    [`difference`][ordito.boolean.difference]
    [`trimesh.boolean.union`][]

    Notes
    -----
    A port of the Boolean of Manifold (https://github.com/elalish/manifold, Apache-2.0), the
    engine [`trimesh.boolean.union`][] calls by default: the same exact predicates and symbolic
    perturbation, the same polygon triangulation and the same topology cleanup, so on the same
    ``float32`` inputs the two return the same mesh up to vertex numbering and face order.

    Examples
    --------
    Two unit cubes overlapping in a 0.5 x 0.7 x 0.8 box enclose 1.72 together:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vertices_b, faces_b = od.creation.box(bounds=[[0.5, 0.3, 0.2], [1.5, 1.3, 1.2]])
    vertices, faces = od.boolean.union(vertices_a, faces_a, vertices_b, faces_b)
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 1.7200
    ```
    """
    return _boolean(
        vertices_a, faces_a, vertices_b, faces_b, "union", collapse_collinear=collapse_collinear
    )


def intersection(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    collapse_collinear: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid both closed meshes enclose in common, as a closed mesh.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` first closed, consistently wound mesh.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` second closed, consistently wound mesh.
    collapse_collinear
        Whether a crossing vertex whose surrounding triangles lie in at most two planes -- one
        on a straight crease or inside a flat face -- is collapsed into a neighbour, as Manifold
        does. ``False`` keeps every vertex of the two surfaces' arrangement but those closer than
        the solids' tolerance: the vertices an exact-arithmetic Boolean such as libigl's CGAL
        one returns. Only flat regions crossed along straight lines have such vertices; on
        curved inputs both settings give the same mesh.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` result vertices: the inputs' vertices it keeps, then the crossings.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` result triangles, wound as the inputs are; empty when the solids do
        not overlap.

    Raises
    ------
    RuntimeError
        If the four arrays are not all on one device.
    ValueError
        If either mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`union`][ordito.boolean.union]
    [`difference`][ordito.boolean.difference]
    [`trimesh.boolean.intersection`][]

    Examples
    --------
    Two unit cubes overlapping in a 0.5 x 0.7 x 0.8 box:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vertices_b, faces_b = od.creation.box(bounds=[[0.5, 0.3, 0.2], [1.5, 1.3, 1.2]])
    vertices, faces = od.boolean.intersection(vertices_a, faces_a, vertices_b, faces_b)
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 0.2800
    ```
    """
    return _boolean(
        vertices_a,
        faces_a,
        vertices_b,
        faces_b,
        "intersection",
        collapse_collinear=collapse_collinear,
    )


def difference(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    collapse_collinear: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid the first closed mesh encloses outside the second, as a closed mesh.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` closed, consistently wound mesh to cut.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` closed, consistently wound mesh to remove.
    collapse_collinear
        Whether a crossing vertex whose surrounding triangles lie in at most two planes -- one
        on a straight crease or inside a flat face -- is collapsed into a neighbour, as Manifold
        does. ``False`` keeps every vertex of the two surfaces' arrangement but those closer than
        the solids' tolerance: the vertices an exact-arithmetic Boolean such as libigl's CGAL
        one returns. Only flat regions crossed along straight lines have such vertices; on
        curved inputs both settings give the same mesh.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` result vertices: the inputs' vertices it keeps, then the crossings.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` result triangles, wound outward; the second mesh's faces it keeps are
        reversed.

    Raises
    ------
    RuntimeError
        If the four arrays are not all on one device.
    ValueError
        If either mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`union`][ordito.boolean.union]
    [`intersection`][ordito.boolean.intersection]
    [`trimesh.boolean.difference`][]

    Examples
    --------
    Cutting a cube with another that overlaps it in a 0.5 x 0.7 x 0.8 box leaves 0.72:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vertices_b, faces_b = od.creation.box(bounds=[[0.5, 0.3, 0.2], [1.5, 1.3, 1.2]])
    vertices, faces = od.boolean.difference(vertices_a, faces_a, vertices_b, faces_b)
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 0.7200
    ```
    """
    return _boolean(
        vertices_a,
        faces_a,
        vertices_b,
        faces_b,
        "difference",
        collapse_collinear=collapse_collinear,
    )


def split_by_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
) -> tuple[
    tuple[wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[wp.vec3], wp.array[wp.int32]]
]:
    """
    Return the first closed mesh cut by the second: its part inside, and its part outside.

    The same two meshes as [`intersection`][ordito.boolean.intersection] and
    [`difference`][ordito.boolean.difference] of the pair, from one pass over the crossings.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` closed, consistently wound mesh to cut.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` closed, consistently wound cutter.

    Returns
    -------
    inside : tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` part of the first mesh inside the cutter.
    outside : tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` part of the first mesh outside the cutter.

    Raises
    ------
    RuntimeError
        If the four arrays are not all on one device.
    ValueError
        If either mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`split_by_plane`][ordito.boolean.split_by_plane]

    Notes
    -----
    Manifold's ``Manifold.split``.

    Examples
    --------
    A unit cube cut by another overlapping it in a 0.5 x 0.7 x 0.8 box:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vertices_b, faces_b = od.creation.box(bounds=[[0.5, 0.3, 0.2], [1.5, 1.3, 1.2]])
    inside, outside = od.boolean.split_by_mesh(vertices_a, faces_a, vertices_b, faces_b)
    print(f"{od.measures.volume(*inside):.4f} {od.measures.volume(*outside):.4f}")  # 0.2800 0.7200
    ```
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    device = vertices_a.device
    p = solid_from_mesh(vertices_a, faces_a, 0, "a")
    q = solid_from_mesh(vertices_b, faces_b, 1, "b")
    inside, outside = _solid_split(p, q)
    return mesh_from_solid(inside, device), mesh_from_solid(outside, device)


def split_by_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normal: Sequence[float],
    origin_offset: float = 0.0,
) -> tuple[
    tuple[wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[wp.vec3], wp.array[wp.int32]]
]:
    """
    Return a closed mesh cut by a plane into two closed meshes, one on each side.

    The plane is the points ``x`` with ``dot(x, normal / |normal|) == origin_offset``. Both parts
    are capped where the plane cuts them, and the cut is exact: the parts are the mesh's
    [`split_by_mesh`][ordito.boolean.split_by_mesh] by a box covering it beyond the plane.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    normal
        ``(3,)`` plane normal, of any nonzero length; the first part lies on its side.
    origin_offset
        The plane's signed distance from the origin along the unit normal.

    Returns
    -------
    above : tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` part on the side ``normal`` points to.
    below : tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` part on the other side.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If the mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound, or if ``normal`` is zero or not finite.

    See Also
    --------
    [`trim_by_plane`][ordito.boolean.trim_by_plane]
    [`intersection.split_mesh_with_plane`][ordito.intersection.split_mesh_with_plane]

    Notes
    -----
    Manifold's ``Manifold.split_by_plane``, whose half-space this builds the same way: the
    answers are the same meshes.

    Examples
    --------
    A unit cube halved at ``z = 0.25``:

    ```python
    import ordito as od

    vertices, faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    above, below = od.boolean.split_by_plane(vertices, faces, (0.0, 0.0, 1.0), 0.25)
    print(f"{od.measures.volume(*above):.4f} {od.measures.volume(*below):.4f}")  # 0.7500 0.2500
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    solid = solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty:
        empty = mesh_from_solid(solid, device)
        return empty, mesh_from_solid(solid, device)
    above, below = _solid_split(solid, _halfspace_solid(solid, normal, origin_offset, 1))
    return mesh_from_solid(above, device), mesh_from_solid(below, device)


def trim_by_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normal: Sequence[float],
    origin_offset: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the part of a closed mesh on one side of a plane, capped, as a closed mesh.

    The first part of [`split_by_plane`][ordito.boolean.split_by_plane], computed alone.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    normal
        ``(3,)`` plane normal, of any nonzero length, pointing to the part kept.
    origin_offset
        The plane's signed distance from the origin along the unit normal.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` vertices of the part kept.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` its triangles.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If the mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound, or if ``normal`` is zero or not finite.

    See Also
    --------
    [`split_by_plane`][ordito.boolean.split_by_plane]

    Notes
    -----
    Manifold's ``Manifold.trim_by_plane``.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    kept = od.boolean.trim_by_plane(vertices, faces, (1.0, 1.0, 0.0), 0.0)
    print(f"{od.measures.volume(*kept):.4f}")  # 1.0000
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    solid = solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty:
        return mesh_from_solid(solid, device)
    halfspace = _halfspace_solid(solid, normal, origin_offset, 1)
    return mesh_from_solid(_solid_boolean(solid, halfspace, "intersection"), device)


def union_all(
    meshes: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid any of several closed meshes encloses, as a closed mesh.

    Meshes whose boxes do not touch are joined without a Boolean, and the rest are united
    pairwise, largest first, so the answer is the one a chain of
    [`union`][ordito.boolean.union] calls in that order gives.

    Parameters
    ----------
    meshes
        ``(vertices, faces)`` pairs of closed, consistently wound meshes, on one device.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` result vertices.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` result triangles.

    Raises
    ------
    RuntimeError
        If the meshes are not all on one device.
    ValueError
        If ``meshes`` is empty, or a mesh is not closed, has an edge shared by more than two
        faces, or is not consistently wound.

    See Also
    --------
    [`union`][ordito.boolean.union]
    [`intersection_all`][ordito.boolean.intersection_all]
    [`difference_all`][ordito.boolean.difference_all]

    Notes
    -----
    Manifold's ``Manifold.batch_boolean`` with ``OpType.Add``: the same grouping of disjoint
    meshes and the same order of unions, so the same mesh.

    Examples
    --------
    Three unit cubes in a row, overlapping by a quarter:

    ```python
    import ordito as od

    cubes = [
        od.creation.box(bounds=[[0.75 * i, 0.0, 0.0], [0.75 * i + 1.0, 1.0, 1.0]])
        for i in range(3)
    ]
    vertices, faces = od.boolean.union_all(cubes)
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 2.5000
    ```
    """
    solids, device = _batch_solids(meshes)
    return mesh_from_solid(_batch_union(solids), device)


def intersection_all(
    meshes: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid every one of several closed meshes encloses, as a closed mesh.

    Parameters
    ----------
    meshes
        ``(vertices, faces)`` pairs of closed, consistently wound meshes, on one device.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` result vertices.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` result triangles; empty when the solids share no volume.

    Raises
    ------
    RuntimeError
        If the meshes are not all on one device.
    ValueError
        If ``meshes`` is empty, or a mesh is not closed, has an edge shared by more than two
        faces, or is not consistently wound.

    See Also
    --------
    [`intersection`][ordito.boolean.intersection]
    [`union_all`][ordito.boolean.union_all]

    Notes
    -----
    Manifold's ``Manifold.batch_boolean`` with ``OpType.Intersect``: pairwise, largest first.

    Examples
    --------
    ```python
    import ordito as od

    cubes = [
        od.creation.box(bounds=[[0.25 * i, 0.0, 0.0], [0.25 * i + 1.0, 1.0, 1.0]])
        for i in range(3)
    ]
    vertices, faces = od.boolean.intersection_all(cubes)
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 0.5000
    ```
    """
    solids, device = _batch_solids(meshes)
    return mesh_from_solid(_batch_boolean(solids, "intersection"), device)


def difference_all(
    meshes: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid the first closed mesh encloses outside all the others, as a closed mesh.

    The others are united first ([`union_all`][ordito.boolean.union_all]), then removed.

    Parameters
    ----------
    meshes
        ``(vertices, faces)`` pairs of closed, consistently wound meshes, on one device: the
        mesh to cut, then the meshes to remove.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` result vertices.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` result triangles, wound outward.

    Raises
    ------
    RuntimeError
        If the meshes are not all on one device.
    ValueError
        If ``meshes`` is empty, or a mesh is not closed, has an edge shared by more than two
        faces, or is not consistently wound.

    See Also
    --------
    [`difference`][ordito.boolean.difference]
    [`union_all`][ordito.boolean.union_all]

    Notes
    -----
    Manifold's ``Manifold.batch_boolean`` with ``OpType.Subtract``.

    Examples
    --------
    A unit cube with two corners cut off:

    ```python
    import ordito as od

    meshes = [
        od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        od.creation.box(bounds=[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]),
        od.creation.box(bounds=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
    ]
    vertices, faces = od.boolean.difference_all(meshes)
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 0.7500
    ```
    """
    solids, device = _batch_solids(meshes)
    first = solids[0]
    if len(solids) == 1:
        return mesh_from_solid(first, device)
    return mesh_from_solid(_solid_boolean(first, _batch_union(solids[1:]), "difference"), device)


class _Operand(NamedTuple):
    """A solid with its halfedge pairing and face tree, ready for the crossing passes."""

    solid: Solid
    mates: wp.array[wp.int32]
    first_halfedge: wp.array[wp.int32]
    bvh: wp.Bvh


class _Crossings(NamedTuple):
    """``Boolean3``: both operands' edge-face crossings and their vertices' windings."""

    p: _Operand
    q: _Operand
    expand: int
    n12: int
    n21: int
    run_offsets_p: wp.array[wp.int32]
    run_offsets_q: wp.array[wp.int32]
    record_edge: wp.array[wp.int32]
    record_face: wp.array[wp.int32]
    record_crossing: wp.array[wp.int32]
    record_point: wp.array[wp.vec3d]
    broken_p: wp.array[wp.int32]
    broken_q: wp.array[wp.int32]
    labels_p: wp.array[wp.int32]
    labels_q: wp.array[wp.int32]
    windings_p: wp.array[wp.int32]
    windings_q: wp.array[wp.int32]
    pad: float


def _boolean(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    operation: _Operation,
    *,
    collapse_collinear: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Run the Boolean ``operation`` on two closed meshes.

    Raises
    ------
    ValueError
        If either mesh is not closed, edge-manifold and consistently wound.
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    p = solid_from_mesh(vertices_a, faces_a, 0, "a")
    q = solid_from_mesh(vertices_b, faces_b, 1, "b")
    if operation == "union":
        # ``a + b`` is a two-child ``BatchUnion``: meshes whose boxes are apart are composed.
        return mesh_from_solid(
            _batch_union([p, q], collapse_collinear=collapse_collinear), vertices_a.device
        )
    return mesh_from_solid(
        _solid_boolean(p, q, operation, collapse_collinear=collapse_collinear), vertices_a.device
    )


def _solid_boolean(
    p: Solid, q: Solid, operation: _Operation, *, collapse_collinear: bool = True
) -> Solid:
    """``SimpleBoolean``: one Boolean of two solids, as a solid."""
    early = _empty_operand_result(p, q, operation)
    if early is not None:
        return early
    crossings = _boolean3(_operand(p), _operand(q), 1 if operation == "union" else 0)
    return _result(crossings, operation, collapse_collinear=collapse_collinear)


def _batch_solids(
    meshes: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[list[Solid], wp.Device]:
    """
    Build each mesh's solid, numbered in order, and return them with their device.

    Raises
    ------
    ValueError
        If ``meshes`` is empty, or a mesh is not closed, edge-manifold and consistently wound.
    """
    if len(meshes) == 0:
        raise ValueError("meshes must hold at least one mesh")
    require_same_device(meshes=[array for mesh in meshes for array in mesh])
    device = wp.get_device(meshes[0][0].device)
    solids = [
        solid_from_mesh(vertices, faces, index, str(index))
        for index, (vertices, faces) in enumerate(meshes)
    ]
    return solids, device


def _batch_union(children: list[Solid], *, collapse_collinear: bool = True) -> Solid:
    """
    ``BatchUnion``: compose the meshes whose boxes are pairwise apart, then unite the groups.

    Each mesh joins the first group none of whose boxes it touches (``Box::DoesOverlap``,
    closed); a group of several is joined by ``Compose``, and the groups go to
    ``BatchBoolean``. Manifold works through its children a thousand at a time from the back.
    """
    children = list(children)
    while len(children) > 1:
        start = max(len(children) - _MAX_UNION_SIZE, 0)
        boxes = [(child.lower, child.upper) for child in children[start:]]
        groups: list[list[int]] = []
        for i, box in enumerate(boxes):
            for group in groups:
                if not any(_boxes_overlap(box, boxes[j]) for j in group):
                    group.append(i)
                    break
            else:
                groups.append([i])
        composed = [
            children[start + group[0]]
            if len(group) == 1
            else _compose([children[start + j] for j in group])
            for group in groups
        ]
        children = [
            *children[:start],
            _batch_boolean(composed, "union", collapse_collinear=collapse_collinear),
        ]
        children[0], children[-1] = children[-1], children[0]
    return children[0]


def _boxes_overlap(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    """``Box::DoesOverlap``: whether two closed boxes share a point."""
    return all(a[0][i] <= b[1][i] and a[1][i] >= b[0][i] for i in range(3))


def _batch_boolean(
    results: list[Solid], operation: _Operation, *, collapse_collinear: bool = True
) -> Solid:
    """
    ``BatchBoolean``: combine pairwise, the two with the most vertices first, four pairs a round.

    Manifold's heap pops the largest by ``(vertex count, insertion number)``; each round pairs up
    to four times and returns its results, numbered on, together.
    """
    if len(results) == 1:
        return results[0]
    if len(results) == 2:
        return _solid_boolean(
            results[0], results[1], operation, collapse_collinear=collapse_collinear
        )
    heap = [(-solid.n_vertices, -serial, serial, solid) for serial, solid in enumerate(results)]
    heapq.heapify(heap)
    next_serial = len(results)
    while len(heap) > 1:
        done: list[tuple[int, int, int, Solid]] = []
        for _ in range(4):
            if len(heap) <= 1:
                break
            a = heapq.heappop(heap)[3]
            b = heapq.heappop(heap)[3]
            result = _solid_boolean(a, b, operation, collapse_collinear=collapse_collinear)
            done.append((-result.n_vertices, -next_serial, next_serial, result))
            next_serial += 1
        for item in done:
            heapq.heappush(heap, item)
    return heap[0][3]


def _compose(solids: list[Solid]) -> Solid:
    """
    ``CsgLeafNode::Compose``: meshes known apart, joined into one solid.

    Their vertices, faces and references concatenated, ``RemoveDegenerates``, ``SortGeometry``.
    """
    device = solids[0].positions.device
    positions = _concatenate([solid.positions for solid in solids], wp.vec3d, device)
    vertex_offsets = np.cumsum([0] + [solid.n_vertices for solid in solids[:-1]])
    faces = _concatenate(
        [
            _offset_faces(solid.faces, int(offset))
            for solid, offset in zip(solids, vertex_offsets, strict=True)
        ],
        wp.int32,
        device,
    )
    face_normals = _concatenate([solid.face_normals for solid in solids], wp.vec3d, device)
    face_mesh = _concatenate([solid.face_mesh for solid in solids], wp.int32, device)
    face_coplanar = _concatenate([solid.face_coplanar for solid in solids], wp.int32, device)
    epsilon = max(max(solid.epsilon, kernel_solid.PRECISION * solid.scale) for solid in solids)
    tolerance = max(solid.tolerance for solid in solids)
    lower = tuple(min(solid.lower[i] for solid in solids) for i in range(3))
    upper = tuple(max(solid.upper[i] for solid in solids) for i in range(3))
    combined = Solid(
        positions,
        faces,
        face_normals,
        _concatenate([solid.vertex_normals for solid in solids], wp.vec3d, device),
        face_mesh,
        face_coplanar,
        epsilon,
        tolerance,
        (lower[0], lower[1], lower[2]),
        (upper[0], upper[1], upper[2]),
    )
    mesh = halfedge_mesh_from_solid(combined)
    sizes = simplify_topology(
        mesh, (faces.size, combined.n_vertices), 0, epsilon, tolerance, device, colinear=False
    )
    # ``Compose`` sorts in the union of its parts' boxes, never recalculated.
    return finish_solid(mesh, sizes, epsilon, tolerance, device, (combined.lower, combined.upper))


def _concatenate(arrays: list[odt.ArrayNd], dtype: object, device: wp.DeviceLike) -> odt.ArrayNd:
    """Return the arrays end to end in one new array."""
    total = sum(int(array.shape[0]) for array in arrays)
    out = _launch.empty(total, dtype=dtype, device=device)
    offset = 0
    for array in arrays:
        count = int(array.shape[0])
        if count > 0:
            _launch.copy(out, array, dest_offset=offset, count=count)
        offset += count
    return out


def _offset_faces(faces: wp.array[wp.int32], offset: int) -> wp.array[wp.int32]:
    """Return ``faces`` with every index moved by ``offset``."""
    if offset == 0 or faces.size == 0:
        return faces
    out = _launch.empty(faces.size, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_boolean.offset_indices,
        dim=faces.size,
        inputs=[faces, offset],
        outputs=[out],
        device=faces.device,
    )
    return out


def _solid_split(p: Solid, q: Solid) -> tuple[Solid, Solid]:
    """``Manifold::Split``: one ``Boolean3`` of a difference, read as both results."""
    if p.is_empty or q.is_empty:
        inside = _empty_operand_result(p, q, "intersection")
        outside = _empty_operand_result(p, q, "difference")
        assert inside is not None
        assert outside is not None
        return inside, outside
    crossings = _boolean3(_operand(p), _operand(q), 0)
    return _result(crossings, "intersection"), _result(crossings, "difference")


def _empty_operand_result(p: Solid, q: Solid, operation: _Operation) -> Solid | None:
    """``Boolean3::Result``'s answer when an operand is empty, else ``None``."""
    if p.is_empty:
        return q if operation == "union" else p
    if q.is_empty:
        return q if operation == "intersection" else p
    return None


def _operand(solid: Solid) -> _Operand:
    """Pair a solid's halfedges and build its face tree."""
    device = solid.positions.device
    n_faces = solid.n_faces
    twins = halfedge_twins(solid.faces, solid.n_vertices, validate=False)
    first_halfedge = _launch.full(solid.n_vertices, 2147483647, dtype=wp.int32, device=device)
    lowers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    uppers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_solid.operand_boxes,
        dim=n_faces,
        inputs=[solid.positions, solid.faces, twins, 0.0],
        outputs=[lowers, uppers, first_halfedge, _launch.zeros(1, dtype=wp.int32, device=device)],
        device=device,
    )
    return _Operand(solid, twins, first_halfedge, wp.Bvh(lowers, uppers))


def _boolean3(p: _Operand, q: _Operand, expand: int) -> _Crossings:
    """``Boolean3``: the edge-face crossings both ways and the vertices' winding numbers."""
    device = p.solid.positions.device
    scale = max(p.solid.scale, q.solid.scale)
    # The face trees' query boxes are padded by a few float32 ulps of the scene: a superset of the
    # candidates the exact predicates then decide.
    pad = max(scale, 1.0) * 4.0 * FLOAT32_EPSILON
    run_offsets_p, n12, pairs_p = _count_crossings(p, q, pad, expand, forward=True)
    run_offsets_q, n21, pairs_q = _count_crossings(q, p, pad, expand, forward=False)
    size = max(n12 + n21, 1)
    record_edge = _launch.empty(size, dtype=wp.int32, device=device)
    record_face = _launch.empty(size, dtype=wp.int32, device=device)
    record_crossing = _launch.empty(size, dtype=wp.int32, device=device)
    record_point = _launch.empty(size, dtype=wp.vec3d, device=device)
    records = [record_edge, record_face, record_crossing, record_point]
    _write_crossings(p, q, pad, expand, True, run_offsets_p, pairs_p, 0, records)
    _write_crossings(q, p, pad, expand, False, run_offsets_q, pairs_q, n12, records)
    broken_p, labels_p, windings_p = _windings(p, q, record_edge, 0, n12, pad, expand, True)
    broken_q, labels_q, windings_q = _windings(q, p, record_edge, n12, n21, pad, expand, False)
    return _Crossings(
        p,
        q,
        expand,
        n12,
        n21,
        run_offsets_p,
        run_offsets_q,
        record_edge,
        record_face,
        record_crossing,
        record_point,
        broken_p,
        broken_q,
        labels_p,
        labels_q,
        windings_p,
        windings_q,
        pad,
    )


def _result(
    crossings: _Crossings, operation: _Operation, *, collapse_collinear: bool = True
) -> Solid:
    """
    ``Boolean3::Result``: one operation's solid from the crossings and windings.

    Without ``collapse_collinear``, the cleanup skips ``CollapseColinearEdges``.
    """
    p, q = crossings.p, crossings.q
    device = p.solid.positions.device
    n12, n21 = crossings.n12, crossings.n21
    record_edge, record_face = crossings.record_edge, crossings.record_face
    c1 = 0 if operation == "intersection" else 1
    c2 = 1 if operation == "union" else 0
    c3 = 1 if operation == "intersection" else -1
    invert_q = operation == "difference"
    epsilon = max(p.solid.epsilon, q.solid.epsilon)
    tolerance = max(p.solid.tolerance, q.solid.tolerance)

    # Level 4: the inclusions and the result's vertices.
    n_records = n12 + n21
    size = max(n_records, 1)
    n_vp = p.solid.n_vertices
    n_vq = q.solid.n_vertices
    n_items = n_vp + n_vq + n_records
    vertex_offsets, abs_inclusion = offsets_buffer(n_items, device)
    inclusion = _launch.empty(n_items, dtype=wp.int32, device=device)
    for operand, labels, windings, base, c_base in (
        (p, crossings.labels_p, crossings.windings_p, 0, c1),
        (q, crossings.labels_q, crossings.windings_q, n_vp, c2),
    ):
        _launch.launch(
            kernel_boolean.vertex_inclusions,
            dim=operand.solid.n_vertices,
            inputs=[
                labels,
                windings,
                operand.first_halfedge,
                operand.solid.faces.size,
                c_base,
                c3,
                _launch.view(inclusion, base, base + operand.solid.n_vertices),
                _launch.view(abs_inclusion, base, base + operand.solid.n_vertices),
            ],
            device=device,
        )
    if n_records > 0:
        _launch.launch(
            kernel_boolean.crossing_inclusions,
            dim=n_records,
            inputs=[
                crossings.record_crossing,
                c3,
                _launch.view(inclusion, n_vp + n_vq, n_items),
                _launch.view(abs_inclusion, n_vp + n_vq, n_items),
            ],
            device=device,
        )
    n_result_vertices = scan_count(abs_inclusion)
    if n_result_vertices == 0:
        return empty_solid(device)
    # Host readback: the retained vertices come first, and the simplification moves only those
    # after them, so it needs their count.
    first_new_vert = int(read_values(vertex_offsets, n_vp + n_vq, 1)[0])
    positions = _launch.empty(n_result_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.place_result_vertices,
        dim=n_items,
        inputs=[
            p.solid.positions,
            q.solid.positions,
            crossings.record_point,
            inclusion,
            vertex_offsets,
            n_vp,
            n_vq,
        ],
        outputs=[positions],
        device=device,
    )
    # The crossings' result vertices and inclusions; a stand-in nobody reads when there are none.
    record_vertex = (
        vertex_offsets if n_records == 0 else _launch.view(vertex_offsets, n_vp + n_vq, n_items)
    )
    record_inclusion = (
        inclusion if n_records == 0 else _launch.view(inclusion, n_vp + n_vq, n_items)
    )

    # The polygon sides of every face of P and Q (``SizeOutput``).
    n_fp = p.solid.n_faces
    n_fq = q.solid.n_faces
    n_faces = n_fp + n_fq
    face_offsets, sides = offsets_buffer(n_faces, device)
    for operand, vertex_base, face_base in ((p, 0, 0), (q, n_vp, n_fp)):
        _launch.launch(
            kernel_boolean.count_face_sides,
            dim=operand.solid.n_faces,
            inputs=[operand.solid.faces, inclusion, vertex_base, face_base],
            outputs=[sides],
            device=device,
        )
    for count, first, edge_mates, edge_face_base, face_face_base in (
        (n12, 0, p.mates, 0, n_fp),
        (n21, n12, q.mates, n_fp, 0),
    ):
        if count > 0:
            _launch.launch(
                kernel_boolean.count_crossing_sides,
                dim=count,
                inputs=[
                    record_edge,
                    record_face,
                    record_inclusion,
                    edge_mates,
                    first,
                    edge_face_base,
                    face_face_base,
                ],
                outputs=[sides],
                device=device,
            )
    n_sides = scan_count(sides)
    face_cursor = _launch.clone(face_offsets)
    starts = _launch.empty(max(n_sides, 1), dtype=wp.int32, device=device)
    ends = _launch.empty(max(n_sides, 1), dtype=wp.int32, device=device)
    partner = _launch.empty(max(n_sides, 1), dtype=wp.int32, device=device)

    # Level 5: the polygon edges -- the broken edges' pieces, the new edges, the whole edges.
    rank = _launch.empty(size, dtype=wp.int32, device=device)
    for operand, run_offsets, record_base, vertex_base, face_base in (
        (p, crossings.run_offsets_p, 0, 0, 0),
        (q, crossings.run_offsets_q, n12, n_vp, n_fp),
    ):
        _launch.launch(
            kernel_boolean.emit_partial_edges,
            dim=operand.solid.faces.size,
            inputs=[
                operand.solid.faces,
                operand.mates,
                operand.solid.positions,
                run_offsets,
                record_base,
                record_inclusion,
                record_vertex,
                inclusion,
                vertex_offsets,
                vertex_base,
                face_base,
                positions,
                face_cursor,
                rank,
            ],
            outputs=[starts, ends, partner],
            device=device,
        )
    if n_records > 0:
        n_new = 2 * n_records
        keys, items = _launch.sort_buffers(n_new, wp.int64, device)
        _launch.launch(
            kernel_boolean.new_edge_keys,
            dim=n_new,
            inputs=[n12, n_fq, record_edge, record_face, record_inclusion, p.mates, q.mates],
            outputs=[keys, items],
            device=device,
        )
        _launch.radix_sort_pairs(keys, items, n_new)
        item_rank = _launch.empty(n_new, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_boolean.emit_new_edges,
            dim=n_new,
            inputs=[
                n12,
                n_fp,
                _launch.view(keys, 0, n_new),
                items,
                record_edge,
                record_face,
                record_inclusion,
                record_vertex,
                p.mates,
                q.mates,
                positions,
                face_cursor,
                item_rank,
            ],
            outputs=[starts, ends, partner],
            device=device,
        )
    for operand, broken, vertex_base, face_base in (
        (p, crossings.broken_p, 0, 0),
        (q, crossings.broken_q, n_vp, n_fp),
    ):
        _launch.launch(
            kernel_boolean.emit_whole_edges,
            dim=operand.solid.faces.size,
            inputs=[
                operand.solid.faces,
                operand.mates,
                broken,
                inclusion,
                vertex_offsets,
                vertex_base,
                face_base,
                face_cursor,
            ],
            outputs=[starts, ends, partner],
            device=device,
        )

    # Level 6: every face's polygons into triangles (``Face2Tri``).
    tri_offsets, capacity = offsets_buffer(n_faces, device)
    _launch.launch(
        kernel_boolean.triangle_capacities,
        dim=n_faces,
        inputs=[face_offsets],
        outputs=[capacity],
        device=device,
    )
    n_capacity = scan_count(capacity)
    count_offsets, counts = offsets_buffer(n_faces, device)
    state, triangles = _ear_clip_state(max(n_capacity, 1), device)
    scratch = max(n_sides, 1)
    heavy, heavy_faces = _heavy_faces(face_offsets, tri_offsets, n_faces)
    on_cpu = wp.get_device(device).is_cpu
    # The heavy passes' face list on the CPU device (all of it when the list overflowed); the
    # other passes run a thread a face.
    unlisted = _launch.empty(0, dtype=wp.int32, device=device)
    listed = unlisted
    if heavy > 0 and on_cpu and heavy_faces is not None:
        listed = _launch.array([face[0] for face in heavy_faces], dtype=wp.int32, device=device)
    heavy_dim = n_faces if heavy_faces is None else len(heavy_faces)
    triangulate_inputs: list[object] = [
        face_offsets,
        tri_offsets,
        starts,
        ends,
        positions,
        p.solid.face_normals,
        q.solid.face_normals,
        n_fp,
        1 if invert_q else 0,
        epsilon,
        _HEAVY_FACE_SIDES,
    ]
    scratch_arrays = [_launch.empty(scratch, dtype=wp.int32, device=device) for _ in range(3)]
    _launch.launch(
        kernel_boolean.triangulate_faces,
        dim=n_faces,
        inputs=[*triangulate_inputs, 0, unlisted, *scratch_arrays, state],
        outputs=[counts],
        device=device,
    )
    held: _HeavyHost | None = None
    if heavy > 0 and on_cpu:
        _launch.launch(
            kernel_boolean.triangulate_faces,
            dim=heavy_dim,
            inputs=[*triangulate_inputs, 1, listed, *scratch_arrays, state],
            outputs=[counts],
            device=device,
        )
    elif heavy > 0 and heavy_faces is not None:
        held = _triangulate_heavy_on_host(
            heavy_faces, triangulate_inputs, scratch, n_capacity, counts, triangles
        )
    elif heavy > 0:
        host_state, host_triangles = _ear_clip_state(max(n_capacity, 1), "cpu")
        _launch_heavy_faces(
            kernel_boolean.triangulate_faces,
            n_faces,
            [*triangulate_inputs, 1, unlisted, *scratch_arrays],
            [(triangles, host_triangles)],
            host_state,
            [counts],
        )
    n_triangles = scan_count(counts)
    if n_triangles == 0:
        return empty_solid(device)
    mesh = halfedge_mesh(n_triangles, n_result_vertices, positions, device)
    slot_halfedge = _launch.full(max(n_sides, 1), -1, dtype=wp.int32, device=device)
    write_inputs: list[object] = [
        tri_offsets,
        count_offsets,
        triangles,
        face_offsets,
        starts,
        ends,
        p.solid.face_normals,
        q.solid.face_normals,
        p.solid.face_coplanar,
        q.solid.face_coplanar,
        p.solid.face_mesh,
        q.solid.face_mesh,
        n_fp,
        1 if invert_q else 0,
        _HEAVY_FACE_SIDES,
    ]
    write_outputs: list[object] = [
        mesh.start,
        mesh.pair,
        mesh.tri_mesh,
        mesh.tri_coplanar,
        mesh.tri_normal,
    ]
    _launch.launch(
        kernel_boolean.write_result_halfedges,
        dim=n_faces,
        inputs=[*write_inputs, 0, unlisted, slot_halfedge],
        outputs=write_outputs,
        device=device,
    )
    if heavy > 0 and on_cpu:
        _launch.launch(
            kernel_boolean.write_result_halfedges,
            dim=heavy_dim,
            inputs=[*write_inputs, 1, listed, slot_halfedge],
            outputs=write_outputs,
            device=device,
        )
    elif held is not None:
        _write_heavy_on_host(held, write_inputs, slot_halfedge, n_triangles, write_outputs)
    elif heavy > 0:
        # Only the written prefixes travel: the result's halfedges and triangles so far.
        n_written = 3 * n_triangles
        _launch_heavy_faces(
            kernel_boolean.write_result_halfedges,
            n_faces,
            [*write_inputs, 1, unlisted, slot_halfedge],
            [],
            None,
            [
                _launch.view(mesh.start, 0, n_written),
                _launch.view(mesh.pair, 0, n_written),
                _launch.view(mesh.tri_mesh, 0, n_triangles),
                _launch.view(mesh.tri_coplanar, 0, n_triangles),
                _launch.view(mesh.tri_normal, 0, n_triangles),
            ],
        )
    _launch.launch(
        kernel_boolean.pair_polygon_halfedges,
        dim=n_sides,
        inputs=[slot_halfedge, partner],
        outputs=[mesh.pair],
        device=device,
    )
    sizes = simplify_topology(
        mesh,
        (3 * n_triangles, n_result_vertices),
        first_new_vert,
        epsilon,
        tolerance,
        device,
        colinear=collapse_collinear,
    )
    return finish_solid(mesh, sizes, epsilon, tolerance, device)


def _count_crossings(
    a: _Operand, b: _Operand, pad: float, expand: int, *, forward: bool
) -> tuple[wp.array[wp.int32], int, wp.array[wp.int32] | _Candidates | None]:
    """
    Find each edge of ``a``'s crossings with ``b``'s faces; their run offsets and total.

    An ``a`` of few edges against many faces goes pair by pair, and the pairs' scanned offsets
    come third. Otherwise each edge's candidate faces from ``b``'s face tree are decided one a
    thread, and the decided candidates come third, to be written with the offsets
    (``_write_crossings``); ``None`` when no edge has a candidate.
    """
    device = a.solid.positions.device
    n_h = a.solid.faces.size
    n_faces_b = b.solid.n_faces
    if _crossings_by_pairs(n_h, n_faces_b):
        offsets, _ = offsets_buffer(n_h, device)
        pair_offsets, flags = offsets_buffer(n_h * n_faces_b, device)
        _launch.launch(
            kernel_boolean.count_pair_crossings,
            dim=n_h * n_faces_b,
            inputs=[n_faces_b, *_crossing_operands(a, b, pad, expand, forward)],
            outputs=[flags],
            device=device,
        )
        total = scan_count(flags)
        _launch.launch(
            kernel_boolean.edge_offsets_from_pairs,
            dim=n_h + 1,
            inputs=[pair_offsets, n_faces_b],
            outputs=[offsets],
            device=device,
        )
        return offsets, total, pair_offsets
    walk = [a.solid.faces, a.solid.positions, b.bvh.id, pad]
    candidate_offsets, counts = offsets_buffer(n_h, device)
    _launch.launch(
        kernel_boolean.count_edge_candidates, dim=n_h, inputs=walk, outputs=[counts], device=device
    )
    n_candidates = scan_count(counts)
    if n_candidates == 0:
        return candidate_offsets, 0, None
    # The candidates, their decisions and the run offsets live until the records are written.
    edges, candidate_faces, crossing_offsets, crossing, point, run_offsets = _launch.carve_typed(
        [
            (wp.int32, n_candidates),
            (wp.int32, n_candidates),
            (wp.int32, n_candidates + 1),
            (wp.int32, n_candidates),
            (wp.vec3d, n_candidates),
            (wp.int32, n_h + 1),
        ],
        device,
    )
    _launch.launch(
        kernel_boolean.write_edge_candidates,
        dim=n_h,
        inputs=[*walk, candidate_offsets],
        outputs=[edges, candidate_faces],
        device=device,
    )
    _launch.launch(
        kernel_boolean.decide_edge_candidates,
        dim=n_candidates,
        inputs=[
            _crossing_operand(a),
            _crossing_operand(b),
            expand,
            1 if forward else 0,
            edges,
            candidate_faces,
        ],
        outputs=[crossing_offsets, crossing, point],
        device=device,
    )
    total = scan_count(_launch.view(crossing_offsets, 1, n_candidates + 1))
    candidates = _Candidates(
        candidate_offsets, crossing_offsets, edges, candidate_faces, crossing, point
    )
    return run_offsets, total, candidates


class _Candidates(NamedTuple):
    """One operand's decided edge-face candidates, waiting for the records to be allocated."""

    candidate_offsets: wp.array[wp.int32]
    crossing_offsets: wp.array[wp.int32]
    edges: wp.array[wp.int32]
    faces: wp.array[wp.int32]
    crossing: wp.array[wp.int32]
    point: wp.array[wp.vec3d]


def _crossings_by_pairs(n_halfedges: int, n_faces_b: int) -> bool:
    """
    Whether an operand's edges cross the other's faces pair by pair rather than by tree walks.

    Pairs pay where few edges each walk much of a large tree (2026-11-21: a half-space cutter's
    18 edges against a sphere of 82 k faces, 263 -> 4 ms); a tree walk a thread fills the device
    otherwise. The pairs' flags take 4 bytes a pair, so at most ``_PAIR_CROSSING_MAX_PAIRS`` go
    that way (268 MB): at ``2**24`` a cutter against a 1.3 M-face sphere walked the tree, each of
    its edges a thread through every face, for minutes.
    """
    return (
        n_halfedges <= _PAIR_CROSSING_HALFEDGES
        and n_halfedges * n_faces_b <= _PAIR_CROSSING_MAX_PAIRS
    )


def _crossing_operands(
    a: _Operand, b: _Operand, pad: float, expand: int, forward: bool
) -> list[object]:
    """Return the pair kernels' shared arguments: both operands, ``b``'s face boxes, the side."""
    return [
        _crossing_operand(a),
        _crossing_operand(b),
        b.bvh.lowers,
        b.bvh.uppers,
        pad,
        expand,
        1 if forward else 0,
    ]


def _crossing_operand(operand: _Operand) -> StructInstance:
    """Bundle one operand's arrays the crossing kernels read into their struct argument."""
    bundle = kernel_boolean.CrossingOperand()
    bundle.faces = operand.solid.faces
    bundle.positions = operand.solid.positions
    bundle.vertex_normals = operand.solid.vertex_normals
    bundle.face_normals = operand.solid.face_normals
    bundle.mates = operand.mates
    return bundle


def _write_crossings(
    a: _Operand,
    b: _Operand,
    pad: float,
    expand: int,
    forward: bool,
    offsets: wp.array[wp.int32],
    found: wp.array[wp.int32] | _Candidates | None,
    record_base: int,
    records: list[odt.ArrayNd],
) -> None:
    """Write the crossings ``_count_crossings`` found from ``record_base`` on."""
    device = a.solid.positions.device
    if found is None:
        return
    if isinstance(found, _Candidates):
        n_candidates = found.edges.size
        _launch.launch(
            kernel_boolean.write_candidate_crossings,
            dim=n_candidates + a.solid.faces.size + 1,
            inputs=[
                found.candidate_offsets,
                found.crossing_offsets,
                found.edges,
                found.faces,
                found.crossing,
                found.point,
                n_candidates,
                record_base,
            ],
            outputs=[offsets, *records],
            device=device,
        )
        return
    n_faces_b = b.solid.n_faces
    _launch.launch(
        kernel_boolean.write_pair_crossings,
        dim=a.solid.faces.size * n_faces_b,
        inputs=[n_faces_b, *_crossing_operands(a, b, pad, expand, forward), found, record_base],
        outputs=records,
        device=device,
    )


def _windings(
    a: _Operand,
    b: _Operand,
    record_edge: wp.array[wp.int32],
    first: int,
    count: int,
    pad: float,
    expand: int,
    forward: bool,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """Return ``a``'s broken-halfedge flags, vertex components and roots' windings in ``b``."""
    device = a.solid.positions.device
    broken = _launch.zeros(a.solid.faces.size, dtype=wp.int32, device=device)
    if count > 0:
        _launch.launch(
            kernel_boolean.mark_broken_edges,
            dim=count,
            inputs=[record_edge, a.mates, first],
            outputs=[broken],
            device=device,
        )
    edges = odt.empty_2d((a.solid.faces.size, 2), wp.int32, device=device)
    _launch.launch(
        kernel_boolean.unbroken_edge_rows,
        dim=a.solid.faces.size,
        inputs=[a.solid.faces, broken],
        outputs=[edges],
        device=device,
    )
    labels = connected_component_labels_from_edges(edges, a.solid.n_vertices, validate=False)
    windings = _launch.zeros(a.solid.n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.component_windings,
        dim=a.solid.n_vertices,
        inputs=[
            labels,
            a.solid.positions,
            a.solid.vertex_normals,
            b.solid.faces,
            b.solid.positions,
            b.solid.vertex_normals,
            b.solid.face_normals,
            b.mates,
            b.bvh.id,
            b.solid.lower[2] - 1.0,
            b.solid.upper[2] + 1.0,
            pad,
            expand,
            1 if forward else 0,
        ],
        outputs=[windings],
        device=device,
    )
    return broken, labels, windings


def _ear_clip_state(
    capacity: int, device: wp.DeviceLike
) -> tuple[StructInstance, wp.array[wp.vec3i]]:
    """Allocate the ear clipper's scratch for ``capacity`` slots over every face, and its output."""
    state = _launch.carve_struct(kernel_polyline.EarClipState, capacity, device)
    return state, state.tris


def _heavy_faces(
    face_offsets: wp.array[wp.int32], tri_offsets: wp.array[wp.int32], n_faces: int
) -> tuple[int, list[tuple[int, int, int, int, int]] | None]:
    """
    Count the result faces of ``_HEAVY_FACE_SIDES`` or more polygon sides, and list them.

    Each listed face is ``(face, first side, side end, first triangle slot, slot end)``; the list
    is ``None`` past ``_HEAVY_FACE_LIST`` faces.
    """
    device = face_offsets.device
    found = _launch.zeros(1 + 5 * _HEAVY_FACE_LIST, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.list_heavy_faces,
        dim=n_faces,
        inputs=[face_offsets, tri_offsets, _HEAVY_FACE_SIDES, _HEAVY_FACE_LIST],
        outputs=[found],
        device=device,
    )
    # Host readback: whether a face is big enough to go to the CPU, and which ranges travel.
    values = [int(x) for x in read_values(found, 0, 1 + 5 * _HEAVY_FACE_LIST)]
    count = values[0]
    if count > _HEAVY_FACE_LIST:
        return count, None
    faces = [
        (values[k], values[k + 1], values[k + 2], values[k + 3], values[k + 4])
        for k in range(1, 1 + 5 * count, 5)
    ]
    return count, faces


class _HeavyHost(NamedTuple):
    """The host copies the heavy faces' triangulation left, for their halfedge pass."""

    faces: list[tuple[int, int, int, int, int]]
    listed: wp.array[wp.int32]
    inputs: list[object]
    triangles: wp.array[wp.vec3i]


def _copy_range(dest: odt.ArrayNd, src: odt.ArrayNd, start: int, count: int) -> None:
    """Copy ``src[start:start + count]`` into the same entries of ``dest``."""
    if count > 0:
        _launch.copy(dest, src, dest_offset=start, src_offset=start, count=count)


def _triangulate_heavy_on_host(
    faces: list[tuple[int, int, int, int, int]],
    inputs: list[object],
    scratch: int,
    n_capacity: int,
    counts: wp.array[wp.int32],
    triangles: wp.array[wp.vec3i],
) -> _HeavyHost:
    """
    Triangulate the listed heavy faces on the CPU, moving only what they read and write.

    A face of many polygon sides costs its thread time quadratic in them, which one CPU thread
    runs far faster than one GPU thread. Host arrays as large as the device's take only the
    listed faces' entries (and every position, which their sides name); their counts and
    triangles go back.
    """
    host = wp.get_device("cpu")
    face_offsets, tri_offsets, starts, ends, positions, normals_p, normals_q = (
        cast(odt.ArrayNd, value) for value in inputs[:7]
    )
    n_fp = cast(int, inputs[7])
    h_face_offsets = _launch.empty(face_offsets.size, dtype=wp.int32, device=host)
    h_tri_offsets = _launch.empty(tri_offsets.size, dtype=wp.int32, device=host)
    h_starts = _launch.empty(starts.size, dtype=wp.int32, device=host)
    h_ends = _launch.empty(ends.size, dtype=wp.int32, device=host)
    h_normals_p = _launch.empty(normals_p.size, dtype=wp.vec3d, device=host)
    h_normals_q = _launch.empty(normals_q.size, dtype=wp.vec3d, device=host)
    for f, first, end, _, _ in faces:
        _copy_range(h_face_offsets, face_offsets, f, 2)
        _copy_range(h_tri_offsets, tri_offsets, f, 2)
        _copy_range(h_starts, starts, first, end - first)
        _copy_range(h_ends, ends, first, end - first)
        if f < n_fp:
            _copy_range(h_normals_p, normals_p, f, 1)
        else:
            _copy_range(h_normals_q, normals_q, f - n_fp, 1)
    host_inputs: list[object] = [
        h_face_offsets,
        h_tri_offsets,
        h_starts,
        h_ends,
        _launch.clone(positions, device=host),
        h_normals_p,
        h_normals_q,
        *inputs[7:],
    ]
    listed = _launch.array([face[0] for face in faces], dtype=wp.int32, device=host)
    state, h_triangles = _ear_clip_state(max(n_capacity, 1), host)
    h_counts = _launch.empty(counts.size, dtype=wp.int32, device=host)
    _launch.launch(
        kernel_boolean.triangulate_faces,
        dim=len(faces),
        inputs=[
            *host_inputs,
            1,
            listed,
            *[_launch.empty(scratch, dtype=wp.int32, device=host) for _ in range(3)],
            state,
        ],
        outputs=[h_counts],
        device=host,
    )
    for f, _, _, base, _ in faces:
        _copy_range(counts, h_counts, f, 1)
        # Host read: the face's triangle count, on the CPU.
        _copy_range(triangles, h_triangles, base, int(read_values(h_counts, f, 1)[0]))
    return _HeavyHost(faces, listed, host_inputs, h_triangles)


def _write_heavy_on_host(
    held: _HeavyHost,
    inputs: list[object],
    slot_halfedge: wp.array[wp.int32],
    n_triangles: int,
    outputs: list[object],
) -> None:
    """
    Write the listed heavy faces' halfedges on the CPU, from their triangulation's host copies.

    Only their entries travel: the counts' offsets, references and polygon slots in, and the
    halfedges and triangles they write out.
    """
    host = wp.get_device("cpu")
    count_offsets = cast(odt.ArrayNd, inputs[1])
    coplanar_p, coplanar_q, mesh_p, mesh_q = (cast(odt.ArrayNd, value) for value in inputs[8:12])
    n_fp = cast(int, inputs[12])
    h_face_offsets, h_tri_offsets, h_starts, h_ends, _, h_normals_p, h_normals_q = held.inputs[:7]
    h_count_offsets = _launch.empty(count_offsets.size, dtype=wp.int32, device=host)
    h_references = [
        _launch.empty(array.size, dtype=wp.int32, device=host)
        for array in (coplanar_p, coplanar_q, mesh_p, mesh_q)
    ]
    h_slots = _launch.empty(slot_halfedge.size, dtype=wp.int32, device=host)
    for f, first, end, _, _ in held.faces:
        _copy_range(h_count_offsets, count_offsets, f, 2)
        if f < n_fp:
            _copy_range(h_references[0], coplanar_p, f, 1)
            _copy_range(h_references[2], mesh_p, f, 1)
        else:
            _copy_range(h_references[1], coplanar_q, f - n_fp, 1)
            _copy_range(h_references[3], mesh_q, f - n_fp, 1)
        _copy_range(h_slots, slot_halfedge, first, end - first)
    start, pair, tri_mesh, tri_coplanar, tri_normal = (
        cast(odt.ArrayNd, value) for value in outputs
    )
    h_outputs = [
        _launch.empty(3 * n_triangles, dtype=wp.int32, device=host),
        _launch.empty(3 * n_triangles, dtype=wp.int32, device=host),
        _launch.empty(n_triangles, dtype=wp.int32, device=host),
        _launch.empty(n_triangles, dtype=wp.int32, device=host),
        _launch.empty(n_triangles, dtype=wp.vec3d, device=host),
    ]
    _launch.launch(
        kernel_boolean.write_result_halfedges,
        dim=len(held.faces),
        inputs=[
            h_tri_offsets,
            h_count_offsets,
            held.triangles,
            h_face_offsets,
            h_starts,
            h_ends,
            h_normals_p,
            h_normals_q,
            *h_references,
            *inputs[12:],
            1,
            held.listed,
            h_slots,
        ],
        outputs=h_outputs,
        device=host,
    )
    for f, first, end, _, _ in held.faces:
        # Host read: the face's triangle range, on the CPU.
        dst, stop = (int(x) for x in read_values(h_count_offsets, f, 2))
        _copy_range(start, h_outputs[0], 3 * dst, 3 * (stop - dst))
        _copy_range(pair, h_outputs[1], 3 * dst, 3 * (stop - dst))
        for array, h_array in zip((tri_mesh, tri_coplanar, tri_normal), h_outputs[2:], strict=True):
            _copy_range(array, h_array, dst, stop - dst)
        _copy_range(slot_halfedge, h_slots, first, end - first)


def _launch_heavy_faces(
    kernel: odt.Kernel,
    n_faces: int,
    inputs: list[object],
    paired: list[tuple[odt.ArrayNd, odt.ArrayNd]],
    struct: StructInstance | None,
    outputs: list[object],
) -> None:
    """
    Run a per-face kernel's heavy pass on the CPU, over the arrays the CUDA pass left.

    A face of many polygon sides costs its thread time quadratic in them, which one CPU thread
    runs far faster than one GPU thread. Every array goes to the host and back whole: the heavy
    pass writes only its faces' ranges, so the rest come back as they were. ``struct`` (the ear
    clipper's scratch, allocated on the host) follows the inputs, and ``paired`` gives the device
    array each of its host fields copies back to.
    """
    host = wp.get_device("cpu")
    moved: list[tuple[odt.ArrayNd, odt.ArrayNd]] = []

    def to_host(value: object) -> object:
        if isinstance(value, wp.array):
            copy = _launch.clone(value, device=host)
            moved.append((value, copy))
            return copy
        return value

    for device_array, host_array in paired:
        _launch.copy(host_array, device_array)
        moved.append((device_array, host_array))
    host_inputs = [to_host(value) for value in inputs]
    if struct is not None:
        host_inputs.append(struct)
    host_outputs = [to_host(value) for value in outputs]
    _launch.launch(kernel, dim=n_faces, inputs=host_inputs, outputs=host_outputs, device=host)
    for array, copy in moved:
        _launch.copy(array, copy)


# An affine transform as Manifold's ``mat3x4``: four columns of three rows.
_Affine = tuple[tuple[float, float, float], ...]


_PIO2_HI = 1.57079632679489655800e00


_PIO2_LO = 6.12323399573676603587e-17


_HALF_PI = 1.57079632679489661923132169163975144


_PI = 3.1415926535897931160e00


_PI_LO = 1.2246467991473531772e-16


_K_PI = 3.14159265358979323846264338327950288


def _halfspace_solid(
    solid: Solid, normal: Sequence[float], origin_offset: float, mesh_id: int
) -> Solid:
    """
    Manifold's ``Halfspace``: a cube past the plane covering ``solid``'s box, as a solid.

    ``Cube(vec3(2), true)`` moved by one along +x, scaled to cover the box, moved to the plane
    and turned onto the normal, every step in Manifold's ``float64`` order, the transforms
    composed lazily and applied once (``Impl::Transform``).

    Raises
    ------
    ValueError
        If ``normal`` is zero or not finite.
    """
    device = solid.positions.device
    nx, ny, nz = (float(c) for c in normal)
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not (length > 0.0 and math.isfinite(length)):
        raise ValueError(f"normal must be finite and nonzero, got {tuple(normal)}")
    n = (nx / length, ny / length, nz / length)
    offset = float(origin_offset)
    cube = _unit_cube_solid(mesh_id, device)
    lower, upper = solid.lower, solid.upper
    center = tuple(0.5 * (upper[i] + lower[i]) for i in range(3))
    size_box = tuple(upper[i] - lower[i] for i in range(3))
    delta = tuple(center[i] - n[i] * offset for i in range(3))
    size = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]) + 0.5 * (
        math.sqrt(size_box[0] * size_box[0] + size_box[1] * size_box[1] + size_box[2] * size_box[2])
    )
    y_deg = -_m_asin(n[2]) * 180 / _K_PI
    z_deg = _m_atan2(n[1], n[0]) * 180 / _K_PI
    transform = _translation(1.0, 0.0, 0.0)
    transform = _compose_affine(_scaling(size), transform)
    transform = _compose_affine(_translation(offset, 0.0, 0.0), transform)
    transform = _compose_affine(_rotation(0.0, y_deg, z_deg), transform)
    return _transformed_solid(cube, transform)


def _unit_cube_solid(mesh_id: int, device: wp.DeviceLike) -> Solid:
    """``Manifold::Cube(vec3(2.0), true)``: Manifold's cube, its vertices and faces in its order."""
    corners = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
    ]
    triangles = [1, 0, 4, 2, 4, 0, 1, 3, 0, 3, 1, 5, 3, 2, 0, 3, 7, 2]
    triangles += [5, 4, 6, 5, 1, 4, 6, 4, 2, 7, 6, 2, 7, 3, 5, 7, 5, 6]
    positions_np = np.array([[2.0 * c - 1.0 for c in corner] for corner in corners])
    positions = _launch.array(positions_np, dtype=wp.vec3d, device=device)
    faces = _launch.array(np.array(triangles, dtype=np.int32), dtype=wp.int32, device=device)
    # ``SetEpsilon()`` on a box of scale one: both tolerances ``kPrecision``.
    epsilon = kernel_solid.PRECISION
    box = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    return solid_from_positions(
        positions, faces, mesh_id, epsilon, epsilon, *box, "cube", sort_first=True
    )


def _m_asin(x: float) -> float:
    """``math::asin``."""
    if not math.isfinite(x) or x < -1.0 or x > 1.0:
        return float("nan")
    if x == 1.0:
        return _HALF_PI
    if x == -1.0:
        return -_HALF_PI
    return _HALF_PI - _m_acos(x)


def _m_acos(x: float) -> float:
    """``math::acos``."""
    xx = _bits(x)
    hx = xx >> 32
    ix = hx & 0x7FFFFFFF
    if ix >= 0x3FF00000:
        if x == 1.0:
            return 0.0
        if x == -1.0:
            return 2 * _PIO2_HI
        return float("nan")
    if ix < 0x3FE00000:
        if ix <= 0x3C600000:
            return _PIO2_HI
        return _PIO2_HI - (x - (_PIO2_LO - x * _acos_ratio(x * x)))
    if hx >> 31:
        z = (1.0 + x) * 0.5
        s = math.sqrt(z)
        w = _acos_ratio(z) * s - _PIO2_LO
        return 2 * (_PIO2_HI - (s + w))
    z = (1.0 - x) * 0.5
    s = math.sqrt(z)
    df = _from_bits(_bits(s) & 0xFFFFFFFF00000000)
    c = (z - df * df) / (s + df)
    w = _acos_ratio(z) * s + c
    return 2 * (df + w)


def _bits(x: float) -> int:
    """Return the IEEE bits of a ``float64``."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _acos_ratio(z: float) -> float:
    p = z * (
        1.66666666666666657415e-01
        + z
        * (
            -3.25565818622400915405e-01
            + z
            * (
                2.01212532134862925881e-01
                + z
                * (
                    -4.00555345006794114027e-02
                    + z * (7.91534994289814532176e-04 + z * 3.47933107596021167570e-05)
                )
            )
        )
    )
    q = 1.0 + z * (
        -2.40339491173441421878e00
        + z
        * (
            2.02094576023350569471e00
            + z * (-6.88283971605453293030e-01 + z * 7.70381505559019352791e-02)
        )
    )
    return p / q


def _from_bits(u: int) -> float:
    """Return the ``float64`` of IEEE bits."""
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def _m_atan2(y: float, x: float) -> float:
    """``math::atan2``."""
    if math.isnan(x) or math.isnan(y):
        return x + y
    bx = _bits(x)
    by = _bits(y)
    ix = bx >> 32
    iy = by >> 32
    lx = bx & 0xFFFFFFFF
    ly = by & 0xFFFFFFFF
    if ((ix - 0x3FF00000) | lx) == 0:
        return _m_atan(y)
    m = ((iy >> 31) & 1) | ((ix >> 30) & 2)
    ix &= 0x7FFFFFFF
    iy &= 0x7FFFFFFF
    if (iy | ly) == 0:
        return y if m in (0, 1) else (_PI if m == 2 else -_PI)
    if (ix | lx) == 0:
        return -_PI / 2 if m & 1 else _PI / 2
    if ix == 0x7FF00000:
        if iy == 0x7FF00000:
            return (_PI / 4, -_PI / 4, 3 * _PI / 4, -3 * _PI / 4)[m]
        return (0.0, -0.0, _PI, -_PI)[m]
    if ix + (64 << 20) < iy or iy == 0x7FF00000:
        return -_PI / 2 if m & 1 else _PI / 2
    z = 0.0 if (m & 2) and iy + (64 << 20) < ix else _m_atan(abs(y / x))
    if m == 0:
        return z
    if m == 1:
        return -z
    if m == 2:
        return _PI - (z - _PI_LO)
    return (z - _PI_LO) - _PI


def _m_atan(x: float) -> float:
    """``math::atan``."""
    atanhi = (
        4.63647609000806093515e-01,
        7.85398163397448278999e-01,
        9.82793723247329054082e-01,
        1.57079632679489655800e00,
    )
    atanlo = (
        2.26987774529616870924e-17,
        3.06161699786838301793e-17,
        1.39033110312309984516e-17,
        6.12323399573676603587e-17,
    )
    at = (
        3.33333333333329318027e-01,
        -1.99999999998764832476e-01,
        1.42857142725034663711e-01,
        -1.11111104054623557880e-01,
        9.09088713343650656196e-02,
        -7.69187620504482999495e-02,
        6.66107313738753120669e-02,
        -5.83357013379057348645e-02,
        4.97687799461593236017e-02,
        -3.65315727442169155270e-02,
        1.62858201153657823623e-02,
    )
    ix = _bits(x) >> 32
    sign = ix >> 31
    ix &= 0x7FFFFFFF
    if ix >= 0x44100000:
        if math.isnan(x):
            return x
        z = atanhi[3]
        return -z if sign else z
    if ix < 0x3FDC0000:
        if ix < 0x3E400000:
            return x
        index = -1
    else:
        x = abs(x)
        if ix < 0x3FF30000:
            if ix < 0x3FE60000:
                index = 0
                x = (2.0 * x - 1.0) / (2.0 + x)
            else:
                index = 1
                x = (x - 1.0) / (x + 1.0)
        elif ix < 0x40038000:
            index = 2
            x = (x - 1.5) / (1.0 + 1.5 * x)
        else:
            index = 3
            x = -1.0 / x
    z = x * x
    w = z * z
    s1 = z * (at[0] + w * (at[2] + w * (at[4] + w * (at[6] + w * (at[8] + w * at[10])))))
    s2 = w * (at[1] + w * (at[3] + w * (at[5] + w * (at[7] + w * at[9]))))
    if index < 0:
        return x - x * (s1 + s2)
    zz = atanhi[index] - (x * (s1 + s2) - atanlo[index] - x)
    return -zz if sign else zz


def _translation(x: float, y: float, z: float) -> _Affine:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (x, y, z))


def _compose_affine(m: _Affine, transform: _Affine) -> _Affine:
    """``CsgLeafNode::Transform``: ``m * Mat4(transform)``."""
    columns4 = [(*transform[0], 0.0), (*transform[1], 0.0), (*transform[2], 0.0)]
    columns4.append((*transform[3], 1.0))
    return tuple(_mat_vec(m, column) for column in columns4)


def _mat_vec(columns: _Affine, v: Sequence[float]) -> tuple[float, float, float]:
    """``la::mul`` of a matrix by a vector: the columns' sum weighted by ``v``, left to right."""
    out = [0.0, 0.0, 0.0]
    for i in range(3):
        acc = columns[0][i] * v[0]
        for k in range(1, len(v)):
            acc = acc + columns[k][i] * v[k]
        out[i] = acc
    return out[0], out[1], out[2]


def _scaling(s: float) -> _Affine:
    return ((s, 0.0, 0.0), (0.0, s, 0.0), (0.0, 0.0, s), (0.0, 0.0, 0.0))


def _rotation(x_degrees: float, y_degrees: float, z_degrees: float) -> _Affine:
    """``CsgNode::Rotate``: ``rZ * rY * rX`` from ``sind`` / ``cosd``."""
    rx = (
        (1.0, 0.0, 0.0),
        (0.0, _cosd(x_degrees), _sind(x_degrees)),
        (0.0, -_sind(x_degrees), _cosd(x_degrees)),
    )
    ry = (
        (_cosd(y_degrees), 0.0, -_sind(y_degrees)),
        (0.0, 1.0, 0.0),
        (_sind(y_degrees), 0.0, _cosd(y_degrees)),
    )
    rz = (
        (_cosd(z_degrees), _sind(z_degrees), 0.0),
        (-_sind(z_degrees), _cosd(z_degrees), 0.0),
        (0.0, 0.0, 1.0),
    )
    rotation = _mat3_mul(_mat3_mul(rz, ry), rx)
    return (*rotation, (0.0, 0.0, 0.0))


def _cosd(x: float) -> float:
    """``cosd``: the cosine of ``x`` degrees, exact at multiples of 90."""
    return _sind(x + 90.0)


def _sind(x: float) -> float:
    """``sind``: the sine of ``x`` degrees, exact at multiples of 90."""
    if not math.isfinite(x):
        return float("nan")
    if x < 0.0:
        return -_sind(-x)
    r = math.remainder(abs(x), 90.0)
    quo = round((abs(x) - r) / 90.0)
    xr = r * _K_PI / 180
    return (_m_sin(xr), _m_cos(xr), -_m_sin(xr), -_m_cos(xr))[quo % 4]


def _m_sin(x: float) -> float:
    """``math::sin`` for ``|x| <= pi / 4``, the only range ``sind`` hands it."""
    ix = (_bits(x) >> 32) & 0x7FFFFFFF
    if ix > 0x3FE921FB:
        raise ValueError("outside the reduced range")
    if ix < 0x3E500000:
        return x
    return _sin_kernel(x, 0.0, 0)


def _sin_kernel(x: float, y: float, iy: int) -> float:
    s1 = -1.66666666666666324348e-01
    s2 = 8.33333333332248946124e-03
    s3 = -1.98412698298579493134e-04
    s4 = 2.75573137070700676789e-06
    s5 = -2.50507602534068634195e-08
    s6 = 1.58969099521155010221e-10
    z = x * x
    w = z * z
    r = s2 + z * (s3 + z * s4) + z * w * (s5 + z * s6)
    v = z * x
    if iy == 0:
        return x + v * (s1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * s1)


def _m_cos(x: float) -> float:
    """``math::cos`` for ``|x| <= pi / 4``."""
    ix = (_bits(x) >> 32) & 0x7FFFFFFF
    if ix > 0x3FE921FB:
        raise ValueError("outside the reduced range")
    if ix < 0x3E46A09E:
        return 1.0
    return _cos_kernel(x, 0.0)


def _cos_kernel(x: float, y: float) -> float:
    c1 = 4.16666666666666019037e-02
    c2 = -1.38888888888741095749e-03
    c3 = 2.48015872894767294178e-05
    c4 = -2.75573143513906633035e-07
    c5 = 2.08757232129817482790e-09
    c6 = -1.13596475577881948265e-11
    z = x * x
    w = z * z
    r = z * (c1 + z * (c2 + z * c3)) + w * w * (c4 + z * (c5 + z * c6))
    hz = 0.5 * z
    w1 = 1.0 - hz
    return w1 + (((1.0 - w1) - hz) + (z * r - x * y))


def _mat3_mul(a: _Affine, b: _Affine) -> _Affine:
    return tuple(_mat_vec(a, column) for column in b)


def _transformed_solid(solid: Solid, transform: _Affine) -> Solid:
    """``Impl::Transform``: positions and normals moved, the order and references kept."""
    device = solid.positions.device
    # Host readback: a cutter of eight vertices is transformed in ``float64`` on the host, in
    # Manifold's operation order.
    positions = read_array(solid.positions)
    face_normals = read_array(solid.face_normals)
    vertex_normals = read_array(solid.vertex_normals)
    moved = np.array([_mat_vec(transform, (*p, 1.0)) for p in positions.tolist()])
    normal_transform = _normal_transform(transform)
    faces_n = np.array([_transform_normal(normal_transform, v) for v in face_normals.tolist()])
    verts_n = np.array([_transform_normal(normal_transform, v) for v in vertex_normals.tolist()])
    lower = (float(moved[:, 0].min()), float(moved[:, 1].min()), float(moved[:, 2].min()))
    upper = (float(moved[:, 0].max()), float(moved[:, 1].max()), float(moved[:, 2].max()))
    linear = np.array([transform[c] for c in range(3)]).T
    spectral = float(np.linalg.svd(linear, compute_uv=False)[0])
    scale = max(abs(x) for x in (*lower, *upper))
    # ``SetEpsilon(epsilon * SpectralNorm)``, the tolerance at least the new epsilon.
    epsilon = max(solid.epsilon * spectral, kernel_solid.PRECISION * scale)
    tolerance = max(solid.tolerance, epsilon)
    return Solid(
        _launch.array(moved, dtype=wp.vec3d, device=device),
        solid.faces,
        _launch.array(faces_n, dtype=wp.vec3d, device=device),
        _launch.array(verts_n, dtype=wp.vec3d, device=device),
        solid.face_mesh,
        solid.face_coplanar,
        epsilon,
        tolerance,
        lower,
        upper,
    )


def _normal_transform(m: _Affine) -> _Affine:
    """``NormalTransform``: ``inverse(transpose(mat3(m)))``, by the adjugate."""
    a = tuple(tuple(m[r][c] for r in range(3)) for c in range(3))
    xx, xy, xz = a[0]
    yx, yy, yz = a[1]
    zx, zy, zz = a[2]
    adjugate = (
        (yy * zz - zy * yz, zy * xz - xy * zz, xy * yz - yy * xz),
        (yz * zx - zz * yx, zz * xx - xz * zx, xz * yx - yz * xx),
        (yx * zy - zx * yy, zx * xy - xx * zy, xx * yy - yx * xy),
    )
    determinant = xx * (yy * zz - zy * yz) + xy * (yz * zx - zz * yx) + xz * (yx * zy - zx * yy)
    return tuple((c[0] / determinant, c[1] / determinant, c[2] / determinant) for c in adjugate)


def _transform_normal(n: _Affine, v: Sequence[float]) -> tuple[float, float, float]:
    """``TransformNormals``: the normal transformed and normalized, or zero."""
    t = _mat_vec(n, v)
    length = math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
    if not length > 0.0:
        return 0.0, 0.0, 0.0
    out = (t[0] / length, t[1] / length, t[2] / length)
    if math.isnan(out[0]):
        return 0.0, 0.0, 0.0
    return out
