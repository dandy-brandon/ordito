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
from typing import Literal, NamedTuple

import numpy as np
import warp as wp
from warp._src.codegen import StructInstance

import ordito.typing as odt
from ordito import _launch
from ordito._common import offsets_buffer, read_array, read_values, require_same_device, scan_count
from ordito.array import split as array_split
from ordito.graph import connected_component_labels_from_edges
from ordito.halfedge import halfedge_twins
from ordito.kernels import boolean as kernel_boolean
from ordito.kernels import polyline as kernel_polyline
from ordito.points import convex_hull
from ordito.repair import remove_unreferenced_vertices

_Operation = Literal["union", "intersection", "difference"]

_FLOAT32_EPSILON = 1.1920929e-07

# ``BatchUnion``'s chunk: it groups at most this many of its last children at a time.
_MAX_UNION_SIZE = 1000
# ``Partition``'s cache: each sorted set of side divisions' barycentrics and sub-triangles.
_PARTITIONS: dict[
    tuple[int, int, int, int],
    tuple[list[tuple[float, float, float, float]], list[tuple[int, int, int]]],
] = {}


def union(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid both closed meshes enclose together, as a closed mesh.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` first closed, consistently wound mesh.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` second closed, consistently wound mesh.

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
    return _boolean(vertices_a, faces_a, vertices_b, faces_b, "union")


def intersection(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid both closed meshes enclose in common, as a closed mesh.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` first closed, consistently wound mesh.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` second closed, consistently wound mesh.

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
    return _boolean(vertices_a, faces_a, vertices_b, faces_b, "intersection")


def difference(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the solid the first closed mesh encloses outside the second, as a closed mesh.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` closed, consistently wound mesh to cut.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` closed, consistently wound mesh to remove.

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
    return _boolean(vertices_a, faces_a, vertices_b, faces_b, "difference")


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
    p = _solid_from_mesh(vertices_a, faces_a, 0, "a")
    q = _solid_from_mesh(vertices_b, faces_b, 1, "b")
    inside, outside = _solid_split(p, q)
    return _mesh_from_solid(inside, device), _mesh_from_solid(outside, device)


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
    solid = _solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty:
        empty = _mesh_from_solid(solid, device)
        return empty, _mesh_from_solid(solid, device)
    above, below = _solid_split(solid, _halfspace_solid(solid, normal, origin_offset, 1))
    return _mesh_from_solid(above, device), _mesh_from_solid(below, device)


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
    solid = _solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty:
        return _mesh_from_solid(solid, device)
    halfspace = _halfspace_solid(solid, normal, origin_offset, 1)
    return _mesh_from_solid(_solid_boolean(solid, halfspace, "intersection"), device)


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
    return _mesh_from_solid(_batch_union(solids), device)


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
    return _mesh_from_solid(_batch_boolean(solids, "intersection"), device)


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
        return _mesh_from_solid(first, device)
    return _mesh_from_solid(_solid_boolean(first, _batch_union(solids[1:]), "difference"), device)


def segment_hits(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    origins: wp.array[wp.vec3],
    endpoints: wp.array[wp.vec3],
) -> tuple[
    wp.array[wp.int32],
    wp.array[wp.float64],
    wp.array[wp.vec3],
    wp.array[wp.vec3],
    wp.array[wp.int32],
]:
    """
    Return every crossing of each segment with a closed mesh's surface, in order along it.

    A crossing is decided by the same exact predicates as the Booleans, ties included, so a
    segment through an edge or a vertex crosses the surface exactly once there, and a segment
    entering and leaving a solid crosses it an even number of times.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    origins, endpoints
        ``(n_segments,)`` segment ends; segment ``s`` runs from ``origins[s]`` to
        ``endpoints[s]``.

    Returns
    -------
    hit_faces : wp.array[wp.int32]
        ``(n_hits,)`` each crossing's face, an index into ``faces``' triangles.
    parameters : wp.array[wp.float64]
        ``(n_hits,)`` each crossing's place along its segment, from 0 at the origin to 1 at the
        endpoint (measured along the segment's dominant axis).
    points : wp.array[wp.vec3]
        ``(n_hits,)`` each crossing.
    normals : wp.array[wp.vec3]
        ``(n_hits,)`` the crossed face's unit normal (shared by every face of a planar region).
    offsets : wp.array[wp.int32]
        ``(n_segments + 1,)`` segment ``s``'s crossings are ``offsets[s]:offsets[s + 1]``, nearest
        the origin first.

    Raises
    ------
    RuntimeError
        If the four arrays are not all on one device.
    ValueError
        If the mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound, or if ``origins`` and ``endpoints`` differ in length.

    See Also
    --------
    [`ray.intersects_location`][ordito.ray.intersects_location]

    Notes
    -----
    Manifold's ``Manifold.ray_cast``, for many segments at once.

    Examples
    --------
    A segment through a unit cube, entering and leaving:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    origins = wp.array([[0.3, 0.4, -1.0]], dtype=wp.vec3)
    endpoints = wp.array([[0.3, 0.4, 2.0]], dtype=wp.vec3)
    _, parameters, points, _, offsets = od.boolean.segment_hits(vertices, faces, origins, endpoints)
    print(offsets.numpy().tolist())  # [0, 2]
    print([round(float(z), 4) for z in points.numpy()[:, 2]])  # [0.0, 1.0]
    ```
    """
    require_same_device(vertices=vertices, faces=faces, origins=origins, endpoints=endpoints)
    n_segments = int(origins.shape[0])
    if int(endpoints.shape[0]) != n_segments:
        raise ValueError(
            f"origins and endpoints must match in length, got {n_segments} and "
            f"{int(endpoints.shape[0])}"
        )
    device = vertices.device
    solid = _solid_from_mesh(vertices, faces, 0, "")
    offsets, counts = offsets_buffer(n_segments, device)
    if solid.is_empty or n_segments == 0:
        return (
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.float64, device=device),
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.vec3, device=device),
            offsets,
        )
    mates = halfedge_twins(solid.faces, solid.n_vertices, validate=False)
    ray_positions = _launch.empty(2 * n_segments, dtype=wp.vec3d, device=device)
    ray_faces = _launch.empty(3 * n_segments, dtype=wp.int32, device=device)
    ray_mates = _launch.empty(3 * n_segments, dtype=wp.int32, device=device)
    ray_normals = _launch.empty(2 * n_segments, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.segment_edge_mesh,
        dim=n_segments,
        inputs=[origins, endpoints],
        outputs=[ray_positions, ray_faces, ray_mates, ray_normals],
        device=device,
    )
    segment_box_lower, segment_box_upper = _positions_box(ray_positions)
    scale = max(solid.scale, *(abs(x) for x in (*segment_box_lower, *segment_box_upper)))
    pad = max(scale, 1.0) * 4.0 * _FLOAT32_EPSILON
    # The face tree for the segments' ray walks: boxes padded past the walk's ``float32`` rounding.
    lowers = _launch.empty(solid.n_faces, dtype=wp.vec3, device=device)
    uppers = _launch.empty(solid.n_faces, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_boolean.operand_boxes,
        dim=solid.n_faces,
        inputs=[solid.positions, solid.faces, mates, pad],
        outputs=[
            lowers,
            uppers,
            _launch.full(solid.n_vertices, 2147483647, dtype=wp.int32, device=device),
            _launch.zeros(1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    bvh = wp.Bvh(lowers, uppers)
    shared = [
        ray_faces,
        ray_positions,
        ray_normals,
        ray_mates,
        solid.faces,
        solid.positions,
        solid.vertex_normals,
        solid.face_normals,
        mates,
        bvh.id,
        pad,
    ]
    _launch.launch(
        kernel_boolean.count_segment_hits,
        dim=n_segments,
        inputs=shared,
        outputs=[counts],
        device=device,
    )
    n_hits = scan_count(counts)
    hit_faces = _launch.empty(n_hits, dtype=wp.int32, device=device)
    parameters = _launch.empty(n_hits, dtype=wp.float64, device=device)
    points = _launch.empty(n_hits, dtype=wp.vec3, device=device)
    normals = _launch.empty(n_hits, dtype=wp.vec3, device=device)
    assert solid.face_source is not None
    if n_hits > 0:
        _launch.launch(
            kernel_boolean.write_segment_hits,
            dim=n_segments,
            inputs=[*shared, offsets, solid.face_source],
            outputs=[hit_faces, parameters, points, normals],
            device=device,
        )
    return hit_faces, parameters, points, normals, offsets


def min_gap(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    search_length: float,
) -> float:
    """
    Return the least distance between two closed meshes' surfaces, up to ``search_length``.

    Zero when the solids overlap; ``search_length`` when nothing is nearer.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` first closed, consistently wound mesh.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` second closed, consistently wound mesh.
    search_length
        The largest gap reported; face pairs farther apart than this are not compared.

    Returns
    -------
    float
        The gap, in ``[0, search_length]``.

    Raises
    ------
    RuntimeError
        If the four arrays are not all on one device.
    ValueError
        If either mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`metrics.hausdorff_mesh_to_mesh`][ordito.metrics.hausdorff_mesh_to_mesh]

    Notes
    -----
    Manifold's ``Manifold.min_gap``: the solids' intersection first, then the least distance
    between any two faces within reach.

    Examples
    --------
    Two unit cubes half a unit apart:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vertices_b, faces_b = od.creation.box(bounds=[[1.5, 0.0, 0.0], [2.5, 1.0, 1.0]])
    print(od.boolean.min_gap(vertices_a, faces_a, vertices_b, faces_b, 1.0))  # 0.5
    ```
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    device = vertices_a.device
    p = _solid_from_mesh(vertices_a, faces_a, 0, "a")
    q = _solid_from_mesh(vertices_b, faces_b, 1, "b")
    search = float(search_length)
    if p.is_empty or q.is_empty:
        return search
    if not _solid_boolean(p, q, "intersection").is_empty:
        return 0.0
    operand = _operand(p)
    least = _launch.array([math.inf], dtype=wp.float64, device=device)
    scale = max(p.scale, q.scale) + abs(search)
    pad = max(scale, 1.0) * 4.0 * _FLOAT32_EPSILON
    _launch.launch(
        kernel_boolean.min_face_distance,
        dim=q.n_faces,
        inputs=[p.faces, p.positions, q.faces, q.positions, operand.bvh.id, search, pad],
        outputs=[least],
        device=device,
    )
    # Host readback: the gap is a host scalar.
    least_squared = float(read_values(least, 0, 1)[0])
    return math.sqrt(min(least_squared, search * search))


def simplify(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], tolerance: float = 0.0
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return a closed mesh with its redundant vertices removed, its surface moved under ``tolerance``.

    Faces within ``tolerance`` of one plane are treated as one face, and the vertices inside
    such a face or along a straight edge between two are collapsed; slivers go too. The result
    keeps a subset of the vertices.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    tolerance
        How far the surface may move. Zero, or less than the mesh's own precision
        (``float32`` epsilon times its largest coordinate), uses that precision.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` the vertices kept.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` result triangles.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If the mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`remesh.quadric_decimate`][ordito.remesh.quadric_decimate]

    Notes
    -----
    Manifold's ``Manifold.simplify``.

    Examples
    --------
    A subdivided box simplifies back to its twelve triangles:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    vertices, faces = od.remesh.subdivide(vertices, faces)
    print(faces.size // 3)  # 48
    vertices, faces = od.boolean.simplify(vertices, faces)
    print(faces.size // 3)  # 12
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    solid = _solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty:
        return _mesh_from_solid(solid, device)
    return _mesh_from_solid(_solid_simplify(solid, float(tolerance)), device)


def cross_section(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], height: float = 0.0
) -> list[wp.array[wp.vec2]]:
    """
    Return the closed loops where the plane ``z = height`` cuts a closed mesh.

    The loops of [`cross_section_with_offsets`][ordito.boolean.cross_section_with_offsets], one
    array each.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    height
        The plane's ``z``.

    Returns
    -------
    list[wp.array[wp.vec2]]
        ``(n_i,)`` each loop's ``(x, y)`` points, outer loops counter-clockwise and holes clockwise.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If the mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`cross_section_with_offsets`][ordito.boolean.cross_section_with_offsets]
    [`intersection.mesh_with_plane`][ordito.intersection.mesh_with_plane]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    loops = od.boolean.cross_section(vertices, faces, 0.5)
    print(len(loops))  # 1
    ```
    """
    points, offsets = cross_section_with_offsets(vertices, faces, height)
    return array_split(points, offsets)


def cross_section_with_offsets(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], height: float = 0.0
) -> tuple[wp.array[wp.vec2], wp.array[wp.int32]]:
    """
    Return the closed loops where the plane ``z = height`` cuts a closed mesh, packed.

    A face belongs to the section when its lowest corner is at or below the plane and its
    highest above it, so a plane through vertices or along faces is decided one way, the same
    for every face: the section of a box at its bottom height is the bottom face, at its top
    nothing. Each loop runs counter-clockwise around material, as seen from above.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    height
        The plane's ``z``.

    Returns
    -------
    points : wp.array[wp.vec2]
        ``(n_points,)`` every loop's ``(x, y)`` points, loop after loop.
    offsets : wp.array[wp.int32]
        ``(n_loops + 1,)`` loop ``r`` is ``points[offsets[r] : offsets[r + 1]]``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If the mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`cross_section`][ordito.boolean.cross_section]
    [`polyline.triangulate_polygons_from_offsets`][ordito.polyline.triangulate_polygons_from_offsets]

    Notes
    -----
    Manifold's ``Manifold::Slice``: the same loops (Manifold starts each where its hash set does;
    here at the crossing on the earliest face). A plane through a vertex reaches it from each face
    around it, so the loop repeats that point once a face; ``manifold3d``'s ``slice`` returns a
    ``CrossSection``, whose cleanup drops the repeats.

    Examples
    --------
    A unit cube cut through its middle: one square loop of eight points (its faces' diagonals
    cross the plane too):

    ```python
    import ordito as od

    vertices, faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    points, offsets = od.boolean.cross_section_with_offsets(vertices, faces, 0.5)
    print(offsets.numpy().tolist())  # [0, 8]
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device), _launch.zeros(
            1, dtype=wp.int32, device=device
        )
    positions, twins = _closed_positions(vertices, faces)
    ranks, flags = offsets_buffer(n_faces, device)
    face_points = _launch.empty(n_faces, dtype=wp.vec2d, device=device)
    face_next = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.slice_crossings,
        dim=n_faces,
        inputs=[faces, twins, positions, float(height)],
        outputs=[flags, face_points, face_next],
        device=device,
    )
    raw_flags = _launch.clone(flags)
    n_crossings = scan_count(flags)
    if n_crossings == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device), _launch.zeros(
            1, dtype=wp.int32, device=device
        )
    packed_points = _launch.empty(n_crossings, dtype=wp.vec2d, device=device)
    packed_next = _launch.empty(n_crossings, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.gather_crossings,
        dim=n_faces,
        inputs=[raw_flags, ranks, face_points, face_next],
        outputs=[packed_points, packed_next],
        device=device,
    )
    # Host readback: the loops are linked by walking each crossing's successor, a serial chase
    # over the section's points only.
    next_np = read_array(packed_next)
    points_np = read_array(packed_points)
    order: list[int] = []
    sizes = [0]
    visited = np.zeros(n_crossings, dtype=bool)
    for start in range(n_crossings):
        if visited[start]:
            continue
        current = start
        count = 0
        while not visited[current]:
            visited[current] = True
            order.append(current)
            count += 1
            current = int(next_np[current])
        sizes.append(count)
    loops_np = points_np[np.asarray(order)].astype(np.float32)
    points = _launch.array(loops_np, dtype=wp.vec2, device=device)
    offsets = _launch.array(np.cumsum(sizes).astype(np.int32), dtype=wp.int32, device=device)
    return points, offsets


def decompose(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]:
    """
    Return a closed mesh's connected pieces, each a closed mesh.

    Pieces are ordered by their earliest vertex, and each is laid out as
    [`union`][ordito.boolean.union] lays out a result: vertices and faces in Morton order within
    the piece's own box (a mesh of one piece keeps the whole mesh's box). Joining them back with
    [`union_all`][ordito.boolean.union_all] gives the mesh.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.

    Returns
    -------
    list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]
        ``(n_vertices_i,)`` and ``(3 * n_faces_i,)`` each piece.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If the mesh is not closed, has an edge shared by more than two faces, or is not
        consistently wound.

    See Also
    --------
    [`combine.split`][ordito.combine.split]

    Notes
    -----
    Manifold's ``Manifold.decompose``: the same pieces, in the same order and layout.

    Examples
    --------
    ```python
    import ordito as od

    a = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    b = od.creation.box(bounds=[[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]])
    pieces = od.boolean.decompose(*od.boolean.union_all([a, b]))
    print(len(pieces))  # 2
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    solid = _solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty:
        return []
    edges = odt.empty_2d((solid.faces.size, 2), wp.int32, device=device)
    _launch.launch(
        kernel_boolean.unbroken_edge_rows,
        dim=solid.faces.size,
        inputs=[solid.faces, _launch.zeros(solid.faces.size, dtype=wp.int32, device=device)],
        outputs=[edges],
        device=device,
    )
    labels = connected_component_labels_from_edges(edges, solid.n_vertices, validate=False)
    # Host readback: each piece is gathered on the host, its vertices and faces in index order.
    labels_np = read_array(labels)
    roots = np.unique(labels_np)
    if roots.size == 1:
        return [_mesh_from_solid(solid, device)]
    faces_np = read_array(solid.faces).reshape(-1, 3)
    positions_np = read_array(solid.positions)
    pieces: list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]] = []
    for root in roots:
        vertex_ids = np.flatnonzero(labels_np == root)
        piece_faces = faces_np[labels_np[faces_np[:, 0]] == root]
        if piece_faces.size == 0:
            continue
        remap = np.full(labels_np.size, -1, dtype=np.int32)
        remap[vertex_ids] = np.arange(vertex_ids.size, dtype=np.int32)
        piece_positions = positions_np[vertex_ids]
        lower = tuple(float(x) for x in piece_positions.min(axis=0))
        upper = tuple(float(x) for x in piece_positions.max(axis=0))
        sorted_positions, sorted_faces, *_ = _morton_sorted(
            _launch.array(piece_positions, dtype=wp.vec3d, device=device),
            _launch.array(remap[piece_faces].reshape(-1), dtype=wp.int32, device=device),
            (lower[0], lower[1], lower[2]),
            (upper[0], upper[1], upper[2]),
        )
        piece_vertices = _launch.empty(vertex_ids.size, dtype=wp.vec3, device=device)
        _launch.launch(
            kernel_boolean.narrow_positions,
            dim=vertex_ids.size,
            inputs=[sorted_positions],
            outputs=[piece_vertices],
            device=device,
        )
        pieces.append((piece_vertices, sorted_faces))
    return pieces


def hull(points: wp.array[wp.vec3]) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return the convex hull of points as a closed mesh, laid out as the Boolean results are.

    The hull's vertices are the extreme points, in Morton order within the hull's box, and its
    faces follow in the same order. Pass a mesh's vertices for the hull of the mesh, or several
    meshes' vertices together for the hull of all of them.

    Parameters
    ----------
    points
        ``(n_points,)`` points, ``n_points >= 4``, not all coplanar.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` the hull's vertices, a subset of ``points``.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` outward-wound triangles.

    Raises
    ------
    ValueError
        If ``points`` holds fewer than 4 points or no 4 of them span a tetrahedron.

    See Also
    --------
    [`points.convex_hull`][ordito.points.convex_hull]

    Notes
    -----
    Manifold's ``Manifold.hull_points`` (``hull`` and ``batch_hull`` on meshes' vertices): the same
    vertices and faces in the same order, each face possibly starting at another corner. The hull
    is exact (see [`points.convex_hull`][ordito.points.convex_hull]) where Manifold's QuickHull
    drops points within a rounding tolerance of a facet, so nearly coplanar clouds can keep a few
    more vertices; Manifold returns a flat two-sided mesh for coplanar points, where this raises.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.boolean.hull(od.creation.icosphere(subdivisions=2)[0])
    print(vertices.shape[0], faces.size // 3)  # 162 320
    ```
    """
    device = points.device
    hull_faces = convex_hull(points)
    vertices, faces, _ = remove_unreferenced_vertices(points, hull_faces)
    positions = _launch.empty(vertices.shape[0], dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.widen_positions,
        dim=vertices.shape[0],
        inputs=[vertices],
        outputs=[positions],
        device=device,
    )
    lower, upper = _positions_box(positions)
    # ``SetEpsilon()``: both tolerances the double-precision floor at the hull's scale.
    epsilon = kernel_boolean.PRECISION * max(abs(x) for x in (*lower, *upper))
    solid = _solid_from_positions(
        positions, faces, 0, epsilon, epsilon, lower, upper, "hull", sort_first=True
    )
    return _mesh_from_solid(solid, device)


def refine(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], n: int
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return a closed mesh with every edge split into ``n`` equal pieces and each face tiled to match.

    Each face becomes ``n * n`` flat triangles in its own plane, so the surface does not move.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    n
        The pieces each edge is split into; ``1`` returns the mesh as
        [`union`][ordito.boolean.union] lays out a result.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices_out,)`` the input vertices and the added ones.
    faces : wp.array[wp.int32]
        ``(3 * n_faces_out,)`` result triangles.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If ``n`` is below one, the mesh is not closed, has an edge shared by more than two faces,
        or is not consistently wound.

    See Also
    --------
    [`refine_to_length`][ordito.boolean.refine_to_length]
    [`remesh.subdivide`][ordito.remesh.subdivide]

    Notes
    -----
    Manifold's ``Manifold.refine``.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.boolean.refine(*od.creation.box(), 3)
    print(faces.size // 3)  # 108
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    device = vertices.device
    solid = _solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty or n == 1:
        return _mesh_from_solid(solid, device)
    return _mesh_from_solid(_solid_refine(solid, int(n) - 1, 0.0), device)


def refine_to_length(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], length: float
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return a closed mesh with each edge split into pieces of about ``length``, faces tiled to match.

    An edge of length ``l`` takes ``floor(l / length)`` added vertices, evenly spaced, and each
    face is tiled flat between its split sides with interior vertices where they keep the new
    edges near that length too.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    length
        The target piece length; its sign is ignored.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices_out,)`` the input vertices and the added ones.
    faces : wp.array[wp.int32]
        ``(3 * n_faces_out,)`` result triangles.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If ``length`` is zero or not finite, the mesh is not closed, has an edge shared by more
        than two faces, or is not consistently wound.

    See Also
    --------
    [`refine`][ordito.boolean.refine]
    [`remesh.isotropic_remesh`][ordito.remesh.isotropic_remesh]

    Notes
    -----
    Manifold's ``Manifold.refine_to_length``.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.boolean.refine_to_length(*od.creation.box(), 0.25)
    print(faces.size // 3)  # 312
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    length = abs(float(length))
    if length == 0.0 or not math.isfinite(length):
        raise ValueError(f"length must be finite and nonzero, got {length}")
    device = vertices.device
    solid = _solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty:
        return _mesh_from_solid(solid, device)
    return _mesh_from_solid(_solid_refine(solid, 0, length), device)


def smooth_out(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n: int | None = None,
    *,
    length: float | None = None,
    tolerance: float | None = None,
    min_sharp_angle: float = 52.5,
    min_smoothness: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return a closed mesh refined onto a smooth surface through its vertices, keeping sharp edges.

    Each edge becomes a curve leaving its ends along the surface's tangent planes, circular where
    it can be, and each face a patch between its edges' curves. Edges bent past
    ``min_sharp_angle`` stay creases; faces of three or more coplanar triangles stay flat, and
    two coplanar triangles are smoothed as one quad. The refinement is chosen by exactly one of
    ``n``, ``length`` and ``tolerance``.

    Parameters
    ----------
    vertices, faces
        ``(n_vertices,)`` and ``(3 * n_faces,)`` closed, consistently wound mesh.
    n
        The pieces each edge is split into, as [`refine`][ordito.boolean.refine] splits them;
        ``1`` returns the mesh as [`union`][ordito.boolean.union] lays out a result.
    length
        Split each edge into pieces of about this length instead, as
        [`refine_to_length`][ordito.boolean.refine_to_length] does.
    tolerance
        Split each edge until the faceted result is within about this distance of the smooth
        surface instead; curved regions take more vertices than flat ones.
    min_sharp_angle
        Degrees: edges whose faces meet at more than this stay sharp.
    min_smoothness
        How smooth a sharp edge still is, from ``0`` (a crease) to ``1`` (as smooth as any).

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices_out,)`` the input vertices and the added ones, on the smooth surface.
    faces : wp.array[wp.int32]
        ``(3 * n_faces_out,)`` result triangles.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    ValueError
        If not exactly one of ``n``, ``length`` and ``tolerance`` is given, ``n`` is below one,
        ``length`` or ``tolerance`` is zero or not finite, or the mesh is not closed, has an edge
        shared by more than two faces, or is not consistently wound.

    See Also
    --------
    [`refine`][ordito.boolean.refine]
    [`remesh.subdivide`][ordito.remesh.subdivide]

    Notes
    -----
    Manifold's ``Manifold.smooth_out`` followed by ``refine``, ``refine_to_length`` or
    ``refine_to_tolerance``.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.boolean.smooth_out(*od.creation.icosphere(subdivisions=1), 3)
    print(faces.size // 3)  # 720
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if sum(x is not None for x in (n, length, tolerance)) != 1:
        raise ValueError("pass exactly one of n, length and tolerance")
    if n is not None and n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    target = abs(float(length if length is not None else tolerance or 0.0))
    if n is None and (target == 0.0 or not math.isfinite(target)):
        raise ValueError(f"length or tolerance must be finite and nonzero, got {target}")
    device = vertices.device
    solid = _solid_from_mesh(vertices, faces, 0, "")
    if solid.is_empty or n == 1:
        return _mesh_from_solid(solid, device)
    smoothed = _solid_smooth_out(
        solid,
        int(n) if n is not None else 1,
        target if length is not None else 0.0,
        target if tolerance is not None else 0.0,
        float(min_sharp_angle),
        float(min_smoothness),
    )
    return _mesh_from_solid(smoothed, device)


def _solid_refine(solid: _Solid, added: int, length: float) -> _Solid:
    """
    ``Impl::Refine`` with no tangents: ``Subdivide``, ``CalculateVertNormals``, ``SortGeometry``.

    Every edge takes ``added`` vertices, or ``int(|edge| / length)`` when ``length`` is positive;
    the new vertices sit at their barycentrics. Sorted in the solid's own box (``Refine`` does
    not recalculate it).
    """
    device = solid.positions.device
    twins = halfedge_twins(solid.faces, solid.n_vertices, validate=False)
    inside = _launch.zeros(solid.faces.size, dtype=wp.int32, device=device)
    sub = _subdivide(solid, twins, inside, added, length)
    positions = _launch.empty(sub.bary_tri.size, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.linear_bary_positions,
        dim=sub.bary_tri.size,
        inputs=[solid.positions, solid.faces, twins, inside, sub.bary_tri, sub.bary_uvw],
        outputs=[positions],
        device=device,
    )
    refined = solid._replace(
        positions=positions,
        faces=sub.faces,
        face_normals=sub.face_normals,
        vertex_normals=positions,
        face_mesh=sub.face_mesh,
        face_coplanar=sub.face_coplanar,
        face_source=None,
    )
    mesh = _halfedge_mesh_from_solid(refined)
    return _finish_solid(mesh, solid.epsilon, solid.tolerance, device, (solid.lower, solid.upper))


def _solid_smooth_out(
    solid: _Solid,
    n: int,
    length: float,
    tolerance: float,
    min_sharp_angle: float,
    min_smoothness: float,
) -> _Solid:
    """
    ``SmoothOut`` then ``Refine(n)``, ``RefineToLength`` or ``RefineToTolerance``.

    Tangents from the geometry, then the refined vertices on the surface they define, normals
    and coplanar groups recomputed and sorted in the new box.
    """
    device = solid.positions.device
    n_h = solid.faces.size
    n_v = solid.n_vertices
    twins = halfedge_twins(solid.faces, n_v, validate=False)
    tangents = _smoothing_tangents(solid, twins, min_sharp_angle, min_smoothness)
    inside = _launch.empty(n_h, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.quad_diagonal_flags,
        dim=n_h,
        inputs=[tangents],
        outputs=[inside],
        device=device,
    )
    sub = _subdivide(solid, twins, inside, n - 1, length, tangents, tolerance)
    positions = _launch.empty(sub.bary_tri.size, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.interp_tri_positions,
        dim=sub.bary_tri.size,
        inputs=[solid.positions, solid.faces, twins, inside, tangents, sub.bary_tri, sub.bary_uvw],
        outputs=[positions],
        device=device,
    )
    lower, upper = _positions_box(positions)
    return _solid_from_positions(
        positions, sub.faces, 0, solid.epsilon, solid.tolerance, lower, upper, ""
    )


def _smoothing_tangents(
    solid: _Solid, twins: wp.array[wp.int32], min_sharp_angle: float, min_smoothness: float
) -> wp.array[wp.vec4d]:
    """
    ``CreateTangents(SharpenEdges(min_sharp_angle, min_smoothness))``.

    Each halfedge's weighted cubic Bezier tangent, a quad's diagonal marked with weight ``-1``.
    """
    device = solid.positions.device
    n_h = solid.faces.size
    n_v = solid.n_vertices
    n_t = solid.n_faces
    first = _launch.full(n_v, 2147483647, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.least_outgoing_halfedges,
        dim=n_h,
        inputs=[solid.faces],
        outputs=[first],
        device=device,
    )
    flat = _launch.zeros(n_t, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.flat_faces,
        dim=n_t,
        inputs=[twins, solid.face_mesh, solid.face_coplanar],
        outputs=[flat],
        device=device,
    )
    normals = _launch.empty(n_v, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.smoothing_vertex_normals,
        dim=n_v,
        inputs=[
            twins,
            solid.face_mesh,
            solid.face_coplanar,
            flat,
            first,
            solid.face_normals,
            solid.vertex_normals,
        ],
        outputs=[normals],
        device=device,
    )
    tangents = _launch.empty(n_h, dtype=wp.vec4d, device=device)
    _launch.launch(
        kernel_boolean.initial_tangents,
        dim=n_h,
        inputs=[
            solid.positions,
            solid.faces,
            twins,
            solid.face_mesh,
            solid.face_coplanar,
            solid.face_normals,
            normals,
        ],
        outputs=[tangents],
        device=device,
    )
    present = _launch.zeros(n_h, dtype=wp.int32, device=device)
    smoothness = _launch.empty(n_h, dtype=wp.vec2d, device=device)
    min_radians = max(min_sharp_angle, 1e-4) * math.pi / 180
    _launch.launch(
        kernel_boolean.edge_smoothness,
        dim=n_h,
        inputs=[
            solid.faces,
            twins,
            solid.face_mesh,
            solid.face_coplanar,
            flat,
            solid.face_normals,
            min_radians,
            min_smoothness,
        ],
        outputs=[present, smoothness],
        device=device,
    )
    fixed = _launch.zeros(n_h, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.sharpen_vertex_tangents,
        dim=n_v,
        inputs=[solid.positions, solid.faces, twins, flat, first, present, smoothness],
        outputs=[tangents, fixed],
        device=device,
    )
    _launch.launch(
        kernel_boolean.linearize_flat_tangents,
        dim=n_h,
        inputs=[solid.positions, solid.faces, twins],
        outputs=[tangents],
        device=device,
    )
    _launch.launch(
        kernel_boolean.distribute_tangents,
        dim=n_h,
        inputs=[solid.positions, solid.faces, twins, solid.vertex_normals, fixed],
        outputs=[tangents],
        device=device,
    )
    return tangents


class _Subdivision(NamedTuple):
    """``Subdivide``'s triangles, their inherited attributes, and every vertex's barycentric."""

    faces: wp.array[wp.int32]
    face_normals: wp.array[wp.vec3d]
    face_mesh: wp.array[wp.int32]
    face_coplanar: wp.array[wp.int32]
    bary_tri: wp.array[wp.int32]
    bary_uvw: wp.array[wp.vec4d]


def _subdivide(
    solid: _Solid,
    twins: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    added: int,
    length: float,
    tangents: wp.array[wp.vec4d] | None = None,
    tolerance: float = 0.0,
) -> _Subdivision:
    """
    ``Impl::Subdivide``: split the edges and tile each face with its sides' cached partition.

    A quad is the face whose diagonal ``inside`` marks. Vertices keep their indices, each edge's
    added vertices follow in edge order, then each face's interior ones; every vertex records the
    face (a quad's lower triangle) and the barycentric it sits at.
    """
    device = solid.positions.device
    n_h = solid.faces.size
    n_v = solid.n_vertices
    n_t = solid.n_faces
    ranks, forward = offsets_buffer(n_h, device)
    _launch.launch(
        kernel_boolean.forward_halfedge_flags,
        dim=n_h,
        inputs=[solid.faces],
        outputs=[forward],
        device=device,
    )
    # ``ranks[h]`` is then each forward halfedge's edge index.
    n_edges = scan_count(forward)
    added_offsets, added_counts = offsets_buffer(n_edges, device)
    edge_halfedge = _launch.empty(n_edges, dtype=wp.int32, device=device)
    if tangents is None:
        tangents = _launch.empty(1, dtype=wp.vec4d, device=device)
    # ``RefineToTolerance`` keeps some interior: its raw counts feed a second pass.
    raw = added_counts
    if tolerance > 0.0:
        raw = _launch.zeros(n_edges, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.refine_edge_added,
        dim=n_h,
        inputs=[
            solid.positions,
            solid.faces,
            twins,
            ranks,
            inside,
            tangents,
            added,
            length,
            tolerance,
        ],
        outputs=[raw, edge_halfedge],
        device=device,
    )
    if tolerance > 0.0:
        _launch.launch(
            kernel_boolean.keep_interior_added,
            dim=n_h,
            inputs=[solid.faces, twins, ranks, inside, raw],
            outputs=[added_counts],
            device=device,
        )
    n_added = scan_count(added_counts)
    keys = _launch.empty(n_t, dtype=wp.vec4i, device=device)
    _launch.launch(
        kernel_boolean.refine_face_keys,
        dim=n_t,
        inputs=[solid.faces, twins, ranks, inside, added_offsets],
        outputs=[keys],
        device=device,
    )
    # Host readback: the partitions are built on the host, one per distinct set of divisions.
    unique_keys, face_key_np = np.unique(read_array(keys), axis=0, return_inverse=True)
    partitions = [_partition((int(a), int(b), int(c), int(d))) for a, b, c, d in unique_keys]
    key_triangles_np = np.cumsum([0] + [len(tris) for _, tris in partitions], dtype=np.int64)
    key_vertices_np = np.cumsum([0] + [len(bary) for bary, _ in partitions], dtype=np.int64)
    key_interior_np = np.diff(key_vertices_np) - unique_keys.sum(axis=1)
    face_key_np = face_key_np.reshape(-1)
    face_triangles_np = np.diff(key_triangles_np)[face_key_np]
    face_interior_np = key_interior_np[face_key_np]
    triangle_offsets_np = np.concatenate([[0], np.cumsum(face_triangles_np)])
    interior_offsets_np = n_v + n_added + np.concatenate([[0], np.cumsum(face_interior_np)])
    n_new_t = int(triangle_offsets_np[-1])
    n_new_v = int(interior_offsets_np[-1])
    partition_triangles = np.asarray(
        [t for _, tris in partitions for t in tris], dtype=np.int32
    ).reshape(-1, 3)
    partition_bary = np.asarray(
        [b for bary, _ in partitions for b in bary], dtype=np.float64
    ).reshape(-1, 4)

    def upload(values: np.ndarray) -> wp.array[wp.int32]:
        return _launch.array(values.astype(np.int32), dtype=wp.int32, device=device)

    bary_tri = _launch.empty(n_new_v, dtype=wp.int32, device=device)
    bary_uvw = _launch.empty(n_new_v, dtype=wp.vec4d, device=device)
    last = _launch.full(n_v, -1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.last_corner_halfedges,
        dim=n_h,
        inputs=[solid.faces, inside],
        outputs=[last],
        device=device,
    )
    _launch.launch(
        kernel_boolean.refine_retained_bary,
        dim=n_v,
        inputs=[twins, inside, last],
        outputs=[bary_tri, bary_uvw],
        device=device,
    )
    _launch.launch(
        kernel_boolean.refine_edge_bary,
        dim=n_edges,
        inputs=[twins, inside, edge_halfedge, added_offsets, n_v],
        outputs=[bary_tri, bary_uvw],
        device=device,
    )
    faces = _launch.empty(3 * n_new_t, dtype=wp.int32, device=device)
    face_normals = _launch.empty(n_new_t, dtype=wp.vec3d, device=device)
    face_mesh = _launch.empty(n_new_t, dtype=wp.int32, device=device)
    face_coplanar = _launch.empty(n_new_t, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.refine_faces,
        dim=n_t,
        inputs=[
            solid.faces,
            twins,
            ranks,
            inside,
            added_offsets,
            n_v,
            upload(face_key_np),
            upload(key_triangles_np),
            upload(key_vertices_np),
            _launch.array(partition_triangles, dtype=wp.vec3i, device=device),
            _launch.array(partition_bary, dtype=wp.vec4d, device=device),
            upload(triangle_offsets_np),
            upload(interior_offsets_np),
            solid.face_normals,
            solid.face_mesh,
            solid.face_coplanar,
        ],
        outputs=[faces, face_normals, face_mesh, face_coplanar, bary_tri, bary_uvw],
        device=device,
    )
    return _Subdivision(faces, face_normals, face_mesh, face_coplanar, bary_tri, bary_uvw)


def _partition(
    n: tuple[int, int, int, int],
) -> tuple[list[tuple[float, float, float, float]], list[tuple[int, int, int]]]:
    """
    ``Partition::GetCachedPartition`` of the sorted side divisions ``n``.

    A quad's when ``n[3]`` is positive, a triangle's (``n[0] >= n[1] >= n[2]``) otherwise,
    nothing for all zeros. Purely topological: each vertex's barycentric (corners, then each
    side's added vertices, then the interior) and the sub-triangles over them, cached by the
    divisions.
    """
    cached = _PARTITIONS.get(n)
    if cached is not None:
        return cached
    bary: list[tuple[float, float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    if n[0] > 0 and n[3] > 0:  # quad
        bary.extend(
            [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
        )
        offsets = [4, 0, 0, 0]
        for i in range(4):
            if i > 0:
                offsets[i] = offsets[i - 1] + n[i - 1] - 1
            next_bary = bary[(i + 1) % 4]
            for j in range(1, n[i]):
                bary.append(_lerp4(bary[i], next_bary, j / n[i]))
        _partition_quad(
            tris, bary, [0, 1, 2, 3], offsets, [k - 1 for k in n], [True, True, True, True]
        )
    elif n[0] > 0:
        _triangle_partition(bary, tris, (n[0], n[1], n[2]))
    _PARTITIONS[n] = (bary, tris)
    return bary, tris


def _triangle_partition(
    bary: list[tuple[float, float, float, float]],
    tris: list[tuple[int, int, int]],
    n: tuple[int, int, int],
) -> None:
    """``GetCachedPartition``'s triangle branch, sides of ``n[0] >= n[1] >= n[2]`` pieces."""
    bary.extend([(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)])
    for i in range(3):
        next_bary = bary[(i + 1) % 3]
        for j in range(1, n[i]):
            bary.append(_lerp4(bary[i], next_bary, j / n[i]))
    offsets = (3, 3 + n[0] - 1, 3 + n[0] - 1 + n[1] - 1)
    f = float(n[2] * n[2] + n[0] * n[0])
    if n[1] == 1:
        if n[0] == 1:
            tris.append((0, 1, 2))
        else:
            _partition_fan(tris, (0, 1, 2), n[0] - 1, offsets[0])
    elif n[1] * n[1] > f - math.sqrt(2.0) * n[0] * n[2]:  # acute-ish
        tris.append((offsets[1] - 1, 1, offsets[1]))
        _partition_quad(
            tris,
            bary,
            [offsets[1] - 1, offsets[1], 2, 0],
            [-1, offsets[1] + 1, offsets[2], offsets[0]],
            [0, n[1] - 2, n[2] - 1, n[0] - 2],
            [True, True, True, True],
        )
    else:  # obtuse: split into two acute
        ns = min(n[0] - 2, int(_round_half_away((f - n[1] * n[1]) / (2 * n[0]))))
        nh = int(max(1.0, _round_half_away(math.sqrt(n[2] * n[2] - ns * ns))))
        h_offset = len(bary)
        middle = bary[offsets[0] + ns - 1]
        for j in range(1, nh):
            bary.append(_lerp4(bary[2], middle, j / nh))
        tris.append((offsets[1] - 1, 1, offsets[1]))
        _partition_quad(
            tris,
            bary,
            [offsets[1] - 1, offsets[1], 2, offsets[0] + ns - 1],
            [-1, offsets[1] + 1, h_offset, offsets[0] + ns],
            [0, n[1] - 2, nh - 1, n[0] - ns - 2],
            [True, True, True, True],
        )
        if n[2] == 1:
            _partition_fan(tris, (0, offsets[0] + ns - 1, 2), ns - 1, offsets[0])
        elif ns == 1:
            tris.append((h_offset, 2, offsets[2]))
            _partition_quad(
                tris,
                bary,
                [h_offset, offsets[2], 0, offsets[0]],
                [-1, offsets[2] + 1, -1, h_offset + nh - 2],
                [0, n[2] - 2, ns - 1, nh - 2],
                [True, True, True, False],
            )
        else:
            tris.append((h_offset - 1, 0, offsets[0]))
            _partition_quad(
                tris,
                bary,
                [h_offset - 1, offsets[0], offsets[0] + ns - 1, 2],
                [-1, offsets[0] + 1, h_offset + nh - 2, offsets[2]],
                [0, ns - 2, nh - 1, n[2] - 2],
                [True, True, False, True],
            )


def _lerp4(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], t: float
) -> tuple[float, float, float, float]:
    """``la::lerp`` of two barycentrics: ``a * (1 - t) + b * t``, component by component."""
    s = 1.0 - t
    return (a[0] * s + b[0] * t, a[1] * s + b[1] * t, a[2] * s + b[2] * t, a[3] * s + b[3] * t)


def _round_half_away(x: float) -> float:
    """``std::round``: to the nearest integer, halves away from zero."""
    r = math.floor(abs(x))
    if abs(x) - r >= 0.5:
        r += 1.0
    return math.copysign(r, x)


def _partition_fan(
    tris: list[tuple[int, int, int]], corners: tuple[int, int, int], added: int, offset: int
) -> None:
    """``PartitionFan``: side 0's added vertices fanned from corner 2."""
    last = corners[0]
    for i in range(added):
        nxt = offset + i
        tris.append((last, nxt, corners[2]))
        last = nxt
    tris.append((last, corners[1], corners[2]))


def _partition_quad(
    tris: list[tuple[int, int, int]],
    bary: list[tuple[float, float, float, float]],
    corners: list[int],
    offsets: list[int],
    added: list[int],
    forward: list[bool],
) -> None:
    """``PartitionQuad``: strips parallel to side 0, or a terminal fan where two sides add none."""

    def edge_vert(edge: int, index: int) -> int:
        return offsets[edge] + (index if forward[edge] else -index)

    corner = -1
    last = 3
    max_edge = -1
    for i in range(4):
        if corner == -1 and added[i] == 0 and added[last] == 0:
            corner = i
        if added[i] > 0:
            max_edge = i if max_edge == -1 else -2
        last = i
    if corner >= 0:  # terminate
        if max_edge >= 0:
            edge = [(k + max_edge) % 4 for k in range(4)]
            middle = added[max_edge] // 2
            tris.append((corners[edge[2]], corners[edge[3]], edge_vert(max_edge, middle)))
            prev = corners[edge[0]]
            for i in range(middle + 1):
                nxt = edge_vert(max_edge, i)
                tris.append((corners[edge[3]], prev, nxt))
                prev = nxt
            prev = corners[edge[1]]
            for i in range(added[max_edge] - 1, middle - 1, -1):
                nxt = edge_vert(max_edge, i)
                tris.append((corners[edge[2]], nxt, prev))
                prev = nxt
        else:
            side_vert = corners[0]
            for j in (1, 2):
                side = (corner + j) % 4
                if j == 2 and added[side] > 0:
                    tris.append((corners[side], edge_vert(side, 0), side_vert))
                else:
                    side_vert = corners[side]
                for i in range(added[side]):
                    nxt = edge_vert(side, i)
                    tris.append((corners[corner], side_vert, nxt))
                    side_vert = nxt
                if j == 2 or added[side] == 0:
                    tris.append((corners[corner], side_vert, corners[(corner + j + 1) % 4]))
        return
    partitions = 1 + min(added[1], added[3])
    new_corners = [corners[1], -1, -1, corners[0]]
    new_offsets = [offsets[1], -1, edge_vert(3, added[3] + 1), offsets[0]]
    new_added = [0, -1, 0, added[0]]
    new_forward = [forward[1], True, forward[3], forward[0]]
    for i in range(1, partitions):
        corner_offset1 = (added[1] * i) // partitions
        corner_offset3 = added[3] - 1 - (added[3] * i) // partitions
        next_offset1 = edge_vert(1, corner_offset1 + 1)
        next_offset3 = edge_vert(3, corner_offset3 + 1)
        t = i / partitions
        count = int(_round_half_away(float(added[0]) * (1.0 - t) + float(added[2]) * t))
        new_corners[1] = edge_vert(1, corner_offset1)
        new_corners[2] = edge_vert(3, corner_offset3)
        new_added[0] = abs(next_offset1 - new_offsets[0]) - 1
        new_added[1] = count
        new_added[2] = abs(next_offset3 - new_offsets[2]) - 1
        new_offsets[1] = len(bary)
        new_offsets[2] = next_offset3
        for j in range(count):
            bary.append(
                _lerp4(bary[new_corners[1]], bary[new_corners[2]], (j + 1.0) / (count + 1.0))
            )
        _partition_quad(
            tris, bary, list(new_corners), list(new_offsets), list(new_added), list(new_forward)
        )
        new_corners[0] = new_corners[1]
        new_corners[3] = new_corners[2]
        new_added[3] = new_added[1]
        new_offsets[0] = next_offset1
        new_offsets[3] = new_offsets[1] + new_added[1] - 1
        new_forward[3] = False
    new_corners[1] = corners[2]
    new_corners[2] = corners[3]
    new_offsets[1] = offsets[2]
    new_added[0] = added[1] - abs(new_offsets[0] - offsets[1])
    new_added[1] = added[2]
    new_added[2] = abs(new_offsets[2] - offsets[3]) - 1
    new_offsets[2] = offsets[3]
    new_forward[1] = forward[2]
    _partition_quad(tris, bary, new_corners, new_offsets, new_added, new_forward)


def _closed_positions(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3d], wp.array[wp.int32]]:
    """
    Return a closed mesh's ``float64`` positions and its twin halfedges, in the caller's order.

    Raises
    ------
    ValueError
        If the mesh is not closed, edge-manifold and consistently wound.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    positions = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.widen_positions,
        dim=n_vertices,
        inputs=[vertices],
        outputs=[positions],
        device=device,
    )
    defects = _launch.zeros(3, dtype=wp.int32, device=device)
    twins = halfedge_twins(faces, n_vertices, defect_counts=defects)
    _launch.launch(
        kernel_boolean.count_unpaired,
        dim=faces.size,
        inputs=[twins],
        outputs=[_launch.view(defects, 2, 3)],
        device=device,
    )
    # Host readback: whether the mesh is closed and paired decides whether it has a section.
    non_manifold, inconsistent, boundary = read_values(defects, 0, 3)
    if non_manifold or inconsistent or boundary:
        raise ValueError(
            f"the mesh must be closed, edge-manifold and consistently wound: {boundary} boundary "
            f"halfedges, {non_manifold} edges with three or more faces, {inconsistent} edges "
            "whose faces wind the same way"
        )
    return positions, twins


class _Solid(NamedTuple):
    """
    A closed mesh as Manifold's ``Impl`` carries it between operations.

    ``float64`` positions and Morton-ordered vertices and faces (``SortGeometry``); each face's
    normal (inherited from the face it was cut from, never recomputed), its mesh's id and its
    coplanar group (``triRef``); each vertex's pseudo-normal; and the tolerances.
    """

    positions: wp.array[wp.vec3d]
    faces: wp.array[wp.int32]
    face_normals: wp.array[wp.vec3d]
    vertex_normals: wp.array[wp.vec3d]
    face_mesh: wp.array[wp.int32]
    face_coplanar: wp.array[wp.int32]
    epsilon: float
    tolerance: float
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    # For a solid built from a caller's mesh: each face's index in that mesh.
    face_source: wp.array[wp.int32] | None = None

    @property
    def n_vertices(self) -> int:
        """The vertex count."""
        return int(self.positions.shape[0])

    @property
    def n_faces(self) -> int:
        """The face count."""
        return self.faces.size // 3

    @property
    def is_empty(self) -> bool:
        """Whether the solid has no faces."""
        return self.faces.size == 0

    @property
    def scale(self) -> float:
        """``Box::Scale``: the largest absolute coordinate of the box."""
        if self.is_empty:
            return 0.0
        return max(abs(x) for x in (*self.lower, *self.upper))


class _Operand(NamedTuple):
    """A solid with its halfedge pairing and face tree, ready for the crossing passes."""

    solid: _Solid
    mates: wp.array[wp.int32]
    first_halfedge: wp.array[wp.int32]
    bvh: wp.Bvh

    @property
    def positions(self) -> wp.array[wp.vec3d]:
        """The solid's positions."""
        return self.solid.positions

    @property
    def faces(self) -> wp.array[wp.int32]:
        """The solid's faces."""
        return self.solid.faces

    @property
    def face_normals(self) -> wp.array[wp.vec3d]:
        """The solid's face normals."""
        return self.solid.face_normals

    @property
    def vertex_normals(self) -> wp.array[wp.vec3d]:
        """The solid's vertex normals."""
        return self.solid.vertex_normals

    @property
    def n_vertices(self) -> int:
        """The vertex count."""
        return self.solid.n_vertices

    @property
    def n_faces(self) -> int:
        """The face count."""
        return self.solid.n_faces


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


class _ResultMesh(NamedTuple):
    """The result's ``HalfedgeMesh`` struct and the arrays it bundles, for host-side reads."""

    struct: StructInstance
    start: wp.array[wp.int32]
    pair: wp.array[wp.int32]
    tri_mesh: wp.array[wp.int32]
    tri_coplanar: wp.array[wp.int32]
    tri_normal: wp.array[wp.vec3d]
    pos: wp.array[wp.vec3d]
    sizes: wp.array[wp.int32]


def _boolean(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    operation: _Operation,
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
    p = _solid_from_mesh(vertices_a, faces_a, 0, "a")
    q = _solid_from_mesh(vertices_b, faces_b, 1, "b")
    if operation == "union":
        # ``a + b`` is a two-child ``BatchUnion``: meshes whose boxes are apart are composed.
        return _mesh_from_solid(_batch_union([p, q]), vertices_a.device)
    return _mesh_from_solid(_solid_boolean(p, q, operation), vertices_a.device)


def _solid_boolean(p: _Solid, q: _Solid, operation: _Operation) -> _Solid:
    """``SimpleBoolean``: one Boolean of two solids, as a solid."""
    early = _empty_operand_result(p, q, operation)
    if early is not None:
        return early
    crossings = _boolean3(_operand(p), _operand(q), 1 if operation == "union" else 0)
    return _result(crossings, operation)


def _batch_solids(
    meshes: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[list[_Solid], wp.Device]:
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
        _solid_from_mesh(vertices, faces, index, str(index))
        for index, (vertices, faces) in enumerate(meshes)
    ]
    return solids, device


def _batch_union(children: list[_Solid]) -> _Solid:
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
        children = [*children[:start], _batch_boolean(composed, "union")]
        children[0], children[-1] = children[-1], children[0]
    return children[0]


def _boxes_overlap(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    """``Box::DoesOverlap``: whether two closed boxes share a point."""
    return all(a[0][i] <= b[1][i] and a[1][i] >= b[0][i] for i in range(3))


def _batch_boolean(results: list[_Solid], operation: _Operation) -> _Solid:
    """
    ``BatchBoolean``: combine pairwise, the two with the most vertices first, four pairs a round.

    Manifold's heap pops the largest by ``(vertex count, insertion number)``; each round pairs up
    to four times and returns its results, numbered on, together.
    """
    if len(results) == 1:
        return results[0]
    if len(results) == 2:
        return _solid_boolean(results[0], results[1], operation)
    heap = [(-solid.n_vertices, -serial, serial, solid) for serial, solid in enumerate(results)]
    heapq.heapify(heap)
    next_serial = len(results)
    while len(heap) > 1:
        done: list[tuple[int, int, int, _Solid]] = []
        for _ in range(4):
            if len(heap) <= 1:
                break
            a = heapq.heappop(heap)[3]
            b = heapq.heappop(heap)[3]
            result = _solid_boolean(a, b, operation)
            done.append((-result.n_vertices, -next_serial, next_serial, result))
            next_serial += 1
        for item in done:
            heapq.heappush(heap, item)
    return heap[0][3]


def _compose(solids: list[_Solid]) -> _Solid:
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
    epsilon = max(max(solid.epsilon, kernel_boolean.PRECISION * solid.scale) for solid in solids)
    tolerance = max(solid.tolerance for solid in solids)
    lower = tuple(min(solid.lower[i] for solid in solids) for i in range(3))
    upper = tuple(max(solid.upper[i] for solid in solids) for i in range(3))
    combined = _Solid(
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
    mesh = _halfedge_mesh_from_solid(combined)
    _simplify_topology(mesh, 0, epsilon, tolerance, device, colinear=False)
    # ``Compose`` sorts in the union of its parts' boxes, never recalculated.
    return _finish_solid(mesh, epsilon, tolerance, device, (combined.lower, combined.upper))


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


def _solid_split(p: _Solid, q: _Solid) -> tuple[_Solid, _Solid]:
    """``Manifold::Split``: one ``Boolean3`` of a difference, read as both results."""
    if p.is_empty or q.is_empty:
        inside = _empty_operand_result(p, q, "intersection")
        outside = _empty_operand_result(p, q, "difference")
        assert inside is not None
        assert outside is not None
        return inside, outside
    crossings = _boolean3(_operand(p), _operand(q), 0)
    return _result(crossings, "intersection"), _result(crossings, "difference")


def _solid_simplify(solid: _Solid, tolerance: float) -> _Solid:
    """
    ``Manifold::Simplify``: the whole mesh's ``SimplifyTopology`` at a raised tolerance.

    A tolerance above the solid's regroups its coplanar faces at it first
    (``SetNormalsAndCoplanar`` on the sorted mesh); the result keeps the solid's own tolerance.
    """
    device = solid.positions.device
    old_tolerance = solid.tolerance
    if tolerance == 0.0:
        tolerance = old_tolerance
    if tolerance > old_tolerance:
        solid = _regrouped_solid(solid, tolerance)
    mesh = _halfedge_mesh_from_solid(solid)
    _simplify_topology(mesh, 0, solid.epsilon, max(tolerance, old_tolerance), device)
    # ``Simplify`` sorts in the box the solid had before (it does not ``CalculateBBox``).
    return _finish_solid(mesh, solid.epsilon, old_tolerance, device, (solid.lower, solid.upper))


def _regrouped_solid(solid: _Solid, tolerance: float) -> _Solid:
    """``SetNormalsAndCoplanar`` again, at ``tolerance``, on the solid as it is ordered."""
    device = solid.positions.device
    n_faces = solid.n_faces
    twins = halfedge_twins(solid.faces, solid.n_vertices, validate=False)
    face_normals = _launch.empty(n_faces, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.mesh_face_normals,
        dim=n_faces,
        inputs=[solid.positions, solid.faces],
        outputs=[face_normals],
        device=device,
    )
    coplanar = _coplanar_groups(solid.positions, solid.faces, twins, face_normals, tolerance)
    first_halfedge = _launch.full(solid.n_vertices, 2147483647, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.operand_boxes,
        dim=n_faces,
        inputs=[solid.positions, solid.faces, twins, 0.0],
        outputs=[
            _launch.empty(n_faces, dtype=wp.vec3, device=device),
            _launch.empty(n_faces, dtype=wp.vec3, device=device),
            first_halfedge,
            _launch.zeros(1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    vertex_normals = _launch.empty(solid.n_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.vertex_pseudo_normals,
        dim=solid.n_vertices,
        inputs=[solid.faces, twins, solid.positions, face_normals, first_halfedge],
        outputs=[vertex_normals],
        device=device,
    )
    return solid._replace(
        face_normals=face_normals,
        vertex_normals=vertex_normals,
        face_coplanar=coplanar,
        tolerance=tolerance,
    )


def _halfedge_mesh_from_solid(solid: _Solid) -> _ResultMesh:
    """Return a solid's halfedges as the topology passes edit them."""
    device = solid.positions.device
    n_faces = solid.n_faces
    mesh = _halfedge_mesh(n_faces, solid.n_vertices, solid.positions, device)
    twins = halfedge_twins(solid.faces, solid.n_vertices, validate=False)
    _launch.copy(mesh.start, solid.faces, count=solid.faces.size)
    _launch.copy(mesh.pair, twins, count=solid.faces.size)
    _launch.copy(mesh.tri_normal, solid.face_normals, count=n_faces)
    _launch.copy(mesh.tri_mesh, solid.face_mesh, count=n_faces)
    _launch.copy(mesh.tri_coplanar, solid.face_coplanar, count=n_faces)
    return mesh


def _empty_operand_result(p: _Solid, q: _Solid, operation: _Operation) -> _Solid | None:
    """``Boolean3::Result``'s answer when an operand is empty, else ``None``."""
    if p.is_empty:
        return q if operation == "union" else p
    if q.is_empty:
        return q if operation == "intersection" else p
    return None


def _solid_from_mesh(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], mesh_id: int, name: str
) -> _Solid:
    """
    Build a solid from a ``float32`` mesh, as Manifold's ``MeshGL`` constructor does.

    Raises
    ------
    ValueError
        If the mesh is not closed, edge-manifold and consistently wound.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if faces.size == 0:
        return _empty_solid(device)
    widened = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.widen_positions,
        dim=n_vertices,
        inputs=[vertices],
        outputs=[widened],
        device=device,
    )
    lower, upper = _positions_box(widened)
    scale = max(abs(x) for x in (*lower, *upper))
    epsilon = kernel_boolean.PRECISION * scale
    # ``SetEpsilon(-1, useSingle=true)``: a single-precision mesh's tolerance.
    tolerance = max(epsilon, _FLOAT32_EPSILON * scale)
    return _solid_from_positions(widened, faces, mesh_id, epsilon, tolerance, lower, upper, name)


def _solid_from_positions(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    mesh_id: int,
    epsilon: float,
    tolerance: float,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    name: str,
    *,
    sort_first: bool = False,
) -> _Solid:
    """
    Build a solid from ``float64`` positions: coplanar groups, normals and Morton order.

    ``SetNormalsAndCoplanar`` (the coplanar groups and their normals, then the vertices'
    pseudo-normals) in the given order, then ``SortGeometry``, as the ``MeshGL`` constructor runs
    them; with ``sort_first`` the other way round, as Manifold's primitive shapes do. The coplanar
    ids are triangle indices in the order the groups were formed in.

    Raises
    ------
    ValueError
        If the mesh is not closed, edge-manifold and consistently wound.
    """
    device = positions.device
    n_vertices = int(positions.shape[0])
    n_faces = faces.size // 3
    if sort_first:
        positions, faces, _, _, _, _, _ = _morton_sorted(positions, faces, lower, upper)
    defects = _launch.zeros(3, dtype=wp.int32, device=device)
    twins = halfedge_twins(faces, n_vertices, defect_counts=defects)
    first_halfedge = _launch.full(n_vertices, 2147483647, dtype=wp.int32, device=device)
    lowers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    uppers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_boolean.operand_boxes,
        dim=n_faces,
        inputs=[positions, faces, twins, 0.0],
        outputs=[lowers, uppers, first_halfedge, _launch.view(defects, 2, 3)],
        device=device,
    )
    # Host readback: whether the mesh is closed and paired decides whether there is a Boolean.
    non_manifold, inconsistent, boundary = read_values(defects, 0, 3)
    if non_manifold or inconsistent or boundary:
        raise ValueError(
            f"mesh {name} must be closed, edge-manifold and consistently wound: {boundary} "
            f"boundary halfedges, {non_manifold} edges with three or more faces, {inconsistent} "
            "edges whose faces wind the same way"
        )
    face_normals = _launch.empty(n_faces, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.mesh_face_normals,
        dim=n_faces,
        inputs=[positions, faces],
        outputs=[face_normals],
        device=device,
    )
    coplanar = _coplanar_groups(positions, faces, twins, face_normals, tolerance)
    vertex_normals = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.vertex_pseudo_normals,
        dim=n_vertices,
        inputs=[faces, twins, positions, face_normals, first_halfedge],
        outputs=[vertex_normals],
        device=device,
    )
    face_mesh = _launch.full(n_faces, mesh_id, dtype=wp.int32, device=device)
    if sort_first:
        return _Solid(
            positions,
            faces,
            face_normals,
            vertex_normals,
            face_mesh,
            coplanar,
            epsilon,
            tolerance,
            lower,
            upper,
        )
    return _sorted_solid(
        positions,
        faces,
        vertex_normals,
        face_normals,
        face_mesh,
        coplanar,
        epsilon,
        tolerance,
        lower,
        upper,
    )


def _empty_solid(device: wp.DeviceLike) -> _Solid:
    """Return the empty solid."""
    nothing = (0.0, 0.0, 0.0)
    return _Solid(
        _launch.empty(0, dtype=wp.vec3d, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
        _launch.empty(0, dtype=wp.vec3d, device=device),
        _launch.empty(0, dtype=wp.vec3d, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
        -1.0,
        -1.0,
        nothing,
        nothing,
    )


def _mesh_from_solid(
    solid: _Solid, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return a solid as a ``float32`` mesh, as Manifold's ``MeshGL`` lays it out.

    Vertices in the solid's Morton order; faces in it too, stably grouped by the input mesh each
    came from (``GetMeshGL``'s runs).
    """
    n_vertices = solid.n_vertices
    n_faces = solid.n_faces
    vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    if n_faces == 0:
        return vertices, _launch.empty(0, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.narrow_positions,
        dim=n_vertices,
        inputs=[solid.positions],
        outputs=[vertices],
        device=device,
    )
    keys, order = _launch.sort_buffers(n_faces, wp.int32, device)
    _launch.launch(
        kernel_boolean.face_mesh_keys,
        dim=n_faces,
        inputs=[solid.face_mesh],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_faces)
    faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.gather_faces,
        dim=n_faces,
        inputs=[solid.faces, order],
        outputs=[faces],
        device=device,
    )
    return vertices, faces


def _positions_box(
    positions: wp.array[wp.vec3d],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``CalculateBBox`` over the finite positions, read back."""
    device = positions.device
    inf = float("inf")
    box = _launch.array([inf, inf, inf, -inf, -inf, -inf], dtype=wp.float64, device=device)
    _launch.launch(
        kernel_boolean.positions_box,
        dim=int(positions.shape[0]),
        inputs=[positions],
        outputs=[box],
        device=device,
    )
    # Host readback: the box frames the Morton codes and sets the tolerances.
    values = [float(x) for x in read_values(box, 0, 6)]
    return (values[0], values[1], values[2]), (values[3], values[4], values[5])


def _coplanar_groups(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3d],
    tolerance: float,
) -> wp.array[wp.int32]:
    """
    ``SetNormalsAndCoplanar``: each face's coplanar group, its normal set to its seed's.

    Faces in decreasing area seed groups that spread across edges to neighbours within
    ``tolerance`` of the seed's plane; ``face_normals`` is updated in place.
    """
    device = positions.device
    n_faces = faces.size // 3
    keys, order = _launch.sort_buffers(n_faces, wp.float64, device)
    _launch.launch(
        kernel_boolean.face_area_keys,
        dim=n_faces,
        inputs=[positions, faces],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_faces)
    rank = _launch.empty(n_faces, dtype=wp.int32, device=device)
    seed_offsets, seed_flags = offsets_buffer(n_faces, device)
    _launch.launch(
        kernel_boolean.coplanar_seed_flags,
        dim=n_faces,
        inputs=[faces, twins, positions, face_normals, order, tolerance],
        outputs=[rank, seed_flags],
        device=device,
    )
    is_seed = _launch.clone(seed_flags)
    n_seeds = scan_count(seed_flags)
    coplanar = _launch.full(n_faces, -1, dtype=wp.int32, device=device)
    if n_seeds > 0:
        seeds = _launch.empty(n_seeds, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_boolean.flagged_halfedges,
            dim=n_faces,
            inputs=[is_seed, seed_offsets],
            outputs=[seeds],
            device=device,
        )
        _launch.launch(
            kernel_boolean.flood_coplanar,
            dim=1,
            inputs=[
                faces,
                twins,
                positions,
                seeds,
                n_seeds,
                order,
                rank,
                is_seed,
                tolerance,
                face_normals,
                _launch.empty(2 * n_faces + 4, dtype=wp.int32, device=device),
            ],
            outputs=[coplanar],
            device=device,
        )
    _launch.launch(
        kernel_boolean.own_coplanar_groups, dim=n_faces, inputs=[coplanar], device=device
    )
    return coplanar


def _morton_sorted(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    vertex_normals: wp.array[wp.vec3d] | None = None,
    face_normals: wp.array[wp.vec3d] | None = None,
    face_mesh: wp.array[wp.int32] | None = None,
    face_coplanar: wp.array[wp.int32] | None = None,
) -> tuple[
    wp.array[wp.vec3d],
    wp.array[wp.int32],
    wp.array[wp.vec3d],
    wp.array[wp.vec3d],
    wp.array[wp.int32],
    wp.array[wp.int32],
    wp.array[wp.int32],
]:
    """
    Renumber vertices and faces in Morton order within the box, as ``SortGeometry`` does.

    Returns the sorted arrays, then each sorted face's old index.

    The order decides nothing geometric; it is the order Manifold's ties fall in (which halfedge of
    an edge is forward, which edge collapses first), so following it reproduces its answers in
    the degenerate cases too. The attributes, where given, move with their vertex or face.
    """
    device = positions.device
    n_vertices = int(positions.shape[0])
    n_faces = faces.size // 3
    low = wp.vec3d(*lower)
    high = wp.vec3d(*upper)
    keys, order = _launch.sort_buffers(n_vertices, wp.uint32, device)
    _launch.launch(
        kernel_boolean.vertex_morton_keys,
        dim=n_vertices,
        inputs=[positions, low, high],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_vertices)
    with_normals = vertex_normals is not None
    sorted_positions = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    sorted_normals = _launch.empty(n_vertices if with_normals else 0, dtype=wp.vec3d, device=device)
    old_to_new = _launch.empty(n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.renumber_vertices,
        dim=n_vertices,
        inputs=[
            positions,
            vertex_normals if vertex_normals is not None else positions,
            order,
            1 if with_normals else 0,
        ],
        outputs=[sorted_positions, sorted_normals, old_to_new],
        device=device,
    )
    keys, order = _launch.sort_buffers(n_faces, wp.uint32, device)
    _launch.launch(
        kernel_boolean.face_morton_keys,
        dim=n_faces,
        inputs=[positions, faces, low, high],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_faces)
    with_attributes = face_normals is not None
    n_attributes = n_faces if with_attributes else 0
    sorted_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    sorted_face_normals = _launch.empty(n_attributes, dtype=wp.vec3d, device=device)
    sorted_face_mesh = _launch.empty(n_attributes, dtype=wp.int32, device=device)
    sorted_face_coplanar = _launch.empty(n_attributes, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.renumber_faces,
        dim=n_faces,
        inputs=[
            faces,
            face_normals if face_normals is not None else positions,
            face_mesh if face_mesh is not None else faces,
            face_coplanar if face_coplanar is not None else faces,
            order,
            old_to_new,
            1 if with_attributes else 0,
        ],
        outputs=[sorted_faces, sorted_face_normals, sorted_face_mesh, sorted_face_coplanar],
        device=device,
    )
    return (
        sorted_positions,
        sorted_faces,
        sorted_normals,
        sorted_face_normals,
        sorted_face_mesh,
        sorted_face_coplanar,
        _launch.view(order, 0, n_faces),
    )


def _operand(solid: _Solid) -> _Operand:
    """Pair a solid's halfedges and build its face tree."""
    device = solid.positions.device
    n_faces = solid.n_faces
    twins = halfedge_twins(solid.faces, solid.n_vertices, validate=False)
    first_halfedge = _launch.full(solid.n_vertices, 2147483647, dtype=wp.int32, device=device)
    lowers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    uppers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_boolean.operand_boxes,
        dim=n_faces,
        inputs=[solid.positions, solid.faces, twins, 0.0],
        outputs=[lowers, uppers, first_halfedge, _launch.zeros(1, dtype=wp.int32, device=device)],
        device=device,
    )
    return _Operand(solid, twins, first_halfedge, wp.Bvh(lowers, uppers))


def _boolean3(p: _Operand, q: _Operand, expand: int) -> _Crossings:
    """``Boolean3``: the edge-face crossings both ways and the vertices' winding numbers."""
    device = p.positions.device
    scale = max(p.solid.scale, q.solid.scale)
    # The face trees' query boxes are padded by a few float32 ulps of the scene: a superset of the
    # candidates the exact predicates then decide.
    pad = max(scale, 1.0) * 4.0 * _FLOAT32_EPSILON
    run_offsets_p, n12 = _count_crossings(p, q, pad, expand, forward=True)
    run_offsets_q, n21 = _count_crossings(q, p, pad, expand, forward=False)
    size = max(n12 + n21, 1)
    record_edge = _launch.empty(size, dtype=wp.int32, device=device)
    record_face = _launch.empty(size, dtype=wp.int32, device=device)
    record_crossing = _launch.empty(size, dtype=wp.int32, device=device)
    record_point = _launch.empty(size, dtype=wp.vec3d, device=device)
    records = [record_edge, record_face, record_crossing, record_point]
    _write_crossings(p, q, pad, expand, True, run_offsets_p, 0, records)
    _write_crossings(q, p, pad, expand, False, run_offsets_q, n12, records)
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


def _result(crossings: _Crossings, operation: _Operation) -> _Solid:
    """``Boolean3::Result``: one operation's solid from the crossings and windings."""
    p, q = crossings.p, crossings.q
    device = p.positions.device
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
    n_vp = p.n_vertices
    n_vq = q.n_vertices
    n_items = n_vp + n_vq + n_records
    vertex_offsets, abs_inclusion = offsets_buffer(n_items, device)
    inclusion = _launch.empty(n_items, dtype=wp.int32, device=device)
    for operand, labels, windings, base, c_base in (
        (p, crossings.labels_p, crossings.windings_p, 0, c1),
        (q, crossings.labels_q, crossings.windings_q, n_vp, c2),
    ):
        _launch.launch(
            kernel_boolean.vertex_inclusions,
            dim=operand.n_vertices,
            inputs=[
                labels,
                windings,
                operand.first_halfedge,
                operand.faces.size,
                c_base,
                c3,
                _launch.view(inclusion, base, base + operand.n_vertices),
                _launch.view(abs_inclusion, base, base + operand.n_vertices),
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
        return _empty_solid(device)
    # Host readback: the retained vertices come first, and the simplification moves only those
    # after them, so it needs their count.
    first_new_vert = int(read_values(vertex_offsets, n_vp + n_vq, 1)[0])
    positions = _launch.empty(n_result_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.place_result_vertices,
        dim=n_items,
        inputs=[
            p.positions,
            q.positions,
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
    n_fp = p.n_faces
    n_fq = q.n_faces
    n_faces = n_fp + n_fq
    face_offsets, sides = offsets_buffer(n_faces, device)
    for operand, vertex_base, face_base in ((p, 0, 0), (q, n_vp, n_fp)):
        _launch.launch(
            kernel_boolean.count_face_sides,
            dim=operand.n_faces,
            inputs=[operand.faces, inclusion, vertex_base, face_base],
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
            dim=operand.faces.size,
            inputs=[
                operand.faces,
                operand.mates,
                operand.positions,
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
            dim=operand.faces.size,
            inputs=[
                operand.faces,
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
    _launch.launch(
        kernel_boolean.triangulate_faces,
        dim=n_faces,
        inputs=[
            face_offsets,
            tri_offsets,
            starts,
            ends,
            positions,
            p.face_normals,
            q.face_normals,
            n_fp,
            1 if invert_q else 0,
            epsilon,
            _launch.empty(scratch, dtype=wp.int32, device=device),
            _launch.empty(scratch, dtype=wp.int32, device=device),
            _launch.empty(scratch, dtype=wp.int32, device=device),
            state,
        ],
        outputs=[counts],
        device=device,
    )
    n_triangles = scan_count(counts)
    if n_triangles == 0:
        return _empty_solid(device)
    mesh = _halfedge_mesh(n_triangles, n_result_vertices, positions, device)
    slot_halfedge = _launch.full(max(n_sides, 1), -1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.write_result_halfedges,
        dim=n_faces,
        inputs=[
            tri_offsets,
            count_offsets,
            triangles,
            face_offsets,
            starts,
            ends,
            p.face_normals,
            q.face_normals,
            p.solid.face_coplanar,
            q.solid.face_coplanar,
            p.solid.face_mesh,
            q.solid.face_mesh,
            n_fp,
            1 if invert_q else 0,
            slot_halfedge,
        ],
        outputs=[mesh.start, mesh.pair, mesh.tri_mesh, mesh.tri_coplanar, mesh.tri_normal],
        device=device,
    )
    _launch.launch(
        kernel_boolean.pair_polygon_halfedges,
        dim=n_sides,
        inputs=[slot_halfedge, partner],
        outputs=[mesh.pair],
        device=device,
    )
    _simplify_topology(mesh, first_new_vert, epsilon, tolerance, device)
    return _finish_solid(mesh, epsilon, tolerance, device)


def _count_crossings(
    a: _Operand, b: _Operand, pad: float, expand: int, *, forward: bool
) -> tuple[wp.array[wp.int32], int]:
    """Count each edge of ``a``'s crossings with ``b``'s faces; their offsets and total."""
    offsets, counts = offsets_buffer(a.faces.size, a.positions.device)
    _launch.launch(
        kernel_boolean.count_edge_crossings,
        dim=a.faces.size,
        inputs=[
            a.faces,
            a.positions,
            a.vertex_normals,
            a.face_normals,
            a.mates,
            b.faces,
            b.positions,
            b.vertex_normals,
            b.face_normals,
            b.mates,
            b.bvh.id,
            pad,
            expand,
            1 if forward else 0,
        ],
        outputs=[counts],
        device=a.positions.device,
    )
    return offsets, scan_count(counts)


def _write_crossings(
    a: _Operand,
    b: _Operand,
    pad: float,
    expand: int,
    forward: bool,
    offsets: wp.array[wp.int32],
    record_base: int,
    records: list[odt.ArrayNd],
) -> None:
    """Write the crossings ``_count_crossings`` counted from ``record_base`` on."""
    _launch.launch(
        kernel_boolean.write_edge_crossings,
        dim=a.faces.size,
        inputs=[
            a.faces,
            a.positions,
            a.vertex_normals,
            a.face_normals,
            a.mates,
            b.faces,
            b.positions,
            b.vertex_normals,
            b.face_normals,
            b.mates,
            b.bvh.id,
            pad,
            expand,
            1 if forward else 0,
            offsets,
            record_base,
        ],
        outputs=records,
        device=a.positions.device,
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
    device = a.positions.device
    broken = _launch.zeros(a.faces.size, dtype=wp.int32, device=device)
    if count > 0:
        _launch.launch(
            kernel_boolean.mark_broken_edges,
            dim=count,
            inputs=[record_edge, a.mates, first],
            outputs=[broken],
            device=device,
        )
    edges = odt.empty_2d((a.faces.size, 2), wp.int32, device=device)
    _launch.launch(
        kernel_boolean.unbroken_edge_rows,
        dim=a.faces.size,
        inputs=[a.faces, broken],
        outputs=[edges],
        device=device,
    )
    labels = connected_component_labels_from_edges(edges, a.n_vertices, validate=False)
    windings = _launch.zeros(a.n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.component_windings,
        dim=a.n_vertices,
        inputs=[
            labels,
            a.positions,
            a.vertex_normals,
            b.faces,
            b.positions,
            b.vertex_normals,
            b.face_normals,
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
    state = kernel_polyline.EarClipState()
    state.pos = _launch.empty(capacity, dtype=wp.vec2d, device=device)
    state.right_dir = _launch.empty(capacity, dtype=wp.vec2d, device=device)
    state.left = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.right = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.idx = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.cost = _launch.empty(capacity, dtype=wp.float64, device=device)
    state.seq = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.visit = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.collider = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.stack = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.starts = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.holes = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.hole_box = _launch.empty(capacity, dtype=wp.vec4d, device=device)
    state.outers = _launch.empty(capacity, dtype=wp.int32, device=device)
    state.simples = _launch.empty(capacity, dtype=wp.int32, device=device)
    tris = _launch.empty(capacity, dtype=wp.vec3i, device=device)
    state.tris = tris
    return state, tris


def _halfedge_mesh(
    n_triangles: int, n_vertices: int, positions: wp.array[wp.vec3d], device: wp.DeviceLike
) -> _ResultMesh:
    """Allocate the result's halfedges with room for the simplification to grow them."""
    n_halfedges = 3 * n_triangles
    capacity = 2 * n_halfedges + 64
    start = _launch.empty(capacity, dtype=wp.int32, device=device)
    pair = _launch.empty(capacity, dtype=wp.int32, device=device)
    tri_mesh = _launch.empty(capacity // 3, dtype=wp.int32, device=device)
    tri_coplanar = _launch.empty(capacity // 3, dtype=wp.int32, device=device)
    tri_normal = _launch.empty(capacity // 3, dtype=wp.vec3d, device=device)
    # The simplification appends vertices too: at most two a collapse or split, fewer than the
    # halfedges.
    pos = _launch.empty(n_vertices + capacity, dtype=wp.vec3d, device=device)
    _launch.copy(pos, positions, count=n_vertices)
    sizes = _launch.array(
        [n_halfedges, n_vertices, capacity // 3 * 3, n_vertices + capacity],
        dtype=wp.int32,
        device=device,
    )
    struct = kernel_boolean.HalfedgeMesh()
    struct.start = start
    struct.pair = pair
    struct.tri_mesh = tri_mesh
    struct.tri_coplanar = tri_coplanar
    struct.tri_normal = tri_normal
    struct.pos = pos
    struct.sizes = sizes
    return _ResultMesh(struct, start, pair, tri_mesh, tri_coplanar, tri_normal, pos, sizes)


def _simplify_topology(
    result: _ResultMesh,
    first_new_vert: int,
    epsilon: float,
    tolerance: float,
    device: wp.DeviceLike,
    *,
    colinear: bool = True,
) -> None:
    """
    ``SimplifyTopology(firstNewVert)`` on the result's halfedges, in place.

    Without ``colinear``, ``RemoveDegenerates``: the same less the collinear-edge collapses.
    """
    mesh = result.struct
    capacity = result.start.size
    flags_offsets, flags = offsets_buffer(capacity, device)
    flagged = _launch.empty(capacity, dtype=wp.int32, device=device)
    scratch = _launch.empty(capacity, dtype=wp.int32, device=device)
    collapsed = _launch.zeros(1, dtype=wp.int32, device=device)

    def sizes() -> tuple[int, int]:
        # Host readback: the serial passes grow the mesh, and the next pass's grid follows it.
        n_h, n_v = read_values(result.sizes, 0, 2)
        return int(n_h), int(n_v)

    def flag(kernel: odt.Kernel, inputs: list[object]) -> int:
        n_h, _ = sizes()
        # Every flag kernel writes its whole range, so the buffer is reused unzeroed.
        _launch.launch(kernel, dim=n_h, inputs=[mesh, *inputs], outputs=[flags], device=device)
        count = scan_count(_launch.view(flags, 0, n_h))
        if count > 0:
            _launch.launch(
                kernel_boolean.flagged_halfedges,
                dim=n_h,
                inputs=[flags, _launch.view(flags_offsets, 0, n_h + 1)],
                outputs=[flagged],
                device=device,
            )
        return count

    # ``CleanupTopology``: split pinched vertices, then duplicate edges until none is left.
    n_h, n_v = sizes()
    fan_min = _launch.empty(capacity, dtype=wp.int32, device=device)
    vertex_min = _launch.full(result.pos.shape[0], 2147483647, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.flag_pinched_fans,
        dim=n_h,
        inputs=[mesh],
        outputs=[fan_min, vertex_min],
        device=device,
    )
    _launch.launch(
        kernel_boolean.count_pinched_fans,
        dim=n_h,
        inputs=[mesh, fan_min, vertex_min],
        outputs=[flags],
        device=device,
    )
    n_pinched = scan_count(_launch.view(flags, 0, n_h))
    if n_pinched > 0:
        _launch.launch(
            kernel_boolean.split_pinched_fans,
            dim=n_h,
            inputs=[mesh, _launch.view(flags_offsets, 0, n_h + 1), n_v],
            device=device,
        )
        _launch.copy(
            result.sizes, _launch.array([n_v + n_pinched], dtype=wp.int32, device=device), 1
        )
    while True:
        count = flag(kernel_boolean.flag_duplicate_edges, [])
        if count == 0:
            break
        _launch.launch(
            kernel_boolean.dedupe_flagged, dim=1, inputs=[mesh, flagged, count], device=device
        )
    # ``CollapseShortEdges`` (to ``tolerance`` only past the new vertices: everywhere it is
    # ``epsilon``), ``CollapseColinearEdges``, ``SwapDegenerates``.
    short_tolerance = tolerance if first_new_vert > 0 else epsilon
    count = flag(kernel_boolean.flag_short_edges, [first_new_vert, short_tolerance, epsilon])
    if count > 0:
        _launch.launch(
            kernel_boolean.collapse_flagged,
            dim=1,
            inputs=[mesh, flagged, count, short_tolerance, first_new_vert, epsilon, scratch],
            outputs=[collapsed],
            device=device,
        )
    while colinear:
        count = flag(kernel_boolean.flag_colinear_edges, [first_new_vert])
        if count == 0:
            break
        _launch.launch(
            kernel_boolean.collapse_flagged,
            dim=1,
            inputs=[mesh, flagged, count, epsilon, 0, epsilon, scratch],
            outputs=[collapsed],
            device=device,
        )
        # Host readback: ``CollapseColinearEdges`` repeats until a pass collapses nothing.
        if read_values(collapsed, 0, 1)[0] == 0:
            break
    count = flag(kernel_boolean.flag_swappable_edges, [first_new_vert, tolerance])
    if count > 0:
        _launch.launch(
            kernel_boolean.swap_flagged,
            dim=1,
            inputs=[
                mesh,
                flagged,
                count,
                tolerance,
                epsilon,
                _launch.full(capacity, -1, dtype=wp.int32, device=device),
                _launch.empty(8 * capacity, dtype=wp.int32, device=device),
                scratch,
            ],
            device=device,
        )


def _finish_solid(
    result: _ResultMesh,
    epsilon: float,
    tolerance: float,
    device: wp.DeviceLike,
    box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> _Solid:
    """
    Return the simplified result as a solid.

    ``CalculateVertNormals`` (the end of ``SimplifyTopology``) on the halfedges as they stand,
    ``RemoveUnreferencedVerts`` with the collapsed triangles, then ``SortGeometry`` in the kept
    vertices' box (``CalculateBBox``), or in ``box`` where the caller never recalculates it.
    """
    mesh = result.struct
    # Host readback: the simplification grew the mesh, and the compaction's grid follows it.
    n_h, n_v = (int(x) for x in read_values(result.sizes, 0, 2))
    n_t = n_h // 3
    first = _launch.full(n_v, 2147483647, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.hm_first_halfedges, dim=n_h, inputs=[mesh], outputs=[first], device=device
    )
    vertex_normals = _launch.empty(n_v, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.hm_vertex_normals,
        dim=n_v,
        inputs=[mesh, first],
        outputs=[vertex_normals],
        device=device,
    )
    tri_offsets, tri_kept = offsets_buffer(n_t, device)
    vert_offsets, vert_used = offsets_buffer(n_v, device)
    _launch.launch(
        kernel_boolean.mark_kept,
        dim=n_t,
        inputs=[mesh],
        outputs=[tri_kept, vert_used],
        device=device,
    )
    n_kept = scan_count(tri_kept)
    n_used = scan_count(vert_used)
    if n_kept == 0:
        return _empty_solid(device)
    faces = _launch.empty(3 * n_kept, dtype=wp.int32, device=device)
    face_normals = _launch.empty(n_kept, dtype=wp.vec3d, device=device)
    face_mesh = _launch.empty(n_kept, dtype=wp.int32, device=device)
    face_coplanar = _launch.empty(n_kept, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.write_kept_faces,
        dim=n_t,
        inputs=[mesh, tri_offsets, vert_offsets],
        outputs=[faces, face_normals, face_mesh, face_coplanar],
        device=device,
    )
    positions = _launch.empty(n_used, dtype=wp.vec3d, device=device)
    kept_normals = _launch.empty(n_used, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.write_kept_vertices,
        dim=n_v,
        inputs=[mesh, vertex_normals, vert_offsets],
        outputs=[positions, kept_normals],
        device=device,
    )
    lower, upper = _positions_box(positions) if box is None else box
    return _sorted_solid(
        positions,
        faces,
        kept_normals,
        face_normals,
        face_mesh,
        face_coplanar,
        epsilon,
        tolerance,
        lower,
        upper,
    )


def _sorted_solid(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    vertex_normals: wp.array[wp.vec3d],
    face_normals: wp.array[wp.vec3d],
    face_mesh: wp.array[wp.int32],
    face_coplanar: wp.array[wp.int32],
    epsilon: float,
    tolerance: float,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> _Solid:
    """``SortGeometry``: the solid with its vertices and faces in Morton order, attributes along."""
    (
        sorted_positions,
        sorted_faces,
        sorted_vertex_normals,
        sorted_face_normals,
        sorted_face_mesh,
        sorted_face_coplanar,
        face_source,
    ) = _morton_sorted(
        positions, faces, lower, upper, vertex_normals, face_normals, face_mesh, face_coplanar
    )
    return _Solid(
        sorted_positions,
        sorted_faces,
        sorted_face_normals,
        sorted_vertex_normals,
        sorted_face_mesh,
        sorted_face_coplanar,
        epsilon,
        tolerance,
        lower,
        upper,
        face_source,
    )


# An affine transform as Manifold's ``mat3x4``: four columns of three rows.
_Affine = tuple[tuple[float, float, float], ...]


# -- Manifold's host math (``include/manifold/math.h``), for the half-space's frame ----------------
#
# The cutter's rotation is built from these, in ``float64``, as Manifold builds it; Python's
# ``math`` would round some of them differently.

_PIO2_HI = 1.57079632679489655800e00
_PIO2_LO = 6.12323399573676603587e-17
_HALF_PI = 1.57079632679489661923132169163975144
_PI = 3.1415926535897931160e00
_PI_LO = 1.2246467991473531772e-16
_K_PI = 3.14159265358979323846264338327950288


def _halfspace_solid(
    solid: _Solid, normal: Sequence[float], origin_offset: float, mesh_id: int
) -> _Solid:
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


def _unit_cube_solid(mesh_id: int, device: wp.DeviceLike) -> _Solid:
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
    epsilon = kernel_boolean.PRECISION
    box = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    return _solid_from_positions(
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


def _transformed_solid(solid: _Solid, transform: _Affine) -> _Solid:
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
    epsilon = max(solid.epsilon * spectral, kernel_boolean.PRECISION * scale)
    tolerance = max(solid.tolerance, epsilon)
    return _Solid(
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
