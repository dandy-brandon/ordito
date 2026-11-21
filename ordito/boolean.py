"""
Boolean operations on closed meshes: union, intersection and difference.

[`union`][ordito.boolean.union], [`intersection`][ordito.boolean.intersection] and
[`difference`][ordito.boolean.difference] combine two closed, consistently wound meshes as solids
and return a closed, consistently wound mesh. Mirrors ``trimesh.boolean``'s three operations, and
gives the answers of its default engine, the Manifold library, whose Boolean this module ports.

Every crossing of an edge of one mesh with a face of the other, and every vertex's winding number
with respect to the other mesh, is decided exactly from the ``float32`` coordinates. Ties --
surfaces that touch, overlap in a plane or coincide -- are broken by one consistent symbolic
perturbation, so the result is closed and manifold whatever the inputs' relative position, and
the slivers such contacts leave are collapsed afterwards.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import warp as wp
from warp._src.codegen import StructInstance

import ordito.typing as odt
from ordito import _launch
from ordito._common import offsets_buffer, read_values, require_same_device, scan_count
from ordito.bounds import aabb
from ordito.graph import connected_component_labels_from_edges
from ordito.halfedge import halfedge_twins
from ordito.kernels import boolean as kernel_boolean

_Operation = Literal["union", "intersection", "difference"]

_FLOAT32_EPSILON = 1.1920929e-07


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


class _Operand(NamedTuple):
    """One operand's ``float64`` geometry, halfedge pairing and face tree."""

    vertices: wp.array[wp.vec3]
    faces: wp.array[wp.int32]
    positions: wp.array[wp.vec3d]
    face_normals: wp.array[wp.vec3d]
    vertex_normals: wp.array[wp.vec3d]
    mates: wp.array[wp.int32]
    first_halfedge: wp.array[wp.int32]
    coplanar: wp.array[wp.int32]
    bvh: wp.Bvh
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    @property
    def n_vertices(self) -> int:
        """The vertex count."""
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        """The face count."""
        return self.faces.size // 3

    def predicate_inputs(self) -> list[object]:
        """Return the arrays every predicate reads of this mesh as an edge mesh."""
        return [self.faces, self.positions, self.vertex_normals, self.face_normals, self.mates]


def _boolean(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    operation: _Operation,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Run the Boolean ``operation`` on two closed meshes (``Boolean3`` and its ``Result``).

    Raises
    ------
    ValueError
        If either mesh is not closed, edge-manifold and consistently wound.
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    device = vertices_a.device
    empty_a = faces_a.size == 0
    empty_b = faces_b.size == 0
    if empty_a or empty_b:
        if empty_a and not empty_b and operation == "union":
            return _launch.clone(vertices_b), _launch.clone(faces_b)
        if empty_b and not empty_a and operation != "intersection":
            return _launch.clone(vertices_a), _launch.clone(faces_a)
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.empty(
            0, dtype=wp.int32, device=device
        )
    p = _prepare_operand(vertices_a, faces_a, "a")
    q = _prepare_operand(vertices_b, faces_b, "b")

    # The symbolic perturbation: a union expands both operands, the other operations contract P.
    expand = 1 if operation == "union" else 0
    c1 = 0 if operation == "intersection" else 1
    c2 = 1 if operation == "union" else 0
    c3 = 1 if operation == "intersection" else -1
    invert_q = operation == "difference"
    scale = max(abs(x) for x in (*p.lower, *p.upper, *q.lower, *q.upper))
    # The face trees' query boxes are padded by a few float32 ulps of the scene: a superset of the
    # candidates the exact predicates then decide.
    pad = max(scale, 1.0) * 4.0 * _FLOAT32_EPSILON
    epsilon = kernel_boolean.PRECISION * scale

    # Level 3: the edge-face crossings, P's edges through Q's faces and Q's through P's.
    run_offsets_p, n12 = _count_crossings(p, q, pad, expand, forward=True)
    run_offsets_q, n21 = _count_crossings(q, p, pad, expand, forward=False)
    n_records = n12 + n21
    size = max(n_records, 1)
    record_edge = _launch.empty(size, dtype=wp.int32, device=device)
    record_face = _launch.empty(size, dtype=wp.int32, device=device)
    record_crossing = _launch.empty(size, dtype=wp.int32, device=device)
    record_point = _launch.empty(size, dtype=wp.vec3d, device=device)
    records = [record_edge, record_face, record_crossing, record_point]
    _write_crossings(p, q, pad, expand, True, run_offsets_p, 0, records)
    _write_crossings(q, p, pad, expand, False, run_offsets_q, n12, records)

    # The winding numbers, a vertex's component's from its least vertex.
    broken_p, labels_p, windings_p = _windings(p, q, record_edge, 0, n12, pad, expand, True)
    broken_q, labels_q, windings_q = _windings(q, p, record_edge, n12, n21, pad, expand, False)

    # Level 4: the inclusions and the result's vertices.
    n_vp = p.n_vertices
    n_vq = q.n_vertices
    n_items = n_vp + n_vq + n_records
    vertex_offsets, abs_inclusion = offsets_buffer(n_items, device)
    inclusion = _launch.empty(n_items, dtype=wp.int32, device=device)
    for operand, labels, windings, base, c_base in (
        (p, labels_p, windings_p, 0, c1),
        (q, labels_q, windings_q, n_vp, c2),
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
                record_crossing,
                c3,
                _launch.view(inclusion, n_vp + n_vq, n_items),
                _launch.view(abs_inclusion, n_vp + n_vq, n_items),
            ],
            device=device,
        )
    n_result_vertices = scan_count(abs_inclusion)
    if n_result_vertices == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.empty(
            0, dtype=wp.int32, device=device
        )
    # Host readback: the retained vertices come first, and the simplification moves only those
    # after them, so it needs their count.
    first_new_vert = int(read_values(vertex_offsets, n_vp + n_vq, 1)[0])
    positions = _launch.empty(n_result_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.place_result_vertices,
        dim=n_items,
        inputs=[p.positions, q.positions, record_point, inclusion, vertex_offsets, n_vp, n_vq],
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
        (p, run_offsets_p, 0, 0, 0),
        (q, run_offsets_q, n12, n_vp, n_fp),
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
    for operand, broken, vertex_base, face_base in ((p, broken_p, 0, 0), (q, broken_q, n_vp, n_fp)):
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
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.empty(
            0, dtype=wp.int32, device=device
        )
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
            p.coplanar,
            q.coplanar,
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
    tolerance = max(epsilon, _FLOAT32_EPSILON * scale)
    return _simplify_topology(mesh, first_new_vert, epsilon, tolerance, device)


def _prepare_operand(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], name: str) -> _Operand:
    """
    Widen one operand to ``float64``, pair its halfedges and build its face tree.

    Raises
    ------
    ValueError
        If the mesh is not closed, edge-manifold and consistently wound.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = faces.size // 3
    vertices, faces, lower, upper = _morton_sorted(vertices, faces)
    defects = _launch.zeros(3, dtype=wp.int32, device=device)
    twins = halfedge_twins(faces, n_vertices, defect_counts=defects)
    positions = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    face_normals = _launch.empty(n_faces, dtype=wp.vec3d, device=device)
    lowers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    uppers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    first_halfedge = _launch.full(n_vertices, 2147483647, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.face_normals_and_boxes,
        dim=n_faces,
        inputs=[vertices, faces, twins],
        outputs=[
            positions,
            face_normals,
            lowers,
            uppers,
            first_halfedge,
            _launch.view(defects, 2, 3),
        ],
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
    scale = max(abs(x) for x in (*lower, *upper))
    # ``SetNormalsAndCoplanar``: coplanar neighbours share their largest triangle's normal, within
    # a single-precision mesh's tolerance (``SetEpsilon(-1, useSingle=true)``).
    tolerance = max(kernel_boolean.PRECISION, _FLOAT32_EPSILON) * scale
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
    vertex_normals = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_boolean.vertex_pseudo_normals,
        dim=n_vertices,
        inputs=[faces, twins, positions, face_normals, first_halfedge],
        outputs=[vertex_normals],
        device=device,
    )
    bvh = wp.Bvh(lowers, uppers)
    return _Operand(
        vertices,
        faces,
        positions,
        face_normals,
        vertex_normals,
        twins,
        first_halfedge,
        coplanar,
        bvh,
        lower,
        upper,
    )


def _morton_sorted(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[
    wp.array[wp.vec3], wp.array[wp.int32], tuple[float, float, float], tuple[float, float, float]
]:
    """
    Renumber a mesh's vertices and faces in Morton order, as Manifold's ``SortGeometry`` does.

    The order decides nothing geometric; it is the order Manifold's ties fall in (which halfedge of
    an edge is forward, which edge collapses first), so following it reproduces its answers in
    the degenerate cases too.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = faces.size // 3
    box_low, box_high = aabb(vertices)
    lower = (float(box_low[0]), float(box_low[1]), float(box_low[2]))
    upper = (float(box_high[0]), float(box_high[1]), float(box_high[2]))
    low = wp.vec3d(*lower)
    high = wp.vec3d(*upper)
    keys, order = _launch.sort_buffers(n_vertices, wp.uint32, device)
    _launch.launch(
        kernel_boolean.vertex_morton_keys,
        dim=n_vertices,
        inputs=[vertices, low, high],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_vertices)
    sorted_vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    old_to_new = _launch.empty(n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.renumber_vertices,
        dim=n_vertices,
        inputs=[vertices, order],
        outputs=[sorted_vertices, old_to_new],
        device=device,
    )
    keys, order = _launch.sort_buffers(n_faces, wp.uint32, device)
    _launch.launch(
        kernel_boolean.face_morton_keys,
        dim=n_faces,
        inputs=[vertices, faces, low, high],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_faces)
    sorted_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_boolean.renumber_faces,
        dim=n_faces,
        inputs=[faces, order, old_to_new],
        outputs=[sorted_faces],
        device=device,
    )
    return sorted_vertices, sorted_faces, lower, upper


def _count_crossings(
    a: _Operand, b: _Operand, pad: float, expand: int, *, forward: bool
) -> tuple[wp.array[wp.int32], int]:
    """Count each edge of ``a``'s crossings with ``b``'s faces; their offsets and total."""
    offsets, counts = offsets_buffer(a.faces.size, a.vertices.device)
    _launch.launch(
        kernel_boolean.count_edge_crossings,
        dim=a.faces.size,
        inputs=[
            a.vertices,
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
        device=a.vertices.device,
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
            a.vertices,
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
        device=a.vertices.device,
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
    device = a.vertices.device
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
            a.vertices,
            a.positions,
            a.vertex_normals,
            b.faces,
            b.positions,
            b.vertex_normals,
            b.face_normals,
            b.mates,
            b.bvh.id,
            b.lower[2] - 1.0,
            b.upper[2] + 1.0,
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
    state = kernel_boolean.EarClipState()
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
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """``SimplifyTopology(firstNewVert)`` and ``RemoveUnreferencedVerts``, then the flat mesh."""
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
    # ``CollapseShortEdges``, ``CollapseColinearEdges``, ``SwapDegenerates``.
    count = flag(kernel_boolean.flag_short_edges, [first_new_vert, tolerance, epsilon])
    if count > 0:
        _launch.launch(
            kernel_boolean.collapse_flagged,
            dim=1,
            inputs=[mesh, flagged, count, tolerance, first_new_vert, epsilon, scratch],
            outputs=[collapsed],
            device=device,
        )
    while True:
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
    # ``RemoveUnreferencedVerts`` and the collapsed triangles.
    n_h, n_v = sizes()
    n_t = n_h // 3
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
    faces = _launch.empty(3 * n_kept, dtype=wp.int32, device=device)
    vertices = _launch.empty(n_used, dtype=wp.vec3, device=device)
    if n_kept > 0:
        _launch.launch(
            kernel_boolean.write_kept_faces,
            dim=n_t,
            inputs=[mesh, tri_offsets, vert_offsets],
            outputs=[faces],
            device=device,
        )
        _launch.launch(
            kernel_boolean.write_kept_vertices,
            dim=n_v,
            inputs=[mesh, vert_offsets],
            outputs=[vertices],
            device=device,
        )
    return vertices, faces
