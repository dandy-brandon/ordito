"""
Kernels for Boolean operations on closed meshes (``ordito.boolean``).

A port of the Boolean of the Manifold library (Apache-2.0, https://github.com/elalish/manifold,
v3.5.5: ``src/boolean3.cpp``, ``src/boolean_result.cpp``, ``src/face_op.cpp``,
``src/polygon.cpp``; Copyright 2021 The Manifold Authors). The notation follows theirs: the
operands are P and Q, the result R; a digit names a dimension (vertex 0, edge 1, face 2,
solid 3), so ``x12`` is the signed crossing of an edge of P with a face of Q and ``w03`` the
winding number of a vertex of P with respect to Q. Every predicate is evaluated in ``float64``
from the ``float32`` input coordinates, which ``float64`` holds exactly, and ties are broken by
Manifold's symbolic perturbation along the vertex and face normals, so every crossing, every
winding number and every polygon the result is cut into is decided consistently with every other.

The whole module builds unfused (``fuse_fp``): the predicates compare interpolated coordinates
for exact equality, and a contraction nvcc chooses by the code around it would decide a tie
differently from the CPU, and from Manifold's own unfused arithmetic.

What differs from the C++, none of it in an answer: the serial ``std::map`` gathers of the result
assembly are a sort of group keys and a thread a group; the ear clipper's ``std::multiset`` of
ears is a scan for the least ``(cost, insertion)`` pair, its 2-d tree a scan of the polygon's
vertices through the same box, and its recursion an explicit stack -- the same choices in the
same order. The serial parts of ``SetNormalsAndCoplanar`` and ``SimplifyTopology`` (``edge_op.cpp``)
run as one thread, over the faces and edges a parallel pass flags, as Manifold's ``FlagStore``
runs them; the coplanar flood visits only the faces that would claim a neighbour, which leaves
every group as the full serial loop forms it. The operands are renumbered in Morton order first
(``SortGeometry``), so every tie Manifold breaks by index falls the same way. Not ported: the
input-side ``CleanupTopology`` / ``RemoveDegenerates`` (a closed, edge-manifold input with no
pinched vertex and no degenerate triangle is what they leave unchanged) and the per-vertex
properties (``CreateProperties``).
"""

import warp as wp

from ordito.kernels import polyline as kernel_polyline
from ordito.kernels.halfedge import halfedge_next, next3
from ordito.kernels.predicates import project, shadow_edge_face, shadow_vertex_face

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

# The collision id of a retained vertex in an edge's sort, after every crossing's.
RETAINED_COLLISION = wp.int32(2147483647)


@wp.struct
class CrossingOperand:
    """One operand of the edge-face crossing kernels: its faces, positions, normals and mates."""

    faces: wp.array[wp.int32]
    positions: wp.array[wp.vec3d]
    vertex_normals: wp.array[wp.vec3d]
    face_normals: wp.array[wp.vec3d]
    mates: wp.array[wp.int32]


# -- edge-face crossings (``Intersect12``) ---------------------------------------------------------


@wp.func
def edge_query_box(
    positions: wp.array[wp.vec3d], start: wp.int32, end: wp.int32, pad: wp.float32
) -> tuple[wp.vec3, wp.vec3]:
    # The edge's box in ``float32``, padded past the rounding of the narrowing.
    a = wp.vec3(positions[start])
    b = wp.vec3(positions[end])
    padding = wp.vec3(pad, pad, pad)
    return wp.min(a, b) - padding, wp.max(a, b) + padding


@wp.kernel
def count_edge_crossings(
    a: CrossingOperand,
    b: CrossingOperand,
    bvh_b: wp.uint64,
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    # How many faces of B each edge of A crosses: a thread a forward halfedge, its box through
    # B's face tree, ``shadow_edge_face`` on each candidate. The box is a superset;
    # ``shadow_edge_face`` decides.
    h = wp.int32(wp.tid())
    start = a.faces[h]
    end = a.faces[halfedge_next(h)]
    count = wp.int32(0)
    if start < end:
        lower, upper = edge_query_box(a.positions, start, end, pad)
        query = wp.bvh_query_aabb(bvh_b, lower, upper)
        face = wp.int32(0)
        while wp.bvh_query_next(query, face):
            x12, _v12 = shadow_edge_face(
                h,
                face,
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
                expand,
                forward,
            )
            if x12 != 0:
                count += 1
    out_counts[h] = count


@wp.kernel
def write_edge_crossings(
    a: CrossingOperand,
    b: CrossingOperand,
    bvh_b: wp.uint64,
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
    offsets: wp.array[wp.int32],
    record_base: wp.int32,
    out_edge: wp.array[wp.int32],
    out_face: wp.array[wp.int32],
    out_crossing: wp.array[wp.int32],
    out_point: wp.array[wp.vec3d],
) -> None:
    # The crossings ``count_edge_crossings`` counted, each edge's run sorted by face as
    # ``Intersect12`` sorts them, from ``record_base + offsets[h]``.
    h = wp.int32(wp.tid())
    first = offsets[h]
    last = offsets[h + 1]
    if first == last:
        return
    start = a.faces[h]
    end = a.faces[halfedge_next(h)]
    lower, upper = edge_query_box(a.positions, start, end, pad)
    query = wp.bvh_query_aabb(bvh_b, lower, upper)
    face = wp.int32(0)
    slot = record_base + first
    stop = record_base + last
    while wp.bvh_query_next(query, face):
        if slot < stop:
            x12, v12 = shadow_edge_face(
                h,
                face,
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
                expand,
                forward,
            )
            if x12 != 0:
                # Insert by face: a run is a handful of faces.
                at = slot
                while at > record_base + first and out_face[at - 1] > face:
                    out_face[at] = out_face[at - 1]
                    out_crossing[at] = out_crossing[at - 1]
                    out_point[at] = out_point[at - 1]
                    at -= 1
                out_edge[at] = h
                out_face[at] = face
                out_crossing[at] = x12
                out_point[at] = v12
                slot += 1
    for r in range(record_base + first, slot):
        out_edge[r] = h


# -- winding numbers (``Winding03``) ---------------------------------------------------------------


@wp.kernel
def mark_broken_edges(
    record_edge: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    first: wp.int32,
    out_broken: wp.array[wp.int32],
) -> None:
    # Both halfedges of every edge a crossing breaks.
    r = wp.int32(wp.tid())
    h = record_edge[first + r]
    out_broken[h] = 1
    out_broken[mates[h]] = 1


@wp.kernel
def unbroken_edge_rows(
    faces: wp.array[wp.int32], broken: wp.array[wp.int32], out_edges: wp.array2d[wp.int32]
) -> None:
    # The union-find edges of ``Winding03``: every unbroken forward halfedge joins its ends; every
    # other halfedge writes a self loop, which joins nothing.
    h = wp.int32(wp.tid())
    start = faces[h]
    end = faces[halfedge_next(h)]
    out_edges[h, 0] = start
    if start < end and broken[h] == 0:
        out_edges[h, 1] = end
    else:
        out_edges[h, 1] = start


@wp.kernel
def component_windings(
    labels: wp.array[wp.int32],
    pos_a: wp.array[wp.vec3d],
    vnormal_a: wp.array[wp.vec3d],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
    bvh_b: wp.uint64,
    z_low: wp.float32,
    z_high: wp.float32,
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
    out_winding: wp.array[wp.int32],
) -> None:
    # The winding number of each component's representative (its least vertex) with respect to
    # B, from the faces of B above or below it in z; every other vertex reads its root's later.
    v = wp.int32(wp.tid())
    if labels[v] != v:
        return
    p = wp.vec3(pos_a[v])
    lower = wp.vec3(p[0] - pad, p[1] - pad, z_low)
    upper = wp.vec3(p[0] + pad, p[1] + pad, z_high)
    query = wp.bvh_query_aabb(bvh_b, lower, upper)
    face = wp.int32(0)
    winding = wp.int32(0)
    while wp.bvh_query_next(query, face):
        s02, _z02, ok = shadow_vertex_face(
            v,
            face,
            pos_a,
            vnormal_a,
            faces_b,
            pos_b,
            vnormal_b,
            fnormal_b,
            mates_b,
            expand,
            forward,
        )
        if ok:
            winding += s02
    out_winding[v] = wp.where(forward != 0, winding, wp.int32(0) - winding)


# -- result sizing (``SizeOutput``) ----------------------------------------------------------------


@wp.kernel
def vertex_inclusions(
    labels: wp.array[wp.int32],
    windings: wp.array[wp.int32],
    first_halfedge: wp.array[wp.int32],
    n_halfedges: wp.int32,
    c_base: wp.int32,
    c_sign: wp.int32,
    out_inclusion: wp.array[wp.int32],
    out_abs: wp.array[wp.int32],
) -> None:
    # Each retained vertex's inclusion ``c_base + c_sign * w03`` (its root's winding, the flood
    # fill), and its absolute value into the vertex count scan. A vertex no face references is
    # left out of the result.
    v = wp.int32(wp.tid())
    inclusion = c_base + c_sign * windings[labels[v]]
    if first_halfedge[v] >= n_halfedges:
        inclusion = wp.int32(0)
    out_inclusion[v] = inclusion
    out_abs[v] = wp.abs(inclusion)


# An operand of few edges crosses the other's faces pair by pair: one thread an (edge, face) pair
# instead of a thread an edge walking the face tree, which leaves a large operand's faces to a
# handful of threads when the few edges are long (a half-space cutter's). Flags scanned in edge-
# major, face-minor order place each edge's run sorted by face, as ``write_edge_crossings``'
# insertion leaves it; ``shadow_edge_face`` alone decides, so the box test only skips work.


@wp.func
def pair_crossing(
    p: wp.int32,
    n_faces_b: wp.int32,
    a: CrossingOperand,
    b: CrossingOperand,
    lowers_b: wp.array[wp.vec3],
    uppers_b: wp.array[wp.vec3],
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
) -> tuple[wp.int32, wp.vec3d]:
    h = p // n_faces_b
    face = p - h * n_faces_b
    start = a.faces[h]
    end = a.faces[halfedge_next(h)]
    if start >= end:
        return 0, wp.vec3d()
    lower, upper = edge_query_box(a.positions, start, end, pad)
    lo = lowers_b[face]
    hi = uppers_b[face]
    if (
        lower[0] > hi[0]
        or lower[1] > hi[1]
        or lower[2] > hi[2]
        or lo[0] > upper[0]
        or lo[1] > upper[1]
        or lo[2] > upper[2]
    ):
        return 0, wp.vec3d()
    return shadow_edge_face(
        h,
        face,
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
        expand,
        forward,
    )


@wp.kernel
def count_pair_crossings(
    n_faces_b: wp.int32,
    a: CrossingOperand,
    b: CrossingOperand,
    lowers_b: wp.array[wp.vec3],
    uppers_b: wp.array[wp.vec3],
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    p = wp.int32(wp.tid())
    x12, _v12 = pair_crossing(p, n_faces_b, a, b, lowers_b, uppers_b, pad, expand, forward)
    out_flags[p] = wp.where(x12 != 0, 1, 0)


@wp.kernel
def edge_offsets_from_pairs(
    pair_offsets: wp.array[wp.int32], n_faces_b: wp.int32, out_offsets: wp.array[wp.int32]
) -> None:
    # Each edge's first record: the scanned pairs at its first face.
    h = wp.int32(wp.tid())
    out_offsets[h] = pair_offsets[h * n_faces_b]


@wp.kernel
def write_pair_crossings(
    n_faces_b: wp.int32,
    a: CrossingOperand,
    b: CrossingOperand,
    lowers_b: wp.array[wp.vec3],
    uppers_b: wp.array[wp.vec3],
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
    pair_offsets: wp.array[wp.int32],
    record_base: wp.int32,
    out_edge: wp.array[wp.int32],
    out_face: wp.array[wp.int32],
    out_crossing: wp.array[wp.int32],
    out_point: wp.array[wp.vec3d],
) -> None:
    p = wp.int32(wp.tid())
    if pair_offsets[p + 1] == pair_offsets[p]:
        return
    x12, v12 = pair_crossing(p, n_faces_b, a, b, lowers_b, uppers_b, pad, expand, forward)
    r = record_base + pair_offsets[p]
    h = p // n_faces_b
    out_edge[r] = h
    out_face[r] = p - h * n_faces_b
    out_crossing[r] = x12
    out_point[r] = v12


@wp.kernel
def crossing_inclusions(
    crossings: wp.array[wp.int32],
    c_sign: wp.int32,
    out_inclusion: wp.array[wp.int32],
    out_abs: wp.array[wp.int32],
) -> None:
    r = wp.int32(wp.tid())
    inclusion = c_sign * crossings[r]
    out_inclusion[r] = inclusion
    out_abs[r] = wp.abs(inclusion)


@wp.kernel
def place_result_vertices(
    pos_p: wp.array[wp.vec3d],
    pos_q: wp.array[wp.vec3d],
    points: wp.array[wp.vec3d],
    inclusion: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    n_p: wp.int32,
    n_q: wp.int32,
    out_positions: wp.array[wp.vec3d],
) -> None:
    # ``DuplicateVerts``: retained vertices of P, of Q, then the crossings, each written
    # ``|inclusion|`` times from its scanned offset.
    i = wp.int32(wp.tid())
    p = wp.vec3d()
    if i < n_p:
        p = pos_p[i]
    elif i < n_p + n_q:
        p = pos_q[i - n_p]
    else:
        p = points[i - n_p - n_q]
    first = vertex_offsets[i]
    for j in range(wp.abs(inclusion[i])):
        out_positions[first + j] = p


@wp.kernel
def count_face_sides(
    faces: wp.array[wp.int32],
    inclusion: wp.array[wp.int32],
    vertex_base: wp.int32,
    face_base: wp.int32,
    out_sides: wp.array[wp.int32],
) -> None:
    # ``CountVerts``: each face gets one side per retained copy of each of its corners.
    f = wp.int32(wp.tid())
    sides = wp.int32(0)
    for i in range(3):
        sides += wp.abs(inclusion[vertex_base + faces[3 * f + i]])
    out_sides[face_base + f] = sides


@wp.kernel
def count_crossing_sides(
    record_edge: wp.array[wp.int32],
    record_face: wp.array[wp.int32],
    inclusion: wp.array[wp.int32],
    mates_edge: wp.array[wp.int32],
    first: wp.int32,
    edge_face_base: wp.int32,
    face_face_base: wp.int32,
    out_sides: wp.array[wp.int32],
) -> None:
    # ``CountNewVerts``: a crossing adds ``|inclusion|`` sides to the face it crosses and to both
    # faces of the edge crossing it.
    r = first + wp.int32(wp.tid())
    n = wp.abs(inclusion[r])
    if n == 0:
        return
    h = record_edge[r]
    wp.atomic_add(out_sides, face_face_base + record_face[r], n)
    wp.atomic_add(out_sides, edge_face_base + h // 3, n)
    wp.atomic_add(out_sides, edge_face_base + mates_edge[h] // 3, n)


# -- polygon edges (``AppendPartialEdges``, ``AppendNewEdges``, ``AppendWholeEdges``) --------------


@wp.func
def emit_edge_pair(
    start: wp.int32,
    end: wp.int32,
    face_left: wp.int32,
    face_right: wp.int32,
    face_cursor: wp.array[wp.int32],
    out_starts: wp.array[wp.int32],
    out_ends: wp.array[wp.int32],
    out_partner: wp.array[wp.int32],
):
    # One polygon edge: start -> end into the left face and end -> start into the right one, each
    # at its face's next free slot, each the other's partner. The order within a face decides
    # nothing: the triangulation assembles its loops by vertex.
    left = wp.atomic_add(face_cursor, face_left, 1)
    out_starts[left] = start
    out_ends[left] = end
    right = wp.atomic_add(face_cursor, face_right, 1)
    out_starts[right] = end
    out_ends[right] = start
    out_partner[left] = right
    out_partner[right] = left


@wp.func
def item_before(
    key_a: wp.float64,
    cid_a: wp.int32,
    item_a: wp.int32,
    key_b: wp.float64,
    cid_b: wp.int32,
    item_b: wp.int32,
) -> wp.bool:
    # ``EdgePos::operator<`` (position, then collision id); the item order breaks what remains,
    # which is only an item against itself.
    if key_a != key_b:
        return key_a < key_b
    if cid_a != cid_b:
        return cid_a < cid_b
    return item_a < item_b


@wp.func
def partial_item(
    item: wp.int32,
    m: wp.int32,
    first: wp.int32,
    record_inclusion: wp.array[wp.int32],
    record_vertex: wp.array[wp.int32],
    vertex_inclusion: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    vertex_base: wp.int32,
    v_start: wp.int32,
    v_end: wp.int32,
) -> wp.vec4i:
    # Item ``item`` of a broken edge as (copies, start flag, first result vertex, collision id):
    # its ``m`` crossings from record ``first`` on, then its retained start and end vertex.
    if item < m:
        inclusion = record_inclusion[first + item]
        return wp.vec4i(
            wp.abs(inclusion),
            wp.where(inclusion < 0, 1, 0),
            record_vertex[first + item],
            first + item,
        )
    v = wp.where(item == m, v_start, v_end)
    inclusion = vertex_inclusion[vertex_base + v]
    is_start = wp.where(item == m, inclusion > 0, inclusion < 0)
    return wp.vec4i(
        wp.abs(inclusion),
        wp.where(is_start, 1, 0),
        vertex_offsets[vertex_base + v],
        RETAINED_COLLISION,
    )


@wp.kernel
def emit_partial_edges(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    pos_in: wp.array[wp.vec3d],
    run_offsets: wp.array[wp.int32],
    record_base: wp.int32,
    record_inclusion: wp.array[wp.int32],
    record_vertex: wp.array[wp.int32],
    vertex_inclusion: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    vertex_base: wp.int32,
    face_base: wp.int32,
    positions: wp.array[wp.vec3d],
    face_cursor: wp.array[wp.int32],
    record_rank: wp.array[wp.int32],
    out_starts: wp.array[wp.int32],
    out_ends: wp.array[wp.int32],
    out_partner: wp.array[wp.int32],
) -> None:
    # ``AppendPartialEdges`` for one broken edge (its forward halfedge): its crossings and its
    # retained ends, each ``|inclusion|`` copies, ordered along the edge, and the k-th start
    # paired with the k-th end (``PairUp``). Each item's rank among its kind is computed once
    # (a crossing's into ``record_rank``), so the pairing is quadratic in the run, not cubic.
    h = wp.int32(wp.tid())
    first = record_base + run_offsets[h]
    m = record_base + run_offsets[h + 1] - first
    if m == 0:
        return
    v_start = faces[h]
    v_end = faces[halfedge_next(h)]
    edge_vec = pos_in[v_end] - pos_in[v_start]
    n_items = m + 2
    rank_start = wp.int32(0)
    rank_end = wp.int32(0)
    for item in range(n_items):
        it = partial_item(
            item,
            m,
            first,
            record_inclusion,
            record_vertex,
            vertex_inclusion,
            vertex_offsets,
            vertex_base,
            v_start,
            v_end,
        )
        if it[0] == 0:
            continue
        key = wp.dot(positions[it[2]], edge_vec)
        rank = wp.int32(0)
        for other in range(n_items):
            ot = partial_item(
                other,
                m,
                first,
                record_inclusion,
                record_vertex,
                vertex_inclusion,
                vertex_offsets,
                vertex_base,
                v_start,
                v_end,
            )
            if ot[0] > 0 and ot[1] == it[1]:
                if item_before(wp.dot(positions[ot[2]], edge_vec), ot[3], other, key, it[3], item):
                    rank += ot[0]
        if item < m:
            record_rank[first + item] = rank
        elif item == m:
            rank_start = rank
        else:
            rank_end = rank
    face_left = face_base + h // 3
    face_right = face_base + mates[h] // 3
    for item in range(n_items):
        it = partial_item(
            item,
            m,
            first,
            record_inclusion,
            record_vertex,
            vertex_inclusion,
            vertex_offsets,
            vertex_base,
            v_start,
            v_end,
        )
        if it[0] == 0 or it[1] == 0:
            continue
        rank = wp.where(item < m, record_rank[first + wp.min(item, m - 1)], rank_start)
        if item == m + 1:
            rank = rank_end
        for j in range(it[0]):
            k = rank + j
            # The end whose copies hold rank k.
            end_vert = wp.int32(-1)
            for other in range(n_items):
                ot = partial_item(
                    other,
                    m,
                    first,
                    record_inclusion,
                    record_vertex,
                    vertex_inclusion,
                    vertex_offsets,
                    vertex_base,
                    v_start,
                    v_end,
                )
                if ot[0] > 0 and ot[1] == 0:
                    o_rank = wp.where(
                        other < m, record_rank[first + wp.min(other, m - 1)], rank_start
                    )
                    if other == m + 1:
                        o_rank = rank_end
                    if o_rank <= k and k < o_rank + ot[0]:
                        end_vert = ot[2] + k - o_rank
            if end_vert >= 0:
                emit_edge_pair(
                    it[2] + j,
                    end_vert,
                    face_left,
                    face_right,
                    face_cursor,
                    out_starts,
                    out_ends,
                    out_partner,
                )


@wp.func
def new_edge_item(
    item: wp.int32,
    n12: wp.int32,
    record_edge: wp.array[wp.int32],
    record_face: wp.array[wp.int32],
    record_inclusion: wp.array[wp.int32],
    mates_p: wp.array[wp.int32],
    mates_q: wp.array[wp.int32],
) -> wp.vec4i:
    # Item ``item`` (``2 * record + side``) of the new edges as (face of P, face of Q, start flag,
    # copies): a crossing lies on the two new edges where its face meets the edge's two faces,
    # keyed by ``AddNewEdgeVerts``' keyRight (side 0) and keyLeft (side 1).
    r = item // 2
    side = item - 2 * r
    inclusion = record_inclusion[r]
    direction = inclusion < 0
    h = record_edge[r]
    f = record_face[r]
    if r < n12:
        edge_face = wp.where(side == 0, mates_p[h] // 3, h // 3)
        is_start = wp.where(side == 0, direction, not direction)
        return wp.vec4i(edge_face, f, wp.where(is_start, 1, 0), wp.abs(inclusion))
    edge_face = wp.where(side == 0, mates_q[h] // 3, h // 3)
    is_start = wp.where(side == 0, not direction, direction)
    return wp.vec4i(f, edge_face, wp.where(is_start, 1, 0), wp.abs(inclusion))


@wp.kernel
def new_edge_keys(
    n12: wp.int32,
    n_faces_q: wp.int32,
    record_edge: wp.array[wp.int32],
    record_face: wp.array[wp.int32],
    record_inclusion: wp.array[wp.int32],
    mates_p: wp.array[wp.int32],
    mates_q: wp.array[wp.int32],
    out_keys: wp.array[wp.int64],
    out_items: wp.array[wp.int32],
) -> None:
    # Each new-edge item's key (face of P, face of Q), for the sort that groups ``edgesNew``.
    item = wp.int32(wp.tid())
    it = new_edge_item(item, n12, record_edge, record_face, record_inclusion, mates_p, mates_q)
    out_keys[item] = wp.int64(it[0]) * wp.int64(n_faces_q) + wp.int64(it[1])
    out_items[item] = item


@wp.kernel
def emit_new_edges(
    n12: wp.int32,
    n_faces_p: wp.int32,
    keys: wp.array[wp.int64],
    items: wp.array[wp.int32],
    record_edge: wp.array[wp.int32],
    record_face: wp.array[wp.int32],
    record_inclusion: wp.array[wp.int32],
    record_vertex: wp.array[wp.int32],
    mates_p: wp.array[wp.int32],
    mates_q: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    face_cursor: wp.array[wp.int32],
    item_rank: wp.array[wp.int32],
    out_starts: wp.array[wp.int32],
    out_ends: wp.array[wp.int32],
    out_partner: wp.array[wp.int32],
) -> None:
    # ``AppendNewEdges`` for one group of sorted items (the thread at its head): its points ordered
    # along the group box's longest axis and paired as ``PairUp`` pairs them, each edge forward
    # into the face of P and backward into the face of Q.
    k0 = wp.int32(wp.tid())
    n = keys.shape[0]
    if k0 > 0 and keys[k0 - 1] == keys[k0]:
        return
    k1 = k0 + 1
    while k1 < n and keys[k1] == keys[k0]:
        k1 += 1
    lower = positions[record_vertex[items[k0] // 2]]
    upper = lower
    for k in range(k0 + 1, k1):
        p = positions[record_vertex[items[k] // 2]]
        lower = wp.min(lower, p)
        upper = wp.max(upper, p)
    size = upper - lower
    axis = wp.int32(2)
    if size[0] > size[1] and size[0] > size[2]:
        axis = wp.int32(0)
    elif size[1] > size[2]:
        axis = wp.int32(1)
    for k in range(k0, k1):
        it = new_edge_item(
            items[k], n12, record_edge, record_face, record_inclusion, mates_p, mates_q
        )
        key = positions[record_vertex[items[k] // 2]][axis]
        rank = wp.int32(0)
        for o in range(k0, k1):
            ot = new_edge_item(
                items[o], n12, record_edge, record_face, record_inclusion, mates_p, mates_q
            )
            if ot[2] == it[2]:
                o_key = positions[record_vertex[items[o] // 2]][axis]
                if item_before(o_key, items[o] // 2, o, key, items[k] // 2, k):
                    rank += ot[3]
        item_rank[k] = rank
    for k in range(k0, k1):
        it = new_edge_item(
            items[k], n12, record_edge, record_face, record_inclusion, mates_p, mates_q
        )
        if it[2] == 0:
            continue
        face_p = it[0]
        vert = record_vertex[items[k] // 2]
        for j in range(it[3]):
            target = item_rank[k] + j
            end_vert = wp.int32(-1)
            for o in range(k0, k1):
                ot = new_edge_item(
                    items[o], n12, record_edge, record_face, record_inclusion, mates_p, mates_q
                )
                if ot[2] == 0 and item_rank[o] <= target and target < item_rank[o] + ot[3]:
                    end_vert = record_vertex[items[o] // 2] + target - item_rank[o]
            if end_vert >= 0:
                emit_edge_pair(
                    vert + j,
                    end_vert,
                    face_p,
                    n_faces_p + it[1],
                    face_cursor,
                    out_starts,
                    out_ends,
                    out_partner,
                )


@wp.kernel
def emit_whole_edges(
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    broken: wp.array[wp.int32],
    vertex_inclusion: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    vertex_base: wp.int32,
    face_base: wp.int32,
    face_cursor: wp.array[wp.int32],
    out_starts: wp.array[wp.int32],
    out_ends: wp.array[wp.int32],
    out_partner: wp.array[wp.int32],
) -> None:
    # ``DuplicateHalfedges``: an unbroken edge whose vertices are retained, once a copy, reversed
    # where the inclusion is negative.
    h = wp.int32(wp.tid())
    if broken[h] != 0:
        return
    start = faces[h]
    end = faces[halfedge_next(h)]
    if start >= end:
        return
    inclusion = vertex_inclusion[vertex_base + start]
    if inclusion == 0:
        return
    a = vertex_offsets[vertex_base + start]
    b = vertex_offsets[vertex_base + end]
    if inclusion < 0:
        t = a
        a = b
        b = t
    for j in range(wp.abs(inclusion)):
        emit_edge_pair(
            a + j,
            b + j,
            face_base + h // 3,
            face_base + mates[h] // 3,
            face_cursor,
            out_starts,
            out_ends,
            out_partner,
        )


# -- triangulation (``Face2Tri``, ``EarClip``) -----------------------------------------------------


@wp.func
def halfedge_before(
    starts: wp.array[wp.int32], ends: wp.array[wp.int32], a: wp.int32, b: wp.int32
) -> wp.bool:
    if starts[a] != starts[b]:
        return starts[a] < starts[b]
    return ends[a] < ends[b]


@wp.func
def assemble_loops(
    starts: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    first: wp.int32,
    n: wp.int32,
    used: wp.array[wp.int32],
    order: wp.array[wp.int32],
    loop_heads: wp.array[wp.int32],
) -> wp.int32:
    # ``AssembleHalfedges``: the face's halfedges into loops, each begun at the remaining halfedge
    # with the least start vertex and closed on returning to it, into ``order[first:]``; loop ``l``
    # begins at ``order[first + loop_heads[first + l]]``. Ties (a vertex two loops touch) go to the
    # least end vertex, so the slot order the emitters raced for decides nothing.
    for i in range(n):
        used[first + i] = 0
    placed = wp.int32(0)
    n_loops = wp.int32(0)
    while placed < n:
        best = wp.int32(-1)
        for i in range(n):
            if used[first + i] == 0:
                if best < 0 or halfedge_before(starts, ends, first + i, first + best):
                    best = i
        loop_heads[first + n_loops] = placed
        n_loops += 1
        loop_vertex = starts[first + best]
        current = best
        while current >= 0:
            used[first + current] = 1
            order[first + placed] = current
            placed += 1
            want = ends[first + current]
            current = wp.int32(-1)
            if want != loop_vertex:
                for i in range(n):
                    if used[first + i] == 0 and starts[first + i] == want:
                        if current < 0 or ends[first + i] < ends[first + current]:
                            current = i
    return n_loops


@wp.kernel
def triangulate_faces(
    face_offsets: wp.array[wp.int32],
    tri_offsets: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    fnormal_p: wp.array[wp.vec3d],
    fnormal_q: wp.array[wp.vec3d],
    n_faces_p: wp.int32,
    invert_q: wp.int32,
    eps: wp.float64,
    heavy_sides: wp.int32,
    heavy_pass: wp.int32,
    used: wp.array[wp.int32],
    order: wp.array[wp.int32],
    loop_heads: wp.array[wp.int32],
    ec: kernel_polyline.EarClipState,
    out_counts: wp.array[wp.int32],
) -> None:
    # ``Face2Tri`` for one face of P or Q: a triangle as it is, a quad by its better diagonal,
    # anything else through the ear clipper, into ``ec.tris[tri_offsets[f]:]`` as result vertices.
    # Faces of ``heavy_sides`` or more sides go in the heavy pass, the rest in the other.
    f = wp.int32(wp.tid())
    first = face_offsets[f]
    n = face_offsets[f + 1] - first
    if (n >= heavy_sides) != (heavy_pass != 0):
        return
    if n == 0:
        out_counts[f] = 0
        return
    normal = wp.vec3d()
    if f < n_faces_p:
        normal = fnormal_p[f]
    else:
        normal = fnormal_q[f - n_faces_p]
        if invert_q != 0:
            normal = wp.vec3d() - normal
    base = tri_offsets[f]
    cap = tri_offsets[f + 1] - base
    n_loops = assemble_loops(starts, ends, first, n, used, order, loop_heads)
    if n == 3 and n_loops == 1:
        ec.tris[base] = wp.vec3i(
            starts[first + order[first]],
            starts[first + order[first + 1]],
            starts[first + order[first + 2]],
        )
        out_counts[f] = 1
        return
    if n == 4 and n_loops == 1:
        q0 = starts[first + order[first]]
        q1 = starts[first + order[first + 1]]
        q2 = starts[first + order[first + 2]]
        q3 = starts[first + order[first + 3]]
        p0 = project(normal, positions[q0])
        p1 = project(normal, positions[q1])
        p2 = project(normal, positions[q2])
        p3 = project(normal, positions[q3])
        choice = wp.int32(0)
        if not (
            kernel_polyline.ccw(p0, p1, p2, eps) >= 0 and kernel_polyline.ccw(p0, p2, p3, eps) >= 0
        ):
            choice = wp.int32(1)
        elif (
            kernel_polyline.ccw(p1, p2, p3, eps) >= 0 and kernel_polyline.ccw(p0, p1, p3, eps) >= 0
        ):
            diag0 = positions[q0] - positions[q2]
            diag1 = positions[q1] - positions[q3]
            if wp.dot(diag0, diag0) > wp.dot(diag1, diag1):
                choice = wp.int32(1)
        if choice == 0:
            ec.tris[base] = wp.vec3i(q0, q1, q2)
            ec.tris[base + 1] = wp.vec3i(q0, q2, q3)
        else:
            ec.tris[base] = wp.vec3i(q1, q2, q3)
            ec.tris[base + 1] = wp.vec3i(q0, q1, q3)
        out_counts[f] = 2
        return
    # ``TriangulateIdxHalfedges``: loops of fewer than three edges have no area and are dropped.
    n_verts = wp.int32(0)
    n_kept = wp.int32(0)
    for loop in range(n_loops):
        head = loop_heads[first + loop]
        tail = n
        if loop + 1 < n_loops:
            tail = loop_heads[first + loop + 1]
        if tail - head < 3:
            continue
        loop_first = base + n_verts
        for i in range(head, tail):
            h = order[first + i]
            v = base + n_verts
            ec.pos[v] = project(normal, positions[starts[first + h]])
            ec.idx[v] = h
            ec.seq[v] = -1
            ec.cost[v] = wp.float64(0.0)
            if i > head:
                kernel_polyline.ec_link(ec, v - 1, v)
            n_verts += 1
        kernel_polyline.ec_link(ec, base + n_verts - 1, loop_first)
        ec.starts[base + n_kept] = loop_first
        n_kept += 1
    count = kernel_polyline.ear_clip(ec, base, cap, n_verts, n_kept, eps)
    for t in range(count):
        tri = ec.tris[base + t]
        ec.tris[base + t] = wp.vec3i(
            starts[first + tri[0]], starts[first + tri[1]], starts[first + tri[2]]
        )
    out_counts[f] = count


@wp.kernel
def triangle_capacities(face_offsets: wp.array[wp.int32], out_capacity: wp.array[wp.int32]) -> None:
    # A face of n sides makes n - 2 triangles as a triangle or quad, and at most 2 n through the
    # ear clipper, whose scratch it also sizes (n vertices, two more a hole).
    f = wp.int32(wp.tid())
    n = face_offsets[f + 1] - face_offsets[f]
    capacity = wp.int32(0)
    if n == 3:
        capacity = wp.int32(1)
    elif n == 4:
        capacity = wp.int32(2)
    elif n > 4:
        capacity = 2 * n
    out_capacity[f] = capacity


@wp.kernel
def offset_indices(
    indices: wp.array[wp.int32], offset: wp.int32, out_indices: wp.array[wp.int32]
) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = indices[i] + offset


# -- the result's halfedges ------------------------------------------------------------------------


@wp.kernel
def count_heavy_faces(
    face_offsets: wp.array[wp.int32], heavy_sides: wp.int32, out_count: wp.array[wp.int32]
) -> None:
    # Faces of ``heavy_sides`` or more polygon sides, whose quadratic passes go to the CPU.
    f = wp.int32(wp.tid())
    if face_offsets[f + 1] - face_offsets[f] >= heavy_sides:
        wp.atomic_add(out_count, 0, 1)


@wp.kernel
def write_result_halfedges(
    tri_offsets: wp.array[wp.int32],
    count_offsets: wp.array[wp.int32],
    tris: wp.array[wp.vec3i],
    face_offsets: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    fnormal_p: wp.array[wp.vec3d],
    fnormal_q: wp.array[wp.vec3d],
    coplanar_p: wp.array[wp.int32],
    coplanar_q: wp.array[wp.int32],
    mesh_p: wp.array[wp.int32],
    mesh_q: wp.array[wp.int32],
    n_faces_p: wp.int32,
    invert_q: wp.int32,
    heavy_sides: wp.int32,
    heavy_pass: wp.int32,
    slot_halfedge: wp.array[wp.int32],
    out_start: wp.array[wp.int32],
    out_pair: wp.array[wp.int32],
    out_tri_mesh: wp.array[wp.int32],
    out_tri_coplanar: wp.array[wp.int32],
    out_tri_normal: wp.array[wp.vec3d],
) -> None:
    # Each face's triangles as halfedges, ``WriteLocalTriangles`` / ``WriteGeneralTriangulation``:
    # an edge on the face's polygon takes that polygon edge's slot (paired across faces by
    # ``pair_polygon_halfedges``), a diagonal pairs with its reverse in the same face; and each
    # triangle's source (``triRef``) and normal.
    f = wp.int32(wp.tid())
    src = tri_offsets[f]
    dst = count_offsets[f]
    n_tris = count_offsets[f + 1] - dst
    if n_tris == 0:
        return
    first = face_offsets[f]
    n = face_offsets[f + 1] - first
    if (n >= heavy_sides) != (heavy_pass != 0):
        return
    mesh = wp.int32(0)
    coplanar = wp.int32(0)
    normal = wp.vec3d()
    if f < n_faces_p:
        mesh = mesh_p[f]
        coplanar = coplanar_p[f]
        normal = fnormal_p[f]
    else:
        mesh = mesh_q[f - n_faces_p]
        coplanar = coplanar_q[f - n_faces_p]
        normal = fnormal_q[f - n_faces_p]
        if invert_q != 0:
            normal = wp.vec3d() - normal
    for t in range(n_tris):
        tri = tris[src + t]
        # ``ReorderHalfedges``: each triangle starts at its least vertex.
        if tri[1] < tri[0] and tri[1] < tri[2]:
            tri = wp.vec3i(tri[1], tri[2], tri[0])
        elif tri[2] < tri[0] and tri[2] < tri[1]:
            tri = wp.vec3i(tri[2], tri[0], tri[1])
        out_tri_mesh[dst + t] = mesh
        out_tri_coplanar[dst + t] = coplanar
        out_tri_normal[dst + t] = normal
        for i in range(3):
            h = 3 * (dst + t) + i
            out_start[h] = tri[i]
            out_pair[h] = -1
            u = tri[i]
            v = tri[next3(i)]
            for k in range(n):
                if slot_halfedge[first + k] < 0 and starts[first + k] == u and ends[first + k] == v:
                    slot_halfedge[first + k] = h
                    out_pair[h] = -2
                    break
    # Diagonals: pair each with its reverse among the face's other unpaired halfedges.
    h0 = 3 * dst
    h1 = 3 * (dst + n_tris)
    for h in range(h0, h1):
        if out_pair[h] != -1:
            continue
        u = out_start[h]
        v = out_start[halfedge_next(h)]
        for g in range(h0, h1):
            if (
                g != h
                and out_pair[g] == -1
                and out_start[g] == v
                and out_start[halfedge_next(g)] == u
            ):
                out_pair[h] = g
                out_pair[g] = h
                break


@wp.kernel
def pair_polygon_halfedges(
    slot_halfedge: wp.array[wp.int32], partner: wp.array[wp.int32], out_pair: wp.array[wp.int32]
) -> None:
    # A polygon edge's triangle halfedge pairs with its partner slot's.
    s = wp.int32(wp.tid())
    h = slot_halfedge[s]
    if h < 0:
        return
    other = slot_halfedge[partner[s]]
    out_pair[h] = other
