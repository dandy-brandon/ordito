"""
A solid's topology passes: duplicate edges, pinched fans, short and colinear edges, thin faces.

The Manifold library's ``SimplifyTopology`` (``edge_op.cpp``) (Apache-2.0,
https://github.com/elalish/manifold, v3.5.5; Copyright 2021 The Manifold Authors): duplicate edges
are split apart, pinched vertex fans separated, edges shorter than the tolerance or colinear with
their neighbours collapsed and triangles thinner than it swapped across their longest edge; then the
kept vertices and faces are compacted. Each pass flags its edges in parallel and edits them serially
in flag order, as Manifold's ``FlagStore`` runs it (``ordito._solid.simplify_topology``).

The whole module builds unfused (``fuse_fp``), as ``kernels/algorithms/solid`` does.
"""

import warp as wp

from ordito.kernels import polyline as kernel_polyline
from ordito.kernels.algorithms.solid import pseudo_normal, same_face
from ordito.kernels.halfedge import halfedge_next
from ordito.kernels.predicates import project

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


# -- topology simplification (``SimplifyTopology``, ``edge_op.cpp``) -------------------------------


@wp.struct
class HalfedgeMesh:
    """The result as Manifold's ``Halfedges`` with its per-triangle references, grown in place."""

    start: wp.array[wp.int32]
    pair: wp.array[wp.int32]
    tri_mesh: wp.array[wp.int32]
    tri_coplanar: wp.array[wp.int32]
    tri_normal: wp.array[wp.vec3d]
    pos: wp.array[wp.vec3d]
    # [halfedge count, vertex count, halfedge capacity, vertex capacity]
    sizes: wp.array[wp.int32]


@wp.func
def hm_end(m: HalfedgeMesh, h: wp.int32) -> wp.int32:
    return m.start[halfedge_next(h)]


@wp.func
def hm_pair_up(m: HalfedgeMesh, a: wp.int32, b: wp.int32):
    m.pair[a] = b
    m.pair[b] = a


@wp.func
def hm_same_face(m: HalfedgeMesh, t0: wp.int32, t1: wp.int32) -> wp.bool:
    return same_face(m.tri_mesh, m.tri_coplanar, t0, t1)


@wp.func
def hm_nan() -> wp.vec3d:
    nan = wp.float64(0.0) / wp.float64(0.0)
    return wp.vec3d(nan, nan, nan)


@wp.func
def hm_new_vert(m: HalfedgeMesh, copy_of: wp.int32) -> wp.int32:
    v = m.sizes[1]
    if v >= m.sizes[3]:
        return copy_of
    m.sizes[1] = v + 1
    m.pos[v] = m.pos[copy_of]
    return v


@wp.func
def hm_update_vert(m: HalfedgeMesh, vert: wp.int32, start_edge: wp.int32, end_edge: wp.int32):
    # ``UpdateVert``: walk CW from start_edge to end_edge, pointing each edge at ``vert``.
    current = start_edge
    steps = wp.int32(0)
    limit = m.sizes[0]
    while current != end_edge and current >= 0 and steps < limit:
        steps += 1
        m.start[halfedge_next(current)] = vert
        current = halfedge_next(current)
        m.start[current] = vert
        current = m.pair[current]


@wp.func
def hm_remove_if_folded(m: HalfedgeMesh, edge: wp.int32):
    # ``RemoveIfFolded``: a pair of triangles folded onto each other across ``edge`` goes.
    if edge < 0 or m.pair[edge] < 0:
        return
    t0 = wp.vec3i(edge, halfedge_next(edge), halfedge_next(halfedge_next(edge)))
    p = m.pair[edge]
    t1 = wp.vec3i(p, halfedge_next(p), halfedge_next(halfedge_next(p)))
    if m.pair[t0[1]] == -1:
        return
    if m.start[t0[2]] == m.start[t1[2]]:
        if m.pair[t0[1]] == t1[2]:
            if m.pair[t0[2]] == t1[1]:
                for i in range(3):
                    m.pos[m.start[t0[i]]] = hm_nan()
            else:
                m.pos[m.start[t0[1]]] = hm_nan()
        else:
            if m.pair[t0[2]] == t1[1]:
                m.pos[m.start[t1[1]]] = hm_nan()
        hm_pair_up(m, m.pair[t0[1]], m.pair[t1[2]])
        hm_pair_up(m, m.pair[t0[2]], m.pair[t1[1]])
        for i in range(3):
            m.start[t0[i]] = -1
            m.pair[t0[i]] = -1
            m.start[t1[i]] = -1
            m.pair[t1[i]] = -1


@wp.func
def hm_form_loop(m: HalfedgeMesh, current: wp.int32, end: wp.int32):
    # ``FormLoop``: where a collapse would make an edge non-manifold, duplicate its two vertices
    # and attach the sheets the other way across it.
    start_vert = hm_new_vert(m, m.start[current])
    end_vert = hm_new_vert(m, hm_end(m, current))
    old_match = m.pair[current]
    new_match = m.pair[end]
    hm_update_vert(m, start_vert, old_match, new_match)
    hm_update_vert(m, end_vert, end, current)
    hm_pair_up(m, current, new_match)
    hm_pair_up(m, end, old_match)
    hm_remove_if_folded(m, end)


@wp.func
def hm_collapse_tri(m: HalfedgeMesh, tri: wp.vec3i):
    # ``CollapseTri``.
    if m.pair[tri[1]] == -1:
        return
    hm_pair_up(m, m.pair[tri[1]], m.pair[tri[2]])
    for i in range(3):
        m.start[tri[i]] = -1
        m.pair[tri[i]] = -1


@wp.func
def hm_project(m: HalfedgeMesh, tri: wp.int32, p: wp.vec3d) -> wp.vec2d:
    return project(m.tri_normal[tri], p)


@wp.func
def hm_collapse_edge(
    m: HalfedgeMesh,
    edge: wp.int32,
    edges: wp.array[wp.int32],
    tol: wp.float64,
    first_new_vert: wp.int32,
    eps: wp.float64,
) -> wp.bool:
    # ``CollapseEdge``: remove ``edge``'s start vertex into its end, unless that would invert a
    # triangle or move a vertex off a feature; may split the mesh where it would pinch.
    pair = m.pair[edge]
    if edge < 0 or pair < 0:
        return False
    t0 = wp.vec3i(edge, halfedge_next(edge), halfedge_next(halfedge_next(edge)))
    t1 = wp.vec3i(pair, halfedge_next(pair), halfedge_next(halfedge_next(pair)))
    start_vert = m.start[t0[0]]
    end_vert = m.start[t0[1]]
    p_new = m.pos[end_vert]
    p_old = m.pos[start_vert]
    delta = p_new - p_old
    max_len = eps * eps
    if end_vert < first_new_vert:
        max_len = tol * tol
    short_edge = wp.dot(delta, delta) < max_len
    limit = m.sizes[0]
    start = m.pair[t1[1]]
    current = t1[2]
    if not short_edge:
        current = start
        ref_check = pair // 3
        p_last = m.pos[m.start[t1[2]]]
        steps = wp.int32(0)
        while current != t1[0] and steps < limit:
            steps += 1
            current = halfedge_next(current)
            p_next = m.pos[hm_end(m, current)]
            tri = current // 3
            if not hm_same_face(m, tri, ref_check):
                old_ref = ref_check
                ref_check = edge // 3
                if not hm_same_face(m, tri, ref_check):
                    return False
                if m.tri_mesh[tri] != m.tri_mesh[old_ref] or wp.dot(
                    m.tri_normal[pair // 3], m.tri_normal[tri]
                ) < wp.float64(-0.5):
                    if (
                        kernel_polyline.ccw(
                            hm_project(m, tri, p_last),
                            hm_project(m, tri, p_old),
                            hm_project(m, tri, p_new),
                            tol,
                        )
                        != 0
                    ):
                        return False
            if (
                kernel_polyline.ccw(
                    hm_project(m, tri, p_next),
                    hm_project(m, tri, p_last),
                    hm_project(m, tri, p_new),
                    eps,
                )
                < 0
            ):
                return False
            p_last = p_next
            current = m.pair[current]
    # Orbit the end vertex.
    n_edges = wp.int32(0)
    current = m.pair[t0[1]]
    steps = wp.int32(0)
    while current != t1[2] and current >= 0 and steps < limit:
        steps += 1
        current = halfedge_next(current)
        if n_edges < edges.shape[0]:
            edges[n_edges] = current
            n_edges += 1
        current = m.pair[current]
    m.pos[start_vert] = hm_nan()
    hm_collapse_tri(m, t1)
    # Orbit the start vertex.
    current = start
    steps = wp.int32(0)
    while current != t0[2] and current >= 0 and steps < limit:
        steps += 1
        current = halfedge_next(current)
        vert = hm_end(m, current)
        nxt = m.pair[current]
        for i in range(n_edges):
            if vert == hm_end(m, edges[i]):
                hm_form_loop(m, edges[i], current)
                start = nxt
                n_edges = i
                break
        current = nxt
    hm_update_vert(m, end_vert, start, t0[2])
    hm_collapse_tri(m, t0)
    hm_remove_if_folded(m, start)
    return True


@wp.func
def is01_longest(v0: wp.vec2d, v1: wp.vec2d, v2: wp.vec2d) -> wp.bool:
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    l0 = wp.dot(e0, e0)
    return l0 > wp.dot(e1, e1) and l0 > wp.dot(e2, e2)


@wp.func
def hm_dedupe_edge(m: HalfedgeMesh, edge: wp.int32):
    # ``DedupeEdge``: make a 4-manifold edge two edges by duplicating its end vertex, adding two
    # triangles where the duplicates share a sheet, and its start vertex if that pinches it.
    next_edge = halfedge_next(edge)
    start_vert = m.start[edge]
    end_vert = m.start[next_edge]
    current = m.pair[next_edge]
    limit = m.sizes[0]
    steps = wp.int32(0)
    while current != edge and current >= 0 and steps < limit:
        steps += 1
        vert = m.start[current]
        if vert == start_vert:
            # One sheet: two faces split it.
            n_h = m.sizes[0]
            if n_h + 6 > m.sizes[2]:
                return
            new_vert = hm_new_vert(m, end_vert)
            current = m.pair[halfedge_next(current)]
            opposite = m.pair[next_edge]
            hm_update_vert(m, new_vert, current, opposite)
            old_face = current // 3
            outside = m.start[current]
            m.start[n_h] = end_vert
            m.start[n_h + 1] = new_vert
            m.start[n_h + 2] = outside
            m.pair[n_h] = -1
            m.pair[n_h + 1] = -1
            m.pair[n_h + 2] = -1
            hm_pair_up(m, n_h + 2, m.pair[current])
            hm_pair_up(m, n_h + 1, current)
            m.tri_mesh[n_h // 3] = m.tri_mesh[old_face]
            m.tri_coplanar[n_h // 3] = m.tri_coplanar[old_face]
            m.tri_normal[n_h // 3] = m.tri_normal[old_face]
            n2 = n_h + 3
            old_face = opposite // 3
            outside = m.start[opposite]
            m.start[n2] = new_vert
            m.start[n2 + 1] = end_vert
            m.start[n2 + 2] = outside
            m.pair[n2] = -1
            m.pair[n2 + 1] = -1
            m.pair[n2 + 2] = -1
            hm_pair_up(m, n2 + 2, m.pair[opposite])
            hm_pair_up(m, n2 + 1, opposite)
            hm_pair_up(m, n2, n2 - 3)
            m.tri_mesh[n2 // 3] = m.tri_mesh[old_face]
            m.tri_coplanar[n2 // 3] = m.tri_coplanar[old_face]
            m.tri_normal[n2 // 3] = m.tri_normal[old_face]
            m.sizes[0] = n_h + 6
            break
        current = m.pair[halfedge_next(current)]
    if current == edge:
        # Separate sheets need no new faces.
        new_vert = hm_new_vert(m, end_vert)
        first = halfedge_next(current)
        c = first
        steps = wp.int32(0)
        while steps < limit:
            steps += 1
            c = halfedge_next(m.pair[c])
            m.start[c] = new_vert
            m.start[halfedge_next(m.pair[c])] = new_vert
            if c == first:
                break
    # Orbit the start vertex: split it if the split above pinched it.
    pair = m.pair[edge]
    current = m.pair[halfedge_next(pair)]
    steps = wp.int32(0)
    while current != pair and current >= 0 and steps < limit:
        steps += 1
        if m.start[current] == end_vert:
            break
        current = m.pair[halfedge_next(current)]
    if current == pair:
        new_vert = hm_new_vert(m, end_vert)
        first = halfedge_next(current)
        c = first
        steps = wp.int32(0)
        while steps < limit:
            steps += 1
            c = halfedge_next(m.pair[c])
            m.start[c] = new_vert
            m.start[halfedge_next(m.pair[c])] = new_vert
            if c == first:
                break


@wp.func
def hm_swap_edge(m: HalfedgeMesh, t0: wp.vec3i, t1: wp.vec3i) -> wp.bool:
    # ``RecursiveEdgeSwap``'s ``SwapEdge``: turn the shared edge of the two triangles. Returns
    # whether the new edge already existed (the mesh was split there instead).
    v0 = m.start[t0[2]]
    v1 = m.start[t1[2]]
    m.start[t0[0]] = v1
    m.start[halfedge_next(t0[2])] = v1
    m.start[t1[0]] = v0
    m.start[halfedge_next(t1[2])] = v0
    hm_pair_up(m, t0[0], m.pair[t1[2]])
    hm_pair_up(m, t1[0], m.pair[t0[2]])
    hm_pair_up(m, t0[2], t1[2])
    tri0 = t0[0] // 3
    tri1 = t1[0] // 3
    m.tri_normal[tri0] = m.tri_normal[tri1]
    m.tri_mesh[tri0] = m.tri_mesh[tri1]
    m.tri_coplanar[tri0] = m.tri_coplanar[tri1]
    current = m.pair[t1[0]]
    end_vert = hm_end(m, t1[1])
    limit = m.sizes[0]
    steps = wp.int32(0)
    while current != t0[1] and current >= 0 and steps < limit:
        steps += 1
        current = halfedge_next(current)
        if hm_end(m, current) == end_vert:
            hm_form_loop(m, t0[2], current)
            hm_remove_if_folded(m, t0[2])
            return True
        current = m.pair[current]
    return False


@wp.func
def hm_recursive_edge_swap(
    m: HalfedgeMesh,
    edge: wp.int32,
    tag: wp.int32,
    visited: wp.array[wp.int32],
    stack: wp.array[wp.int32],
    top: wp.int32,
    edges: wp.array[wp.int32],
    tolerance: wp.float64,
    eps: wp.float64,
) -> wp.vec2i:
    # ``RecursiveEdgeSwap`` for one edge: swap the long edge of a degenerate triangle, pushing the
    # edges that may have become swappable. Returns (stack top, tag).
    if edge < 0:
        return wp.vec2i(top, tag)
    pair = m.pair[edge]
    if pair < 0:
        return wp.vec2i(top, tag)
    if visited[edge] == tag and visited[pair] == tag:
        return wp.vec2i(top, tag)
    t0 = wp.vec3i(edge, halfedge_next(edge), halfedge_next(halfedge_next(edge)))
    t1 = wp.vec3i(pair, halfedge_next(pair), halfedge_next(halfedge_next(pair)))
    tri = edge // 3
    a0 = hm_project(m, tri, m.pos[m.start[t0[0]]])
    a1 = hm_project(m, tri, m.pos[m.start[t0[1]]])
    a2 = hm_project(m, tri, m.pos[m.start[t0[2]]])
    if kernel_polyline.ccw(a0, a1, a2, tolerance) > 0 or not is01_longest(a0, a1, a2):
        return wp.vec2i(top, tag)
    ptri = pair // 3
    v0 = hm_project(m, ptri, m.pos[m.start[t0[0]]])
    v1 = hm_project(m, ptri, m.pos[m.start[t0[1]]])
    v2 = hm_project(m, ptri, m.pos[m.start[t0[2]]])
    v3 = hm_project(m, ptri, m.pos[m.start[t1[2]]])
    new_tag = tag
    new_top = top
    if kernel_polyline.ccw(v1, v0, v3, tolerance) <= 0:
        if not is01_longest(v1, v0, v3):
            return wp.vec2i(top, tag)
        # Two facing, long-edge degenerates can swap. (A swap that splits the mesh instead
        # returns from ``SwapEdge`` alone; the caller goes on, as here.)
        hm_swap_edge(m, t0, t1)
        e23 = v3 - v2
        if wp.dot(e23, e23) < tolerance * tolerance:
            new_tag += 1
            hm_collapse_edge(m, t0[2], edges, eps, 0, eps)
        else:
            visited[edge] = new_tag
            visited[pair] = new_tag
            if new_top + 4 <= stack.shape[0]:
                stack[new_top] = t1[1]
                stack[new_top + 1] = t1[0]
                stack[new_top + 2] = t0[1]
                stack[new_top + 3] = t0[0]
                new_top += 4
        return wp.vec2i(new_top, new_tag)
    if (
        kernel_polyline.ccw(v0, v3, v2, tolerance) <= 0
        or kernel_polyline.ccw(v1, v2, v3, tolerance) <= 0
    ):
        return wp.vec2i(top, tag)
    hm_swap_edge(m, t0, t1)
    visited[edge] = new_tag
    visited[pair] = new_tag
    if new_top + 2 <= stack.shape[0]:
        stack[new_top] = m.pair[t1[0]]
        stack[new_top + 1] = m.pair[t0[1]]
        new_top += 2
    return wp.vec2i(new_top, new_tag)


@wp.kernel
def flag_pinched_fans(
    m: HalfedgeMesh, out_fan_min: wp.array[wp.int32], out_vertex_min: wp.array[wp.int32]
) -> None:
    # ``SplitPinchedVerts``, parallel: each halfedge finds the least halfedge of its fan around
    # its start vertex; each fan's least halfedge bids for its vertex, the least fan keeping it.
    h = wp.int32(wp.tid())
    least = h
    if m.start[h] < 0:
        out_fan_min[h] = -1
        return
    current = h
    steps = wp.int32(0)
    limit = m.sizes[0]
    while steps < limit:
        steps += 1
        p = m.pair[current]
        if p < 0:
            break
        current = halfedge_next(p)
        least = wp.min(least, current)
        if current == h:
            break
    out_fan_min[h] = least
    if least == h:
        wp.atomic_min(out_vertex_min, m.start[h], h)


@wp.kernel
def count_pinched_fans(
    m: HalfedgeMesh,
    fan_min: wp.array[wp.int32],
    vertex_min: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # A fan that is its vertex's second or later, in index order, takes a vertex of its own.
    h = wp.int32(wp.tid())
    out_flags[h] = wp.where(
        fan_min[h] == h and vertex_min[m.start[h]] != h, wp.int32(1), wp.int32(0)
    )


@wp.kernel
def split_pinched_fans(m: HalfedgeMesh, ranks: wp.array[wp.int32], n_verts: wp.int32) -> None:
    # Each such fan's halfedges move to a fresh copy of the vertex, numbered in fan order.
    h = wp.int32(wp.tid())
    rank = ranks[h]
    if rank == ranks[h + 1]:
        return
    vert = n_verts + rank
    m.pos[vert] = m.pos[m.start[h]]
    current = h
    steps = wp.int32(0)
    limit = m.sizes[0]
    while steps < limit:
        steps += 1
        current = halfedge_next(m.pair[current])
        m.start[current] = vert
        m.start[halfedge_next(m.pair[current])] = vert
        if current == h:
            break


@wp.kernel
def flag_duplicate_edges(m: HalfedgeMesh, out_flags: wp.array[wp.int32]) -> None:
    # ``DedupeEdges``' test: a halfedge whose start and end another, lesser halfedge shares.
    h = wp.int32(wp.tid())
    out_flags[h] = 0
    if m.start[h] < 0:
        return
    end = hm_end(m, h)
    if end < 0:
        return
    current = h
    steps = wp.int32(0)
    limit = m.sizes[0]
    while steps < limit:
        steps += 1
        p = m.pair[current]
        if p < 0:
            break
        current = halfedge_next(p)
        if current == h:
            break
        if current < h and m.start[current] >= 0 and hm_end(m, current) == end:
            out_flags[h] = 1
            break


@wp.kernel
def flag_short_edges(
    m: HalfedgeMesh,
    first_new_vert: wp.int32,
    tol: wp.float64,
    eps: wp.float64,
    out_flags: wp.array[wp.int32],
) -> None:
    # ``CollapseShortEdges``' test.
    h = wp.int32(wp.tid())
    out_flags[h] = 0
    pair = m.pair[h]
    if pair < 0:
        return
    start = m.start[h]
    end = hm_end(m, h)
    if start < first_new_vert and end < first_new_vert:
        return
    delta = m.pos[end] - m.pos[start]
    max_len = eps * eps
    if end < first_new_vert:
        max_len = tol * tol
    if wp.dot(delta, delta) < max_len:
        out_flags[h] = 1


@wp.kernel
def flag_colinear_edges(
    m: HalfedgeMesh, first_new_vert: wp.int32, out_flags: wp.array[wp.int32]
) -> None:
    # ``CollapseColinearEdges``' test: a new vertex surrounded by triangles of at most two
    # original faces.
    edge = wp.int32(wp.tid())
    out_flags[edge] = 0
    pair = m.pair[edge]
    if pair < 0 or m.start[edge] < first_new_vert:
        return
    ref0 = edge // 3
    current = halfedge_next(pair)
    ref1 = current // 3
    ref1_updated = not hm_same_face(m, ref0, ref1)
    steps = wp.int32(0)
    limit = m.sizes[0]
    while current != edge and steps < limit:
        steps += 1
        p = m.pair[current]
        if p < 0:
            return
        current = halfedge_next(p)
        tri = current // 3
        if not hm_same_face(m, tri, ref0) and not hm_same_face(m, tri, ref1):
            if not ref1_updated:
                ref1 = tri
                ref1_updated = True
            else:
                return
    out_flags[edge] = 1


@wp.kernel
def flag_swappable_edges(
    m: HalfedgeMesh, first_new_vert: wp.int32, tolerance: wp.float64, out_flags: wp.array[wp.int32]
) -> None:
    # ``SwapDegenerates``' test: the long edge of a degenerate triangle touching a new vertex.
    edge = wp.int32(wp.tid())
    out_flags[edge] = 0
    pair = m.pair[edge]
    if pair < 0:
        return
    t0 = wp.vec3i(edge, halfedge_next(edge), halfedge_next(halfedge_next(edge)))
    t1 = wp.vec3i(pair, halfedge_next(pair), halfedge_next(halfedge_next(pair)))
    if (
        m.start[t0[0]] < first_new_vert
        and m.start[t0[1]] < first_new_vert
        and m.start[t0[2]] < first_new_vert
        and m.start[t1[2]] < first_new_vert
    ):
        return
    tri = edge // 3
    a0 = hm_project(m, tri, m.pos[m.start[t0[0]]])
    a1 = hm_project(m, tri, m.pos[m.start[t0[1]]])
    a2 = hm_project(m, tri, m.pos[m.start[t0[2]]])
    if kernel_polyline.ccw(a0, a1, a2, tolerance) > 0 or not is01_longest(a0, a1, a2):
        return
    ptri = pair // 3
    b0 = hm_project(m, ptri, m.pos[m.start[t1[0]]])
    b1 = hm_project(m, ptri, m.pos[m.start[t1[1]]])
    b2 = hm_project(m, ptri, m.pos[m.start[t1[2]]])
    if kernel_polyline.ccw(b0, b1, b2, tolerance) > 0 or is01_longest(b0, b1, b2):
        out_flags[edge] = 1


@wp.kernel
def dedupe_flagged(m: HalfedgeMesh, flagged: wp.array[wp.int32], count: wp.int32) -> None:
    # ``DedupeEdges``' serial pass over the flagged edges, in index order.
    for i in range(count):
        hm_dedupe_edge(m, flagged[i])


@wp.kernel
def collapse_flagged(
    m: HalfedgeMesh,
    flagged: wp.array[wp.int32],
    count: wp.int32,
    tol: wp.float64,
    first_new_vert: wp.int32,
    eps: wp.float64,
    edges: wp.array[wp.int32],
    out_collapsed: wp.array[wp.int32],
) -> None:
    # ``CollapseShortEdges`` / ``CollapseColinearEdges``' serial pass, in index order.
    collapsed = wp.int32(0)
    for i in range(count):
        if hm_collapse_edge(m, flagged[i], edges, tol, first_new_vert, eps):
            collapsed += 1
    out_collapsed[0] = collapsed


@wp.kernel
def swap_flagged(
    m: HalfedgeMesh,
    flagged: wp.array[wp.int32],
    count: wp.int32,
    tolerance: wp.float64,
    eps: wp.float64,
    visited: wp.array[wp.int32],
    stack: wp.array[wp.int32],
    edges: wp.array[wp.int32],
) -> None:
    # ``SwapDegenerates``' serial pass, each flagged edge's swaps run to exhaustion off the stack.
    # ``visited`` arrives filled with -1.
    tag = wp.int32(0)
    for i in range(count):
        tag += 1
        state = hm_recursive_edge_swap(m, flagged[i], tag, visited, stack, 0, edges, tolerance, eps)
        top = state[0]
        tag = state[1]
        steps = wp.int32(0)
        while top > 0 and steps < 64 * m.sizes[0] + 64:
            steps += 1
            top -= 1
            state = hm_recursive_edge_swap(
                m, stack[top], tag, visited, stack, top, edges, tolerance, eps
            )
            top = state[0]
            tag = state[1]


# -- the simplified result -------------------------------------------------------------------------


@wp.kernel
def mark_kept(
    m: HalfedgeMesh, out_tri_kept: wp.array[wp.int32], out_vert_used: wp.array[wp.int32]
) -> None:
    # A triangle survives unless collapsed; a vertex survives if a survivor uses it.
    t = wp.int32(wp.tid())
    kept = m.start[3 * t] >= 0 and m.start[3 * t + 1] >= 0 and m.start[3 * t + 2] >= 0
    out_tri_kept[t] = wp.where(kept, wp.int32(1), wp.int32(0))
    if kept:
        for i in range(3):
            out_vert_used[m.start[3 * t + i]] = 1


@wp.kernel
def write_kept_faces(
    m: HalfedgeMesh,
    tri_ranks: wp.array[wp.int32],
    vert_ranks: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_face_normals: wp.array[wp.vec3d],
    out_face_mesh: wp.array[wp.int32],
    out_face_coplanar: wp.array[wp.int32],
) -> None:
    t = wp.int32(wp.tid())
    r = tri_ranks[t]
    if tri_ranks[t + 1] == r:
        return
    for i in range(3):
        out_faces[3 * r + i] = vert_ranks[m.start[3 * t + i]]
    out_face_normals[r] = m.tri_normal[t]
    out_face_mesh[r] = m.tri_mesh[t]
    out_face_coplanar[r] = m.tri_coplanar[t]


@wp.kernel
def write_kept_vertices(
    m: HalfedgeMesh,
    vertex_normals: wp.array[wp.vec3d],
    vert_ranks: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3d],
    out_vertex_normals: wp.array[wp.vec3d],
) -> None:
    v = wp.int32(wp.tid())
    r = vert_ranks[v]
    if vert_ranks[v + 1] != r:
        out_positions[r] = m.pos[v]
        out_vertex_normals[r] = vertex_normals[v]


@wp.kernel
def hm_vertex_normals(
    m: HalfedgeMesh, first: wp.array[wp.int32], out_normals: wp.array[wp.vec3d]
) -> None:
    # Each vertex's pseudo-normal on the simplified result, with the faces' inherited normals.
    v = wp.int32(wp.tid())
    out_normals[v] = pseudo_normal(m.start, m.pair, m.pos, m.tri_normal, first[v], m.sizes[0])
