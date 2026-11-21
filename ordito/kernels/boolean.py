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

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

# Manifold's ``kPrecision``: the triangulator's epsilon is this times the inputs' largest
# absolute coordinate.
PRECISION = 1e-12

# The collision id of a retained vertex in an edge's sort, after every crossing's.
RETAINED_COLLISION = wp.int32(2147483647)

# -- exact predicates (``boolean3.cpp``, ``shared.h``) ---------------------------------------------


@wp.func
def next3(i: wp.int32) -> wp.int32:
    return wp.where(i == 2, wp.int32(0), i + 1)


@wp.func
def next_halfedge(h: wp.int32) -> wp.int32:
    return wp.where(h % 3 == 2, h - 2, h + 1)


@wp.func
def shadows(p: wp.float64, q: wp.float64, direction: wp.float64) -> wp.bool:
    # ``Shadows``: p < q, a tie decided by the sign of the perturbation's direction.
    if p == q:
        return direction < wp.float64(0.0)
    return p < q


@wp.func
def with_sign(positive: wp.int32, v: wp.float64) -> wp.float64:
    return wp.where(positive != 0, v, wp.float64(0.0) - v)


@wp.func
def shadow_step(flag: wp.bool) -> wp.int32:
    return wp.where(flag, wp.int32(1), wp.int32(0))


@wp.func
def interpolate(a_left: wp.vec3d, a_right: wp.vec3d, x: wp.float64) -> wp.vec2d:
    # ``Interpolate``: the (y, z) of segment a_left-a_right at x, from the nearer end.
    dx_left = x - a_left[0]
    dx_right = x - a_right[0]
    use_left = wp.abs(dx_left) < wp.abs(dx_right)
    d = a_right - a_left
    lam = wp.where(use_left, dx_left, dx_right) / d[0]
    if not (wp.isfinite(lam) and wp.isfinite(d[1]) and wp.isfinite(d[2])):
        return wp.vec2d(a_left[1], a_left[2])
    y = lam * d[1] + wp.where(use_left, a_left[1], a_right[1])
    z = lam * d[2] + wp.where(use_left, a_left[2], a_right[2])
    return wp.vec2d(y, z)


@wp.func
def intersect(a_left: wp.vec3d, a_right: wp.vec3d, b_left: wp.vec3d, b_right: wp.vec3d) -> wp.vec4d:
    # ``Intersect``: where segments a and b, given over the same two x, cross in y; their z there.
    dy_left = b_left[1] - a_left[1]
    dy_right = b_right[1] - a_right[1]
    use_left = wp.abs(dy_left) < wp.abs(dy_right)
    dx = a_right[0] - a_left[0]
    lam = wp.where(use_left, dy_left, dy_right) / (dy_left - dy_right)
    if not wp.isfinite(lam):
        lam = wp.float64(0.0)
    x = lam * dx + wp.where(use_left, a_left[0], a_right[0])
    a_dy = a_right[1] - a_left[1]
    b_dy = b_right[1] - b_left[1]
    use_a = wp.abs(a_dy) < wp.abs(b_dy)
    y0 = wp.where(use_a, a_left[1], b_left[1])
    y1 = wp.where(use_a, a_right[1], b_right[1])
    y = lam * wp.where(use_a, a_dy, b_dy) + wp.where(use_left, y0, y1)
    z = lam * (a_right[2] - a_left[2]) + wp.where(use_left, a_left[2], a_right[2])
    w = lam * (b_right[2] - b_left[2]) + wp.where(use_left, b_left[2], b_right[2])
    return wp.vec4d(x, y, z, w)


@wp.func
def face_edge(
    faces: wp.array[wp.int32], mates: wp.array[wp.int32], tri: wp.int32, i: wp.int32
) -> wp.vec4i:
    # ``LoadFaceEdges``: edge ``i`` of ``tri`` as (forward halfedge, lower vertex, upper vertex,
    # whether the face's own halfedge is the forward one).
    h = 3 * tri + i
    start = faces[h]
    end = faces[3 * tri + next3(i)]
    if start < end:
        return wp.vec4i(h, start, end, 1)
    return wp.vec4i(mates[h], end, start, 0)


@wp.func
def shadow01(
    a0: wp.int32,
    b1: wp.int32,
    b1s: wp.int32,
    b1e: wp.int32,
    pos_a: wp.array[wp.vec3d],
    vnormal_a: wp.array[wp.vec3d],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
    expand: wp.int32,
    forward: wp.int32,
) -> tuple[wp.int32, wp.vec2d, wp.bool]:
    # ``Shadow01``: whether vertex a0 shadows edge b1 (in x, then y), and the edge's (y, z) at the
    # vertex's x. The flag is Manifold's "yz01 is finite": the vertex's x lies within the edge's.
    a0x = pos_a[a0][0]
    b1sx = pos_b[b1s][0]
    b1ex = pos_b[b1e][0]
    a0xp = vnormal_a[a0][0]
    b1sxp = vnormal_b[b1s][0]
    b1exp = vnormal_b[b1e][0]
    s01 = wp.int32(0)
    if forward != 0:
        s01 = shadow_step(shadows(a0x, b1ex, with_sign(expand, a0xp) - b1exp)) - shadow_step(
            shadows(a0x, b1sx, with_sign(expand, a0xp) - b1sxp)
        )
    else:
        s01 = shadow_step(shadows(b1sx, a0x, with_sign(expand, b1sxp) - a0xp)) - shadow_step(
            shadows(b1ex, a0x, with_sign(expand, b1exp) - a0xp)
        )
    yz = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    valid = wp.bool(False)
    if s01 != 0:
        valid = True
        yz = interpolate(pos_b[b1s], pos_b[b1e], a0x)
        direction = fnormal_b[b1 // 3][1] + fnormal_b[mates_b[b1] // 3][1]
        if forward != 0:
            if not shadows(pos_a[a0][1], yz[0], wp.float64(0.0) - direction):
                s01 = wp.int32(0)
        else:
            if not shadows(yz[0], pos_a[a0][1], with_sign(expand, direction)):
                s01 = wp.int32(0)
    return s01, yz, valid


@wp.func
def kernel11(
    p1: wp.int32,
    p1s: wp.int32,
    p1e: wp.int32,
    q1: wp.int32,
    q1s: wp.int32,
    q1e: wp.int32,
    pos_p: wp.array[wp.vec3d],
    vnormal_p: wp.array[wp.vec3d],
    fnormal_p: wp.array[wp.vec3d],
    mates_p: wp.array[wp.int32],
    pos_q: wp.array[wp.vec3d],
    vnormal_q: wp.array[wp.vec3d],
    fnormal_q: wp.array[wp.vec3d],
    mates_q: wp.array[wp.int32],
    expand: wp.int32,
) -> tuple[wp.int32, wp.vec4d, wp.bool]:
    # ``Kernel11``: whether edge p1 of P passes above or below edge q1 of Q where they cross in the
    # xy projection, and the crossing (x, y, z of p1, z of q1).
    k = wp.int32(0)
    s11 = wp.int32(0)
    flag = wp.bool(False)
    p_left = wp.vec3d()
    p_right = wp.vec3d()
    q_left = wp.vec3d()
    q_right = wp.vec3d()
    for i in range(2):
        p0 = wp.where(i == 0, p1s, p1e)
        s01, yz, ok = shadow01(
            p0, q1, q1s, q1e, pos_p, vnormal_p, pos_q, vnormal_q, fnormal_q, mates_q, expand, 1
        )
        if ok:
            s11 += s01 * wp.where(i == 0, wp.int32(-1), wp.int32(1))
            if k < 2 and (k == 0 or (s01 != 0) != flag):
                flag = s01 != 0
                p = pos_p[p0]
                q = wp.vec3d(p[0], yz[0], yz[1])
                if k == 0:
                    p_left = p
                    q_left = q
                else:
                    p_right = p
                    q_right = q
                k += 1
    for i in range(2):
        q0 = wp.where(i == 0, q1s, q1e)
        s10, yz, ok = shadow01(
            q0, p1, p1s, p1e, pos_q, vnormal_q, pos_p, vnormal_p, fnormal_p, mates_p, expand, 0
        )
        if ok:
            s11 += s10 * wp.where(i == 0, wp.int32(-1), wp.int32(1))
            if k < 2 and (k == 0 or (s10 != 0) != flag):
                flag = s10 != 0
                q = pos_q[q0]
                p = wp.vec3d(q[0], yz[0], yz[1])
                if k == 0:
                    p_left = p
                    q_left = q
                else:
                    p_right = p
                    q_right = q
                k += 1
    xyzz = wp.vec4d()
    if s11 == 0:
        return s11, xyzz, False
    xyzz = intersect(p_left, p_right, q_left, q_right)
    dir_p = fnormal_p[p1 // 3][2] + fnormal_p[mates_p[p1] // 3][2]
    dir_q = fnormal_q[q1 // 3][2] + fnormal_q[mates_q[q1] // 3][2]
    if not shadows(xyzz[2], xyzz[3], with_sign(expand, dir_p) - dir_q):
        s11 = wp.int32(0)
    return s11, xyzz, True


@wp.func
def kernel02(
    a0: wp.int32,
    b2: wp.int32,
    pos_a: wp.array[wp.vec3d],
    vnormal_a: wp.array[wp.vec3d],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
    expand: wp.int32,
    forward: wp.int32,
) -> tuple[wp.int32, wp.float64, wp.bool]:
    # ``Kernel02``: whether vertex a0 lies below (forward) or above face b2 in z, and the face's z
    # under the vertex.
    k = wp.int32(0)
    s02 = wp.int32(0)
    flag = wp.bool(False)
    yzz_left = wp.vec3d()
    yzz_right = wp.vec3d()
    for i in range(3):
        edge = face_edge(faces_b, mates_b, b2, i)
        s01, yz, ok = shadow01(
            a0,
            edge[0],
            edge[1],
            edge[2],
            pos_a,
            vnormal_a,
            pos_b,
            vnormal_b,
            fnormal_b,
            mates_b,
            expand,
            forward,
        )
        if ok:
            same = (forward != 0) == (edge[3] != 0)
            s02 += s01 * wp.where(same, wp.int32(-1), wp.int32(1))
            if k < 2 and (k == 0 or (s01 != 0) != flag):
                flag = s01 != 0
                yzz = wp.vec3d(yz[0], yz[1], yz[1])
                if k == 0:
                    yzz_left = yzz
                else:
                    yzz_right = yzz
                k += 1
    if s02 == 0:
        return s02, wp.float64(0.0), False
    a = pos_a[a0]
    z02 = interpolate(yzz_left, yzz_right, a[1])[1]
    if forward != 0:
        if not shadows(a[2], z02, wp.float64(0.0) - fnormal_b[b2][2]):
            s02 = wp.int32(0)
    else:
        if not shadows(z02, a[2], with_sign(expand, fnormal_b[b2][2])):
            s02 = wp.int32(0)
    return s02, z02, True


@wp.func
def kernel12(
    a1: wp.int32,
    b2: wp.int32,
    faces_a: wp.array[wp.int32],
    pos_a: wp.array[wp.vec3d],
    vnormal_a: wp.array[wp.vec3d],
    fnormal_a: wp.array[wp.vec3d],
    mates_a: wp.array[wp.int32],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
    expand: wp.int32,
    forward: wp.int32,
) -> tuple[wp.int32, wp.vec3d]:
    # ``Kernel12``: the signed crossing of edge a1 (its forward halfedge) with face b2, and where.
    # A is P and B is Q when ``forward``, the other way round otherwise. Zero is no crossing.
    k = wp.int32(0)
    x12 = wp.int32(0)
    flag = wp.bool(False)
    lr0_left = wp.vec3d()
    lr0_right = wp.vec3d()
    lr1_left = wp.vec3d()
    lr1_right = wp.vec3d()
    edge_start = faces_a[a1]
    edge_end = faces_a[next_halfedge(a1)]
    for j in range(2):
        vert = wp.where(j == 0, edge_start, edge_end)
        s, z, ok = kernel02(
            vert,
            b2,
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
            x12 += s * wp.where((j == 0) == (forward != 0), wp.int32(1), wp.int32(-1))
            if k < 2 and (k == 0 or (s != 0) != flag):
                flag = s != 0
                p = pos_a[vert]
                l0 = wp.vec3d(p[0], p[2], p[1])
                l1 = wp.vec3d(p[0], z, p[1])
                if k == 0:
                    lr0_left = l0
                    lr1_left = l1
                else:
                    lr0_right = l0
                    lr1_right = l1
                k += 1
    for i in range(3):
        edge = face_edge(faces_b, mates_b, b2, i)
        s = wp.int32(0)
        xyzz = wp.vec4d()
        ok = wp.bool(False)
        if forward != 0:
            s, xyzz, ok = kernel11(
                a1,
                edge_start,
                edge_end,
                edge[0],
                edge[1],
                edge[2],
                pos_a,
                vnormal_a,
                fnormal_a,
                mates_a,
                pos_b,
                vnormal_b,
                fnormal_b,
                mates_b,
                expand,
            )
        else:
            s, xyzz, ok = kernel11(
                edge[0],
                edge[1],
                edge[2],
                a1,
                edge_start,
                edge_end,
                pos_b,
                vnormal_b,
                fnormal_b,
                mates_b,
                pos_a,
                vnormal_a,
                fnormal_a,
                mates_a,
                expand,
            )
        if ok:
            x12 -= s * wp.where(edge[3] != 0, wp.int32(1), wp.int32(-1))
            if k < 2 and (k == 0 or (s != 0) != flag):
                flag = s != 0
                z0 = xyzz[2]
                z1 = xyzz[3]
                if forward == 0:
                    z0 = xyzz[3]
                    z1 = xyzz[2]
                l0 = wp.vec3d(xyzz[0], z0, xyzz[1])
                l1 = wp.vec3d(xyzz[0], z1, xyzz[1])
                if k == 0:
                    lr0_left = l0
                    lr1_left = l1
                else:
                    lr0_right = l0
                    lr1_right = l1
                k += 1
    v12 = wp.vec3d()
    if x12 != 0:
        xzyy = intersect(lr0_left, lr0_right, lr1_left, lr1_right)
        v12 = wp.vec3d(xzyy[0], xzyy[2], xzyy[1])
    return x12, v12


# -- per-mesh preparation --------------------------------------------------------------------------


@wp.func
def spread_bits3(v: wp.uint32) -> wp.uint32:
    # ``collider_internal::SpreadBits3``: ten bits spread to every third bit.
    v = wp.uint32(0xFF0000FF) & (v * wp.uint32(0x00010001))
    v = wp.uint32(0x0F00F00F) & (v * wp.uint32(0x00000101))
    v = wp.uint32(0xC30C30C3) & (v * wp.uint32(0x00000011))
    v = wp.uint32(0x49249249) & (v * wp.uint32(0x00000005))
    return v


@wp.func
def morton_axis(x: wp.float64, low: wp.float64, high: wp.float64) -> wp.uint32:
    t = wp.float64(1024.0) * ((x - low) / (high - low))
    if not (t > wp.float64(0.0)):
        t = wp.float64(0.0)
    if not (t < wp.float64(1023.0)):
        t = wp.float64(1023.0)
    return spread_bits3(wp.uint32(t))


@wp.func
def morton_code(p: wp.vec3d, low: wp.vec3d, high: wp.vec3d) -> wp.uint32:
    # ``Collider::MortonCode``: the 30-bit Morton code of ``p`` in the box.
    x = morton_axis(p[0], low[0], high[0])
    y = morton_axis(p[1], low[1], high[1])
    z = morton_axis(p[2], low[2], high[2])
    return x * wp.uint32(4) + y * wp.uint32(2) + z


@wp.kernel
def vertex_morton_keys(
    positions: wp.array[wp.vec3d],
    low: wp.vec3d,
    high: wp.vec3d,
    out_keys: wp.array[wp.uint32],
    out_order: wp.array[wp.int32],
) -> None:
    # ``SortVerts``: each vertex's Morton code in the box, for a stable sort.
    v = wp.int32(wp.tid())
    out_keys[v] = morton_code(positions[v], low, high)
    out_order[v] = v


@wp.kernel
def renumber_vertices(
    positions: wp.array[wp.vec3d],
    normals: wp.array[wp.vec3d],
    new_to_old: wp.array[wp.int32],
    with_normals: wp.int32,
    out_positions: wp.array[wp.vec3d],
    out_normals: wp.array[wp.vec3d],
    out_old_to_new: wp.array[wp.int32],
) -> None:
    v = wp.int32(wp.tid())
    old = new_to_old[v]
    out_positions[v] = positions[old]
    if with_normals != 0:
        out_normals[v] = normals[old]
    out_old_to_new[old] = v


@wp.kernel
def face_morton_keys(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    low: wp.vec3d,
    high: wp.vec3d,
    out_keys: wp.array[wp.uint32],
    out_order: wp.array[wp.int32],
) -> None:
    # ``GetFaceBoxMorton``: each face's centroid's Morton code, for ``SortFaces``' stable sort.
    f = wp.int32(wp.tid())
    center = positions[faces[3 * f]]
    center = center + positions[faces[3 * f + 1]]
    center = center + positions[faces[3 * f + 2]]
    center = center / wp.float64(3.0)
    out_keys[f] = morton_code(center, low, high)
    out_order[f] = f


@wp.kernel
def renumber_faces(
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3d],
    face_mesh: wp.array[wp.int32],
    face_coplanar: wp.array[wp.int32],
    new_to_old: wp.array[wp.int32],
    old_to_new: wp.array[wp.int32],
    with_attributes: wp.int32,
    out_faces: wp.array[wp.int32],
    out_face_normals: wp.array[wp.vec3d],
    out_face_mesh: wp.array[wp.int32],
    out_face_coplanar: wp.array[wp.int32],
) -> None:
    # ``GatherFaces`` with ``ReindexVerts``: each face moved to its sorted slot, its corners
    # renamed, its rotation and its references kept.
    f = wp.int32(wp.tid())
    old = new_to_old[f]
    for i in range(3):
        out_faces[3 * f + i] = old_to_new[faces[3 * old + i]]
    if with_attributes != 0:
        out_face_normals[f] = face_normals[old]
        out_face_mesh[f] = face_mesh[old]
        out_face_coplanar[f] = face_coplanar[old]


@wp.kernel
def widen_positions(vertices: wp.array[wp.vec3], out_positions: wp.array[wp.vec3d]) -> None:
    # A ``float32`` mesh's exact ``float64`` positions.
    v = wp.int32(wp.tid())
    out_positions[v] = wp.vec3d(vertices[v])


@wp.kernel
def mesh_face_normals(
    positions: wp.array[wp.vec3d], faces: wp.array[wp.int32], out_face_normals: wp.array[wp.vec3d]
) -> None:
    # Each face's ``float64`` normal (``SetNormalsAndCoplanar`` before the coplanar merge); a
    # degenerate face's is +z.
    f = wp.int32(wp.tid())
    a = positions[faces[3 * f]]
    n = wp.cross(positions[faces[3 * f + 1]] - a, positions[faces[3 * f + 2]] - a)
    length = wp.length(n)
    normal = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    if length > wp.float64(0.0) and wp.isfinite(length):
        normal = n / length
    out_face_normals[f] = normal


@wp.kernel
def operand_boxes(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    pad: wp.float32,
    out_lowers: wp.array[wp.vec3],
    out_uppers: wp.array[wp.vec3],
    out_first_halfedge: wp.array[wp.int32],
    out_boundary: wp.array[wp.int32],
) -> None:
    # Each face's box for the broad phase (``float32``, widened by ``pad``), its
    # corners' bids for the least halfedge leaving each vertex, and a count of unpaired halfedges.
    f = wp.int32(wp.tid())
    a = wp.vec3(positions[faces[3 * f]])
    b = wp.vec3(positions[faces[3 * f + 1]])
    c = wp.vec3(positions[faces[3 * f + 2]])
    for i in range(3):
        wp.atomic_min(out_first_halfedge, faces[3 * f + i], 3 * f + i)
        if twins[3 * f + i] < 0:
            wp.atomic_add(out_boundary, 0, 1)
    padding = wp.vec3(pad, pad, pad)
    out_lowers[f] = wp.min(wp.min(a, b), c) - padding
    out_uppers[f] = wp.max(wp.max(a, b), c) + padding


@wp.kernel
def positions_box(positions: wp.array[wp.vec3d], out_box: wp.array[wp.float64]) -> None:
    # ``CalculateBBox``: the box of the vertices, into ``(min x, y, z, max x, y, z)`` (filled with
    # +-inf); a removed (NaN) vertex is skipped.
    v = wp.int32(wp.tid())
    p = positions[v]
    if not wp.isfinite(p[0]):
        return
    for i in range(3):
        wp.atomic_min(out_box, i, p[i])
        wp.atomic_max(out_box, 3 + i, p[i])


# -- Manifold's arc cosine (the pseudo-normals' angle weights) ------------------------------------
#
# The angle weights of a vertex's pseudo-normal decide ties wherever two normals cancel (two
# boxes' matching corners), so they must round as Manifold's do, on every device. Manifold does
# not call the C library: ``include/manifold/math.h`` carries its own ``acos`` (fdlibm's, in plain
# ``float64`` arithmetic), which this ports. CUDA's ``acos`` and glibc's differ from it and from
# each other in the last bit on a tenth of inputs (2026-11-21).


@wp.func
def acos_ratio(z: wp.float64) -> wp.float64:
    p = z * (
        wp.float64(1.66666666666666657415e-01)
        + z
        * (
            wp.float64(-3.25565818622400915405e-01)
            + z
            * (
                wp.float64(2.01212532134862925881e-01)
                + z
                * (
                    wp.float64(-4.00555345006794114027e-02)
                    + z
                    * (
                        wp.float64(7.91534994289814532176e-04)
                        + z * wp.float64(3.47933107596021167570e-05)
                    )
                )
            )
        )
    )
    q = wp.float64(1.0) + z * (
        wp.float64(-2.40339491173441421878e00)
        + z
        * (
            wp.float64(2.02094576023350569471e00)
            + z
            * (wp.float64(-6.88283971605453293030e-01) + z * wp.float64(7.70381505559019352791e-02))
        )
    )
    return p / q


@wp.func
def manifold_acos(x: wp.float64) -> wp.float64:
    # ``math::acos``: three ranges, the outer two through ``2 asin(sqrt((1 -+ x) / 2))`` with the
    # square root's low word split off.
    pio2_hi = wp.float64(1.57079632679489655800e00)
    pio2_lo = wp.float64(6.12323399573676603587e-17)
    bits = wp.cast(x, wp.uint64)
    hx = wp.uint32(bits >> wp.uint64(32))
    ix = hx & wp.uint32(0x7FFFFFFF)
    if ix >= wp.uint32(0x3FF00000):
        if x == wp.float64(1.0):
            return wp.float64(0.0)
        if x == wp.float64(-1.0):
            return wp.float64(2.0) * pio2_hi
        return wp.float64(0.0) / (x - x)
    if ix < wp.uint32(0x3FE00000):
        if ix <= wp.uint32(0x3C600000):
            return pio2_hi
        return pio2_hi - (x - (pio2_lo - x * acos_ratio(x * x)))
    if (hx >> wp.uint32(31)) != wp.uint32(0):
        z = (wp.float64(1.0) + x) * wp.float64(0.5)
        s = wp.sqrt(z)
        w = acos_ratio(z) * s - pio2_lo
        return wp.float64(2.0) * (pio2_hi - (s + w))
    z = (wp.float64(1.0) - x) * wp.float64(0.5)
    s = wp.sqrt(z)
    df = wp.cast(wp.cast(s, wp.uint64) & wp.uint64(0xFFFFFFFF00000000), wp.float64)
    c = (z - df * df) / (s + df)
    w = acos_ratio(z) * s + c
    return wp.float64(2.0) * (df + w)


@wp.kernel
def vertex_pseudo_normals(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    face_normals: wp.array[wp.vec3d],
    first_halfedge: wp.array[wp.int32],
    out_vertex_normals: wp.array[wp.vec3d],
) -> None:
    # ``CalculateVertNormals``: the angle-weighted pseudo-normal, summed around the vertex from
    # its least outgoing halfedge in ``ForVert``'s order. An unreferenced vertex gets zero.
    v = wp.int32(wp.tid())
    first = first_halfedge[v]
    normal = wp.vec3d()
    if first >= faces.shape[0]:
        out_vertex_normals[v] = normal
        return
    current = first
    steps = wp.int32(0)
    while steps < faces.shape[0]:
        current = next_halfedge(twins[current])
        tri = current // 3
        start = faces[current]
        end = faces[next_halfedge(current)]
        third = faces[next_halfedge(next_halfedge(current))]
        curr_edge = wp.normalize(positions[end] - positions[start])
        prev_edge = wp.normalize(positions[start] - positions[third])
        if wp.isfinite(curr_edge[0]) and wp.isfinite(prev_edge[0]):
            d = wp.float64(0.0) - wp.dot(prev_edge, curr_edge)
            phi = wp.float64(0.0)
            if d <= wp.float64(-1.0):
                phi = wp.float64(3.141592653589793)
            elif d < wp.float64(1.0):
                phi = manifold_acos(d)
            normal = normal + phi * face_normals[tri]
        steps += 1
        if current == first:
            break
    length = wp.length(normal)
    if length > wp.float64(0.0) and wp.isfinite(length):
        normal = normal / length
    else:
        normal = wp.vec3d()
    out_vertex_normals[v] = normal


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
    faces_a: wp.array[wp.int32],
    pos_a: wp.array[wp.vec3d],
    vnormal_a: wp.array[wp.vec3d],
    fnormal_a: wp.array[wp.vec3d],
    mates_a: wp.array[wp.int32],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
    bvh_b: wp.uint64,
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    # How many faces of B each edge of A crosses: a thread a forward halfedge, its box through
    # B's face tree, ``kernel12`` on each candidate. The box is a superset; ``kernel12`` decides.
    h = wp.int32(wp.tid())
    start = faces_a[h]
    end = faces_a[next_halfedge(h)]
    count = wp.int32(0)
    if start < end:
        lower, upper = edge_query_box(pos_a, start, end, pad)
        query = wp.bvh_query_aabb(bvh_b, lower, upper)
        face = wp.int32(0)
        while wp.bvh_query_next(query, face):
            x12, _v12 = kernel12(
                h,
                face,
                faces_a,
                pos_a,
                vnormal_a,
                fnormal_a,
                mates_a,
                faces_b,
                pos_b,
                vnormal_b,
                fnormal_b,
                mates_b,
                expand,
                forward,
            )
            if x12 != 0:
                count += 1
    out_counts[h] = count


@wp.kernel
def write_edge_crossings(
    faces_a: wp.array[wp.int32],
    pos_a: wp.array[wp.vec3d],
    vnormal_a: wp.array[wp.vec3d],
    fnormal_a: wp.array[wp.vec3d],
    mates_a: wp.array[wp.int32],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
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
    start = faces_a[h]
    end = faces_a[next_halfedge(h)]
    lower, upper = edge_query_box(pos_a, start, end, pad)
    query = wp.bvh_query_aabb(bvh_b, lower, upper)
    face = wp.int32(0)
    slot = record_base + first
    stop = record_base + last
    while wp.bvh_query_next(query, face):
        if slot < stop:
            x12, v12 = kernel12(
                h,
                face,
                faces_a,
                pos_a,
                vnormal_a,
                fnormal_a,
                mates_a,
                faces_b,
                pos_b,
                vnormal_b,
                fnormal_b,
                mates_b,
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
    end = faces[next_halfedge(h)]
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
        s02, _z02, ok = kernel02(
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
    v_end = faces[next_halfedge(h)]
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
    end = faces[next_halfedge(h)]
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
def project(normal: wp.vec3d, p: wp.vec3d) -> wp.vec2d:
    # ``GetAxisAlignedProjection``: drop the normal's dominant axis, keeping the polygon CCW.
    a = wp.vec3d(wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2]))
    r = wp.vec2d(p[1], p[2])
    m = normal[0]
    if a[2] > a[0] and a[2] > a[1]:
        r = wp.vec2d(p[0], p[1])
        m = normal[2]
    elif a[1] > a[0]:
        r = wp.vec2d(p[2], p[0])
        m = normal[1]
    if m < wp.float64(0.0):
        r = wp.vec2d(wp.float64(0.0) - r[0], r[1])
    return r


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
    used: wp.array[wp.int32],
    order: wp.array[wp.int32],
    loop_heads: wp.array[wp.int32],
    ec: kernel_polyline.EarClipState,
    out_counts: wp.array[wp.int32],
) -> None:
    # ``Face2Tri`` for one face of P or Q: a triangle as it is, a quad by its better diagonal,
    # anything else through the ear clipper, into ``ec.tris[tri_offsets[f]:]`` as result vertices.
    f = wp.int32(wp.tid())
    first = face_offsets[f]
    n = face_offsets[f + 1] - first
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
def face_mesh_keys(
    face_mesh: wp.array[wp.int32], out_keys: wp.array[wp.int32], out_order: wp.array[wp.int32]
) -> None:
    f = wp.int32(wp.tid())
    out_keys[f] = face_mesh[f]
    out_order[f] = f


@wp.kernel
def gather_faces(
    faces: wp.array[wp.int32], order: wp.array[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    f = wp.int32(wp.tid())
    old = order[f]
    for i in range(3):
        out_faces[3 * f + i] = faces[3 * old + i]


@wp.kernel
def offset_indices(
    indices: wp.array[wp.int32], offset: wp.int32, out_indices: wp.array[wp.int32]
) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = indices[i] + offset


@wp.kernel
def narrow_positions(positions: wp.array[wp.vec3d], out_vertices: wp.array[wp.vec3]) -> None:
    i = wp.int32(wp.tid())
    out_vertices[i] = wp.vec3(positions[i])


# -- coplanar groups of the inputs (``SetNormalsAndCoplanar``) -------------------------------------


@wp.kernel
def face_area_keys(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    out_keys: wp.array[wp.float64],
    out_order: wp.array[wp.int32],
) -> None:
    # Each face's negated squared double area, so an ascending stable sort visits the largest
    # first and ties in index order, as ``SetNormalsAndCoplanar``'s ``stable_sort`` does.
    f = wp.int32(wp.tid())
    a = positions[faces[3 * f]]
    n = wp.cross(positions[faces[3 * f + 1]] - a, positions[faces[3 * f + 2]] - a)
    out_keys[f] = wp.float64(0.0) - wp.dot(n, n)
    out_order[f] = f


@wp.kernel
def coplanar_seed_flags(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    face_normals: wp.array[wp.vec3d],
    order: wp.array[wp.int32],
    tolerance: wp.float64,
    out_rank: wp.array[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # Each face's place in the area order, and whether as a seed it would claim a neighbour (one
    # whose far vertex lies within ``tolerance`` of its plane). Only those seeds can grow a group:
    # every other face ends its own group of one, unless a seed's flood reaches it first.
    k = wp.int32(wp.tid())
    f = order[k]
    out_rank[f] = k
    base = positions[faces[3 * f]]
    normal = face_normals[f]
    claims = wp.int32(0)
    for i in range(3):
        h = next_halfedge(twins[3 * f + i])
        v = positions[faces[next_halfedge(h)]]
        if wp.abs(wp.dot(v - base, normal)) < tolerance:
            claims = wp.int32(1)
    out_flags[k] = claims


@wp.kernel
def flood_coplanar(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    seeds: wp.array[wp.int32],
    n_seeds: wp.int32,
    order: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    is_seed: wp.array[wp.int32],
    tolerance: wp.float64,
    face_normals: wp.array[wp.vec3d],
    stack: wp.array[wp.int32],
    coplanar: wp.array[wp.int32],
) -> None:
    # ``SetNormalsAndCoplanar``'s flood, one thread over the claiming seeds in area order: each
    # spreads across edges to faces whose far vertex lies within ``tolerance`` of its plane,
    # which take its normal. A face that claims nothing has taken itself by the time a later seed
    # arrives (its rank is lower), as the serial loop over every face would have it; which faces
    # a flood reaches depends only on its seed's plane, not on the walk's order.
    for i in range(n_seeds):
        k = seeds[i]
        seed = order[k]
        if coplanar[seed] >= 0:
            continue
        coplanar[seed] = seed
        base = positions[faces[3 * seed]]
        normal = face_normals[seed]
        top = wp.int32(3)
        stack[0] = 3 * seed
        stack[1] = 3 * seed + 1
        stack[2] = 3 * seed + 2
        while top > 0:
            h = next_halfedge(twins[stack[top - 1]])
            top -= 1
            tri = h // 3
            if coplanar[tri] >= 0:
                continue
            if is_seed[rank[tri]] == 0 and rank[tri] < k:
                continue
            v = positions[faces[next_halfedge(h)]]
            if wp.abs(wp.dot(v - base, normal)) < tolerance:
                coplanar[tri] = seed
                face_normals[tri] = normal
                if top == 0 or h != twins[stack[top - 1]]:
                    stack[top] = h
                    top += 1
                else:
                    top -= 1
                stack[top] = next_halfedge(h)
                top += 1


@wp.kernel
def own_coplanar_groups(coplanar: wp.array[wp.int32]) -> None:
    # A face no flood reached is its own group.
    f = wp.int32(wp.tid())
    if coplanar[f] < 0:
        coplanar[f] = f


# -- the result's halfedges ------------------------------------------------------------------------


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
        v = out_start[next_halfedge(h)]
        for g in range(h0, h1):
            if (
                g != h
                and out_pair[g] == -1
                and out_start[g] == v
                and out_start[next_halfedge(g)] == u
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
    return m.start[next_halfedge(h)]


@wp.func
def hm_pair_up(m: HalfedgeMesh, a: wp.int32, b: wp.int32):
    m.pair[a] = b
    m.pair[b] = a


@wp.func
def hm_same_face(m: HalfedgeMesh, t0: wp.int32, t1: wp.int32) -> wp.bool:
    return m.tri_mesh[t0] == m.tri_mesh[t1] and m.tri_coplanar[t0] == m.tri_coplanar[t1]


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
        m.start[next_halfedge(current)] = vert
        current = next_halfedge(current)
        m.start[current] = vert
        current = m.pair[current]


@wp.func
def hm_remove_if_folded(m: HalfedgeMesh, edge: wp.int32):
    # ``RemoveIfFolded``: a pair of triangles folded onto each other across ``edge`` goes.
    if edge < 0 or m.pair[edge] < 0:
        return
    t0 = wp.vec3i(edge, next_halfedge(edge), next_halfedge(next_halfedge(edge)))
    p = m.pair[edge]
    t1 = wp.vec3i(p, next_halfedge(p), next_halfedge(next_halfedge(p)))
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
    t0 = wp.vec3i(edge, next_halfedge(edge), next_halfedge(next_halfedge(edge)))
    t1 = wp.vec3i(pair, next_halfedge(pair), next_halfedge(next_halfedge(pair)))
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
            current = next_halfedge(current)
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
        current = next_halfedge(current)
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
        current = next_halfedge(current)
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
    next_edge = next_halfedge(edge)
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
            current = m.pair[next_halfedge(current)]
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
        current = m.pair[next_halfedge(current)]
    if current == edge:
        # Separate sheets need no new faces.
        new_vert = hm_new_vert(m, end_vert)
        first = next_halfedge(current)
        c = first
        steps = wp.int32(0)
        while steps < limit:
            steps += 1
            c = next_halfedge(m.pair[c])
            m.start[c] = new_vert
            m.start[next_halfedge(m.pair[c])] = new_vert
            if c == first:
                break
    # Orbit the start vertex: split it if the split above pinched it.
    pair = m.pair[edge]
    current = m.pair[next_halfedge(pair)]
    steps = wp.int32(0)
    while current != pair and current >= 0 and steps < limit:
        steps += 1
        if m.start[current] == end_vert:
            break
        current = m.pair[next_halfedge(current)]
    if current == pair:
        new_vert = hm_new_vert(m, end_vert)
        first = next_halfedge(current)
        c = first
        steps = wp.int32(0)
        while steps < limit:
            steps += 1
            c = next_halfedge(m.pair[c])
            m.start[c] = new_vert
            m.start[next_halfedge(m.pair[c])] = new_vert
            if c == first:
                break


@wp.func
def hm_swap_edge(m: HalfedgeMesh, t0: wp.vec3i, t1: wp.vec3i) -> wp.bool:
    # ``RecursiveEdgeSwap``'s ``SwapEdge``: turn the shared edge of the two triangles. Returns
    # whether the new edge already existed (the mesh was split there instead).
    v0 = m.start[t0[2]]
    v1 = m.start[t1[2]]
    m.start[t0[0]] = v1
    m.start[next_halfedge(t0[2])] = v1
    m.start[t1[0]] = v0
    m.start[next_halfedge(t1[2])] = v0
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
        current = next_halfedge(current)
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
    t0 = wp.vec3i(edge, next_halfedge(edge), next_halfedge(next_halfedge(edge)))
    t1 = wp.vec3i(pair, next_halfedge(pair), next_halfedge(next_halfedge(pair)))
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
        current = next_halfedge(p)
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
        current = next_halfedge(m.pair[current])
        m.start[current] = vert
        m.start[next_halfedge(m.pair[current])] = vert
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
        current = next_halfedge(p)
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
    current = next_halfedge(pair)
    ref1 = current // 3
    ref1_updated = not hm_same_face(m, ref0, ref1)
    steps = wp.int32(0)
    limit = m.sizes[0]
    while current != edge and steps < limit:
        steps += 1
        p = m.pair[current]
        if p < 0:
            return
        current = next_halfedge(p)
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
    t0 = wp.vec3i(edge, next_halfedge(edge), next_halfedge(next_halfedge(edge)))
    t1 = wp.vec3i(pair, next_halfedge(pair), next_halfedge(next_halfedge(pair)))
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
def flagged_halfedges(
    flags: wp.array[wp.int32], ranks: wp.array[wp.int32], out_list: wp.array[wp.int32]
) -> None:
    h = wp.int32(wp.tid())
    if ranks[h + 1] != ranks[h]:
        out_list[ranks[h]] = h


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
def hm_first_halfedges(m: HalfedgeMesh, out_first: wp.array[wp.int32]) -> None:
    # ``CalculateVertNormals``' least live halfedge leaving each vertex.
    h = wp.int32(wp.tid())
    v = m.start[h]
    if v >= 0:
        wp.atomic_min(out_first, v, h)


@wp.kernel
def hm_vertex_normals(
    m: HalfedgeMesh, first: wp.array[wp.int32], out_normals: wp.array[wp.vec3d]
) -> None:
    # ``CalculateVertNormals`` on the simplified result, in ``ForVert``'s order from the least
    # halfedge, with the faces' inherited normals.
    v = wp.int32(wp.tid())
    start = first[v]
    normal = wp.vec3d()
    if start >= m.sizes[0]:
        out_normals[v] = normal
        return
    current = start
    steps = wp.int32(0)
    while steps < m.sizes[0]:
        steps += 1
        p = m.pair[current]
        if p < 0:
            break
        current = next_halfedge(p)
        a = m.start[current]
        b = m.start[next_halfedge(current)]
        c = m.start[next_halfedge(next_halfedge(current))]
        curr_edge = wp.normalize(m.pos[b] - m.pos[a])
        prev_edge = wp.normalize(m.pos[a] - m.pos[c])
        if wp.isfinite(curr_edge[0]) and wp.isfinite(prev_edge[0]):
            d = wp.float64(0.0) - wp.dot(prev_edge, curr_edge)
            phi = wp.float64(0.0)
            if d <= wp.float64(-1.0):
                phi = wp.float64(3.141592653589793)
            elif d < wp.float64(1.0):
                phi = manifold_acos(d)
            normal = normal + phi * m.tri_normal[current // 3]
        if current == start:
            break
    length = wp.length(normal)
    if length > wp.float64(0.0) and wp.isfinite(length):
        normal = normal / length
    else:
        normal = wp.vec3d()
    out_normals[v] = normal


# -- segment casts (``Impl::RayCast``) -------------------------------------------------------------


@wp.kernel
def segment_edge_mesh(
    origins: wp.array[wp.vec3],
    endpoints: wp.array[wp.vec3],
    out_positions: wp.array[wp.vec3d],
    out_faces: wp.array[wp.int32],
    out_mates: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3d],
) -> None:
    # ``RayCast``'s one-edge mesh a segment: halfedge ``3 s`` runs origin -> endpoint, ``3 s + 1``
    # back, the third closes the storage; every normal zero, so the segment perturbs nothing.
    s = wp.int32(wp.tid())
    out_positions[2 * s] = wp.vec3d(origins[s])
    out_positions[2 * s + 1] = wp.vec3d(endpoints[s])
    out_faces[3 * s] = 2 * s
    out_faces[3 * s + 1] = 2 * s + 1
    out_faces[3 * s + 2] = 2 * s
    out_mates[3 * s] = 3 * s + 1
    out_mates[3 * s + 1] = 3 * s
    out_mates[3 * s + 2] = -1
    out_normals[2 * s] = wp.vec3d()
    out_normals[2 * s + 1] = wp.vec3d()


@wp.func
def segment_hit(
    s: wp.int32,
    face: wp.int32,
    ray_faces: wp.array[wp.int32],
    ray_pos: wp.array[wp.vec3d],
    ray_normals: wp.array[wp.vec3d],
    ray_mates: wp.array[wp.int32],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
) -> tuple[wp.bool, wp.float64, wp.vec3d]:
    # Whether segment ``s`` crosses ``face``, its parameter there in [0, 1] along the segment's
    # dominant axis, and the crossing: ``Kernel12`` of the segment's forward halfedge.
    x12, v12 = kernel12(
        3 * s,
        face,
        ray_faces,
        ray_pos,
        ray_normals,
        ray_normals,
        ray_mates,
        faces_b,
        pos_b,
        vnormal_b,
        fnormal_b,
        mates_b,
        0,
        1,
    )
    if x12 == 0 or not wp.isfinite(v12[0]):
        return False, wp.float64(0.0), v12
    origin = ray_pos[2 * s]
    direction = ray_pos[2 * s + 1] - origin
    a = wp.vec3d(wp.abs(direction[0]), wp.abs(direction[1]), wp.abs(direction[2]))
    axis = wp.int32(2)
    if a[0] > a[1] and a[0] > a[2]:
        axis = wp.int32(0)
    elif a[1] > a[2]:
        axis = wp.int32(1)
    t = (v12[axis] - origin[axis]) / direction[axis]
    return t >= wp.float64(0.0) and t <= wp.float64(1.0), t, v12


@wp.func
def segment_direction(ray_pos: wp.array[wp.vec3d], s: wp.int32) -> wp.vec3:
    return wp.vec3(ray_pos[2 * s + 1]) - wp.vec3(ray_pos[2 * s])


# The ray walks' reach, in segment lengths: a little past the endpoint, so the walk's ``float32``
# rounding never drops a face the exact predicate would keep.
SEGMENT_REACH = wp.float32(1.0001)


@wp.kernel
def count_segment_hits(
    ray_faces: wp.array[wp.int32],
    ray_pos: wp.array[wp.vec3d],
    ray_normals: wp.array[wp.vec3d],
    ray_mates: wp.array[wp.int32],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
    bvh_b: wp.uint64,
    pad: wp.float32,
    out_counts: wp.array[wp.int32],
) -> None:
    # How many faces each segment crosses within its length.
    s = wp.int32(wp.tid())
    # The faces whose boxes (padded past ``float32`` rounding) the segment passes through; the
    # exact predicate decides.
    query = wp.bvh_query_ray(bvh_b, wp.vec3(ray_pos[2 * s]), segment_direction(ray_pos, s))
    face = wp.int32(0)
    count = wp.int32(0)
    while wp.bvh_query_next(query, face, SEGMENT_REACH + wp.float32(0.0)):
        hit, _t, _point = segment_hit(
            s,
            face,
            ray_faces,
            ray_pos,
            ray_normals,
            ray_mates,
            faces_b,
            pos_b,
            vnormal_b,
            fnormal_b,
            mates_b,
        )
        if hit:
            count += 1
    out_counts[s] = count


@wp.kernel
def write_segment_hits(
    ray_faces: wp.array[wp.int32],
    ray_pos: wp.array[wp.vec3d],
    ray_normals: wp.array[wp.vec3d],
    ray_mates: wp.array[wp.int32],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
    bvh_b: wp.uint64,
    pad: wp.float32,
    offsets: wp.array[wp.int32],
    face_to_input: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_parameters: wp.array[wp.float64],
    out_points: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # Each segment's hits in order along it, ties by the face's input index: the face (as the
    # caller numbered it), the parameter, the crossing and the face's normal.
    s = wp.int32(wp.tid())
    first = offsets[s]
    stop = offsets[s + 1]
    if first == stop:
        return
    query = wp.bvh_query_ray(bvh_b, wp.vec3(ray_pos[2 * s]), segment_direction(ray_pos, s))
    face = wp.int32(0)
    slot = first
    while wp.bvh_query_next(query, face, SEGMENT_REACH + wp.float32(0.0)):
        if slot >= stop:
            continue
        hit, t, point = segment_hit(
            s,
            face,
            ray_faces,
            ray_pos,
            ray_normals,
            ray_mates,
            faces_b,
            pos_b,
            vnormal_b,
            fnormal_b,
            mates_b,
        )
        if not hit:
            continue
        index = face_to_input[face]
        at = slot
        while at > first and (
            out_parameters[at - 1] > t
            or (out_parameters[at - 1] == t and out_faces[at - 1] > index)
        ):
            out_faces[at] = out_faces[at - 1]
            out_parameters[at] = out_parameters[at - 1]
            out_points[at] = out_points[at - 1]
            out_normals[at] = out_normals[at - 1]
            at -= 1
        out_faces[at] = index
        out_parameters[at] = t
        out_points[at] = wp.vec3(point)
        out_normals[at] = wp.vec3(fnormal_b[face])
        slot += 1


# -- the gap between two solids (``Impl::MinGap``, ``tri_dist.h``) --------------------------------


@wp.func
def clamp01(x: wp.float64) -> wp.float64:
    return wp.min(wp.max(x, wp.float64(0.0)), wp.float64(1.0))


@wp.func
def edge_edge_closest(
    p: wp.vec3d, a: wp.vec3d, q: wp.vec3d, b: wp.vec3d
) -> tuple[wp.vec3d, wp.vec3d]:
    # ``EdgeEdgeDist``: the closest points of segments p + t a and q + u b.
    big_t = q - p
    a_dot_a = wp.dot(a, a)
    b_dot_b = wp.dot(b, b)
    a_dot_b = wp.dot(a, b)
    a_dot_t = wp.dot(a, big_t)
    b_dot_t = wp.dot(b, big_t)
    denominator = a_dot_a * b_dot_b - a_dot_b * a_dot_b
    t = wp.float64(0.0)
    if denominator != wp.float64(0.0):
        t = clamp01((a_dot_t * b_dot_b - b_dot_t * a_dot_b) / denominator)
    u = wp.float64(0.0)
    if b_dot_b != wp.float64(0.0):
        u = (t * a_dot_b - b_dot_t) / b_dot_b
        if u < wp.float64(0.0):
            u = wp.float64(0.0)
            t = wp.float64(0.0)
            if a_dot_a != wp.float64(0.0):
                t = clamp01(a_dot_t / a_dot_a)
        elif u > wp.float64(1.0):
            u = wp.float64(1.0)
            t = wp.float64(0.0)
            if a_dot_a != wp.float64(0.0):
                t = clamp01((a_dot_b + a_dot_t) / a_dot_a)
    else:
        u = wp.float64(0.0)
        t = wp.float64(0.0)
        if a_dot_a != wp.float64(0.0):
            t = clamp01(a_dot_t / a_dot_a)
    return p + a * t, q + b * u


@wp.func
def triangle_vertex(t: wp.mat33d, i: wp.int32) -> wp.vec3d:
    return wp.vec3d(t[i, 0], t[i, 1], t[i, 2])


@wp.func
def projected_vertex_distance(p: wp.mat33d, q: wp.mat33d) -> tuple[wp.bool, wp.bool, wp.float64]:
    # ``DistanceTriangleTriangleSquared``'s face test: whether q lies wholly on one side of p's
    # plane (``shown_disjoint``), and whether its nearest vertex projects inside p, with that
    # squared distance.
    p0 = triangle_vertex(p, 0)
    p1 = triangle_vertex(p, 1)
    p2 = triangle_vertex(p, 2)
    s0 = p1 - p0
    s1 = p2 - p1
    s2 = p0 - p2
    sn = wp.cross(s0, s1)
    snl = wp.dot(sn, sn)
    if not (snl > wp.float64(1e-15)):
        return False, False, wp.float64(0.0)
    tp = wp.vec3d(
        wp.dot(p0 - triangle_vertex(q, 0), sn),
        wp.dot(p0 - triangle_vertex(q, 1), sn),
        wp.dot(p0 - triangle_vertex(q, 2), sn),
    )
    index = wp.int32(-1)
    if tp[0] > wp.float64(0.0) and tp[1] > wp.float64(0.0) and tp[2] > wp.float64(0.0):
        index = wp.where(tp[0] < tp[1], wp.int32(0), wp.int32(1))
        if tp[2] < tp[index]:
            index = wp.int32(2)
    elif tp[0] < wp.float64(0.0) and tp[1] < wp.float64(0.0) and tp[2] < wp.float64(0.0):
        index = wp.where(tp[0] > tp[1], wp.int32(0), wp.int32(1))
        if tp[2] > tp[index]:
            index = wp.int32(2)
    if index < 0:
        return False, False, wp.float64(0.0)
    vertex = triangle_vertex(q, index)
    if wp.dot(vertex - p0, wp.cross(sn, s0)) > wp.float64(0.0):
        if wp.dot(vertex - p1, wp.cross(sn, s1)) > wp.float64(0.0):
            if wp.dot(vertex - p2, wp.cross(sn, s2)) > wp.float64(0.0):
                cp = vertex + sn * tp[index] / snl
                d = cp - vertex
                return True, True, wp.dot(d, d)
    return True, False, wp.float64(0.0)


@wp.func
def triangle_distance_squared(p: wp.mat33d, q: wp.mat33d) -> wp.float64:
    # ``DistanceTriangleTriangleSquared``: the squared distance between two triangles.
    shown_disjoint = wp.bool(False)
    mindd = wp.float64(1.7976931348623157e308)
    for i in range(3):
        pi = triangle_vertex(p, i)
        si = triangle_vertex(p, (i + 1) % 3) - pi
        for j in range(3):
            qj = triangle_vertex(q, j)
            tj = triangle_vertex(q, (j + 1) % 3) - qj
            cp, cq = edge_edge_closest(pi, si, qj, tj)
            v = cq - cp
            dd = wp.dot(v, v)
            if dd <= mindd:
                mindd = dd
                a = wp.dot(triangle_vertex(p, (i + 2) % 3) - cp, v)
                b = wp.dot(triangle_vertex(q, (j + 2) % 3) - cq, v)
                if a <= wp.float64(0.0) and b >= wp.float64(0.0):
                    return wp.dot(v, v)
                if a <= wp.float64(0.0):
                    a = wp.float64(0.0)
                elif b > wp.float64(0.0):
                    b = wp.float64(0.0)
                if (mindd - a + b) > wp.float64(0.0):
                    shown_disjoint = True
    disjoint, inside, dd = projected_vertex_distance(p, q)
    if disjoint:
        shown_disjoint = True
    if inside:
        return dd
    disjoint, inside, dd = projected_vertex_distance(q, p)
    if disjoint:
        shown_disjoint = True
    if inside:
        return dd
    if shown_disjoint:
        return mindd
    return wp.float64(0.0)


@wp.func
def face_triangle(
    faces: wp.array[wp.int32], positions: wp.array[wp.vec3d], f: wp.int32
) -> wp.mat33d:
    a = positions[faces[3 * f]]
    b = positions[faces[3 * f + 1]]
    c = positions[faces[3 * f + 2]]
    return wp.mat33d(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])


@wp.kernel
def min_face_distance(
    faces_self: wp.array[wp.int32],
    pos_self: wp.array[wp.vec3d],
    faces_other: wp.array[wp.int32],
    pos_other: wp.array[wp.vec3d],
    bvh_self: wp.uint64,
    search_length: wp.float64,
    pad: wp.float32,
    out_min: wp.array[wp.float64],
) -> None:
    # ``MinDistanceRecorder``: each face of the other solid against the faces of this one within
    # ``search_length`` of its box; the least squared distance (a minimum, so in any order).
    f = wp.int32(wp.tid())
    q = face_triangle(faces_other, pos_other, f)
    lower = wp.vec3d(
        wp.min(wp.min(q[0, 0], q[1, 0]), q[2, 0]),
        wp.min(wp.min(q[0, 1], q[1, 1]), q[2, 1]),
        wp.min(wp.min(q[0, 2], q[1, 2]), q[2, 2]),
    )
    upper = wp.vec3d(
        wp.max(wp.max(q[0, 0], q[1, 0]), q[2, 0]),
        wp.max(wp.max(q[0, 1], q[1, 1]), q[2, 1]),
        wp.max(wp.max(q[0, 2], q[1, 2]), q[2, 2]),
    )
    reach = wp.vec3d(search_length, search_length, search_length)
    padding = wp.vec3(pad, pad, pad)
    query = wp.bvh_query_aabb(
        bvh_self, wp.vec3(lower - reach) - padding, wp.vec3(upper + reach) + padding
    )
    face = wp.int32(0)
    best = wp.float64(1.7976931348623157e308) * wp.float64(10.0)
    while wp.bvh_query_next(query, face):
        p = face_triangle(faces_self, pos_self, face)
        best = wp.min(best, triangle_distance_squared(p, q))
    if best < out_min[0]:
        wp.atomic_min(out_min, 0, best)
