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


@wp.func
def pseudo_normal(
    starts: wp.array[wp.int32],
    pairs: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    face_normals: wp.array[wp.vec3d],
    first: wp.int32,
    n_halfedges: wp.int32,
) -> wp.vec3d:
    # ``CalculateVertNormals``: the angle-weighted pseudo-normal around the vertex from its least
    # outgoing halfedge in ``ForVert``'s order; zero for an unreferenced vertex. A halfedge with no
    # pair (removed by the topology passes) ends the walk.
    normal = wp.vec3d()
    if first >= n_halfedges:
        return normal
    current = first
    steps = wp.int32(0)
    while steps < n_halfedges:
        steps += 1
        p = pairs[current]
        if p < 0:
            break
        current = next_halfedge(p)
        a = starts[current]
        b = starts[next_halfedge(current)]
        c = starts[next_halfedge(next_halfedge(current))]
        curr_edge = wp.normalize(positions[b] - positions[a])
        prev_edge = wp.normalize(positions[a] - positions[c])
        if wp.isfinite(curr_edge[0]) and wp.isfinite(prev_edge[0]):
            d = wp.float64(0.0) - wp.dot(prev_edge, curr_edge)
            phi = wp.float64(0.0)
            if d <= wp.float64(-1.0):
                phi = wp.float64(3.141592653589793)
            elif d < wp.float64(1.0):
                phi = manifold_acos(d)
            normal = normal + phi * face_normals[current // 3]
        if current == first:
            break
    length = wp.length(normal)
    if length > wp.float64(0.0) and wp.isfinite(length):
        return normal / length
    return wp.vec3d()


@wp.kernel
def vertex_pseudo_normals(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    face_normals: wp.array[wp.vec3d],
    first_halfedge: wp.array[wp.int32],
    out_vertex_normals: wp.array[wp.vec3d],
) -> None:
    # Each vertex's pseudo-normal, on a solid's faces and twins.
    v = wp.int32(wp.tid())
    out_vertex_normals[v] = pseudo_normal(
        faces, twins, positions, face_normals, first_halfedge[v], faces.shape[0]
    )


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


# An operand of few edges crosses the other's faces pair by pair: one thread an (edge, face) pair
# instead of a thread an edge walking the face tree, which leaves a large operand's faces to a
# handful of threads when the few edges are long (a half-space cutter's). Flags scanned in edge-
# major, face-minor order place each edge's run sorted by face, as ``write_edge_crossings``'
# insertion leaves it; ``kernel12`` alone decides, so the box test only skips work.


@wp.func
def pair_crossing(
    p: wp.int32,
    n_faces_b: wp.int32,
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
    lowers_b: wp.array[wp.vec3],
    uppers_b: wp.array[wp.vec3],
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
) -> tuple[wp.int32, wp.vec3d]:
    h = p // n_faces_b
    face = p - h * n_faces_b
    start = faces_a[h]
    end = faces_a[next_halfedge(h)]
    if start >= end:
        return 0, wp.vec3d()
    lower, upper = edge_query_box(pos_a, start, end, pad)
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
    return kernel12(
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


@wp.kernel
def count_pair_crossings(
    n_faces_b: wp.int32,
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
    lowers_b: wp.array[wp.vec3],
    uppers_b: wp.array[wp.vec3],
    pad: wp.float32,
    expand: wp.int32,
    forward: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    p = wp.int32(wp.tid())
    x12, _v12 = pair_crossing(
        p,
        n_faces_b,
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
        lowers_b,
        uppers_b,
        pad,
        expand,
        forward,
    )
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
    x12, v12 = pair_crossing(
        p,
        n_faces_b,
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
        lowers_b,
        uppers_b,
        pad,
        expand,
        forward,
    )
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


# The flood in parallel rounds. Every seed not yet resolved floods tentatively from the claims
# made so far: claims only ever block a flood, so the region it will take is a subset of this one.
# It bids its rank on every face of the region; a seed whose region holds no lower bid takes
# exactly that region in the serial order, and the regions such seeds take are disjoint, so they
# commit together. A neighbour the flood rejects needs no bid: the plane test rejects it whether or
# not it is claimed. A region past ``COPLANAR_REGION`` faces stops the round's later seeds, and
# when it is the earliest left, the serial flood takes it alone.

COPLANAR_REGION = 64
COPLANAR_BID = 0
COPLANAR_CHECK = 1
COPLANAR_COMMIT = 2


@wp.func
def tentative_flood(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    order: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    is_seed: wp.array[wp.int32],
    tolerance: wp.float64,
    face_normals: wp.array[wp.vec3d],
    coplanar: wp.array[wp.int32],
    bids: wp.array[wp.int32],
    scratch: wp.array[wp.int32],
    slot: wp.int32,
    k: wp.int32,
    mode: wp.int32,
) -> wp.int32:
    # One seed's flood as ``flood_coplanar`` walks it, its region and stack in the thread's
    # scratch. ``COPLANAR_BID`` bids ``k`` on the region, ``COPLANAR_CHECK`` returns 1 when no
    # face of it holds a lower bid, ``COPLANAR_COMMIT`` claims it. Returns -1 on overflow.
    seed = order[k]
    region = slot * 4 * COPLANAR_REGION
    stack = region + COPLANAR_REGION
    base = positions[faces[3 * seed]]
    normal = face_normals[seed]
    n_region = wp.int32(1)
    scratch[region] = seed
    scratch[stack] = 3 * seed
    scratch[stack + 1] = 3 * seed + 1
    scratch[stack + 2] = 3 * seed + 2
    top = wp.int32(3)
    while top > 0:
        h = next_halfedge(twins[scratch[stack + top - 1]])
        top -= 1
        tri = h // 3
        if coplanar[tri] >= 0:
            continue
        taken = wp.int32(0)
        for j in range(n_region):
            if scratch[region + j] == tri:
                taken = 1
        if taken != 0:
            continue
        if is_seed[rank[tri]] == 0 and rank[tri] < k:
            continue
        v = positions[faces[next_halfedge(h)]]
        if wp.abs(wp.dot(v - base, normal)) < tolerance:
            if n_region == COPLANAR_REGION or top + 2 > 3 * COPLANAR_REGION:
                return -1
            scratch[region + n_region] = tri
            n_region += 1
            if top == 0 or h != twins[scratch[stack + top - 1]]:
                scratch[stack + top] = h
                top += 1
            else:
                top -= 1
            scratch[stack + top] = next_halfedge(h)
            top += 1
    ok = wp.int32(1)
    for j in range(n_region):
        tri = scratch[region + j]
        if mode == COPLANAR_BID:
            wp.atomic_min(bids, tri, k)
        elif mode == COPLANAR_CHECK:
            if bids[tri] < k:
                ok = 0
        else:
            coplanar[tri] = seed
            if tri != seed:
                face_normals[tri] = normal
    return ok


@wp.kernel
def coplanar_round_bids(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    order: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    is_seed: wp.array[wp.int32],
    tolerance: wp.float64,
    face_normals: wp.array[wp.vec3d],
    coplanar: wp.array[wp.int32],
    pending: wp.array[wp.int32],
    bids: wp.array[wp.int32],
    scratch: wp.array[wp.int32],
    out_first_overflow: wp.array[wp.int32],
) -> None:
    # Each pending seed's tentative region, bid on; a seed an earlier commit claimed is skipped.
    i = wp.int32(wp.tid())
    k = pending[i]
    if coplanar[order[k]] >= 0:
        return
    result = tentative_flood(
        faces,
        twins,
        positions,
        order,
        rank,
        is_seed,
        tolerance,
        face_normals,
        coplanar,
        bids,
        scratch,
        i,
        k,
        COPLANAR_BID,
    )
    if result < 0:
        wp.atomic_min(out_first_overflow, 0, k)


@wp.kernel
def coplanar_round_checks(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    order: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    is_seed: wp.array[wp.int32],
    tolerance: wp.float64,
    face_normals: wp.array[wp.vec3d],
    coplanar: wp.array[wp.int32],
    pending: wp.array[wp.int32],
    bids: wp.array[wp.int32],
    scratch: wp.array[wp.int32],
    first_overflow: wp.array[wp.int32],
    out_commit: wp.array[wp.int32],
) -> None:
    # Whether each seed before the first overflow holds the lowest bid on all of its region.
    i = wp.int32(wp.tid())
    k = pending[i]
    commit = wp.int32(0)
    if coplanar[order[k]] < 0 and k < first_overflow[0]:
        commit = tentative_flood(
            faces,
            twins,
            positions,
            order,
            rank,
            is_seed,
            tolerance,
            face_normals,
            coplanar,
            bids,
            scratch,
            i,
            k,
            COPLANAR_CHECK,
        )
    out_commit[i] = wp.max(commit, 0)


@wp.kernel
def coplanar_round_commits(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    order: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    is_seed: wp.array[wp.int32],
    tolerance: wp.float64,
    face_normals: wp.array[wp.vec3d],
    coplanar: wp.array[wp.int32],
    pending: wp.array[wp.int32],
    bids: wp.array[wp.int32],
    scratch: wp.array[wp.int32],
    commit: wp.array[wp.int32],
    out_left: wp.array[wp.int32],
) -> None:
    # The checked seeds claim their disjoint regions; the rest stay pending unless already
    # claimed.
    i = wp.int32(wp.tid())
    k = pending[i]
    left = wp.int32(0)
    if commit[i] != 0:
        tentative_flood(
            faces,
            twins,
            positions,
            order,
            rank,
            is_seed,
            tolerance,
            face_normals,
            coplanar,
            bids,
            scratch,
            i,
            k,
            COPLANAR_COMMIT,
        )
    elif coplanar[order[k]] < 0:
        left = 1
    out_left[i] = left


@wp.kernel
def compact_pending(
    pending: wp.array[wp.int32], offsets: wp.array[wp.int32], out_pending: wp.array[wp.int32]
) -> None:
    # The seeds still pending, in order: those whose scanned flag steps the offsets.
    i = wp.int32(wp.tid())
    if offsets[i + 1] != offsets[i]:
        out_pending[offsets[i]] = pending[i]


@wp.kernel
def own_coplanar_groups(coplanar: wp.array[wp.int32]) -> None:
    # A face no flood reached is its own group.
    f = wp.int32(wp.tid())
    if coplanar[f] < 0:
        coplanar[f] = f


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
def hm_vertex_normals(
    m: HalfedgeMesh, first: wp.array[wp.int32], out_normals: wp.array[wp.vec3d]
) -> None:
    # Each vertex's pseudo-normal on the simplified result, with the faces' inherited normals.
    v = wp.int32(wp.tid())
    out_normals[v] = pseudo_normal(m.start, m.pair, m.pos, m.tri_normal, first[v], m.sizes[0])


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


# -- cross-sections (``Impl::Slice``) -------------------------------------------------------------


# -- float64 vector helpers, in linalg's operation order -----------------------------------------


@wp.func
def dot3(a: wp.vec3d, b: wp.vec3d) -> wp.float64:
    # linalg's ``dot``: ``wp.dot`` starts from ``0 + a[0] * b[0]``, which turns a first term of
    # ``-0.0`` into ``+0.0`` where Manifold keeps it (``cross``, ``lerp``, ``*`` and ``/`` round
    # as linalg's and are Warp's own).
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@wp.func
def negated3(a: wp.vec3d) -> wp.vec3d:
    return (a) * (wp.float64(-1.0))


@wp.func
def la_normalize(a: wp.vec3d) -> wp.vec3d:
    # ``la::normalize``: divided by its length, NaN for zero.
    return a / wp.sqrt(dot3(a, a))


@wp.func
def safe_normalize(a: wp.vec3d) -> wp.vec3d:
    # ``SafeNormalize``: zero where the division is not finite.
    n = la_normalize(a)
    if wp.isfinite(n[0]):
        return n
    return wp.vec3d()


@wp.func
def xyz(v: wp.vec4d) -> wp.vec3d:
    return wp.vec3d(v[0], v[1], v[2])


# -- refinement (``Subdivide``) --------------------------------------------------------------------
#
# A halfedge flagged ``inside`` is the diagonal of a quad (two triangles of one face): the quad is
# refined as one face by its lower triangle, and the diagonal takes no vertices. Plain refinement
# flags none.


@wp.kernel
def forward_halfedge_flags(faces: wp.array[wp.int32], out_flags: wp.array[wp.int32]) -> None:
    # ``CreateTmpEdges``: an edge is its forward halfedge, ranked in halfedge order.
    h = wp.int32(wp.tid())
    out_flags[h] = wp.where(faces[h] < faces[next_halfedge(h)], 1, 0)


@wp.func
def refine_edge_of(
    faces: wp.array[wp.int32], twins: wp.array[wp.int32], ranks: wp.array[wp.int32], h: wp.int32
) -> wp.int32:
    # ``half2Edge``: the rank of the halfedge's forward copy.
    forward = h
    if faces[h] > faces[next_halfedge(h)]:
        forward = twins[h]
    return ranks[forward]


@wp.func
def quad_neighbor(inside: wp.array[wp.int32], tri: wp.int32) -> wp.int32:
    # ``GetNeighbor``: the side shared with the quad's other triangle, -1 for none, -2 for two.
    neighbor = wp.int32(-1)
    for i in range(3):
        if inside[3 * tri + i] != 0:
            neighbor = wp.where(neighbor == -1, wp.int32(i), wp.int32(-2))
    return neighbor


@wp.func
def face_halfedges(
    inside: wp.array[wp.int32], twins: wp.array[wp.int32], tri: wp.int32
) -> wp.vec4i:
    # ``GetHalfedges``: a triangle's three halfedges and -1, a quad's four from its lower
    # triangle, or all -1 for a quad's upper triangle.
    halfedges = wp.vec4i(3 * tri, 3 * tri + 1, 3 * tri + 2, -1)
    neighbor = quad_neighbor(inside, tri)
    if neighbor >= 0:
        pair = twins[3 * tri + neighbor]
        if pair // 3 < tri:
            return wp.vec4i(-1, -1, -1, -1)
        h2 = next_halfedge(3 * tri + neighbor)
        h0 = next_halfedge(pair)
        halfedges = wp.vec4i(h0, next_halfedge(h0), h2, next_halfedge(h2))
    return halfedges


@wp.func
def bary_indices(inside: wp.array[wp.int32], twins: wp.array[wp.int32], h: wp.int32) -> wp.vec3i:
    # ``GetIndices``: the face (lower triangle for a quad) and the halfedge's start and end corner
    # in it; -1s for a quad's diagonal.
    tri = h // 3
    idx = h % 3
    neighbor = quad_neighbor(inside, tri)
    if idx == neighbor:
        return wp.vec3i(-1, -1, -1)
    if neighbor < 0:
        return wp.vec3i(tri, idx, next3(idx))
    pair = twins[3 * tri + neighbor]
    if pair // 3 < tri:
        tri = pair // 3
        idx = wp.where(next3(neighbor) == idx, 0, 1)
    else:
        idx = wp.where(next3(neighbor) == idx, 2, 3)
    return wp.vec3i(tri, idx, (idx + 1) % 4)


@wp.kernel
def refine_edge_added(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    tangents: wp.array[wp.vec4d],
    added: wp.int32,
    length: wp.float64,
    tolerance: wp.float64,
    out_added: wp.array[wp.int32],
    out_edge_halfedge: wp.array[wp.int32],
) -> None:
    # Each edge's added vertices, by a warp-uniform selector: ``Refine``'s ``n - 1``,
    # ``RefineToLength``'s truncated ``length(edge) / length`` when ``length`` is positive, or
    # ``RefineToTolerance``'s count from the tangents' bulge when ``tolerance`` is; none on a
    # quad's diagonal.
    h = wp.int32(wp.tid())
    start = faces[h]
    end = faces[next_halfedge(h)]
    if start >= end:
        return
    e = ranks[h]
    out_edge_halfedge[e] = h
    if inside[h] != 0:
        return
    count = added
    d = positions[start] - positions[end]
    if length > wp.float64(0.0):
        count = wp.int32(wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / length)
    elif tolerance > wp.float64(0.0):
        edge_norm = la_normalize(d)
        t_start = xyz(tangents[h])
        t_end = xyz(tangents[twins[h]])
        a = t_start - (edge_norm) * (dot3(edge_norm, t_start))
        b = t_end - (edge_norm) * (dot3(edge_norm, t_end))
        ab = a - b
        bulge = wp.float64(0.5) * (wp.sqrt(dot3(a, a)) + wp.sqrt(dot3(b, b))) + wp.sqrt(
            dot3(ab, ab)
        )
        # A crease corner's ``0 / 0`` smoothness leaves NaN tangents, whose count Manifold
        # converts without a check (undefined, and in practice unbounded); none are added here.
        count = 0
        if wp.isfinite(bulge):
            count = wp.int32(wp.sqrt(wp.float64(3.0) * bulge / (wp.float64(4.0) * tolerance)))
    out_added[e] = count


@wp.func
def interior_extra(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    added: wp.array[wp.int32],
    h: wp.int32,
    this_added: wp.int32,
) -> wp.int32:
    # ``Subdivide``'s ``Added``: the extra divisions a short side of ``h``'s triangle takes so a
    # long side's strip has some thickness.
    longest = wp.int32(0)
    total = wp.int32(0)
    current = h
    for _ in range(3):
        a = added[refine_edge_of(faces, twins, ranks, current)]
        longest = wp.max(longest, a)
        total = total + a
        current = next_halfedge(current)
        if inside[current] != 0:
            longest = 0
            total = 1
            break
    min_extra = wp.int32(wp.float64(longest) * wp.float64(0.2) + wp.float64(1.0))
    extra = 2 * longest + min_extra - total
    if longest == 0:
        return 0
    if extra > 0:
        return (extra * (longest - this_added)) // longest
    return 0


@wp.kernel
def keep_interior_added(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    added: wp.array[wp.int32],
    out_added: wp.array[wp.int32],
) -> None:
    # ``keepInterior``: each edge's count raised by the larger of its two triangles' extras.
    h = wp.int32(wp.tid())
    if faces[h] >= faces[next_halfedge(h)]:
        return
    e = ranks[h]
    this_added = added[e]
    if inside[h] != 0:
        out_added[e] = this_added
        return
    extra = wp.max(
        interior_extra(faces, twins, ranks, inside, added, h, this_added),
        interior_extra(faces, twins, ranks, inside, added, twins[h], this_added),
    )
    out_added[e] = this_added + extra


@wp.func
def partition_sides(d: wp.vec4i) -> wp.vec4i:
    # ``Partition::GetPartition``: which side each sorted division came from. A triangle's sides
    # sort by divisions, largest first; a quad's rotate to start at its least side.
    idx = wp.vec4i(0, 1, 2, 3)
    if d[0] == 0:
        return idx
    if d[3] == 0:
        s = wp.vec4i(d[0], d[1], d[2], 0)
        if s[2] > s[1]:
            t = s[2]
            s[2] = s[1]
            s[1] = t
            t = idx[2]
            idx[2] = idx[1]
            idx[1] = t
        if s[1] > s[0]:
            t = s[1]
            s[1] = s[0]
            s[0] = t
            t = idx[1]
            idx[1] = idx[0]
            idx[0] = t
            if s[2] > s[1]:
                t = s[2]
                s[2] = s[1]
                s[1] = t
                t = idx[2]
                idx[2] = idx[1]
                idx[1] = t
        return idx
    min_idx = wp.int32(0)
    least = d[0]
    after = d[1]
    for i in range(1, 4):
        n = d[(i + 1) % 4]
        if d[i] < least or (d[i] == least and n < after):
            min_idx = i
            least = d[i]
            after = n
    for i in range(4):
        idx[i] = (i + min_idx) % 4
    return idx


@wp.func
def refine_face_divisions(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    added_offsets: wp.array[wp.int32],
    halfedges: wp.vec4i,
) -> wp.vec4i:
    d = wp.vec4i(0, 0, 0, 0)
    for i in range(4):
        h = halfedges[i]
        if h >= 0:
            e = refine_edge_of(faces, twins, ranks, h)
            d[i] = added_offsets[e + 1] - added_offsets[e] + 1
    return d


@wp.kernel
def refine_face_keys(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    added_offsets: wp.array[wp.int32],
    out_keys: wp.array[wp.vec4i],
) -> None:
    # Each face's sorted divisions, the partition it takes (all zero for a quad's upper triangle).
    f = wp.int32(wp.tid())
    d = refine_face_divisions(faces, twins, ranks, added_offsets, face_halfedges(inside, twins, f))
    idx = partition_sides(d)
    out_keys[f] = wp.vec4i(d[idx[0]], d[idx[1]], d[idx[2]], d[idx[3]])


@wp.kernel
def last_corner_halfedges(
    faces: wp.array[wp.int32], inside: wp.array[wp.int32], out_last: wp.array[wp.int32]
) -> None:
    # ``FillRetainedVerts``: the last face through a vertex, by its halfedge, names the vertex's
    # barycentric; a quad's diagonal names none.
    h = wp.int32(wp.tid())
    if inside[h] == 0:
        wp.atomic_max(out_last, faces[h], h)


@wp.kernel
def refine_retained_bary(
    twins: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    last: wp.array[wp.int32],
    out_tri: wp.array[wp.int32],
    out_uvw: wp.array[wp.vec4d],
) -> None:
    v = wp.int32(wp.tid())
    indices = bary_indices(inside, twins, last[v])
    uvw = wp.vec4d()
    uvw[indices[1]] = wp.float64(1.0)
    out_tri[v] = indices[0]
    out_uvw[v] = uvw


@wp.kernel
def refine_edge_bary(
    twins: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    edge_halfedge: wp.array[wp.int32],
    added_offsets: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_tri: wp.array[wp.int32],
    out_uvw: wp.array[wp.vec4d],
) -> None:
    # An edge's added vertices, evenly along it from its forward halfedge's start.
    e = wp.int32(wp.tid())
    first = added_offsets[e]
    n = added_offsets[e + 1] - first
    indices = bary_indices(inside, twins, edge_halfedge[e])
    if indices[0] < 0:
        return
    frac = wp.float64(1.0) / wp.float64(n + 1)
    for i in range(n):
        uvw = wp.vec4d()
        uvw[indices[2]] = wp.float64(i + 1) * frac
        uvw[indices[1]] = wp.float64(1.0) - uvw[indices[2]]
        out_tri[n_vertices + first + i] = indices[0]
        out_uvw[n_vertices + first + i] = uvw


@wp.func
def refine_local_vertex(
    v: wp.int32,
    n_corners: wp.int32,
    corners: wp.vec4i,
    sorted_divisions: wp.vec4i,
    idx: wp.vec4i,
    edge_offsets: wp.vec4i,
    edge_forward: wp.vec4i,
    interior_offset: wp.int32,
) -> wp.int32:
    # ``Partition::Reindex``'s ``newVerts``: the corners, each sorted side's added vertices in its
    # direction, then the interior.
    if v < n_corners:
        return corners[v]
    base = n_corners
    for i in range(4):
        n = sorted_divisions[i] - 1
        if n > 0:
            if v < base + n:
                j = v - base
                side = idx[i]
                if edge_forward[side] != 0:
                    return edge_offsets[side] + j
                return edge_offsets[side] + n - 1 - j
            base = base + n
    return interior_offset + v - base


@wp.kernel
def refine_faces(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    added_offsets: wp.array[wp.int32],
    n_vertices: wp.int32,
    face_key: wp.array[wp.int32],
    key_triangles: wp.array[wp.int32],
    key_vertices: wp.array[wp.int32],
    partition_triangles: wp.array[wp.vec3i],
    partition_bary: wp.array[wp.vec4d],
    triangle_offsets: wp.array[wp.int32],
    interior_offsets: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3d],
    face_mesh: wp.array[wp.int32],
    face_coplanar: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3d],
    out_mesh: wp.array[wp.int32],
    out_coplanar: wp.array[wp.int32],
    out_tri: wp.array[wp.int32],
    out_uvw: wp.array[wp.vec4d],
) -> None:
    # ``Subdivide``'s per-face pass: the cached partition reindexed onto the face's corners,
    # sides and interior, each sub-triangle inheriting the triangle's normal and reference, and
    # the interior vertices' barycentrics in the face.
    f = wp.int32(wp.tid())
    halfedges = face_halfedges(inside, twins, f)
    if halfedges[0] < 0:
        return
    d = refine_face_divisions(faces, twins, ranks, added_offsets, halfedges)
    idx = partition_sides(d)
    sorted_divisions = wp.vec4i(d[idx[0]], d[idx[1]], d[idx[2]], d[idx[3]])
    corners = wp.vec4i(-1, -1, -1, -1)
    edge_offsets = wp.vec4i(0, 0, 0, 0)
    edge_forward = wp.vec4i(0, 0, 0, 0)
    for i in range(4):
        h = halfedges[i]
        if h >= 0:
            corners[i] = faces[h]
            edge_offsets[i] = n_vertices + added_offsets[refine_edge_of(faces, twins, ranks, h)]
            edge_forward[i] = wp.where(faces[h] < faces[next_halfedge(h)], 1, 0)
    is_quad = halfedges[3] >= 0
    n_corners = wp.where(is_quad, 4, 3)
    tri_idx = idx
    out0 = wp.int32(0)
    out1 = wp.int32(1)
    if (not is_quad) and idx[1] != next3(idx[0]):
        tri_idx = wp.vec4i(idx[2], idx[0], idx[1], idx[3])
        for i in range(4):
            edge_forward[i] = 1 - edge_forward[i]
        out0 = 1
        out1 = 0
    ordered = wp.vec4i(-1, -1, -1, -1)
    k = wp.int32(0)
    for i in range(4):
        c = corners[tri_idx[i]]
        if c >= 0:
            ordered[k] = c
            k = k + 1
    key = face_key[f]
    tri_first = key_triangles[key]
    tri_count = key_triangles[key + 1] - tri_first
    interior = interior_offsets[f]
    out_first = triangle_offsets[f]
    for j in range(tri_count):
        local = partition_triangles[tri_first + j]
        o = 3 * (out_first + j)
        out_faces[o + out0] = refine_local_vertex(
            local[0],
            n_corners,
            ordered,
            sorted_divisions,
            idx,
            edge_offsets,
            edge_forward,
            interior,
        )
        out_faces[o + out1] = refine_local_vertex(
            local[1],
            n_corners,
            ordered,
            sorted_divisions,
            idx,
            edge_offsets,
            edge_forward,
            interior,
        )
        out_faces[o + 2] = refine_local_vertex(
            local[2],
            n_corners,
            ordered,
            sorted_divisions,
            idx,
            edge_offsets,
            edge_forward,
            interior,
        )
        out_normals[out_first + j] = face_normals[f]
        out_mesh[out_first + j] = face_mesh[f]
        out_coplanar[out_first + j] = face_coplanar[f]
    v_idx = idx
    if (not is_quad) and idx[1] != next3(idx[0]):
        v_idx = wp.vec4i(idx[2], idx[0], idx[1], idx[3])
    r_idx = wp.vec4i(0, 0, 0, 0)
    for i in range(4):
        r_idx[v_idx[i]] = i
    vert_first = key_vertices[key]
    vert_count = key_vertices[key + 1] - vert_first
    boundary = sorted_divisions[0] + sorted_divisions[1] + sorted_divisions[2] + sorted_divisions[3]
    for s in range(vert_count - boundary):
        b = partition_bary[vert_first + boundary + s]
        out_tri[interior + s] = f
        out_uvw[interior + s] = wp.vec4d(b[r_idx[0]], b[r_idx[1]], b[r_idx[2]], b[r_idx[3]])


# -- smoothing (``SmoothOut``'s tangents and ``InterpTri``) ----------------------------------------
#
# Manifold's own ``sin`` and ``cos`` (``include/manifold/math.h``, fdlibm's) for the rotations, and
# exact negation (``-1 *``, where kernel-scope unary minus is ``0 - x`` and loses a negative zero).

SMOOTH_PI = wp.float64(3.14159265358979323846264338327950288)
SMOOTH_TWO_PI = wp.float64(6.28318530717958647692528676655900576)
SMOOTH_PRECISION = wp.float64(1e-12)
SMOOTH_DBL_EPSILON = wp.float64(2.220446049250313e-16)

# A Bezier's homogeneous point and its cross tangent, as rows.
mat24d = wp.types.matrix(shape=(2, 4), dtype=wp.float64)


@wp.func
def manifold_sin_kernel(x: wp.float64, y: wp.float64, iy: wp.int32) -> wp.float64:
    z = x * x
    w = z * z
    r = (
        wp.float64(8.33333333332248946124e-03)
        + z * (wp.float64(-1.98412698298579493134e-04) + z * wp.float64(2.75573137070700676789e-06))
        + z
        * w
        * (wp.float64(-2.50507602534068634195e-08) + z * wp.float64(1.58969099521155010221e-10))
    )
    v = z * x
    if iy == 0:
        return x + v * (wp.float64(-1.66666666666666324348e-01) + z * r)
    return x - (
        (z * (wp.float64(0.5) * y - v * r) - y) - v * wp.float64(-1.66666666666666324348e-01)
    )


@wp.func
def manifold_cos_kernel(x: wp.float64, y: wp.float64) -> wp.float64:
    z = x * x
    w = z * z
    r = z * (
        wp.float64(4.16666666666666019037e-02)
        + z * (wp.float64(-1.38888888888741095749e-03) + z * wp.float64(2.48015872894767294178e-05))
    ) + w * w * (
        wp.float64(-2.75573143513906633035e-07)
        + z * (wp.float64(2.08757232129817482790e-09) + z * wp.float64(-1.13596475577881948265e-11))
    )
    hz = wp.float64(0.5) * z
    w1 = wp.float64(1.0) - hz
    return w1 + (((wp.float64(1.0) - w1) - hz) + (z * r - x * y))


@wp.func
def high_word(x: wp.float64) -> wp.uint32:
    return wp.uint32((wp.cast(x, wp.uint64) >> wp.uint64(32)) & wp.uint64(0x7FFFFFFF))


@wp.func
def exponent_bits(x: wp.float64) -> wp.int32:
    return wp.int32((wp.cast(x, wp.uint64) >> wp.uint64(52)) & wp.uint64(0x7FF))


@wp.func
def rem_pio2(x: wp.float64) -> wp.vec3d:
    # ``RemPio2`` below ``2**20 * pi / 2``: ``(y0, y1, n)``, ``x = n pi / 2 + y0 + y1``.
    pio4 = wp.float64(0.78539816339744827900)
    invpio2 = wp.float64(6.36619772367581382433e-01)
    pio2_1 = wp.float64(1.57079632673412561417e00)
    pio2_1t = wp.float64(6.07710050650619224932e-11)
    pio2_2 = wp.float64(6.07710050630396597660e-11)
    pio2_2t = wp.float64(2.02226624879595063154e-21)
    pio2_3 = wp.float64(2.02226624871116645580e-21)
    pio2_3t = wp.float64(8.47842766036889956997e-32)
    toint = wp.float64(6755399441055744.0)
    ix = high_word(x)
    negative = (wp.cast(x, wp.uint64) >> wp.uint64(63)) != wp.uint64(0)
    medium = False
    k = wp.float64(0.0)
    if ix <= wp.uint32(0x400F6A7A):
        if (ix & wp.uint32(0xFFFFF)) == wp.uint32(0x921FB):
            medium = True
        elif ix <= wp.uint32(0x4002D97C):
            k = wp.float64(1.0)
        else:
            k = wp.float64(2.0)
    elif ix <= wp.uint32(0x401C463B):
        if ix <= wp.uint32(0x4015FDBC):
            if ix == wp.uint32(0x4012D97C):
                medium = True
            else:
                k = wp.float64(3.0)
        elif ix == wp.uint32(0x401921FB):
            medium = True
        else:
            k = wp.float64(4.0)
    else:
        medium = True
    if not medium:
        if not negative:
            z = x - k * pio2_1
            y0 = z - k * pio2_1t
            return wp.vec3d(y0, (z - y0) - k * pio2_1t, k)
        z = x + k * pio2_1
        y0 = z + k * pio2_1t
        return wp.vec3d(y0, (z - y0) + k * pio2_1t, wp.float64(-1.0) * k)
    fn = (x * invpio2 + toint) - toint
    n = wp.float64(wp.int32(fn))
    r = x - fn * pio2_1
    w = fn * pio2_1t
    if r - w < wp.float64(-1.0) * pio4:
        n = n - wp.float64(1.0)
        fn = fn - wp.float64(1.0)
        r = x - fn * pio2_1
        w = fn * pio2_1t
    elif r - w > pio4:
        n = n + wp.float64(1.0)
        fn = fn + wp.float64(1.0)
        r = x - fn * pio2_1
        w = fn * pio2_1t
    y0 = r - w
    ex = wp.int32(ix >> wp.uint32(20))
    if ex - exponent_bits(y0) > 16:
        t = r
        w = fn * pio2_2
        r = t - w
        w = fn * pio2_2t - ((t - r) - w)
        y0 = r - w
        if ex - exponent_bits(y0) > 49:
            t2 = r
            w = fn * pio2_3
            r = t2 - w
            w = fn * pio2_3t - ((t2 - r) - w)
            y0 = r - w
    return wp.vec3d(y0, (r - y0) - w, n)


@wp.func
def manifold_sin(x: wp.float64) -> wp.float64:
    ix = high_word(x)
    if ix <= wp.uint32(0x3FE921FB):
        if ix < wp.uint32(0x3E500000):
            return x
        return manifold_sin_kernel(x, wp.float64(0.0), 0)
    if ix >= wp.uint32(0x7FF00000):
        return x - x
    y = rem_pio2(x)
    q = wp.int32(y[2]) & 3
    if q == 0:
        return manifold_sin_kernel(y[0], y[1], 1)
    if q == 1:
        return manifold_cos_kernel(y[0], y[1])
    if q == 2:
        return wp.float64(-1.0) * manifold_sin_kernel(y[0], y[1], 1)
    return wp.float64(-1.0) * manifold_cos_kernel(y[0], y[1])


@wp.func
def manifold_cos(x: wp.float64) -> wp.float64:
    ix = high_word(x)
    if ix <= wp.uint32(0x3FE921FB):
        if ix < wp.uint32(0x3E46A09E):
            return wp.float64(1.0)
        return manifold_cos_kernel(x, wp.float64(0.0))
    if ix >= wp.uint32(0x7FF00000):
        return x - x
    y = rem_pio2(x)
    q = wp.int32(y[2]) & 3
    if q == 0:
        return manifold_cos_kernel(y[0], y[1])
    if q == 1:
        return wp.float64(-1.0) * manifold_sin_kernel(y[0], y[1], 1)
    if q == 2:
        return wp.float64(-1.0) * manifold_cos_kernel(y[0], y[1])
    return manifold_sin_kernel(y[0], y[1], 1)


@wp.func
def angle_between(a: wp.vec3d, b: wp.vec3d) -> wp.float64:
    # ``AngleBetween`` of unit vectors.
    d = dot3(a, b)
    if d >= wp.float64(1.0):
        return wp.float64(0.0)
    if d <= wp.float64(-1.0):
        return SMOOTH_PI
    return manifold_acos(d)


@wp.func
def circular_tangent(tangent: wp.vec3d, edge: wp.vec3d) -> wp.vec4d:
    # ``CircularTangent``: the weighted cubic Bezier tangent of a circular arc along ``edge``.
    direction = safe_normalize(tangent)
    weight = wp.max(wp.float64(0.5), dot3(direction, safe_normalize(edge)))
    bz2 = ((direction) * (wp.float64(0.5))) * (wp.sqrt(dot3(edge, edge)))
    t = wp.float64(2.0) / wp.float64(3.0)
    bz3 = wp.lerp(
        wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)),
        wp.vec4d(bz2[0], bz2[1], bz2[2], weight),
        t,
    )
    return wp.vec4d(bz3[0] / bz3[3], bz3[1] / bz3[3], bz3[2] / bz3[3], bz3[3])


@wp.func
def same_face(
    mesh: wp.array[wp.int32], coplanar: wp.array[wp.int32], a: wp.int32, b: wp.int32
) -> wp.bool:
    # ``TriRef::SameFace`` (no face ids on an imported mesh).
    return mesh[a] == mesh[b] and coplanar[a] == coplanar[b]


@wp.func
def is_inside_quad(
    twins: wp.array[wp.int32], mesh: wp.array[wp.int32], coplanar: wp.array[wp.int32], h: wp.int32
) -> wp.bool:
    # ``IsInsideQuad`` before tangents: the halfedge joins two triangles of one face, and neither
    # has another neighbour in it.
    tri = h // 3
    pair = twins[h]
    pair_tri = pair // 3
    if not same_face(mesh, coplanar, tri, pair_tri):
        return False
    n1 = next_halfedge(h)
    if same_face(mesh, coplanar, tri, twins[n1] // 3):
        return False
    if same_face(mesh, coplanar, tri, twins[next_halfedge(n1)] // 3):
        return False
    m1 = next_halfedge(pair)
    if same_face(mesh, coplanar, pair_tri, twins[m1] // 3):
        return False
    if same_face(mesh, coplanar, pair_tri, twins[next_halfedge(m1)] // 3):
        return False
    return True


@wp.kernel
def flat_faces(
    twins: wp.array[wp.int32],
    mesh: wp.array[wp.int32],
    coplanar: wp.array[wp.int32],
    out_flat: wp.array[wp.int32],
) -> None:
    # ``FlatFaces``: a triangle with two or more neighbours in its face, and those neighbours.
    tri = wp.int32(wp.tid())
    count = wp.int32(0)
    for j in range(3):
        if same_face(mesh, coplanar, tri, twins[3 * tri + j] // 3):
            count += 1
    if count > 1:
        out_flat[tri] = 1
        for j in range(3):
            other = twins[3 * tri + j] // 3
            if same_face(mesh, coplanar, tri, other):
                out_flat[other] = 1


@wp.kernel
def smoothing_vertex_normals(
    twins: wp.array[wp.int32],
    mesh: wp.array[wp.int32],
    coplanar: wp.array[wp.int32],
    flat: wp.array[wp.int32],
    first_halfedge: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3d],
    vertex_normals: wp.array[wp.vec3d],
    out_normals: wp.array[wp.vec3d],
) -> None:
    # ``VertFlatFace``: a vertex on one flat face only takes that face's normal (the face's
    # first triangle in index order), otherwise its pseudo-normal.
    v = wp.int32(wp.tid())
    first = first_halfedge[v]
    least = wp.int32(2147483647)
    current = first
    steps = wp.int32(0)
    while steps < twins.shape[0]:
        current = next_halfedge(twins[current])
        if flat[current // 3] != 0:
            least = wp.min(least, current // 3)
        steps += 1
        if current == first:
            break
    normal = vertex_normals[v]
    if least < 2147483647:
        one = wp.int32(1)
        current = first
        steps = wp.int32(0)
        while steps < twins.shape[0]:
            current = next_halfedge(twins[current])
            tri = current // 3
            if flat[tri] != 0 and not same_face(mesh, coplanar, tri, least):
                one = 0
            steps += 1
            if current == first:
                break
        if one != 0:
            normal = face_normals[least]
    out_normals[v] = normal


@wp.kernel
def initial_tangents(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    mesh: wp.array[wp.int32],
    coplanar: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3d],
    normals: wp.array[wp.vec3d],
    out_tangents: wp.array[wp.vec4d],
) -> None:
    # A quad's diagonal is marked ``w = -1``; every other halfedge takes ``TangentFromNormal``:
    # the circular arc leaving its start orthogonal to the vertex's normal.
    h = wp.int32(wp.tid())
    if is_inside_quad(twins, mesh, coplanar, h):
        out_tangents[h] = wp.vec4d(
            wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(-1.0)
        )
        return
    start = faces[h]
    normal = normals[start]
    edge = positions[faces[next_halfedge(h)]] - positions[start]
    edge_normal = face_normals[h // 3] + face_normals[twins[h] // 3]
    bi_tangent = wp.cross(normal, edge)
    if dot3(normal, edge_normal) < wp.float64(0.0):
        bi_tangent = wp.cross(edge_normal, edge)
    out_tangents[h] = circular_tangent(wp.cross(bi_tangent, normal), edge)


@wp.kernel
def edge_smoothness(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    mesh: wp.array[wp.int32],
    coplanar: wp.array[wp.int32],
    flat: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3d],
    min_radians: wp.float64,
    min_smoothness: wp.float64,
    out_present: wp.array[wp.int32],
    out_smoothness: wp.array[wp.vec2d],
) -> None:
    # ``SharpenEdges`` and ``CreateTangents``' edge map, per forward halfedge: whether the edge is
    # sharpened, and each side's least smoothness (1 where none). An edge is sharp past the
    # dihedral angle; a flat face's rim is sharp (0) on the face's side.
    f = wp.int32(wp.tid())
    if faces[f] > faces[next_halfedge(f)]:
        return
    p = twins[f]
    tf = f // 3
    tp = p // 3
    present = wp.int32(0)
    sf = wp.float64(1.0)
    sp = wp.float64(1.0)
    if angle_between(
        face_normals[tf], face_normals[tp]
    ) > min_radians and min_smoothness < wp.float64(1.0):
        present = 1
        sf = wp.min(sf, min_smoothness)
        sp = wp.min(sp, min_smoothness)
    same = same_face(mesh, coplanar, tf, tp)
    if flat[tf] != 0 and (flat[tp] == 0 or not same):
        present = 1
        sf = wp.min(sf, wp.float64(0.0))
    if flat[tp] != 0 and (flat[tf] == 0 or not same):
        present = 1
        sp = wp.min(sp, wp.float64(0.0))
    out_present[f] = present
    out_smoothness[f] = wp.vec2d(sf, sp)


@wp.func
def sharpen_tangent(tangents: wp.array[wp.vec4d], h: wp.int32, smoothness: wp.float64) -> None:
    # ``SharpenTangent``: the tangent scaled, its weight zero only for zero smoothness.
    t = tangents[h]
    w = t[3]
    if smoothness == wp.float64(0.0):
        w = wp.float64(0.0)
    tangents[h] = wp.vec4d(smoothness * t[0], smoothness * t[1], smoothness * t[2], w)


@wp.func
def forward_of(faces: wp.array[wp.int32], twins: wp.array[wp.int32], h: wp.int32) -> wp.int32:
    if faces[h] < faces[next_halfedge(h)]:
        return h
    return twins[h]


@wp.func
def outgoing_smoothness(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    smoothness: wp.array[wp.vec2d],
    h: wp.int32,
) -> wp.vec2d:
    # The edge's smoothness on the side of outgoing ``h``, then the other side.
    f = forward_of(faces, twins, h)
    s = smoothness[f]
    if f == h:
        return s
    return wp.vec2d(s[1], s[0])


@wp.func
def next_sharp_outgoing(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    present: wp.array[wp.int32],
    first: wp.int32,
    after: wp.int32,
) -> wp.int32:
    # The outgoing halfedge around the vertex whose sharpened edge has the least forward index
    # above ``after`` (the order of ``vertTangents``), or -1.
    best = wp.int32(-1)
    best_edge = wp.int32(2147483647)
    current = first
    steps = wp.int32(0)
    while steps < twins.shape[0]:
        current = next_halfedge(twins[current])
        f = forward_of(faces, twins, current)
        if present[f] != 0 and f > after and f < best_edge:
            best = current
            best_edge = f
        steps += 1
        if current == first:
            break
    return best


@wp.kernel
def sharpen_vertex_tangents(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    flat: wp.array[wp.int32],
    first_halfedge: wp.array[wp.int32],
    present: wp.array[wp.int32],
    smoothness: wp.array[wp.vec2d],
    tangents: wp.array[wp.vec4d],
    fixed: wp.array[wp.int32],
) -> None:
    # ``CreateTangents``' vertex pass over its sharpened edges in edge order: none fixes the
    # vertex's first halfedge; two make the crease continuous and sharpen the tangents between
    # them; three or more sharpen every tangent uniformly. Each vertex writes only its own
    # outgoing halfedges.
    v = wp.int32(wp.tid())
    first = first_halfedge[v]
    if first >= faces.shape[0]:
        return
    h0 = next_sharp_outgoing(faces, twins, present, first, -1)
    if h0 < 0:
        fixed[first] = 1
        return
    h1 = next_sharp_outgoing(faces, twins, present, first, forward_of(faces, twins, h0))
    if h1 < 0:
        return
    h2 = next_sharp_outgoing(faces, twins, present, first, forward_of(faces, twins, h1))
    s0 = outgoing_smoothness(faces, twins, smoothness, h0)
    s1 = outgoing_smoothness(faces, twins, smoothness, h1)
    if h2 < 0:
        fixed[h0] = 1
        fixed[h1] = 1
        new_tangent = la_normalize(xyz(tangents[h0]) - xyz(tangents[h1]))
        pos = positions[faces[h0]]
        tangents[h0] = circular_tangent(new_tangent, positions[faces[next_halfedge(h0)]] - pos)
        tangents[h1] = circular_tangent(
            negated3(new_tangent), positions[faces[next_halfedge(h1)]] - pos
        )
        s = (s0[1] + s1[0]) / wp.float64(2.0)
        current = h0
        steps = wp.int32(0)
        while steps < twins.shape[0]:
            current = next_halfedge(twins[current])
            if current == h1:
                s = (s1[1] + s0[0]) / wp.float64(2.0)
            elif current != h0 and tangents[current][3] >= wp.float64(0.0):
                sharpen_tangent(tangents, current, s)
            steps += 1
            if current == h0:
                break
        return
    total = wp.float64(0.0)
    denom = wp.float64(0.0)
    h = h0
    while h >= 0:
        sides = outgoing_smoothness(faces, twins, smoothness, h)
        total = total + sides[0]
        total = total + sides[1]
        denom = denom + wp.where(sides[0] == wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        denom = denom + wp.where(sides[1] == wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        h = next_sharp_outgoing(faces, twins, present, first, forward_of(faces, twins, h))
    total = total / denom
    current = h0
    steps = wp.int32(0)
    while steps < twins.shape[0]:
        current = next_halfedge(twins[current])
        if tangents[current][3] >= wp.float64(0.0):
            uniform = total
            if flat[current // 3] != 0 or flat[twins[current] // 3] != 0:
                uniform = wp.float64(0.0)
            sharpen_tangent(tangents, current, uniform)
        steps += 1
        if current == h0:
            break


@wp.kernel
def linearize_flat_tangents(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    tangents: wp.array[wp.vec4d],
) -> None:
    # ``LinearizeFlatTangents``: an edge with a zero-weight end becomes a straight segment there.
    h = wp.int32(wp.tid())
    if faces[h] > faces[next_halfedge(h)]:
        return
    p = twins[h]
    t = tangents[h]
    o = tangents[p]
    flat0 = t[3] == wp.float64(0.0)
    flat1 = o[3] == wp.float64(0.0)
    if not flat0 and not flat1:
        return
    edge = positions[faces[next_halfedge(h)]] - positions[faces[h]]
    one = wp.float64(1.0)
    if flat0 and flat1:
        e = (edge) / (wp.float64(3.0))
        tangents[h] = wp.vec4d(e[0], e[1], e[2], one)
        n = (negated3(edge)) / (wp.float64(3.0))
        tangents[p] = wp.vec4d(n[0], n[1], n[2], one)
    elif flat0:
        e = (edge + xyz(o)) / (wp.float64(2.0))
        tangents[h] = wp.vec4d(e[0], e[1], e[2], one)
    else:
        e = (negated3(edge) + xyz(t)) / (wp.float64(2.0))
        tangents[p] = wp.vec4d(e[0], e[1], e[2], one)


@wp.func
def wrap_angle(radians: wp.float64) -> wp.float64:
    if radians < wp.float64(-1.0) * SMOOTH_PI:
        return radians + SMOOTH_TWO_PI
    if radians > SMOOTH_PI:
        return radians - SMOOTH_TWO_PI
    return radians


@wp.func
def qrot(q: wp.vec4d, v: wp.vec3d) -> wp.vec3d:
    # ``la::qrot``: ``qxdir * v.x + qydir * v.y + qzdir * v.z``.
    two = wp.float64(2.0)
    x = wp.vec3d(
        q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2],
        (q[0] * q[1] + q[2] * q[3]) * two,
        (q[2] * q[0] - q[1] * q[3]) * two,
    )
    y = wp.vec3d(
        (q[0] * q[1] - q[2] * q[3]) * two,
        q[3] * q[3] - q[0] * q[0] + q[1] * q[1] - q[2] * q[2],
        (q[1] * q[2] + q[0] * q[3]) * two,
    )
    z = wp.vec3d(
        (q[2] * q[0] + q[1] * q[3]) * two,
        (q[1] * q[2] - q[0] * q[3]) * two,
        q[3] * q[3] - q[0] * q[0] - q[1] * q[1] + q[2] * q[2],
    )
    return (x) * (v[0]) + (y) * (v[1]) + (z) * (v[2])


@wp.func
def axis_angle_quat(axis: wp.vec3d, angle: wp.float64) -> wp.vec4d:
    # ``rotation_quat(axis, angle)``.
    s = manifold_sin(angle / wp.float64(2.0))
    return wp.vec4d(axis[0] * s, axis[1] * s, axis[2] * s, manifold_cos(angle / wp.float64(2.0)))


@wp.func
def distribute_step(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    tangents: wp.array[wp.vec4d],
    current: wp.int32,
    center: wp.vec3d,
) -> wp.vec3d:
    return safe_normalize(positions[faces[next_halfedge(current)]] - center)


@wp.kernel
def distribute_tangents(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    vertex_normals: wp.array[wp.vec3d],
    fixed: wp.array[wp.int32],
    tangents: wp.array[wp.vec4d],
) -> None:
    # ``DistributeTangents``: from each fixed halfedge to the next, rotate the free tangents so
    # their angles share the ratios of the triangle angles between their edges. With one fixed
    # halfedge the whole circle turns to an average offset. The arcs are disjoint.
    h = wp.int32(wp.tid())
    if fixed[h] == 0 or tangents[h][3] < wp.float64(0.0):
        return
    approx_normal = vertex_normals[faces[h]]
    center = positions[faces[h]]
    first_tangent = safe_normalize(xyz(tangents[h]))
    # First walk: the normal, the count, and the last current and desired angles.
    normal = wp.vec3d()
    last_edge = distribute_step(positions, faces, tangents, h, center)
    last_tangent = first_tangent
    desired = wp.float64(0.0)
    current_angle = wp.float64(0.0)
    count = wp.int32(0)
    current = h
    steps = wp.int32(0)
    while steps < twins.shape[0]:
        current = next_halfedge(twins[current])
        if tangents[current][3] >= wp.float64(0.0):
            this_edge = distribute_step(positions, faces, tangents, current, center)
            this_tangent = safe_normalize(xyz(tangents[current]))
            normal = normal + wp.cross(this_tangent, last_tangent)
            desired = angle_between(this_edge, last_edge) + desired
            if current == h:
                current_angle = SMOOTH_TWO_PI
            else:
                current_angle = angle_between(this_tangent, first_tangent)
                if dot3(approx_normal, wp.cross(this_tangent, first_tangent)) < wp.float64(0.0):
                    current_angle = SMOOTH_TWO_PI - current_angle
            count += 1
            last_edge = this_edge
            last_tangent = this_tangent
        steps += 1
        if fixed[current] != 0:
            break
    if count == 1 or dot3(normal, normal) == wp.float64(0.0):
        return
    end = current
    scale = current_angle / desired
    offset = wp.float64(0.0)
    if end == h:
        # Second walk: the average offset.
        last_edge = distribute_step(positions, faces, tangents, h, center)
        desired = wp.float64(0.0)
        current = h
        steps = wp.int32(0)
        while steps < twins.shape[0]:
            current = next_halfedge(twins[current])
            if tangents[current][3] >= wp.float64(0.0):
                this_edge = distribute_step(positions, faces, tangents, current, center)
                this_tangent = safe_normalize(xyz(tangents[current]))
                desired = angle_between(this_edge, last_edge) + desired
                angle = SMOOTH_TWO_PI
                if current != h:
                    angle = angle_between(this_tangent, first_tangent)
                    if dot3(approx_normal, wp.cross(this_tangent, first_tangent)) < wp.float64(0.0):
                        angle = SMOOTH_TWO_PI - angle
                offset = offset + wrap_angle(angle - scale * desired)
                last_edge = this_edge
            steps += 1
            if fixed[current] != 0:
                break
        offset = offset / wp.float64(count)
    axis = la_normalize(normal)
    # Third walk: rotate, the desired angles scaled and clamped against their neighbours. Each
    # entry's current angle and raw cumulative desired angle are recomputed from the tangents
    # as they were, the next entry's looked up one step ahead.
    last_edge = distribute_step(positions, faces, tangents, h, center)
    raw = wp.float64(0.0)
    previous = wp.float64(0.0)
    i = wp.int32(0)
    current = h
    steps = wp.int32(0)
    while steps < twins.shape[0]:
        current = next_halfedge(twins[current])
        if current != h and fixed[current] != 0:
            break
        if tangents[current][3] >= wp.float64(0.0):
            this_edge = distribute_step(positions, faces, tangents, current, center)
            tangent = xyz(tangents[current])
            this_tangent = safe_normalize(tangent)
            raw = angle_between(this_edge, last_edge) + raw
            angle_now = SMOOTH_TWO_PI
            if current != h:
                angle_now = angle_between(this_tangent, first_tangent)
                if dot3(approx_normal, wp.cross(this_tangent, first_tangent)) < wp.float64(0.0):
                    angle_now = SMOOTH_TWO_PI - angle_now
            d = raw * scale
            if d - previous > SMOOTH_PI:
                d = previous + SMOOTH_PI
            elif i + 1 < count:
                # The next entry's raw cumulative angle.
                look = current
                look_steps = wp.int32(0)
                next_raw = raw
                while look_steps < twins.shape[0]:
                    look = next_halfedge(twins[look])
                    if tangents[look][3] >= wp.float64(0.0):
                        next_raw = (
                            angle_between(
                                distribute_step(positions, faces, tangents, look, center), this_edge
                            )
                            + raw
                        )
                        break
                    look_steps += 1
                if scale * next_raw - d > SMOOTH_PI:
                    d = scale * next_raw - SMOOTH_PI
            angle = angle_now - d - offset
            q = axis_angle_quat(axis, angle)
            rotated = qrot(q, tangent)
            tangents[current] = wp.vec4d(rotated[0], rotated[1], rotated[2], tangents[current][3])
            previous = d
            last_edge = this_edge
            i += 1
        steps += 1
        if fixed[current] != 0:
            break


@wp.func
def homogeneous4(v: wp.vec4d) -> wp.vec4d:
    return wp.vec4d(v[0] * v[3], v[1] * v[3], v[2] * v[3], v[3])


@wp.func
def homogeneous3(v: wp.vec3d) -> wp.vec4d:
    return wp.vec4d(v[0], v[1], v[2], wp.float64(1.0))


@wp.func
def h_normalize(v: wp.vec4d) -> wp.vec3d:
    if v[3] == wp.float64(0.0):
        return xyz(v)
    return wp.vec3d(v[0] / v[3], v[1] / v[3], v[2] / v[3])


@wp.func
def bezier_control(point: wp.vec3d, tangent: wp.vec4d) -> wp.vec4d:
    # ``Bezier``: ``Homogeneous(vec4(point, 0) + tangent)``.
    return homogeneous4(
        wp.vec4d(
            point[0] + tangent[0],
            point[1] + tangent[1],
            point[2] + tangent[2],
            wp.float64(0.0) + tangent[3],
        )
    )


@wp.func
def cubic_first(p0: wp.vec4d, p1: wp.vec4d, p2: wp.vec4d, x: wp.float64) -> wp.vec4d:
    # ``CubicBezier2Linear``'s first point: ``lerp(lerp(p0, p1, x), lerp(p1, p2, x), x)``.
    return wp.lerp(wp.lerp(p0, p1, x), wp.lerp(p1, p2, x), x)


@wp.func
def cubic_second(p1: wp.vec4d, p2: wp.vec4d, p3: wp.vec4d, x: wp.float64) -> wp.vec4d:
    return wp.lerp(wp.lerp(p1, p2, x), wp.lerp(p2, p3, x), x)


@wp.func
def orthogonal_to(v: wp.vec3d, alt: wp.vec3d, ref: wp.vec3d) -> wp.vec3d:
    # ``OrthogonalTo``.
    out = v - (ref) * (dot3(v, ref))
    if dot3(out, out) < SMOOTH_PRECISION * dot3(v, v):
        out = alt - (ref) * (dot3(alt, ref))
    return safe_normalize(out)


@wp.func
def matrix_quat(cx: wp.vec3d, cy: wp.vec3d, cz: wp.vec3d) -> wp.vec4d:
    # ``rotation_quat(mat3)`` of the matrix with columns ``cx``, ``cy``, ``cz``.
    q = wp.vec4d(
        cx[0] - cy[1] - cz[2], cy[1] - cx[0] - cz[2], cz[2] - cx[0] - cy[1], cx[0] + cy[1] + cz[2]
    )
    j = wp.int32(0)
    for i in range(1, 4):
        if q[i] > q[j]:
            j = i
    one = wp.float64(1.0)
    s = wp.vec4d(one, cx[1] + cy[0], cz[0] + cx[2], cy[2] - cz[1])
    if j == 1:
        s = wp.vec4d(cx[1] + cy[0], one, cy[2] + cz[1], cz[0] - cx[2])
    elif j == 2:
        s = wp.vec4d(cx[2] + cz[0], cy[2] + cz[1], one, cx[1] - cy[0])
    elif j == 3:
        s = wp.vec4d(cy[2] - cz[1], cz[0] - cx[2], cx[1] - cy[0], one)
    r = wp.vec4d(
        wp.sqrt(wp.max(wp.float64(0.0), one + q[0])),
        wp.sqrt(wp.max(wp.float64(0.0), one + q[1])),
        wp.sqrt(wp.max(wp.float64(0.0), one + q[2])),
        wp.sqrt(wp.max(wp.float64(0.0), one + q[3])),
    )
    length = wp.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3])
    return wp.vec4d(
        wp.copysign(r[0] / length, s[0]),
        wp.copysign(r[1] / length, s[1]),
        wp.copysign(r[2] / length, s[2]),
        wp.copysign(r[3] / length, s[3]),
    )


@wp.func
def qmul(a: wp.vec4d, b: wp.vec4d) -> wp.vec4d:
    return wp.vec4d(
        a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
        a[1] * b[3] + a[3] * b[1] + a[2] * b[0] - a[0] * b[2],
        a[2] * b[3] + a[3] * b[2] + a[0] * b[1] - a[1] * b[0],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )


@wp.func
def qconj(q: wp.vec4d) -> wp.vec4d:
    m = wp.float64(-1.0)
    return wp.vec4d(m * q[0], m * q[1], m * q[2], q[3])


@wp.func
def qxdir(q: wp.vec4d) -> wp.vec3d:
    two = wp.float64(2.0)
    return wp.vec3d(
        q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2],
        (q[0] * q[1] + q[2] * q[3]) * two,
        (q[2] * q[0] - q[1] * q[3]) * two,
    )


@wp.func
def vector_quat(orig: wp.vec3d, dest: wp.vec3d) -> wp.vec4d:
    # ``rotation_quat(orig, dest)``: the shortest rotation between unit vectors.
    cos_theta = dot3(orig, dest)
    if cos_theta >= wp.float64(1.0) - SMOOTH_DBL_EPSILON:
        return wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    if cos_theta < wp.float64(-1.0) + SMOOTH_DBL_EPSILON:
        axis = wp.cross(wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)), orig)
        if dot3(axis, axis) < SMOOTH_DBL_EPSILON:
            axis = wp.cross(wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)), orig)
        return axis_angle_quat(la_normalize(axis), SMOOTH_PI)
    axis = wp.cross(orig, dest)
    s = wp.sqrt((wp.float64(1.0) + cos_theta) * wp.float64(2.0))
    return wp.vec4d(
        axis[0] * (wp.float64(1.0) / s),
        axis[1] * (wp.float64(1.0) / s),
        axis[2] * (wp.float64(1.0) / s),
        s * wp.float64(0.5),
    )


@wp.func
def slerp(x: wp.vec4d, y: wp.vec4d, a: wp.float64, long_way: wp.bool) -> wp.vec4d:
    z = y
    cos_theta = x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
    if (cos_theta < wp.float64(0.0)) != long_way:
        m = wp.float64(-1.0)
        z = wp.vec4d(m * y[0], m * y[1], m * y[2], m * y[3])
        cos_theta = m * cos_theta
    if wp.abs(cos_theta) > wp.float64(1.0) - SMOOTH_DBL_EPSILON:
        return wp.lerp(x, z, a)
    angle = manifold_acos(cos_theta)
    sa = manifold_sin((wp.float64(1.0) - a) * angle)
    sb = manifold_sin(a * angle)
    sc = manifold_sin(angle)
    return wp.vec4d(
        (sa * x[0] + sb * z[0]) / sc,
        (sa * x[1] + sb * z[1]) / sc,
        (sa * x[2] + sb * z[2]) / sc,
        (sa * x[3] + sb * z[3]) / sc,
    )


@wp.func
def rotate_from_to(v: wp.vec3d, start: wp.vec4d, end: wp.vec4d) -> wp.vec3d:
    return qrot(end, qrot(qconj(start), v))


@wp.func
def bezier_to_bezier(
    c0: wp.vec3d,
    c1: wp.vec3d,
    tx0: wp.vec4d,
    tx1: wp.vec4d,
    ty0: wp.vec4d,
    ty1: wp.vec4d,
    x: wp.float64,
    anchor: wp.vec3d,
) -> mat24d:
    # ``Bezier2Bezier``: the point at ``x`` along the edge's Bezier (homogeneous) and the
    # cross tangent there, rotated between the ends' frames; rows of the result.
    p0 = homogeneous3(c0)
    p1 = bezier_control(c0, tx0)
    p2 = bezier_control(c1, tx1)
    p3 = homogeneous3(c1)
    b0 = cubic_first(p0, p1, p2, x)
    b1 = cubic_second(p1, p2, p3, x)
    end = h_normalize(wp.lerp(b0, b1, x))
    tangent = safe_normalize(h_normalize(b1) - h_normalize(b0))
    n0 = safe_normalize(xyz(tx0))
    n1 = negated3(safe_normalize(xyz(tx1)))
    bi0 = orthogonal_to(xyz(ty0), anchor - c0, n0)
    bi1 = orthogonal_to(xyz(ty1), anchor - c1, n1)
    q0 = matrix_quat(n0, bi0, wp.cross(n0, bi0))
    q1 = matrix_quat(n1, bi1, wp.cross(n1, bi1))
    edge = c1 - c0
    long_way = dot3(n0, edge) + dot3(n1, edge) < wp.float64(0.0)
    q_tmp = slerp(q0, q1, x, long_way)
    q = qmul(vector_quat(qxdir(q_tmp), tangent), q_tmp)
    delta = wp.lerp(rotate_from_to(xyz(ty0), q0, q), rotate_from_to(xyz(ty1), q1, q), x)
    delta_w = wp.lerp(ty0[3], ty1[3], x)
    e = homogeneous3(end)
    return mat24d(e[0], e[1], e[2], e[3], delta[0], delta[1], delta[2], delta_w)


@wp.func
def row4(m: mat24d, i: wp.int32) -> wp.vec4d:
    return wp.vec4d(m[i, 0], m[i, 1], m[i, 2], m[i, 3])


@wp.func
def bezier_2d(
    c0: wp.vec3d,
    c1: wp.vec3d,
    c2: wp.vec3d,
    c3: wp.vec3d,
    tx0: wp.vec4d,
    tx1: wp.vec4d,
    tx2: wp.vec4d,
    tx3: wp.vec4d,
    ty0: wp.vec4d,
    ty1: wp.vec4d,
    ty2: wp.vec4d,
    ty3: wp.vec4d,
    x: wp.float64,
    y: wp.float64,
    centroid: wp.vec3d,
) -> wp.vec3d:
    # ``Bezier2D``: a quad's point, Bezier across between its two opposite edges' Beziers.
    bez0 = bezier_to_bezier(c0, c1, tx0, tx1, ty0, ty1, x, centroid)
    bez1 = bezier_to_bezier(c2, c3, tx2, tx3, ty2, ty3, wp.float64(1.0) - x, centroid)
    a0 = row4(bez0, 0)
    a3 = row4(bez1, 0)
    a1 = bezier_control(xyz(a0), row4(bez0, 1))
    a2 = bezier_control(xyz(a3), row4(bez1, 1))
    return h_normalize(wp.lerp(cubic_first(a0, a1, a2, y), cubic_second(a1, a2, a3, y), y))


@wp.kernel
def interp_tri_positions(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    inside: wp.array[wp.int32],
    tangents: wp.array[wp.vec4d],
    bary_tri: wp.array[wp.int32],
    bary_uvw: wp.array[wp.vec4d],
    out_positions: wp.array[wp.vec3d],
) -> None:
    # ``InterpTri``: a refined vertex on the smooth surface the tangents define over its face.
    v = wp.int32(wp.tid())
    halfedges = face_halfedges(inside, twins, bary_tri[v])
    uvw = bary_uvw[v]
    c = wp.mat44d()
    for i in range(4):
        if halfedges[i] >= 0:
            p = positions[faces[halfedges[i]]]
            c[i, 0] = p[0]
            c[i, 1] = p[1]
            c[i, 2] = p[2]
    for i in range(4):
        if uvw[i] == wp.float64(1.0):
            out_positions[v] = wp.vec3d(c[i, 0], c[i, 1], c[i, 2])
            return
    c0 = wp.vec3d(c[0, 0], c[0, 1], c[0, 2])
    c1 = wp.vec3d(c[1, 0], c[1, 1], c[1, 2])
    c2 = wp.vec3d(c[2, 0], c[2, 1], c[2, 2])
    c3 = wp.vec3d(c[3, 0], c[3, 1], c[3, 2])
    pos_h = wp.vec4d()
    if halfedges[3] < 0:
        third = wp.float64(1.0) / wp.float64(3.0)
        centroid = (c0) * (third) + (c1) * (third) + (c2) * (third)
        for i in range(3):
            j = next3(i)
            k = (i + 2) % 3
            ci = wp.vec3d(c[i, 0], c[i, 1], c[i, 2])
            cj = wp.vec3d(c[j, 0], c[j, 1], c[j, 2])
            ck = wp.vec3d(c[k, 0], c[k, 1], c[k, 2])
            # tangentR[m] = T[h_m], tangentL[m] = T[pair(h_{m-1})].
            r_i = tangents[halfedges[i]]
            r_j = tangents[halfedges[j]]
            r_k = tangents[halfedges[k]]
            l_i = tangents[twins[halfedges[(i + 2) % 3]]]
            l_j = tangents[twins[halfedges[(j + 2) % 3]]]
            l_k = tangents[twins[halfedges[(k + 2) % 3]]]
            x = uvw[k] / (wp.float64(1.0) - uvw[i])
            bez = bezier_to_bezier(cj, ck, r_j, l_k, l_j, r_k, x, centroid)
            b0 = row4(bez, 0)
            b1 = bezier_control(xyz(b0), row4(bez, 1))
            b2 = bezier_control(ci, wp.lerp(r_i, l_i, x))
            b3 = homogeneous3(ci)
            p = h_normalize(
                wp.lerp(cubic_first(b0, b1, b2, uvw[i]), cubic_second(b1, b2, b3, uvw[i]), uvw[i])
            )
            pos_h = pos_h + homogeneous4(wp.vec4d(p[0], p[1], p[2], uvw[j] * uvw[k]))
    else:
        quarter = wp.float64(0.25)
        centroid = (c0) * (quarter) + (c1) * (quarter) + (c2) * (quarter)
        centroid = centroid + (c3) * (quarter)
        tx0 = tangents[halfedges[0]]
        tx1 = tangents[twins[halfedges[0]]]
        tx2 = tangents[halfedges[2]]
        tx3 = tangents[twins[halfedges[2]]]
        ty0 = tangents[twins[halfedges[3]]]
        ty1 = tangents[halfedges[1]]
        ty2 = tangents[twins[halfedges[1]]]
        ty3 = tangents[halfedges[3]]
        x = uvw[1] + uvw[2]
        y = uvw[2] + uvw[3]
        px = bezier_2d(c0, c1, c2, c3, tx0, tx1, tx2, tx3, ty0, ty1, ty2, ty3, x, y, centroid)
        py = bezier_2d(
            c1, c2, c3, c0, ty1, ty2, ty3, ty0, tx1, tx2, tx3, tx0, y, wp.float64(1.0) - x, centroid
        )
        pos_h = pos_h + homogeneous4(wp.vec4d(px[0], px[1], px[2], x * (wp.float64(1.0) - x)))
        pos_h = pos_h + homogeneous4(wp.vec4d(py[0], py[1], py[2], y * (wp.float64(1.0) - y)))
    pos = h_normalize(pos_h)
    if not (wp.isfinite(pos[0]) and wp.isfinite(pos[1]) and wp.isfinite(pos[2])):
        pos = c0
    out_positions[v] = pos


@wp.kernel
def least_outgoing_halfedges(starts: wp.array[wp.int32], out_first: wp.array[wp.int32]) -> None:
    # ``VertHalfedge`` / ``CalculateVertNormals``: each vertex's least live outgoing halfedge.
    h = wp.int32(wp.tid())
    v = starts[h]
    if v >= 0:
        wp.atomic_min(out_first, v, h)


@wp.kernel
def quad_diagonal_flags(tangents: wp.array[wp.vec4d], out_inside: wp.array[wp.int32]) -> None:
    # ``IsMarkedInsideQuad``: a negative tangent weight marks a quad's diagonal.
    h = wp.int32(wp.tid())
    out_inside[h] = wp.where(tangents[h][3] < wp.float64(0.0), 1, 0)


# -- level sets (``LevelSet``: marching tetrahedra on a body-centred cubic grid) -------------------
#
# Grid points are ``(x, y, z, w)``: ``w = 1`` the cube corners, ``w = 0`` the centres half a cell
# back. Each point owns seven of its fourteen edges (``level_set_neighbor`` 0-6, the rest their
# opposites). The near-surface points go into Manifold's linear-probing hash table in index order
# (one thread, as ``canParallel = false`` inserts them), and the edge vertices and triangles are
# numbered in the table's slot order, as its serial loops number them.

LEVEL_SET_OPEN = wp.uint64(0xFFFFFFFFFFFFFFFF)
LEVEL_SET_KS = wp.float64(0.25)
LEVEL_SET_KD = wp.float64(3.0)


@wp.func
def level_set_encode(p: wp.vec4i, pw: wp.vec3i) -> wp.uint64:
    # ``EncodeIndex``: a negative coordinate wraps as Manifold's unsigned casts do.
    return (
        wp.uint64(p[3])
        | (wp.uint64(p[2]) << wp.uint64(1))
        | (wp.uint64(p[1]) << wp.uint64(1 + pw[2]))
        | (wp.uint64(p[0]) << wp.uint64(1 + pw[2] + pw[1]))
    )


@wp.func
def level_set_decode(index: wp.uint64, pw: wp.vec3i) -> wp.vec4i:
    w = wp.int32(index & wp.uint64(1))
    rest = index >> wp.uint64(1)
    z = wp.int32(rest & ((wp.uint64(1) << wp.uint64(pw[2])) - wp.uint64(1)))
    rest = rest >> wp.uint64(pw[2])
    y = wp.int32(rest & ((wp.uint64(1) << wp.uint64(pw[1])) - wp.uint64(1)))
    rest = rest >> wp.uint64(pw[1])
    x = wp.int32(rest & ((wp.uint64(1) << wp.uint64(pw[0])) - wp.uint64(1)))
    return wp.vec4i(x, y, z, w)


@wp.func
def level_set_neighbor(base: wp.vec4i, i: wp.int32) -> wp.vec4i:
    # ``Neighbor``: the 14 neighbours, the first seven owned; ``w`` wrapping to the next cube.
    o = wp.vec4i(0, 0, 0, 1)
    if i == 1:
        o = wp.vec4i(1, 0, 0, 0)
    elif i == 2:
        o = wp.vec4i(0, 1, 0, 0)
    elif i == 3:
        o = wp.vec4i(0, 0, 1, 0)
    elif i == 4:
        o = wp.vec4i(-1, 0, 0, 1)
    elif i == 5:
        o = wp.vec4i(0, -1, 0, 1)
    elif i == 6:
        o = wp.vec4i(0, 0, -1, 1)
    elif i == 7:
        o = wp.vec4i(-1, -1, -1, 1)
    elif i == 8:
        o = wp.vec4i(-1, 0, 0, 0)
    elif i == 9:
        o = wp.vec4i(0, -1, 0, 0)
    elif i == 10:
        o = wp.vec4i(0, 0, -1, 0)
    elif i == 11:
        o = wp.vec4i(0, -1, -1, 1)
    elif i == 12:
        o = wp.vec4i(-1, 0, -1, 1)
    elif i == 13:
        o = wp.vec4i(-1, -1, 0, 1)
    n = base + o
    if n[3] == 2:
        n = n + wp.vec4i(1, 1, 1, 1)
        n[3] = 0
    return n


@wp.func
def level_set_position(g: wp.vec4i, origin: wp.vec3d, spacing: wp.vec3d) -> wp.vec3d:
    off = wp.where(g[3] == 1, wp.float64(0.0), wp.float64(-0.5))
    return wp.vec3d(
        origin[0] + spacing[0] * (wp.float64(g[0]) + off),
        origin[1] + spacing[1] * (wp.float64(g[1]) + off),
        origin[2] + spacing[2] * (wp.float64(g[2]) + off),
    )


@wp.func
def level_set_bound(p: wp.vec3d, origin: wp.vec3d, spacing: wp.vec3d, grid: wp.vec3i) -> wp.vec3d:
    # ``Bound``: clamped into the grid's box.
    out = wp.vec3d()
    for k in range(3):
        hi = origin[k] + spacing[k] * (wp.float64(grid[k]) - wp.float64(1.0))
        v = p[k]
        if v < origin[k]:
            v = origin[k]
        if hi < v:
            v = hi
        out[k] = v
    return out


@wp.func
def level_set_bounded(
    g: wp.vec4i,
    origin: wp.vec3d,
    spacing: wp.vec3d,
    grid: wp.vec3i,
    level: wp.float64,
    sdf: wp.Function,
) -> wp.float64:
    # ``BoundedSDF``: zero outside the grid, never positive on its boundary.
    lower = wp.min(wp.min(g[0], g[1]), g[2])
    upper = wp.min(wp.min(grid[0] - g[0], grid[1] - g[1]), grid[2] - g[2])
    bound_dist = wp.min(lower, upper - g[3])
    if bound_dist < 0:
        return wp.float64(0.0)
    d = sdf(level_set_position(g, origin, spacing)) - level
    if bound_dist == 0:
        if wp.float64(0.0) < d:
            return wp.float64(0.0)
    return d


@wp.func
def level_set_find_surface(
    p0_in: wp.vec3d,
    d0_in: wp.float64,
    p1_in: wp.vec3d,
    d1_in: wp.float64,
    tol: wp.float64,
    level: wp.float64,
    sdf: wp.Function,
) -> wp.vec3d:
    # ``FindSurface``: simplified ITP root finding along the edge, to within ``tol``.
    p0 = p0_in
    p1 = p1_in
    d0 = d0_in
    d1 = d1_in
    if d0 == wp.float64(0.0):
        return p0
    if d1 == wp.float64(0.0):
        return p1
    k = wp.float64(0.1)
    e = p0 - p1
    check = wp.float64(2.0) * tol / wp.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
    frac = wp.float64(1.0)
    bi_frac = wp.float64(1.0)
    half = wp.float64(0.5)
    while frac > check:
        t = (d0 / (d0 - d1)) * (wp.float64(1.0) - k) + half * k
        r = bi_frac / frac - half
        x = t
        if not (wp.abs(t - half) < r):
            x = half - r * wp.where(t < half, wp.float64(1.0), wp.float64(-1.0))
        mid = wp.lerp(p0, p1, x)
        d = sdf(mid) - level
        if (d > wp.float64(0.0)) == (d0 > wp.float64(0.0)):
            d0 = d
            p0 = mid
            frac = frac * (wp.float64(1.0) - x)
        else:
            d1 = d
            p1 = mid
            frac = frac * x
        bi_frac = bi_frac / wp.float64(2.0)
    return wp.lerp(p0, p1, d0 / (d0 - d1))


@wp.func
def level_set_voxel(voxels: wp.array[wp.float64], g: wp.vec4i, pw: wp.vec3i) -> wp.float64:
    return voxels[wp.int64(level_set_encode(g + wp.vec4i(1, 1, 1, 0), pw))]


@wp.func
def level_set_same_side(a: wp.float64, b: wp.float64) -> wp.bool:
    return (a > wp.float64(0.0)) == (b > wp.float64(0.0))


@wp.func
def level_set_classify(
    voxels: wp.array[wp.float64], g: wp.vec4i, pw: wp.vec3i, distance: wp.float64
) -> wp.vec4i:
    # ``NearSurface``'s scan of the owned edges: ``(crossing bits, keep, closest neighbour,
    # opposed count)``; the closest neighbour's value is recomputed by ``level_set_closest_value``.
    mask = wp.int32(0)
    keep = wp.int32(0)
    v_max = wp.float64(0.0)
    closest = wp.int32(-1)
    opposed = wp.int32(0)
    for i in range(7):
        val = level_set_voxel(voxels, level_set_neighbor(g, i), pw)
        val_op = level_set_voxel(voxels, level_set_neighbor(g, i + 7), pw)
        if not level_set_same_side(val, distance):
            mask = mask | (1 << i)
            keep = 1
            if not level_set_same_side(val_op, distance):
                opposed += 1
            if wp.abs(val) > LEVEL_SET_KD * wp.abs(distance) and wp.abs(val) > wp.abs(v_max):
                v_max = val
                closest = i
        elif (
            not level_set_same_side(val_op, distance)
            and wp.abs(val_op) > LEVEL_SET_KD * wp.abs(distance)
            and wp.abs(val_op) > wp.abs(v_max)
        ):
            v_max = val_op
            closest = i + 7
    return wp.vec4i(mask, keep, closest, opposed)


@wp.kernel
def level_set_insert(
    keys_in: wp.array[wp.uint64],
    n_entries: wp.int32,
    size: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_slot_entry: wp.array[wp.int32],
) -> None:
    # The table as Manifold's serial ``Insert`` loop leaves it: each key in order into its hashed
    # slot or the next open one. One thread.
    mask = wp.uint64(size - 1)
    for e in range(n_entries):
        key = keys_in[e]
        x = key
        x = (x ^ (x >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
        x = x ^ (x >> wp.uint64(31))
        slot = wp.int32(x & mask)
        while out_keys[slot] != LEVEL_SET_OPEN:
            slot = (slot + 1) & (size - 1)
        out_keys[slot] = key
        out_slot_entry[slot] = e


@wp.func
def level_set_lookup(
    keys: wp.array[wp.uint64], slot_entry: wp.array[wp.int32], key: wp.uint64
) -> wp.int32:
    # ``operator[]``: the entry under ``key``, or -1 (the empty slot's default ``GridVert``).
    size = keys.shape[0]
    x = key
    x = (x ^ (x >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    x = x ^ (x >> wp.uint64(31))
    slot = wp.int32(x & wp.uint64(size - 1))
    while True:
        k = keys[slot]
        if k == key:
            return slot_entry[slot]
        if k == LEVEL_SET_OPEN:
            return -1
        slot = (slot + 1) & (size - 1)
    return -1


@wp.func
def level_set_edge_target(
    voxels: wp.array[wp.float64],
    keys: wp.array[wp.uint64],
    slot_entry: wp.array[wp.int32],
    entry_distance: wp.array[wp.float64],
    entry_moved: wp.array[wp.int32],
    g: wp.vec4i,
    i: wp.int32,
    pw: wp.vec3i,
    distance: wp.float64,
) -> wp.vec2i:
    # ``ComputeVerts`` for one owned edge: ``(0, _)`` for no vertex, ``(1, moved vertex)`` for the
    # neighbour's snapped one, ``(2, _)`` for a new vertex on the edge.
    n = level_set_neighbor(g, i)
    e = level_set_lookup(keys, slot_entry, level_set_encode(n, pw))
    val = wp.float64(0.0)
    if e >= 0:
        val = entry_distance[e]
    else:
        val = level_set_voxel(voxels, n, pw)
    if level_set_same_side(val, distance):
        return wp.vec2i(0, 0)
    if e >= 0 and entry_moved[e] >= 0:
        return wp.vec2i(1, entry_moved[e])
    return wp.vec2i(2, 0)


@wp.kernel
def level_set_count_verts(
    voxels: wp.array[wp.float64],
    keys: wp.array[wp.uint64],
    slot_entry: wp.array[wp.int32],
    entry_keys: wp.array[wp.uint64],
    entry_distance: wp.array[wp.float64],
    entry_moved: wp.array[wp.int32],
    pw: wp.vec3i,
    out_counts: wp.array[wp.int32],
) -> None:
    # Each slot's new edge vertices, numbered in slot order after the snapped ones.
    slot = wp.int32(wp.tid())
    e = slot_entry[slot]
    count = wp.int32(0)
    if e >= 0 and entry_moved[e] < 0:
        g = level_set_decode(entry_keys[e], pw)
        for i in range(7):
            target = level_set_edge_target(
                voxels, keys, slot_entry, entry_distance, entry_moved, g, i, pw, entry_distance[e]
            )
            if target[0] == 2:
                count += 1
    out_counts[slot] = count


@wp.func
def level_set_inside(entry_distance: wp.array[wp.float64], e: wp.int32) -> wp.int32:
    # ``GridVert::Inside``: a missing point's NaN distance counts as outside.
    if e >= 0 and entry_distance[e] > wp.float64(0.0):
        return 1
    return -1


@wp.func
def level_set_edge(edge_verts: wp.array[wp.int32], e: wp.int32, i: wp.int32) -> wp.int32:
    if e < 0:
        return -1
    return edge_verts[7 * e + i]


@wp.func
def level_set_neighbor_inside(
    entry_distance: wp.array[wp.float64], edge_verts: wp.array[wp.int32], e: wp.int32, i: wp.int32
) -> wp.int32:
    return level_set_inside(entry_distance, e) * wp.where(
        level_set_edge(edge_verts, e, i) == -1, 1, -1
    )


@wp.func
def tet_tri(table: wp.int32, i: wp.int32) -> wp.vec3i:
    # ``TetTri0`` / ``TetTri1``: a tetrahedron's triangles by its corners' inside bits.
    t = wp.vec3i(-1, -1, -1)
    if table == 0:
        if i == 1:
            t = wp.vec3i(0, 3, 4)
        elif i == 2:
            t = wp.vec3i(0, 1, 5)
        elif i == 3:
            t = wp.vec3i(1, 5, 3)
        elif i == 4:
            t = wp.vec3i(1, 4, 2)
        elif i == 5:
            t = wp.vec3i(1, 0, 3)
        elif i == 6:
            t = wp.vec3i(2, 5, 0)
        elif i == 7:
            t = wp.vec3i(5, 3, 2)
        elif i == 8:
            t = wp.vec3i(2, 3, 5)
        elif i == 9:
            t = wp.vec3i(0, 5, 2)
        elif i == 10:
            t = wp.vec3i(3, 0, 1)
        elif i == 11:
            t = wp.vec3i(2, 4, 1)
        elif i == 12:
            t = wp.vec3i(3, 5, 1)
        elif i == 13:
            t = wp.vec3i(5, 1, 0)
        elif i == 14:
            t = wp.vec3i(4, 3, 0)
    else:
        if i == 3:
            t = wp.vec3i(3, 4, 1)
        elif i == 5:
            t = wp.vec3i(3, 2, 1)
        elif i == 6:
            t = wp.vec3i(0, 4, 2)
        elif i == 9:
            t = wp.vec3i(2, 4, 0)
        elif i == 10:
            t = wp.vec3i(1, 2, 3)
        elif i == 12:
            t = wp.vec3i(1, 4, 3)
    return t


@wp.func
def level_set_tet_tris(
    tet: wp.vec4i,
    edges_a: wp.vec3i,
    edges_b: wp.vec3i,
    out_tris: wp.array[wp.vec3i],
    write: wp.int32,
    at: wp.int32,
) -> wp.int32:
    # ``CreateTris``: up to two triangles over the tetrahedron's six edge vertices (``edges_a``
    # then ``edges_b``), skipping a degenerate one; returns how many, written from ``at``.
    i = wp.where(tet[0] > 0, 1, 0) + wp.where(tet[1] > 0, 2, 0)
    i = i + wp.where(tet[2] > 0, 4, 0) + wp.where(tet[3] > 0, 8, 0)
    n = wp.int32(0)
    for table in range(2):
        tri = tet_tri(table, i)
        if tri[0] >= 0:
            v = wp.vec3i(0, 0, 0)
            for c in range(3):
                j = tri[c]
                v[c] = wp.where(j < 3, edges_a[j % 3], edges_b[j % 3])
            if v[0] != v[1] and v[1] != v[2] and v[2] != v[0]:
                if write != 0:
                    out_tris[at + n] = v
                n += 1
    return n


@wp.func
def level_set_slot_tris(
    keys: wp.array[wp.uint64],
    slot_entry: wp.array[wp.int32],
    entry_keys: wp.array[wp.uint64],
    entry_distance: wp.array[wp.float64],
    edge_verts: wp.array[wp.int32],
    pw: wp.vec3i,
    slot: wp.int32,
    out_tris: wp.array[wp.vec3i],
    write: wp.int32,
    at: wp.int32,
) -> wp.int32:
    # ``BuildTris`` for one slot: the six tetrahedra around its point's (1, 1, 1) edge.
    b = slot_entry[slot]
    if b < 0:
        return 0
    base = level_set_decode(entry_keys[b], pw)
    lead = base
    if lead[3] == 0:
        lead[3] = 1
    else:
        lead = lead + wp.vec4i(1, 1, 1, 1)
        lead[3] = 0
    tet = wp.vec4i(
        level_set_neighbor_inside(entry_distance, edge_verts, b, 0),
        level_set_inside(entry_distance, b),
        -2,
        -2,
    )
    this_index = base
    this_index[0] = this_index[0] + 1
    this_vert = level_set_lookup(keys, slot_entry, level_set_encode(this_index, pw))
    tet[2] = level_set_neighbor_inside(entry_distance, edge_verts, b, 1)
    n = wp.int32(0)
    for i in range(3):
        prev = (i + 2) % 3
        nxt = (i + 1) % 3
        this_index = lead
        this_index[prev] = this_index[prev] - 1
        next_vert = wp.int32(-1)
        if this_index[prev] >= 0:
            next_vert = level_set_lookup(keys, slot_entry, level_set_encode(this_index, pw))
        tet[3] = level_set_neighbor_inside(entry_distance, edge_verts, b, prev + 4)
        e0 = level_set_edge(edge_verts, b, 0)
        e3 = level_set_edge(edge_verts, next_vert, prev + 1)
        e5 = level_set_edge(edge_verts, b, prev + 4)
        edges1_a = wp.vec3i(
            e0, level_set_edge(edge_verts, b, i + 1), level_set_edge(edge_verts, next_vert, nxt + 4)
        )
        edges1_b = wp.vec3i(e3, level_set_edge(edge_verts, this_vert, i + 4), e5)
        this_vert = next_vert
        n += level_set_tet_tris(tet, edges1_a, edges1_b, out_tris, write, at + n)
        this_index = base
        this_index[nxt] = this_index[nxt] + 1
        next_vert = level_set_lookup(keys, slot_entry, level_set_encode(this_index, pw))
        tet[2] = tet[3]
        tet[3] = level_set_neighbor_inside(entry_distance, edge_verts, b, nxt + 1)
        edges2_a = wp.vec3i(e0, e5, level_set_edge(edge_verts, this_vert, i + 4))
        edges2_b = wp.vec3i(
            level_set_edge(edge_verts, next_vert, nxt + 4),
            e3,
            level_set_edge(edge_verts, b, nxt + 1),
        )
        this_vert = next_vert
        n += level_set_tet_tris(tet, edges2_a, edges2_b, out_tris, write, at + n)
        tet[2] = tet[3]
    return n


@wp.kernel
def level_set_count_tris(
    keys: wp.array[wp.uint64],
    slot_entry: wp.array[wp.int32],
    entry_keys: wp.array[wp.uint64],
    entry_distance: wp.array[wp.float64],
    edge_verts: wp.array[wp.int32],
    pw: wp.vec3i,
    out_tris: wp.array[wp.vec3i],
    out_counts: wp.array[wp.int32],
) -> None:
    slot = wp.int32(wp.tid())
    out_counts[slot] = level_set_slot_tris(
        keys, slot_entry, entry_keys, entry_distance, edge_verts, pw, slot, out_tris, 0, 0
    )


@wp.kernel
def level_set_write_tris(
    keys: wp.array[wp.uint64],
    slot_entry: wp.array[wp.int32],
    entry_keys: wp.array[wp.uint64],
    entry_distance: wp.array[wp.float64],
    edge_verts: wp.array[wp.int32],
    pw: wp.vec3i,
    offsets: wp.array[wp.int32],
    out_tris: wp.array[wp.vec3i],
) -> None:
    slot = wp.int32(wp.tid())
    level_set_slot_tris(
        keys,
        slot_entry,
        entry_keys,
        entry_distance,
        edge_verts,
        pw,
        slot,
        out_tris,
        1,
        offsets[slot],
    )


@wp.kernel
def level_set_entries(
    offsets: wp.array[wp.int32],
    moved_flag: wp.array[wp.int32],
    mask: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_moved: wp.array[wp.int32],
    out_edge_verts: wp.array[wp.int32],
) -> None:
    # The kept points in index order (those whose scanned flag steps ``offsets``): key, snapped
    # flag (numbered by the caller's scan), and their crossing edges marked -2.
    idx = wp.int32(wp.tid())
    e = offsets[idx]
    if offsets[idx + 1] == e:
        return
    out_keys[e] = wp.uint64(idx)
    out_moved[e] = moved_flag[idx]
    for i in range(7):
        out_edge_verts[7 * e + i] = wp.where((mask[idx] >> i) & 1 != 0, -2, -1)


@wp.kernel
def level_set_number_moved(
    moved_offsets: wp.array[wp.int32],
    entry_keys: wp.array[wp.uint64],
    voxels: wp.array[wp.float64],
    pw: wp.vec3i,
    moved_pos_dense: wp.array[wp.vec3d],
    out_moved: wp.array[wp.int32],
    out_distance: wp.array[wp.float64],
    out_edge_verts: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3d],
) -> None:
    # Snapped points' vertex numbers (``NearSurface``'s serial counter), their crossing edges set
    # to that vertex and the rest cleared; every entry's distance from the voxels.
    e = wp.int32(wp.tid())
    key = entry_keys[e]
    g = level_set_decode(key, pw)
    out_distance[e] = level_set_voxel(voxels, g, pw)
    moved = wp.int32(-1)
    if moved_offsets[e + 1] != moved_offsets[e]:
        moved = moved_offsets[e]
        out_positions[moved] = moved_pos_dense[wp.int32(key)]
    out_moved[e] = moved
    for i in range(7):
        v = out_edge_verts[7 * e + i]
        if moved >= 0 and v == -2:
            out_edge_verts[7 * e + i] = moved
        else:
            out_edge_verts[7 * e + i] = -1


_LEVEL_SET_KERNELS: dict[object, tuple[object, object, object]] = {}


def level_set_kernels(sdf: object) -> tuple[object, object, object]:
    """
    Return the level set's kernels that evaluate ``sdf``.

    Voxels, near-surface points and edge vertices: one unfused module per function, built on
    first use.
    """
    cached = _LEVEL_SET_KERNELS.get(sdf)
    if cached is not None:
        return cached
    tag = "".join(c if c.isalnum() else "_" for c in getattr(sdf, "key", "sdf"))

    def voxels(
        origin: wp.vec3d,
        spacing: wp.vec3d,
        grid: wp.vec3i,
        pw: wp.vec3i,
        level: wp.float64,
        out_voxels: wp.array[wp.float64],
    ) -> None:
        # Every voxel's bounded value, the grid padded by one point each way.
        idx = wp.int32(wp.tid())
        g = level_set_decode(wp.uint64(idx), pw) - wp.vec4i(1, 1, 1, 0)
        out_voxels[idx] = level_set_bounded(g, origin, spacing, grid, level, sdf)

    def near_surface(
        voxels: wp.array[wp.float64],
        origin: wp.vec3d,
        spacing: wp.vec3d,
        grid: wp.vec3i,
        pw: wp.vec3i,
        level: wp.float64,
        tol: wp.float64,
        out_keep: wp.array[wp.int32],
        out_moved: wp.array[wp.int32],
        out_moved_pos: wp.array[wp.vec3d],
        out_mask: wp.array[wp.int32],
    ) -> None:
        # ``NearSurface``: whether the point has a crossing edge, and whether it snaps onto the
        # surface (taking every crossing edge's vertex) within a quarter cell.
        idx = wp.int32(wp.tid())
        g = level_set_decode(wp.uint64(idx), pw)
        out_keep[idx] = 0
        out_moved[idx] = 0
        out_mask[idx] = 0
        if g[0] > grid[0] or g[1] > grid[1] or g[2] > grid[2]:
            return
        distance = level_set_voxel(voxels, g, pw)
        c = level_set_classify(voxels, g, pw, distance)
        keep = c[1]
        closest = c[2]
        if closest >= 0 and c[3] <= 3:
            grid_pos = level_set_position(g, origin, spacing)
            n = level_set_neighbor(g, closest)
            v_max = level_set_voxel(voxels, n, pw)
            pos = level_set_find_surface(
                grid_pos, distance, level_set_position(n, origin, spacing), v_max, tol, level, sdf
            )
            d = pos - grid_pos
            if (
                wp.abs(d[0]) < LEVEL_SET_KS * spacing[0]
                and wp.abs(d[1]) < LEVEL_SET_KS * spacing[1]
                and wp.abs(d[2]) < LEVEL_SET_KS * spacing[2]
            ):
                out_moved[idx] = 1
                out_moved_pos[idx] = level_set_bound(pos, origin, spacing, grid)
                keep = 1
        out_keep[idx] = keep
        out_mask[idx] = c[0]

    def edge_verts(
        voxels: wp.array[wp.float64],
        keys: wp.array[wp.uint64],
        slot_entry: wp.array[wp.int32],
        entry_keys: wp.array[wp.uint64],
        entry_distance: wp.array[wp.float64],
        entry_moved: wp.array[wp.int32],
        offsets: wp.array[wp.int32],
        n_moved: wp.int32,
        origin: wp.vec3d,
        spacing: wp.vec3d,
        grid: wp.vec3i,
        pw: wp.vec3i,
        level: wp.float64,
        tol: wp.float64,
        out_edge_verts: wp.array[wp.int32],
        out_positions: wp.array[wp.vec3d],
    ) -> None:
        # ``ComputeVerts``: each unsnapped point's crossing edges take the neighbour's snapped
        # vertex or a new one on the surface, numbered in slot order.
        slot = wp.int32(wp.tid())
        e = slot_entry[slot]
        if e < 0 or entry_moved[e] >= 0:
            return
        g = level_set_decode(entry_keys[e], pw)
        distance = entry_distance[e]
        position = level_set_position(g, origin, spacing)
        at = n_moved + offsets[slot]
        for i in range(7):
            target = level_set_edge_target(
                voxels, keys, slot_entry, entry_distance, entry_moved, g, i, pw, distance
            )
            if target[0] == 1:
                out_edge_verts[7 * e + i] = target[1]
            elif target[0] == 2:
                n = level_set_neighbor(g, i)
                e2 = level_set_lookup(keys, slot_entry, level_set_encode(n, pw))
                val = wp.float64(0.0)
                if e2 >= 0:
                    val = entry_distance[e2]
                else:
                    val = level_set_voxel(voxels, n, pw)
                pos = level_set_find_surface(
                    position, distance, level_set_position(n, origin, spacing), val, tol, level, sdf
                )
                out_positions[at] = level_set_bound(pos, origin, spacing, grid)
                out_edge_verts[7 * e + i] = at
                at += 1

    options = {"fuse_fp": False}
    built = tuple(
        wp.kernel(
            f,
            name=f"level_set_{f.__name__}_{tag}",
            enable_backward=False,
            module="unique",
            module_options=options,
        )
        for f in (voxels, near_surface, edge_verts)
    )
    _LEVEL_SET_KERNELS[sdf] = built
    return built
