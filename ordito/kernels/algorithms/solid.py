"""
Closed solids in halfedge form: built from a mesh, and written back as one.

A solid is a closed, edge-manifold mesh as ``ordito._solid`` carries it: ``float64`` positions
renumbered in Morton order, faces sorted by their centres' Morton codes, each face's normal and
coplanar group and each vertex's pseudo-normal. The Booleans, ``remesh.simplify`` and
``subdivide_smooth``, ``levelset.function_level_set`` and ``ray.segment_hits`` share it. A port of
the Manifold library's ``SortGeometry`` and ``SetNormalsAndCoplanar`` (Apache-2.0,
https://github.com/elalish/manifold, v3.5.5; Copyright 2021 The Manifold Authors), so every tie
Manifold breaks by index falls the same way.

The whole module builds unfused (``fuse_fp``): the ports compare their results with Manifold's bit
for bit, and a contraction nvcc chooses by the code around it would round differently from the CPU
and from Manifold's own unfused arithmetic. The coplanar flood, serial in Manifold, runs here as
exact parallel rounds, over the faces that would claim a neighbour, and leaves every group as the
serial loop forms it.
"""

import warp as wp

from ordito.kernels.algorithms.portable_trigonometry import portable_acos
from ordito.kernels.halfedge import halfedge_next

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


# Manifold's ``kPrecision``: the triangulator's epsilon is this times the inputs' largest
# absolute coordinate.
PRECISION = 1e-12


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
        current = halfedge_next(p)
        a = starts[current]
        b = starts[halfedge_next(current)]
        c = starts[halfedge_next(halfedge_next(current))]
        curr_edge = wp.normalize(positions[b] - positions[a])
        prev_edge = wp.normalize(positions[a] - positions[c])
        if wp.isfinite(curr_edge[0]) and wp.isfinite(prev_edge[0]):
            d = wp.float64(0.0) - wp.dot(prev_edge, curr_edge)
            phi = wp.float64(0.0)
            if d <= wp.float64(-1.0):
                phi = wp.float64(3.141592653589793)
            elif d < wp.float64(1.0):
                phi = portable_acos(d)
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
        h = halfedge_next(twins[3 * f + i])
        v = positions[faces[halfedge_next(h)]]
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
            h = halfedge_next(twins[stack[top - 1]])
            top -= 1
            tri = h // 3
            if coplanar[tri] >= 0:
                continue
            if is_seed[rank[tri]] == 0 and rank[tri] < k:
                continue
            v = positions[faces[halfedge_next(h)]]
            if wp.abs(wp.dot(v - base, normal)) < tolerance:
                coplanar[tri] = seed
                face_normals[tri] = normal
                if top == 0 or h != twins[stack[top - 1]]:
                    stack[top] = h
                    top += 1
                else:
                    top -= 1
                stack[top] = halfedge_next(h)
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
        h = halfedge_next(twins[scratch[stack + top - 1]])
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
        v = positions[faces[halfedge_next(h)]]
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
            scratch[stack + top] = halfedge_next(h)
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


@wp.func
def same_face(
    mesh: wp.array[wp.int32], coplanar: wp.array[wp.int32], a: wp.int32, b: wp.int32
) -> wp.bool:
    # ``TriRef::SameFace`` (no face ids on an imported mesh).
    return mesh[a] == mesh[b] and coplanar[a] == coplanar[b]


@wp.kernel
def least_outgoing_halfedges(starts: wp.array[wp.int32], out_first: wp.array[wp.int32]) -> None:
    # ``VertHalfedge`` / ``CalculateVertNormals``: each vertex's least live outgoing halfedge.
    h = wp.int32(wp.tid())
    v = starts[h]
    if v >= 0:
        wp.atomic_min(out_first, v, h)


@wp.kernel
def narrow_positions(positions: wp.array[wp.vec3d], out_vertices: wp.array[wp.vec3]) -> None:
    i = wp.int32(wp.tid())
    out_vertices[i] = wp.vec3(positions[i])


@wp.kernel
def gather_faces(
    faces: wp.array[wp.int32], order: wp.array[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    f = wp.int32(wp.tid())
    old = order[f]
    for i in range(3):
        out_faces[3 * f + i] = faces[3 * old + i]
