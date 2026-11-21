import warp as wp

from ordito.kernels.array import (
    append_slot,
    declare_lazy_kernels,
    declare_map_signatures,
    map_probe,
    map_probe_single,
)
from ordito.kernels.predicates import shadow_edge_face

wp.set_module_options({"enable_backward": False})


@wp.func
def first_hit_record(
    mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32, both_ways: wp.bool
) -> tuple[wp.vec3, wp.float32, wp.int32, wp.vec3]:
    # A ray's first hit: its location, its distance along the unit ray, the face (-1 on a miss)
    # and the hit's weights on that face's three corners in order (``mesh_eval_position``'s
    # ``(u, v, 1 - u - v)``); ``direction`` need not be unit length. A miss is the origin, an
    # infinite distance, ``-1`` and zero weights. With ``both_ways`` the ray is also cast along
    # ``-direction`` and the nearer hit wins, a backward one carrying a negative distance; a tie
    # keeps the forward hit.
    #
    # The face index carries the hit test -- every consumer reads it that way -- so there is no
    # separate hit flag to keep in step with it. The location is ``origin + t * direction``, the
    # same ``float32`` expression on both sides of the cast.
    #
    # The casts take their rays in index order. Taking them in the Hilbert order of their origins
    # (``_common.space_filling_order``, as the tree searches do) was measured and declined
    # (2026-11-11, CUDA, the sort included): rays fired from a shell around the mesh lose at every
    # size (1.04-1.18x at 2**20 rays, 1.1-1.65x below), the ordered cast itself at par, while rays
    # leaving the surface gain from 2**18 (0.73-0.87x; 0.46-0.59x at 2**20). Telling the two apart
    # takes a sampled test and a readback, a host sync these casts do not otherwise make.
    unit_direction = wp.normalize(direction)
    location = origin
    distance = wp.float32(wp.inf)
    face = wp.int32(-1)
    weights = wp.vec3(0.0, 0.0, 0.0)
    query = wp.mesh_query_ray(mesh_id, origin, unit_direction, max_t)
    if query.result:
        location = origin + unit_direction * query.t
        distance = query.t
        face = query.face
        weights = wp.vec3(query.u, query.v, wp.float32(1.0) - query.u - query.v)
    if both_ways:
        backward = wp.mesh_query_ray(mesh_id, origin, -unit_direction, max_t)
        if backward.result:
            if backward.t < wp.abs(distance):
                location = origin - unit_direction * backward.t
                distance = -backward.t
                face = backward.face
                weights = wp.vec3(backward.u, backward.v, wp.float32(1.0) - backward.u - backward.v)
    return location, distance, face, weights


@wp.func
def first_hit(
    mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32
) -> tuple[wp.int32, wp.vec3]:
    # ``first_hit_record``'s face and location, forward only: what ``intersects_first`` and the
    # sparse ``intersects_location`` kernels read.
    location, _distance, face, _weights = first_hit_record(mesh_id, origin, direction, max_t, False)
    return face, location


@wp.kernel
def first_hits(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.float32,
    both_ways: wp.bool,
    out_locations: wp.array[wp.vec3],
    out_distances: wp.array[wp.float32],
    out_faces: wp.array[wp.int32],
    out_barycentric: wp.array[wp.vec3],
) -> None:
    # ``ray.first_hits``: every ray's ``first_hit_record`` at its own index.
    i = wp.int32(wp.tid())
    location, distance, face, weights = first_hit_record(
        mesh_id, ray_origins[i], ray_directions[i], max_t, both_ways
    )
    out_locations[i] = location
    out_distances[i] = distance
    out_faces[i] = face
    out_barycentric[i] = weights


@wp.kernel
def first_hit_append(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.float32,
    out_index_ray: wp.array[wp.int32],
    out_index_tri: wp.array[wp.int32],
    out_locations: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    # Single-pass compaction: each ray that hits claims one output slot, in arrival order (one
    # atomic a warp, ``append_slot``); the deterministic setting takes ``first_hit_flags`` instead.
    i = wp.int32(wp.tid())
    face, location = first_hit(mesh_id, ray_origins[i], ray_directions[i], max_t)
    if face >= 0:
        slot = append_slot(out_count, 0)
        out_index_ray[slot] = i
        out_index_tri[slot] = face
        out_locations[slot] = location


@wp.kernel
def first_hit_flags(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.float32,
    out_faces: wp.array[wp.int32],
    out_locations: wp.array[wp.vec3],
    out_flags: wp.array[wp.int32],
) -> None:
    # ``first_hit_append``'s ray-ordered form, for the deterministic setting: every ray's hit at
    # its own index and a ``0`` / ``1`` hit flag, which the wrapper scans in place and
    # ``compact_scanned_hits`` compacts in ascending ray order.
    i = wp.int32(wp.tid())
    face, location = first_hit(mesh_id, ray_origins[i], ray_directions[i], max_t)
    out_faces[i] = face
    out_locations[i] = location
    out_flags[i] = wp.where(face >= 0, 1, 0)


@wp.kernel
def compact_scanned_hits(
    faces: wp.array[wp.int32],
    locations: wp.array[wp.vec3],
    inclusive: wp.array[wp.int32],
    out_index_ray: wp.array[wp.int32],
    out_index_tri: wp.array[wp.int32],
    out_locations: wp.array[wp.vec3],
) -> None:
    # The hits ``first_hit_flags`` wrote, at their rank in its scanned flags.
    i = wp.int32(wp.tid())
    face = faces[i]
    if face >= 0:
        slot = inclusive[i] - 1
        out_index_ray[slot] = i
        out_index_tri[slot] = face
        out_locations[slot] = locations[i]


@wp.func
def any_hit(mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32) -> wp.bool:
    return wp.mesh_query_ray_anyhit(mesh_id, origin, wp.normalize(direction), max_t)


@wp.func
def longest_ray_distance(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    direction: wp.vec3,
    max_t: wp.float32,
    planar_tol: wp.float32,
) -> wp.float32:
    # ``direction`` need not be unit length; the offset walk below assumes a unit ray.
    #
    # The 64-step cap bounds how many hits inside ``planar_tol`` of the origin the walk will step
    # over before giving up, and exhausting it returns ``wp.inf`` -- i.e. reports the ray as
    # unobstructed when it is not. Reaching that needs 64 surfaces stacked within one
    # ``planar_tol`` band along one ray, which takes coincident or duplicated geometry rather
    # than a fine mesh; a caller who has that should dedupe (``repair.merge_vertices``) instead of
    # raising the cap, since every extra step is a full BVH descent for every ray in the launch.
    unit_direction = wp.normalize(direction)
    t_offset = wp.float32(0.0)
    cur_origin = origin
    for _i in range(64):
        remaining = max_t - t_offset
        if remaining <= wp.float32(0.0):
            break
        query = wp.mesh_query_ray(mesh_id, cur_origin, unit_direction, remaining)
        if not query.result:
            break
        dist = t_offset + query.t
        if dist > planar_tol:
            return dist
        t_offset = t_offset + query.t + planar_tol
        cur_origin = origin + unit_direction * t_offset
        if t_offset >= max_t:
            break
    return wp.inf


# -- segment casts (``Impl::RayCast``) -------------------------------------------------------------
# ``segment_hits``: Manifold's ``RayCast``, through ``predicates.shadow_edge_face``. Its kernels
# build unfused, each in a module of its own (``fuse_fp`` is a module option), so every hit is
# Manifold's bit for bit.


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
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
    x12, v12 = shadow_edge_face(
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


# Each segment's hits are found in four passes, as ``boolean``'s edge crossings are: its ray walks
# the face tree twice, counting its candidate faces and then writing them; ``segment_hit``
# decides each (segment, face) candidate in a thread of its own; and a thread a segment moves its
# hits into place in order along it. A thread a segment deciding its candidates in turn, once to
# count and again to write, cost the walks twice over (2026-11-21: 10.9 + 5.3 ms device time for
# 262 k random segments through a 328 k-face sphere).


@wp.kernel(enable_backward=False, module="unique")
def count_segment_candidates(
    ray_pos: wp.array[wp.vec3d], bvh_b: wp.uint64, out_counts: wp.array[wp.int32]
) -> None:
    # How many faces' boxes (padded past ``float32`` rounding) each segment passes through.
    s = wp.int32(wp.tid())
    query = wp.bvh_query_ray(bvh_b, wp.vec3(ray_pos[2 * s]), segment_direction(ray_pos, s))
    face = wp.int32(0)
    count = wp.int32(0)
    while wp.bvh_query_next(query, face, SEGMENT_REACH + wp.float32(0.0)):
        count += 1
    out_counts[s] = count


@wp.kernel(enable_backward=False, module="unique")
def write_segment_candidates(
    ray_pos: wp.array[wp.vec3d],
    bvh_b: wp.uint64,
    offsets: wp.array[wp.int32],
    out_segment: wp.array[wp.int32],
    out_face: wp.array[wp.int32],
) -> None:
    # The candidates ``count_segment_candidates`` counted, in the walk's order.
    s = wp.int32(wp.tid())
    slot = offsets[s]
    stop = offsets[s + 1]
    if slot == stop:
        return
    query = wp.bvh_query_ray(bvh_b, wp.vec3(ray_pos[2 * s]), segment_direction(ray_pos, s))
    face = wp.int32(0)
    while wp.bvh_query_next(query, face, SEGMENT_REACH + wp.float32(0.0)):
        if slot < stop:
            out_segment[slot] = s
            out_face[slot] = face
            slot += 1


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def decide_segment_candidates(
    ray_faces: wp.array[wp.int32],
    ray_pos: wp.array[wp.vec3d],
    ray_normals: wp.array[wp.vec3d],
    ray_mates: wp.array[wp.int32],
    faces_b: wp.array[wp.int32],
    pos_b: wp.array[wp.vec3d],
    vnormal_b: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    mates_b: wp.array[wp.int32],
    segments: wp.array[wp.int32],
    candidate_faces: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_parameters: wp.array[wp.float64],
    out_points: wp.array[wp.vec3d],
) -> None:
    # ``segment_hit`` on one (segment, face) candidate: its parameter and crossing, and a hit flag
    # into the unzeroed ``(n_candidates + 1,)`` offsets after their leading zero.
    c = wp.int32(wp.tid())
    if c == 0:
        out_offsets[0] = 0
    hit, t, point = segment_hit(
        segments[c],
        candidate_faces[c],
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
    out_offsets[c + 1] = wp.where(hit, 1, 0)
    out_parameters[c] = t
    out_points[c] = point


@wp.kernel(enable_backward=False, module="unique")
def write_segment_hits(
    candidate_offsets: wp.array[wp.int32],
    hit_offsets: wp.array[wp.int32],
    candidate_faces: wp.array[wp.int32],
    parameters: wp.array[wp.float64],
    points: wp.array[wp.vec3d],
    fnormal_b: wp.array[wp.vec3d],
    face_to_input: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
    out_parameters: wp.array[wp.float64],
    out_points: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # Each segment's hits in order along it, ties by the face's input index then the walk's
    # order: the face (as the caller numbered it), the parameter, the crossing and the face's
    # normal; and its first hit's slot, a thread past the last segment writing the total.
    s = wp.int32(wp.tid())
    first_candidate = candidate_offsets[s]
    first = hit_offsets[first_candidate]
    out_offsets[s] = first
    if s == candidate_offsets.shape[0] - 1:
        return
    slot = first
    for c in range(first_candidate, candidate_offsets[s + 1]):
        if hit_offsets[c + 1] == hit_offsets[c]:
            continue
        face = candidate_faces[c]
        index = face_to_input[face]
        t = parameters[c]
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
        out_points[at] = wp.vec3(points[c])
        out_normals[at] = wp.vec3(fnormal_b[face])
        slot += 1


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (any_hit, (wp.uint64(1), dense(wp.vec3), dense(wp.vec3), wp.float32(1)), wp.bool),
            (any_hit, (wp.uint64(1), single(wp.vec3), single(wp.vec3), wp.float32(1)), wp.bool),
        ]
    )


_declare_map_kernels()


@declare_lazy_kernels
def _declare_lazy_map_kernels() -> None:
    """Declare the ``wp.map`` signatures this module's ops are reached at, each its own module."""
    dense = map_probe
    declare_map_signatures(
        [
            (
                first_hit,
                (wp.uint64(1), dense(wp.vec3), dense(wp.vec3), wp.float32(1)),
                (wp.int32, wp.vec3),
            ),
            (
                longest_ray_distance,
                (wp.uint64(1), dense(wp.vec3), dense(wp.vec3), wp.float32(1), wp.float32(1)),
                wp.float32,
            ),
        ]
    )
