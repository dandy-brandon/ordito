import warp as wp

from ordito import constants as twc
from ordito.kernels.array import cross2, csr_key, declare_block_dims, to_vec3d
from ordito.kernels.laplacian import signed_cotmatrix_rows
from ordito.kernels.linalg import free_row
from ordito.kernels.reduce import (
    block_chunk,
    block_chunk_1d,
    block_max,
    block_min,
    block_running_sum,
    block_sum,
    last_ticket,
    last_ticket_vector,
    store_block_total,
)
from ordito.kernels.triangles import (
    corner_triple,
    face_vertices,
    face_vertices_vec3d,
    known_local_corner,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def face_is_flipped(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], fi: wp.int32
) -> wp.bool:
    # Whether face ``fi`` has negative signed area in the plane. Shared by the mask kernel and the
    # flag kernel, which differ only in the dtype they store it as.
    v0, v1, v2 = face_vertices(vertices, faces, fi)
    e0 = v1 - v0
    e1 = v2 - v0
    # 2D signed area * 2 == det of libigl's homogeneous 3x3 matrix
    return cross2(e0, e1) < 0.0


@wp.kernel
def face_flipped_mask(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    fi = wp.int32(wp.tid())
    out_mask[fi] = face_is_flipped(vertices, faces, fi)


@wp.kernel
def face_flipped_flags(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # ``face_flipped_mask`` as the ``int32`` flag a caller scans in place to compact the flipped
    # face indices, with no bool mask in between.
    fi = wp.int32(wp.tid())
    out_flags[fi] = wp.where(face_is_flipped(vertices, faces, fi), wp.int32(1), wp.int32(0))


@wp.kernel
def scatter_fixed_uv(
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    out_fixed_mask: wp.array[wp.bool],
    out_fixed_values: wp.array2d[wp.float64],
) -> None:
    # Scatter the prescribed boundary positions into a ``(2, n_vertices)`` buffer (row 0 = u,
    # row 1 = v) so the system-assembly kernel can look up ``bc[c, j]`` by right-hand-side column
    # ``c`` and original vertex index ``j``. float64 to match the float64 conjugate-gradient path.
    #
    # The fixed-vertex mask is marked in the same pass -- the ``array.indices_to_mask`` of the same
    # indices, whose out-of-range rule (skip negative and ``>= n``) is the one applied here -- as
    # ``scatter_pinned_stacked`` does for LSCM's stacked layout. The skip is memory safety only: a
    # skipped index, or a repeated one (whose writes race), marks fewer vertices than the indices
    # list, and the wrapper rejects the call from the mask's free count, which it reads anyway.
    b = wp.int32(wp.tid())
    i = boundary_indices[b]
    if i < wp.int32(0) or i >= out_fixed_values.shape[1]:
        return
    uv = boundary_uv[b]
    out_fixed_mask[i] = wp.bool(True)
    out_fixed_values[0, i] = wp.float64(uv[0])
    out_fixed_values[1, i] = wp.float64(uv[1])


@wp.func
def column_uv(columns: wp.array2d[wp.float64], i: wp.int32) -> wp.vec2:
    # Entry ``i`` of a ``(2, n)`` ``float64`` buffer (rows u, v) narrowed to a ``float32`` UV: a
    # fixed vertex's prescribed position, or a free one's solved value at its compact index.
    return wp.vec2(wp.float32(columns[0, i]), wp.float32(columns[1, i]))


@wp.kernel
def negated_cotmatrix_rows(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    run_start: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    count: wp.int32,
    order: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    out_values: wp.array[wp.float64],
) -> None:
    # The ``k == 1`` harmonic system ``-L`` over a mesh operator pattern, one thread per row:
    # ``kernels/laplacian.cotmatrix_rows``' ``float64`` row with each entry negated as it is
    # written (``signed_cotmatrix_rows``, a long row by the warp, as there), so ``L`` is never
    # stored. Negation is exact, so the entries are ``energies.k_harmonic(L, k=1)``'s bit for bit.
    i = wp.int32(wp.tid())
    signed_cotmatrix_rows(
        i, offsets, columns, run_start, keys, count, order, cot_entries, 1, out_values
    )


@wp.kernel
def free_mass_roots(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    mass: wp.array[wp.float64],
    out_roots: wp.array[wp.float64],
) -> None:
    # ``sqrt(M_ii)`` at each free vertex's compact index: the diagonal ``D`` for which the k = 2
    # operator ``L M^-1 L`` is the ``L D^-2 L`` a squared-Laplacian preconditioner inverts. A
    # vertex no face refers to has zero mass and an empty row; 1 keeps ``D`` positive there.
    i = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    out_roots[ri] = wp.where(mass[i] > wp.float64(0.0), wp.sqrt(mass[i]), wp.float64(1.0))


@wp.func
def solved_uv(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array2d[wp.float64],
    fixed_values: wp.array2d[wp.float64],
    i: wp.int32,
) -> wp.vec2:
    # Vertex ``i``'s UV in a reduced solution: its prescribed position when fixed, else the solved
    # value at its compact index, narrowed to ``float32``. ``scatter_solution`` writes it; ARAP's
    # local step reads it in place of a scattered UV field.
    if fixed_mask[i]:
        return column_uv(fixed_values, i)
    return column_uv(sol, free_map[i])


@wp.kernel
def scatter_solution(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array2d[wp.float64],
    fixed_values: wp.array2d[wp.float64],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Reassemble the full ``(n_vertices,)`` UV field: fixed vertices keep their prescribed position
    # (``fixed_values`` is ``(2, n_vertices)``), free vertices read the solved value at their
    # compact index (``sol`` is ``(2, n_free)``). Handles the all-fixed case (``n_free == 0``): the
    # free branch is then never taken, so the empty ``sol`` is never indexed.
    i = wp.int32(wp.tid())
    out_uv[i] = solved_uv(fixed_mask, free_map, sol, fixed_values, i)


@wp.func
def boundary_segment_length(
    boundary: wp.array[wp.int32], vertices: wp.array[wp.vec3], i: wp.int32
) -> wp.float32:
    # Length of the loop segment ending at boundary vertex ``i``; ``0`` at ``i == 0``, which seeds
    # the arc-length prefix sum (matches ``igl::map_vertices_to_circle``).
    if i == 0:
        return wp.float32(0.0)
    return wp.length(vertices[boundary[i]] - vertices[boundary[i - 1]])


@wp.func
def loop_perimeter(
    boundary: wp.array[wp.int32], vertices: wp.array[wp.vec3], open_length: wp.float32
) -> wp.float32:
    # The loop's perimeter from the arc length to its last vertex, closing over the wrap edge
    # ``bnd[0] -> bnd[n-1]``. A single-vertex loop, or one whose vertices are all coincident, has
    # zero perimeter -- every arc length is then 0 too -- so 1 stands in for it: the only guard
    # needed against the 0/0 that would otherwise write NaN into every position.
    n = boundary.shape[0]
    total = open_length + wp.length(vertices[boundary[0]] - vertices[boundary[n - 1]])
    return wp.where(total <= 0.0, wp.float32(1.0), total)


@wp.func
def circle_position(arc_length: wp.float32, perimeter: wp.float32) -> wp.vec2:
    # Arc-length parametrization onto the unit circle: ``frac = len[i] * 2pi / total``.
    frac = arc_length * twc.TWO_PI / perimeter
    return wp.vec2(wp.cos(frac), wp.sin(frac))


@wp.kernel
def boundary_edge_lengths(
    boundary: wp.array[wp.int32], vertices: wp.array[wp.vec3], out_len: wp.array[wp.float32]
) -> None:
    # Segment length between consecutive boundary vertices, scanned by the wrapper into each
    # vertex's arc length.
    i = wp.int32(wp.tid())
    out_len[i] = boundary_segment_length(boundary, vertices, i)


@wp.kernel
def circle_positions(
    boundary: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    cumulative_length: wp.array[wp.float32],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Each boundary vertex's circle position from its scanned arc length.
    i = wp.int32(wp.tid())
    n = cumulative_length.shape[0]
    perimeter = loop_perimeter(boundary, vertices, cumulative_length[n - 1])
    out_uv[i] = circle_position(cumulative_length[i], perimeter)


@wp.kernel
def circle_positions_block(
    boundary: wp.array[wp.int32], vertices: wp.array[wp.vec3], out_uv: wp.array[wp.vec2]
) -> None:
    # ``boundary_edge_lengths``, the scan and ``circle_positions`` as one block, for a short loop:
    # the segment lengths are ranked a block-wide chunk at a time by an inclusive tile scan carried
    # across chunks, each arc length staged in its own vertex's ``out_uv`` slot, and once the
    # perimeter is known every lane rewrites the slots it staged (no barrier: a lane reads only
    # its own). The two forms differ in the scan's summation order alone; both are fixed. Every
    # lane strides by ``wp.block_dim()``, which reads 1 on the CPU device.
    _block, lane = wp.tid()
    lanes = wp.block_dim()
    n = boundary.shape[0]
    carry = wp.float32(0.0)
    for start in range(0, n, lanes):
        i = start + lane
        segment = wp.float32(0.0)
        if i < n:
            segment = boundary_segment_length(boundary, vertices, i)
        running, carry = block_running_sum(segment, carry)
        if i < n:
            out_uv[i] = wp.vec2(running, wp.float32(0.0))
    perimeter = loop_perimeter(boundary, vertices, carry)
    for start in range(0, n, lanes):
        i = start + lane
        if i < n:
            out_uv[i] = circle_position(out_uv[i][0], perimeter)


@wp.kernel
def scatter_pinned_stacked(
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
    n_vertices: wp.int32,
    out_fixed_mask: wp.array[wp.bool],
    out_fixed_values: wp.array2d[wp.float64],
) -> None:
    # Fused mask + value scatter for LSCM's stacked ``[u; v]`` DOFs: pin ``i`` fixes DOF ``i`` (u)
    # and ``i + n`` (v). ``out_fixed_values`` is ``(1, 2n)`` (single right-hand-side column). An
    # out-of-range pin is skipped and a repeated one marks its vertex once, which the wrapper
    # rejects from the free count, as for ``scatter_fixed_uv``.
    b = wp.int32(wp.tid())
    i = pinned_indices[b]
    if i < wp.int32(0) or i >= n_vertices:
        return
    uv = pinned_uv[b]
    out_fixed_mask[i] = True
    out_fixed_mask[i + n_vertices] = True
    out_fixed_values[0, i] = wp.float64(uv[0])
    out_fixed_values[0, i + n_vertices] = wp.float64(uv[1])


@wp.func
def stacked_dof(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array[wp.float64],
    fixed_values: wp.array[wp.float64],
    j: wp.int32,
) -> wp.float64:
    # DOF ``j`` of a stacked ``[u; v]`` solution: its pinned value, or the solved free value at
    # its compact index.
    if fixed_mask[j]:
        return fixed_values[j]
    return sol[free_map[j]]


@wp.kernel
def scatter_solution_stacked(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array[wp.float64],
    fixed_values: wp.array[wp.float64],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Unstack the solved ``[u; v]`` DOF vector into ``(n, 2)`` UV. DOF ``i`` holds u, DOF ``i + n``
    # holds v; each is either a pinned value (``fixed_values``) or a solved free value (``sol`` at
    # the compact free index), ``stacked_dof``.
    i = wp.int32(wp.tid())
    n = out_uv.shape[0]
    u = stacked_dof(fixed_mask, free_map, sol, fixed_values, i)
    v = stacked_dof(fixed_mask, free_map, sol, fixed_values, i + n)
    out_uv[i] = wp.vec2(wp.float32(u), wp.float32(v))


@wp.func
def gather_interior_uv(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    i: wp.int32,
    out_sol: wp.array2d[wp.float64],
) -> None:
    # Seed the conjugate-gradient warm start with the current interior UV at vertex ``i``:
    # interior vertex ``i`` writes its UV into the two rows (u, v) of ``out_sol`` at the compact
    # free index. ``out_sol`` is (2, n_interior), and every row is written, because ``free_map`` is
    # a bijection of the free vertices onto it. The first local step reads the working UV field
    # off this seed (``solved_uv``), which gives a free vertex ``uv[i]`` itself: a ``float32``
    # widened to ``float64`` and narrowed again is exact.
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    p = uv[i]
    out_sol[0, ri] = wp.float64(p[0])
    out_sol[1, ri] = wp.float64(p[1])


# The element ARAP pieces every ARAP here shares (``arap``, ``low_distortion_parametrization``,
# ``arap_parametrization``): one rest shape per face (its shape-function gradients
# and area), one local step (``face_rotation``) and one right-hand side, gathered per vertex
# (``rotation_row``) rather than scattered, so a vertex's sum is one launch with no zeroing and,
# under ``ordito.config``'s deterministic setting, in the adjacency rows' ascending face order.
# Charts are ``(2, n)`` ``float64`` buffers (rows u, v), the layout the solves take.


@wp.func
def face_gradients(
    p0: wp.vec3d, p1: wp.vec3d, p2: wp.vec3d
) -> tuple[wp.vec2d, wp.vec2d, wp.vec2d, wp.float64]:
    # A face's three shape-function gradients in an isometric frame of its own plane (first axis
    # along the edge from corner 0 to corner 1), and its area; all zero for a face of no area, so
    # such a face adds nothing to any sum.
    first = p1 - p0
    second = p2 - p0
    double_area = wp.length(wp.cross(first, second))
    zero = wp.vec2d()
    if not (double_area > wp.float64(0.0) and wp.isfinite(double_area)):
        return zero, zero, zero, wp.float64(0.0)
    length = wp.length(first)
    projection = wp.dot(first, second) / length
    height = double_area / length
    gradient1 = wp.vec2d(wp.float64(1.0) / length, -projection / double_area)
    gradient2 = wp.vec2d(wp.float64(0.0), wp.float64(1.0) / height)
    return -gradient1 - gradient2, gradient1, gradient2, wp.float64(0.5) * double_area


@wp.func
def store_face_gradients(
    f: wp.int32,
    g0: wp.vec2d,
    g1: wp.vec2d,
    g2: wp.vec2d,
    area: wp.float64,
    out_gradients: wp.array2d[wp.vec2d],
    out_areas: wp.array[wp.float64],
) -> None:
    # Row ``f`` of the per-face gradient table and its area.
    out_gradients[f, 0] = g0
    out_gradients[f, 1] = g1
    out_gradients[f, 2] = g2
    out_areas[f] = area


@wp.func
def chart_point(chart: wp.array2d[wp.float64], i: wp.int32) -> wp.vec2d:
    # Vertex ``i``'s UV in a ``(2, n)`` chart.
    return wp.vec2d(chart[0, i], chart[1, i])


@wp.func
def store_chart_point(chart: wp.array2d[wp.float64], i: wp.int32, uv: wp.vec2d) -> None:
    # Write vertex ``i``'s UV into a ``(2, n)`` chart.
    chart[0, i] = uv[0]
    chart[1, i] = uv[1]


@wp.func
def chart_vertex(
    chart: wp.array2d[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array2d[wp.float64],
    i: wp.int32,
) -> wp.vec2d:
    # Vertex ``i``'s UV, the one reader of every ARAP local step. With an empty ``fixed_mask`` it is
    # the full ``(2, n)`` chart's row; otherwise (``arap``'s pinned system) a free vertex reads the
    # reduced ``solution`` at its compact index and a pinned one its prescribed position in
    # ``chart``. The branch is warp-uniform.
    if fixed_mask.shape[0] > 0:
        if not fixed_mask[i]:
            return chart_point(solution, free_map[i])
    return chart_point(chart, i)


@wp.func
def face_rotation(
    uv0: wp.vec2d, uv1: wp.vec2d, uv2: wp.vec2d, g1: wp.vec2d, g2: wp.vec2d
) -> wp.vec2d:
    # ``(cos, sin)`` of the proper rotation nearest a face's Jacobian ``J = [grad u; grad v]``:
    # ``cos ~ a + d``, ``sin ~ c - b`` normalized. A collapsed Jacobian, or an exactly ambiguous
    # reflection, admits any rotation: the identity.
    first = uv1 - uv0
    second = uv2 - uv0
    gradient_u = first[0] * g1 + second[0] * g2
    gradient_v = first[1] * g1 + second[1] * g2
    cosine = gradient_u[0] + gradient_v[1]
    sine = gradient_v[0] - gradient_u[1]
    magnitude = wp.sqrt(cosine * cosine + sine * sine)
    if magnitude == wp.float64(0.0):
        return wp.vec2d(wp.float64(1.0), wp.float64(0.0))
    return wp.vec2d(cosine / magnitude, sine / magnitude)


@wp.func
def chart_face_rotation(
    faces: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    chart: wp.array2d[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array2d[wp.float64],
    f: wp.int32,
) -> wp.vec2d:
    # Face ``f``'s rotation fitted to the UV ``chart_vertex`` reads: the local step.
    i0, i1, i2 = corner_triple(faces, f)
    return face_rotation(
        chart_vertex(chart, fixed_mask, free_map, solution, i0),
        chart_vertex(chart, fixed_mask, free_map, solution, i1),
        chart_vertex(chart, fixed_mask, free_map, solution, i2),
        gradients[f, 1], gradients[f, 2],
    )  # fmt: skip


@wp.func
def rotated_gradient(
    faces: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    rotation: wp.vec2d,
    f: wp.int32,
    v: wp.int32,
) -> wp.vec2d:
    # Face ``f``'s term of vertex ``v``'s ``rotation_row``: its ``rotation`` applied to the
    # gradient at ``v``'s corner, times its area.
    g = gradients[f, known_local_corner(faces, f, v)]
    return areas[f] * wp.vec2d(
        g[0] * rotation[0] - g[1] * rotation[1], g[0] * rotation[1] + g[1] * rotation[0]
    )


@wp.func
def rotation_row(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    chart: wp.array2d[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array2d[wp.float64],
    v: wp.int32,
) -> wp.vec2d:
    # Vertex ``v``'s row of the global step's right-hand side ``sum_f A_f R_f g_k``: each incident
    # face's rotation fitted to the UV ``chart_vertex`` reads (the local step, recomputed by each of
    # its corners) applied to the gradient at ``v``'s corner, in the adjacency row's order. For an
    # isometric flattening ``P`` of the face, ``A G G^T P = A g_k``: the rotated rest edges
    # ``sum_e c_e R (P_i - P_j)`` libigl's element ARAP scatters, gathered per vertex. The proximal
    # loop reads a table instead (``rotation_row_from_table``); ``arap_rhs`` and
    # ``smoothing_start`` keep the refit, their launches 2 % of ``arap``'s device time on
    # ``saddle`` and ``hemisphere`` and of a ``dragon`` chart (2026-11-20). Walking the row four
    # faces a step (``array.row_chunk_entries``) cost the proximal step 1.2x: the step is
    # ``float64``-bound, not load-bound.
    total = wp.vec2d()
    for s in range(vertex_offsets[v], vertex_offsets[v + 1]):
        f = vertex_faces[s]
        rotation = chart_face_rotation(faces, gradients, chart, fixed_mask, free_map, solution, f)
        total += rotated_gradient(faces, gradients, areas, rotation, f, v)
    return total


@wp.func
def rotation_row_from_table(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    rotations: wp.array[wp.vec2d],
    v: wp.int32,
) -> wp.vec2d:
    # ``rotation_row`` with each face's rotation read from ``rotations`` (``face_rotations``),
    # formed once a face rather than once a corner.
    total = wp.vec2d()
    for s in range(vertex_offsets[v], vertex_offsets[v + 1]):
        f = vertex_faces[s]
        total += rotated_gradient(faces, gradients, areas, rotations[f], f, v)
    return total


@wp.kernel
def arap_setup(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    out_gradients: wp.array2d[wp.vec2d],
    out_areas: wp.array[wp.float64],
    out_sol: wp.array2d[wp.float64],
) -> None:
    # ``arap``'s two one-time inputs in one launch over ``max(n_faces, n_vertices)`` threads: each
    # face's gradients and area, and the warm start per vertex (``gather_interior_uv``).
    t = wp.int32(wp.tid())
    if t < faces.shape[0] // 3:
        p0, p1, p2 = face_vertices_vec3d(vertices, faces, t)
        g0, g1, g2, area = face_gradients(p0, p1, p2)
        store_face_gradients(t, g0, g1, g2, area, out_gradients, out_areas)
    if t < vertices.shape[0]:
        gather_interior_uv(fixed_mask, free_map, uv, t, out_sol)


@wp.kernel
def arap_rhs(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    fixed_values: wp.array2d[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array2d[wp.float64],
    rhs_const: wp.array2d[wp.float64],
    out_b: wp.array2d[wp.float64],
) -> None:
    # ``arap``'s local step and global right-hand side at each free vertex: the constant boundary
    # term plus its ``rotation_row``, the pinned vertices read at their prescribed positions.
    i = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    rotated = rotation_row(
        faces, vertex_faces, vertex_offsets, gradients, areas, fixed_values, fixed_mask, free_map,
        sol, i,
    )  # fmt: skip
    out_b[0, ri] = rhs_const[0, ri] + rotated[0]
    out_b[1, ri] = rhs_const[1, ri] + rotated[1]


# ``slim``: symmetric Dirichlet ``sum_f A_f (s1^2 + s2^2 + s1^-2 + s2^-2)`` by scalable locally
# injective maps (Rabinovich et al. 2017). Each face's proxy is the reweighted ARAP term
# ``A_f |W_f (J_f - R_f)|^2`` with ``W_f = U diag(w) U^T``, ``w_i^2 = (s_i - s_i^-3) / (s_i - 1)``,
# whose gradient at the current chart is the energy's own; its Hessian over the vertices is the
# ``wp.mat22d`` operator ``sum_f A_f (G_f G_f^T) (x) W_f^2`` over the stiffness pattern, plus the
# proximal anchor. A step solves that operator against the proxy's gradient (a Newton step on the
# proxy, the energy's own gradient wherever the weight is uncapped); the operator is refactored
# every few steps and the steps between take the held factor against the current gradient, so the
# line search still descends. Charts are ``(n,)`` ``wp.vec2d``,
# the interleaved unknowns of the operator's scalar expansion. Every per-vertex and per-entry sum
# is a gather in a fixed order; the trial energies fold through ``store_block_total``.
SLIM_TRIALS = 8
# The proxy's squared weight is 4 at rest and grows as ``s^-3`` as a face collapses; capped so the
# operator stays factorable (a face of ``s ~ 1e-8`` gave ``1e24`` and an indefinite pivot), and
# the step's gradient with it (``slim_face_terms``). The energy and the barrier are exact whatever
# the cap.
SLIM_WEIGHT_CAP = wp.float64(1.0e9)
# Of the largest step that inverts no face, the line search tries this fraction and its halvings.
SLIM_STEP_FRACTION = wp.float64(0.9)
SLIM_NO_BOUND = wp.float64(1.0e30)
# A face of the start squashed below this singular value, where its weight would pass the cap
# (``SLIM_WEIGHT_CAP ** (-1 / 3)``), is held like an inverted one, by the unbarriered ARAP term.
# Barriered, its capped stiffness lets the step crush it while its bound cuts the step to nothing:
# SLIM from Tutte, harmonic and LSCM starts of jaw scans with handles stopped where it began on 4
# of 32 runs (2026-11-20, 8 crops); tying the threshold to the cap ended every stall, at 1e-3
# with fewer unbarriered faces (inverted after: Tutte median 69, against 85 at 1e-2 with a cap of
# 1e6). A cap of 1e6 alone left 1e-4 and 1e-3 thresholds stalling.
SLIM_COLLAPSED = wp.float64(1.0e-3)
slim_trials_vector = wp.types.vector(length=SLIM_TRIALS + 1, dtype=wp.float64)


@wp.func
def slim_jacobian(
    uv0: wp.vec2d, uv1: wp.vec2d, uv2: wp.vec2d, g0: wp.vec2d, g1: wp.vec2d, g2: wp.vec2d
) -> wp.mat22d:
    # A face's Jacobian from its isometric frame to the chart, ``J[r, c] = sum_k uv_k[r] g_k[c]``.
    return wp.outer(uv0, g0) + wp.outer(uv1, g1) + wp.outer(uv2, g2)


@wp.func
def chart_jacobian(
    chart: wp.array[wp.vec2d],
    faces: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    f: wp.int32,
) -> wp.mat22d:
    # Face ``f``'s ``slim_jacobian`` on a ``wp.vec2d`` chart.
    i0, i1, i2 = corner_triple(faces, f)
    return slim_jacobian(
        chart[i0], chart[i1], chart[i2], gradients[f, 0], gradients[f, 1], gradients[f, 2]
    )


@wp.func
def signed_svd2(j: wp.mat22d) -> wp.vec4d:
    # ``J = Rot(phi) diag(s1, s2) Rot(theta)`` in closed form, ``s1 >= |s2|`` and ``s2`` carrying
    # ``det J``'s sign: ``(s1, s2, phi, rotation)``, where ``theta = rotation - phi`` and
    # ``Rot(rotation) = U V^T`` is the proper rotation nearest ``J``.
    half = wp.float64(0.5)
    e = (j[0, 0] + j[1, 1]) * half
    f = (j[0, 0] - j[1, 1]) * half
    g = (j[1, 0] + j[0, 1]) * half
    h = (j[1, 0] - j[0, 1]) * half
    q = wp.sqrt(e * e + h * h)
    r = wp.sqrt(f * f + g * g)
    a1 = wp.atan2(g, f)
    a2 = wp.atan2(h, e)
    return wp.vec4d(q + r, q - r, (a2 + a1) * half, a2)


@wp.func
def rotation2(angle: wp.float64) -> wp.mat22d:
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.mat22d(c, -s, s, c)


@wp.func
def slim_weight_squared(s: wp.float64) -> wp.float64:
    # The symmetric Dirichlet proxy weight ``(s - s^-3) / (s - 1)``, 4 at ``s = 1``, capped.
    one = wp.float64(1.0)
    x = wp.max(wp.abs(s), wp.float64(1.0e-8))
    if wp.abs(x - one) < wp.float64(1.0e-6):
        return wp.float64(4.0)
    return wp.min((x - one / (x * x * x)) / (x - one), SLIM_WEIGHT_CAP)


@wp.func
def slim_face_energy(j: wp.mat22d, inverted: wp.int32) -> wp.float64:
    # A face's energy less its rest value: ``|J|^2 (1 + det^-2) - 4``, infinite once inverted (the
    # barrier); a face inverted in the start takes ``4 |J - R|^2``, the proxy at rest, unbarriered.
    squared = wp.ddot(j, j)
    if inverted != 0:
        nearest = wp.float64(0.5) * wp.sqrt(
            (j[0, 0] + j[1, 1]) * (j[0, 0] + j[1, 1]) + (j[1, 0] - j[0, 1]) * (j[1, 0] - j[0, 1])
        )
        return wp.float64(4.0) * (squared - wp.float64(4.0) * nearest + wp.float64(2.0))
    det = wp.determinant(j)
    if det <= wp.float64(0.0):
        return wp.float64(wp.inf)
    return squared * (wp.float64(1.0) + wp.float64(1.0) / (det * det)) - wp.float64(4.0)


@wp.kernel
def slim_start(
    uv: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    start_areas: wp.array[wp.vec3d],
    out_chart: wp.array[wp.vec2d],
    out_inverted: wp.array[wp.int32],
) -> None:
    # Over ``max(n_vertices, n_faces)``: the start as a ``float64`` chart, mirrored in v when its
    # faces wind clockwise on the whole (``start_chart_areas``' signed area), since the barrier
    # keeps every face's orientation; and each face's flag, set for a face of no area, one
    # inverted in the (mirrored) start or one squashed past ``SLIM_COLLAPSED``, which the barrier
    # then leaves free.
    t = wp.int32(wp.tid())
    mirror = wp.where(start_areas[0][2] < wp.float64(0.0), wp.float64(-1.0), wp.float64(1.0))
    if t < uv.shape[0]:
        p = uv[t]
        out_chart[t] = wp.vec2d(wp.float64(p[0]), mirror * wp.float64(p[1]))
    if t < faces.shape[0] // 3:
        i0, i1, i2 = corner_triple(faces, t)
        u0 = uv[i0]
        u1 = uv[i1]
        u2 = uv[i2]
        j = slim_jacobian(
            wp.vec2d(wp.float64(u0[0]), wp.float64(u0[1])),
            wp.vec2d(wp.float64(u1[0]), wp.float64(u1[1])),
            wp.vec2d(wp.float64(u2[0]), wp.float64(u2[1])),
            gradients[t, 0], gradients[t, 1], gradients[t, 2],
        )  # fmt: skip
        squashed = wp.abs(signed_svd2(j)[1]) < SLIM_COLLAPSED
        out_inverted[t] = wp.where(
            areas[t] == wp.float64(0.0)
            or mirror * wp.determinant(j) <= wp.float64(0.0)
            or squashed,
            1,
            0,
        )


@wp.kernel
def slim_face_terms(
    chart: wp.array[wp.vec2d],
    faces: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    inverted: wp.array[wp.int32],
    out_metric: wp.array[wp.mat22d],
    out_force: wp.array[wp.mat22d],
) -> None:
    # A face's proxy metric ``A W^2`` (the Hessian block's factor) and its gradient factor
    # ``A U diag(w^2 (s - 1)) V^T`` (half ``d proxy / d J``, the energy's ``A U diag(s - s^-3) V^T``
    # where the weight is uncapped; a vertex's gradient is the sum of its faces' factor times the
    # corner's shape gradient). A start-inverted face takes the ARAP proxy, ``4 A I`` and
    # ``4 A (J - R)``.
    f = wp.int32(wp.tid())
    area = areas[f]
    j = chart_jacobian(chart, faces, gradients, f)
    svd = signed_svd2(j)
    if inverted[f] != 0:
        out_metric[f] = wp.float64(4.0) * area * wp.identity(n=2, dtype=wp.float64)
        out_force[f] = wp.float64(4.0) * area * (j - rotation2(svd[3]))
        return
    s1 = svd[0]
    s2 = svd[1]
    w1 = slim_weight_squared(s1)
    w2 = slim_weight_squared(s2)
    u = rotation2(svd[2])
    vt = rotation2(svd[3] - svd[2])
    out_metric[f] = area * (u * wp.diag(wp.vec2d(w1, w2)) * wp.transpose(u))
    # The proxy's gradient ``w^2 (s - 1)``: the energy's own ``s - s^-3`` wherever the weight is
    # uncapped, and capped with it near collapse. The energy's own there (``~s^-3``) against the
    # capped stiffness made a nearly collapsed face's step ~1e18 too long, which the step bound
    # then cut to nothing for the whole chart (a Tutte start of a jaw scan with handles did not
    # move). A start's collapsed faces are now unbarriered (``SLIM_COLLAPSED``), so this matters
    # for a face that collapses during the iteration.
    one = wp.float64(1.0)
    out_force[f] = area * (u * wp.diag(wp.vec2d(w1 * (s1 - one), w2 * (s2 - one))) * vt)


@wp.kernel
def slim_block_values(
    offsets: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    order: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    metric: wp.array[wp.mat22d],
    anchor: wp.array[wp.float64],
    n_rows: wp.int32,
    out_values: wp.array[wp.mat22d],
) -> None:
    # Each stored block of the proxy Hessian ``sum_f (g_a . g_b) A W^2 + epsilon I`` over the
    # stiffness pattern (``chart_stiffness_setup``'s keys), its contributions added in key order
    # as ``chart_stiffness_values`` adds the scalar stiffness's. Launched over the key count.
    e = wp.int32(wp.tid())
    if e >= offsets[n_rows]:
        return
    face_keys = TWO_LEVEL_FACE_KEYS * gradients.shape[0]
    value = wp.mat22d()
    for s in range(starts[e], starts[e + 1]):
        payload = order[s]
        if payload >= face_keys:
            value += anchor[0] * wp.identity(n=2, dtype=wp.float64)
        else:
            f = payload // TWO_LEVEL_FACE_KEYS
            a = (payload // 3) % 3
            b = payload % 3
            value += wp.dot(gradients[f, a], gradients[f, b]) * metric[f]
    out_values[e] = value


@wp.kernel
def slim_rhs(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    force: wp.array[wp.mat22d],
    out_rhs: wp.array[wp.vec2d],
) -> None:
    # Minus half the proxy's gradient at each vertex, ``-sum_f F_f g_k``, gathered over the
    # adjacency row in its order: the step's right-hand side.
    v = wp.int32(wp.tid())
    total = wp.vec2d()
    for s in range(vertex_offsets[v], vertex_offsets[v + 1]):
        f = vertex_faces[s]
        total += force[f] * gradients[f, known_local_corner(faces, f, v)]
    out_rhs[v] = -total


@wp.func
def slim_face_bound(j: wp.mat22d, d: wp.mat22d) -> wp.float64:
    # The least positive ``t`` with ``det(J + t D) = 0``, ``SLIM_NO_BOUND`` if none: the step past
    # which this face inverts. ``det(J + t D) = det J + t c + t^2 det D``.
    a2 = wp.determinant(d)
    a1 = j[0, 0] * d[1, 1] + j[1, 1] * d[0, 0] - j[0, 1] * d[1, 0] - j[1, 0] * d[0, 1]
    a0 = wp.determinant(j)
    root = SLIM_NO_BOUND
    if wp.abs(a2) <= wp.float64(1.0e-14) * wp.abs(a1):
        if a1 < wp.float64(0.0):
            root = -a0 / a1
    else:
        discriminant = a1 * a1 - wp.float64(4.0) * a2 * a0
        if discriminant >= wp.float64(0.0):
            sq = wp.sqrt(discriminant)
            for r in range(2):
                sign = wp.where(r == 0, wp.float64(-1.0), wp.float64(1.0))
                t = (-a1 + sign * sq) / (wp.float64(2.0) * a2)
                if t > wp.float64(0.0):
                    root = wp.min(root, t)
    return root


@wp.kernel
def slim_step_bound(
    chart: wp.array[wp.vec2d],
    step: wp.array[wp.vec2d],
    faces: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    inverted: wp.array[wp.int32],
    iteration: wp.int32,
    width: wp.int32,
    out_bounds: wp.array[wp.float64],
) -> None:
    # The largest step inverting no barriered face into ``out_bounds[iteration]`` (filled with
    # ``SLIM_NO_BOUND``), lane-strided over ``width`` faces a block (``spanned_blocks_1d``, as
    # ``slim_trial_energies``): a block minimum, then ``atomic_min``, exact in any order.
    block, lane = wp.tid()
    offset, count = block_chunk(faces.shape[0] // 3, block, width)
    least = SLIM_NO_BOUND
    for k in range(lane, count, wp.block_dim()):
        f = offset + k
        if inverted[f] == 0:
            least = wp.min(
                least,
                slim_face_bound(
                    chart_jacobian(chart, faces, gradients, f),
                    chart_jacobian(step, faces, gradients, f),
                ),
            )
    block_least = block_min(least)
    if lane == 0 and block_least < SLIM_NO_BOUND:
        wp.atomic_min(out_bounds, iteration, block_least)


@wp.func
def slim_trial_step(bound: wp.float64, k: wp.int32) -> wp.float64:
    # Trial ``k``'s step: ``SLIM_STEP_FRACTION`` of the bound (at most a full step) halved ``k``
    # times; trial ``SLIM_TRIALS`` stays put.
    if k == SLIM_TRIALS:
        return wp.float64(0.0)
    first = wp.min(wp.float64(1.0), SLIM_STEP_FRACTION * bound)
    return first * wp.pow(wp.float64(0.5), wp.float64(k))


@wp.kernel
def slim_trial_energies(
    chart: wp.array[wp.vec2d],
    step: wp.array[wp.vec2d],
    faces: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    inverted: wp.array[wp.int32],
    bounds: wp.array[wp.float64],
    iteration: wp.int32,
    width: wp.int32,
    out_partials: wp.array[slim_trials_vector],
) -> None:
    # The energy at every trial step at once, lane-strided over the faces into slot 0 of a
    # ``_launch.block_partials`` buffer: one launch and no readback for the whole line search.
    # ``width`` faces a block (``reduce.spanned_blocks_1d``): a face is nine ``float64`` energies
    # in a chain, so a lane walking the standard fold's sixteen held the launch at ~230 us from
    # 9 k to 164 k faces (2026-11-20).
    block, lane = wp.tid()
    offset, count = block_chunk(faces.shape[0] // 3, block, width)
    bound = bounds[iteration]
    sums = slim_trials_vector()
    for k in range(lane, count, wp.block_dim()):
        f = offset + k
        area = areas[f]
        if area > wp.float64(0.0):
            j = chart_jacobian(chart, faces, gradients, f)
            d = chart_jacobian(step, faces, gradients, f)
            flag = inverted[f]
            for trial in range(SLIM_TRIALS + 1):
                sums[trial] += area * slim_face_energy(j + slim_trial_step(bound, trial) * d, flag)
    store_block_total(block, lane, sums, out_partials, 0, 1, last_ticket_vector)


@wp.kernel
def slim_accept(
    partials: wp.array[slim_trials_vector],
    bounds: wp.array[wp.float64],
    iteration: wp.int32,
    out_steps: wp.array[wp.float64],
    out_energies: wp.array[wp.float64],
) -> None:
    # One thread: the longest trial step whose energy is at most the current one (0 if none), into
    # ``out_steps[iteration]``, and the energy before and after it into
    # ``out_energies[2 * iteration]`` and ``[2 * iteration + 1]``: the stop test reads a period's.
    energies = partials[0]
    accepted = wp.float64(0.0)
    reached = energies[SLIM_TRIALS]
    found = wp.int32(0)
    for trial in range(SLIM_TRIALS):
        if found == 0 and energies[trial] <= energies[SLIM_TRIALS]:
            found = 1
            accepted = slim_trial_step(bounds[iteration], trial)
            reached = energies[trial]
    out_steps[iteration] = accepted
    out_energies[2 * iteration] = energies[SLIM_TRIALS]
    out_energies[2 * iteration + 1] = reached


@wp.kernel
def slim_advance(
    step: wp.array[wp.vec2d],
    steps: wp.array[wp.float64],
    iteration: wp.int32,
    chart: wp.array[wp.vec2d],
) -> None:
    # Move the chart by the accepted fraction of the step, in place.
    v = wp.int32(wp.tid())
    chart[v] = chart[v] + steps[iteration] * step[v]


@wp.kernel
def slim_uv(
    chart: wp.array[wp.vec2d], start_areas: wp.array[wp.vec3d], out_uv: wp.array[wp.vec2]
) -> None:
    # The chart narrowed to ``float32`` UV, mirrored back to the start's winding.
    v = wp.int32(wp.tid())
    mirror = wp.where(start_areas[0][2] < wp.float64(0.0), wp.float64(-1.0), wp.float64(1.0))
    p = chart[v]
    out_uv[v] = wp.vec2(wp.float32(p[0]), wp.float32(mirror * p[1]))


# The low-distortion charts: pin-free element ARAP from a coarse level or a classic map's start,
# on the shared pieces above. Their per-vertex sums are gathers too (Jacobians, stiffness rows).

# A coarse face whose doubled area is at most this times ``coarse_cell ** 2`` joins three nearly
# collinear centroids and is dropped (the reference's threshold).
TWO_LEVEL_FLAT_FACE = wp.float64(2.0e-9)
# The proximal anchor ``epsilon`` is this fraction of the mean diagonal of the stiffness matrix.
TWO_LEVEL_ANCHOR = wp.float64(1.0e-6)
# Contributions a face makes to the coarse system's keys: its ``3 x 3`` block.
TWO_LEVEL_FACE_KEYS = 9

mat23d = wp.types.matrix(shape=(2, 3), dtype=wp.float64)


@wp.kernel
def two_level_bounds(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_bounds: wp.array[wp.float32]
) -> None:
    # The validation and grid inputs of the low-distortion charts in one lane-strided pass over
    # ``max(n_vertices, n_faces)`` items, committed by ``atomic_min`` (exact in any order) into
    # ``out_bounds``, eight slots filled with ``+inf``: the lower corner of the finite vertices
    # (0-2), the negated upper corner (3-5), 0 in slot 6 if any coordinate is not finite and 0 in
    # slot 7 if any face has a positive area (its ``float64`` cross product is not zero, the test
    # ``face_gradients`` zeroes a face by).
    block, lane = wp.tid()
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0] // 3
    offset, count = block_chunk_1d(wp.max(n_vertices, n_faces), block)
    infinity = wp.vec3(twc.FLOAT32_INF_CONSTANT, twc.FLOAT32_INF_CONSTANT, twc.FLOAT32_INF_CONSTANT)
    low = infinity
    negated_high = infinity
    # (non-finite seen, positive face seen) as 0 where seen
    flags = wp.vec2(twc.FLOAT32_INF_CONSTANT, twc.FLOAT32_INF_CONSTANT)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        if i < n_vertices:
            p = vertices[i]
            if wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]):
                low = wp.min(low, p)
                negated_high = wp.min(negated_high, -p)
            else:
                flags[0] = wp.float32(0.0)
        if i < n_faces:
            p0, p1, p2 = face_vertices_vec3d(vertices, faces, i)
            if wp.length_sq(wp.cross(p1 - p0, p2 - p0)) > wp.float64(0.0):
                flags[1] = wp.float32(0.0)
    for axis in range(3):
        block_low = block_min(low[axis])
        block_high = block_min(negated_high[axis])
        if lane == 0:
            wp.atomic_min(out_bounds, axis, block_low)
            wp.atomic_min(out_bounds, 3 + axis, block_high)
    for slot in range(2):
        seen = block_min(flags[slot])
        if lane == 0 and seen == wp.float32(0.0):
            wp.atomic_min(out_bounds, 6 + slot, seen)


@wp.kernel
def cluster_cell_keys(
    vertices: wp.array[wp.vec3],
    origin: wp.vec3d,
    cell: wp.float64,
    extent_y: wp.int64,
    extent_z: wp.int64,
    out_keys: wp.array[wp.int64],
    out_positions: wp.array[wp.vec3d],
) -> None:
    # Each vertex's grid cell ``floor((x - origin) / cell)`` packed as ``(cx * ey + cy) * ez + cz``,
    # so the sorted distinct keys number the clusters in the reference's order; and its position
    # in ``float64``, which the clusters' sums add.
    i = wp.int32(wp.tid())
    p = to_vec3d(vertices[i])
    out_positions[i] = p
    cx = wp.int64(wp.floor((p[0] - origin[0]) / cell))
    cy = wp.int64(wp.floor((p[1] - origin[1]) / cell))
    cz = wp.int64(wp.floor((p[2] - origin[2]) / cell))
    out_keys[i] = (cx * extent_y + cy) * extent_z + cz


@wp.kernel
def cluster_centroids(
    sums: wp.array[wp.vec3d],
    counts: wp.array[wp.int32],
    out_centroids: wp.array[wp.vec3d],
    out_centroids_f32: wp.array[wp.vec3],
) -> None:
    # Each cluster's plain mean of its vertices, and its ``float32`` copy for the factorization's
    # geometric ordering.
    c = wp.int32(wp.tid())
    centroid = sums[c] / wp.float64(counts[c])
    out_centroids[c] = centroid
    out_centroids_f32[c] = wp.vec3(
        wp.float32(centroid[0]), wp.float32(centroid[1]), wp.float32(centroid[2])
    )


@wp.func
def oriented_cluster_key(
    c0: wp.int32, c1: wp.int32, c2: wp.int32, flipped: wp.bool, n_clusters: wp.int64
) -> wp.int64:
    # The key of the coarse face ``(c0, c1, c2)``, wound the other way when ``flipped``, rotated so
    # its smallest cluster comes first (a rotation keeps the winding): ``(a * n + b) * n + c``. A
    # triple and its reversed twin keep distinct keys.
    b0 = wp.where(flipped, c2, c1)
    b1 = wp.where(flipped, c1, c2)
    second_least = b0 < c0 and b0 < b1
    third_least = b1 < c0 and b1 < b0
    a = wp.where(second_least, b0, wp.where(third_least, b1, c0))
    b = wp.where(second_least, b1, wp.where(third_least, c0, b0))
    c = wp.where(second_least, c0, wp.where(third_least, b0, b1))
    return (wp.int64(a) * n_clusters + wp.int64(b)) * n_clusters + wp.int64(c)


@wp.kernel
def coarse_face_keys(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cluster: wp.array[wp.int32],
    centroids: wp.array[wp.vec3d],
    n_clusters: wp.int64,
    out_keys: wp.array[wp.int64],
) -> None:
    # The coarse face a fine face spans, as ``oriented_cluster_key``: its corners' three distinct
    # clusters, wound so the centroids' normal agrees with the fine face's (three centroids can
    # turn the other way from the face joining them, and a flat sheet's coarse mesh would then be
    # folded). A face of no area, or whose corners share a cluster, writes the sentinel
    # ``n_clusters ** 3``, which sorts past every key.
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    c0 = cluster[i0]
    c1 = cluster[i1]
    c2 = cluster[i2]
    p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
    fine = wp.cross(p1 - p0, p2 - p0)
    if c0 == c1 or c1 == c2 or c2 == c0 or wp.length_sq(fine) == wp.float64(0.0):
        out_keys[f] = n_clusters * n_clusters * n_clusters
        return
    q0 = centroids[c0]
    coarse = wp.cross(centroids[c1] - q0, centroids[c2] - q0)
    out_keys[f] = oriented_cluster_key(
        c0, c1, c2, wp.dot(fine, coarse) < wp.float64(0.0), n_clusters
    )


@wp.func
def decoded_cluster_key(key: wp.int64, n_clusters: wp.int64) -> wp.vec3i:
    # The three clusters ``oriented_cluster_key`` packed.
    return wp.vec3i(
        wp.int32(key // (n_clusters * n_clusters)),
        wp.int32((key // n_clusters) % n_clusters),
        wp.int32(key % n_clusters),
    )


@wp.kernel
def coarse_face_flags(
    keys: wp.array[wp.int64],
    centroids: wp.array[wp.vec3d],
    n_clusters: wp.int64,
    flat_bound: wp.float64,
    out_flags: wp.array[wp.int32],
) -> None:
    # 1 for each distinct coarse face kept: not the sentinel, and its centroids' cross product
    # longer than ``flat_bound``. Written into the buffer the caller scans in place.
    t = wp.int32(wp.tid())
    key = keys[t]
    kept = wp.int32(0)
    if key < n_clusters * n_clusters * n_clusters:
        c = decoded_cluster_key(key, n_clusters)
        q0 = centroids[c[0]]
        normal = wp.cross(centroids[c[1]] - q0, centroids[c[2]] - q0)
        if wp.length(normal) > flat_bound:
            kept = wp.int32(1)
    out_flags[t] = kept


@wp.kernel
def coarse_faces_emit(
    keys: wp.array[wp.int64],
    kept: wp.array[wp.int32],
    n_clusters: wp.int64,
    out_faces: wp.array[wp.int32],
) -> None:
    # The kept coarse faces as a flat triangle buffer, in ascending key order.
    t = wp.int32(wp.tid())
    c = decoded_cluster_key(keys[kept[t]], n_clusters)
    out_faces[3 * t] = c[0]
    out_faces[3 * t + 1] = c[1]
    out_faces[3 * t + 2] = c[2]


@wp.func
def plane_axes(normal: wp.vec3d) -> tuple[wp.vec3d, wp.vec3d]:
    # Two orthonormal axes of the plane of unit ``normal``, ``u x v = normal``: ``u`` crosses the
    # normal with the coordinate axis it is least aligned with.
    magnitude = wp.vec3d(wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2]))
    seed = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    if magnitude[0] <= magnitude[1] and magnitude[0] <= magnitude[2]:
        seed = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    elif magnitude[1] <= magnitude[2]:
        seed = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
    axis_u = wp.normalize(wp.cross(seed, normal))
    return axis_u, wp.cross(normal, axis_u)


@wp.kernel
def plane_projection_block(
    points: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    normal: wp.vec3d,
    out_chart: wp.array2d[wp.float64],
) -> None:
    # The points' coordinates in a plane through their mean, as one block (the CPU's one lane
    # walks every point): the plane of unit ``normal``, or with a zero ``normal`` their best-fit
    # plane (the centred covariance's two leading eigenvectors, the top two right singular vectors
    # of the centred points). Then v is mirrored if the faces' summed signed area in that frame is
    # negative, since ARAP fits proper rotations and cannot unmirror a start.
    _block, lane = wp.tid()
    lanes = wp.block_dim()
    n = points.shape[0]
    total = wp.vec3d()
    for i in range(lane, n, lanes):
        total += points[i]
    mean = block_sum(total) / wp.float64(n)
    axis_u = wp.vec3d()
    axis_v = wp.vec3d()
    if wp.length_sq(normal) > wp.float64(0.0):
        axis_u, axis_v = plane_axes(normal)
    else:
        scatter = wp.mat33d()
        for i in range(lane, n, lanes):
            d = points[i] - mean
            scatter += wp.outer(d, d)
        covariance = block_sum(scatter)
        vectors, values = wp.eig3(covariance)
        # The largest eigenvalue's column is u, the middle one's v (``eig3`` does not sort them).
        largest = wp.where(
            values[0] >= values[1] and values[0] >= values[2],
            0,
            wp.where(values[1] >= values[2], 1, 2),
        )
        rest_a = (largest + 1) % 3
        rest_b = (largest + 2) % 3
        middle = wp.where(values[rest_a] >= values[rest_b], rest_a, rest_b)
        axis_u = wp.vec3d(vectors[0, largest], vectors[1, largest], vectors[2, largest])
        axis_v = wp.vec3d(vectors[0, middle], vectors[1, middle], vectors[2, middle])
    signed = wp.float64(0.0)
    for f in range(lane, faces.shape[0] // 3, lanes):
        p0, p1, p2 = face_vertices(points, faces, f)
        e1 = p1 - p0
        e2 = p2 - p0
        signed += wp.dot(e1, axis_u) * wp.dot(e2, axis_v) - wp.dot(e1, axis_v) * wp.dot(e2, axis_u)
    mirror = wp.where(block_sum(signed) < wp.float64(0.0), wp.float64(-1.0), wp.float64(1.0))
    for i in range(lane, n, lanes):
        d = points[i] - mean
        out_chart[0, i] = wp.dot(d, axis_u)
        out_chart[1, i] = mirror * wp.dot(d, axis_v)


@wp.kernel
def chart_stiffness_setup(
    points: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    out_gradients: wp.array2d[wp.vec2d],
    out_areas: wp.array[wp.float64],
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
    out_partials: wp.array[wp.float64],
) -> None:
    # The stiffness pattern's one-time inputs, for the proximal ARAP level and ``slim`` (whose
    # ``wp.mat22d`` operator is stored over this scalar pattern), lane-strided over
    # ``max(n_faces, n_points)``:
    # each face's gradients and area, its nine stiffness keys ``(i_a, i_b)`` (payload
    # ``9 f + 3 a + b``), one diagonal key per point (payload ``9 n_faces + i``, so the anchor
    # sorts after a row's face terms and every row has a diagonal, a point of no face too), and the
    # sum of the stiffness diagonal ``sum_f A_f sum_k |g_k|^2`` into ``out_partials`` (a
    # ``_launch.block_partials`` buffer of one slot), whose mean sets the anchor.
    block, lane = wp.tid()
    n_points = points.shape[0]
    n_faces = faces.shape[0] // 3
    offset, count = block_chunk_1d(wp.max(n_faces, n_points), block)
    trace = wp.float64(0.0)
    for k in range(lane, count, wp.block_dim()):
        t = offset + k
        if t < n_faces:
            p0, p1, p2 = face_vertices(points, faces, t)
            g0, g1, g2, area = face_gradients(p0, p1, p2)
            store_face_gradients(t, g0, g1, g2, area, out_gradients, out_areas)
            trace += area * (wp.dot(g0, g0) + wp.dot(g1, g1) + wp.dot(g2, g2))
            i0, i1, i2 = corner_triple(faces, t)
            corners = wp.vec3i(i0, i1, i2)
            for a in range(3):
                for b in range(3):
                    slot = TWO_LEVEL_FACE_KEYS * t + 3 * a + b
                    out_keys[slot] = csr_key(corners[a], corners[b], n_points, n_points)
                    out_order[slot] = slot
        if t < n_points:
            slot = TWO_LEVEL_FACE_KEYS * n_faces + t
            out_keys[slot] = csr_key(t, t, n_points, n_points)
            out_order[slot] = slot
    store_block_total(block, lane, trace, out_partials, 0, 1, last_ticket)


@wp.kernel
def chart_stiffness_values(
    offsets: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    order: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    trace: wp.array[wp.float64],
    n_rows: wp.int32,
    out_values: wp.array[wp.float64],
    out_anchor: wp.array[wp.float64],
) -> None:
    # Each stored entry of ``K + epsilon I``, ``K = sum_f A_f G_f G_f^T`` (the proximal system, and
    # the scalar operator ``slim``'s analysis is built over): its contributions added
    # in the order the keys were written (a row's face terms in face order, then the anchor),
    # ``epsilon`` being ``TWO_LEVEL_ANCHOR`` times the mean of ``K``'s diagonal over every row.
    # Launched over the key count, an upper bound on the entries.
    e = wp.int32(wp.tid())
    anchor = TWO_LEVEL_ANCHOR * trace[0] / wp.float64(n_rows)
    if e == 0:
        out_anchor[0] = anchor
    if e >= offsets[n_rows]:
        return
    face_keys = TWO_LEVEL_FACE_KEYS * (gradients.shape[0])
    value = wp.float64(0.0)
    for s in range(starts[e], starts[e + 1]):
        payload = order[s]
        if payload >= face_keys:
            value += anchor
        else:
            f = payload // TWO_LEVEL_FACE_KEYS
            a = (payload // 3) % 3
            b = payload % 3
            value += areas[f] * wp.dot(gradients[f, a], gradients[f, b])
    out_values[e] = value


@wp.kernel
def proximal_rotations(
    faces: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    moves: wp.array[wp.float64],
    iteration: wp.int32,
    settled: wp.float64,
    no_mask: wp.array[wp.bool],
    no_map: wp.array[wp.int32],
    chart: wp.array2d[wp.float64],
    out_rotations: wp.array[wp.vec2d],
) -> None:
    # The proximal iteration's local step, a face a thread: each face's rotation, which
    # ``proximal_rhs`` reads at every corner. A thread a vertex refitting each incident face's
    # rotation itself spent its time in ``float64`` square roots and divisions three times over,
    # on as few warps as the coarse mesh has vertices: the two launches take 0.43x the one's
    # device time on bunny's coarse level (2026-11-20). Skipped once the loop has settled, as
    # ``proximal_rhs`` is.
    f = wp.int32(wp.tid())
    if iteration > 0:
        if moves[iteration - 1] <= settled:
            return
    out_rotations[f] = chart_face_rotation(faces, gradients, chart, no_mask, no_map, chart, f)


@wp.kernel
def proximal_rhs(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    rotations: wp.array[wp.vec2d],
    anchor: wp.array[wp.float64],
    moves: wp.array[wp.float64],
    iteration: wp.int32,
    settled: wp.float64,
    chart: wp.array2d[wp.float64],
    out_rhs: wp.array2d[wp.float64],
    out_previous: wp.array2d[wp.float64],
) -> None:
    # The proximal global step's right-hand side ``B + epsilon U``, with ``U`` kept in
    # ``out_previous`` for the move test; ``B`` is ``rotation_row`` over ``proximal_rotations``'
    # table. Once an iteration has moved no vertex by more than ``settled`` the loop has converged
    # and every later iteration leaves both buffers as they are: the solve then returns the chart
    # unchanged and ``proximal_moves`` repeats the settled move, so the host may read the moves
    # only every few iterations and still stop exactly.
    v = wp.int32(wp.tid())
    if iteration > 0:
        if moves[iteration - 1] <= settled:
            return
    uv = chart_point(chart, v)
    rotated = rotation_row_from_table(
        faces, vertex_faces, vertex_offsets, gradients, areas, rotations, v
    )
    store_chart_point(out_rhs, v, rotated + anchor[0] * uv)
    store_chart_point(out_previous, v, uv)


@wp.kernel
def proximal_moves(
    chart: wp.array2d[wp.float64],
    previous: wp.array2d[wp.float64],
    iteration: wp.int32,
    width: wp.int32,
    out_moves: wp.array[wp.float64],
) -> None:
    # The largest coordinate move of the iteration, ``max |U_new - U_old|``, into the zeroed
    # ``out_moves[iteration]`` (``atomic_max`` of non-negative values: exact in any order), over
    # blocks of ``width`` points (``reduce.spanned_blocks_1d``).
    block, lane = wp.tid()
    offset, count = block_chunk(chart.shape[1], block, width)
    largest = wp.float64(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        largest = wp.max(
            largest,
            wp.max(wp.abs(chart[0, i] - previous[0, i]), wp.abs(chart[1, i] - previous[1, i])),
        )
    block_largest = block_max(largest)
    if lane == 0:
        wp.atomic_max(out_moves, iteration, block_largest)


@wp.func
def face_jacobian(
    q0: wp.vec3d, q1: wp.vec3d, q2: wp.vec3d, uv0: wp.vec2d, uv1: wp.vec2d, uv2: wp.vec2d
) -> tuple[mat23d, wp.float64]:
    # A coarse face's ``2 x 3`` Jacobian from space to the chart,
    # ``J = E_uv (E^T E)^-1 E^T`` over its edge vectors from corner 0, and its area
    # ``sqrt(det(E^T E)) / 2``.
    e1 = q1 - q0
    e2 = q2 - q0
    d1 = uv1 - uv0
    d2 = uv2 - uv0
    g11 = wp.dot(e1, e1)
    g12 = wp.dot(e1, e2)
    g22 = wp.dot(e2, e2)
    det = g11 * g22 - g12 * g12
    # ``E_uv (E^T E)^-1``, by the adjugate
    m00 = (d1[0] * g22 - d2[0] * g12) / det
    m01 = (d2[0] * g11 - d1[0] * g12) / det
    m10 = (d1[1] * g22 - d2[1] * g12) / det
    m11 = (d2[1] * g11 - d1[1] * g12) / det
    row_u = m00 * e1 + m01 * e2
    row_v = m10 * e1 + m11 * e2
    jacobian = mat23d(row_u[0], row_u[1], row_u[2], row_v[0], row_v[1], row_v[2])
    return jacobian, wp.float64(0.5) * wp.sqrt(det)


@wp.kernel
def cluster_jacobians(
    centroids: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    chart: wp.array2d[wp.float64],
    out_jacobians: wp.array[mat23d],
    out_covered: wp.array[wp.bool],
    out_uncovered: wp.array[wp.bool],
    out_source: wp.array[wp.int32],
) -> None:
    # Each cluster's area-weighted mean of its coarse faces' Jacobians (``face_jacobian``,
    # recomputed by each corner), in the adjacency row's order. A cluster no coarse face touches
    # is uncovered: flagged, and its vertices will follow the nearest covered cluster
    # (``out_source``, the cluster itself until then).
    c = wp.int32(wp.tid())
    weighted = mat23d()
    total = wp.float64(0.0)
    for s in range(vertex_offsets[c], vertex_offsets[c + 1]):
        f = vertex_faces[s]
        i0, i1, i2 = corner_triple(faces, f)
        jacobian, weight = face_jacobian(
            centroids[i0], centroids[i1], centroids[i2],
            chart_point(chart, i0), chart_point(chart, i1), chart_point(chart, i2),
        )  # fmt: skip
        weighted += weight * jacobian
        total += weight
    covered = total > wp.float64(0.0)
    if covered:
        weighted = weighted / total
    out_jacobians[c] = weighted
    out_covered[c] = covered
    out_uncovered[c] = not covered
    out_source[c] = c


@wp.kernel
def borrow_cluster_sources(
    uncovered: wp.array[wp.int32],
    covered: wp.array[wp.int32],
    nearest: wp.array[wp.int32],
    out_source: wp.array[wp.int32],
) -> None:
    # An uncovered cluster takes the nearest covered cluster as the source of its vertices' UV.
    t = wp.int32(wp.tid())
    out_source[uncovered[t]] = covered[nearest[t]]


@wp.kernel
def prolong_chart(
    vertices: wp.array[wp.vec3],
    cluster: wp.array[wp.int32],
    source: wp.array[wp.int32],
    centroids: wp.array[wp.vec3d],
    jacobians: wp.array[mat23d],
    coarse_chart: wp.array2d[wp.float64],
    out_chart: wp.array2d[wp.float64],
) -> None:
    # Each vertex's UV: its source cluster's UV plus its offset from that cluster's centroid,
    # carried by the cluster's Jacobian (an uncovered cluster's vertices take the borrowed
    # cluster's centroid as their origin too).
    i = wp.int32(wp.tid())
    c = source[cluster[i]]
    offset = to_vec3d(vertices[i]) - centroids[c]
    store_chart_point(out_chart, i, chart_point(coarse_chart, c) + jacobians[c] * offset)


@wp.kernel
def smoothing_face_terms(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_gradients: wp.array2d[wp.vec2d],
    out_areas: wp.array[wp.float64],
    out_edge_weights: wp.array[wp.vec3d],
) -> None:
    # The fine level's per-face gradients and areas, and its stiffness as edge weights:
    # ``w_k = -A g_{k+1} . g_{k+2}`` weighs the edge opposite corner ``k`` (half its cotangent).
    f = wp.int32(wp.tid())
    p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
    g0, g1, g2, area = face_gradients(p0, p1, p2)
    store_face_gradients(f, g0, g1, g2, area, out_gradients, out_areas)
    out_edge_weights[f] = wp.vec3d(
        -area * wp.dot(g1, g2), -area * wp.dot(g2, g0), -area * wp.dot(g0, g1)
    )


@wp.func
def stiffness_row(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    edge_weights: wp.array[wp.vec3d],
    values: wp.array2d[wp.float64],
    v: wp.int32,
) -> wp.vec2d:
    # Vertex ``v``'s row of the stiffness matrix applied to ``values``: over each incident face,
    # the two edges at ``v``'s corner ``k`` weighted by ``w_{k+2}`` (to corner ``k + 1``) and
    # ``w_{k+1}`` (to corner ``k + 2``).
    own = chart_point(values, v)
    total = wp.vec2d()
    for s in range(vertex_offsets[v], vertex_offsets[v + 1]):
        f = vertex_faces[s]
        k = known_local_corner(faces, f, v)
        weights = edge_weights[f]
        following = faces[3 * f + (k + 1) % 3]
        preceding = faces[3 * f + (k + 2) % 3]
        total += weights[(k + 2) % 3] * (own - chart_point(values, following))
        total += weights[(k + 1) % 3] * (own - chart_point(values, preceding))
    return total


@wp.func
def stiffness_diagonal(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    v: wp.int32,
) -> wp.float64:
    # Vertex ``v``'s stiffness diagonal ``sum_f A_f |g_k|^2``.
    total = wp.float64(0.0)
    for s in range(vertex_offsets[v], vertex_offsets[v + 1]):
        f = vertex_faces[s]
        g = gradients[f, known_local_corner(faces, f, v)]
        total += areas[f] * wp.dot(g, g)
    return total


@wp.func
def step_ratio(numerator: wp.vec2d, denominator: wp.vec2d) -> wp.vec2d:
    # Component-wise ``numerator / denominator``, zero where the denominator is not positive.
    result = wp.vec2d()
    for coordinate in range(2):
        if denominator[coordinate] > wp.float64(0.0):
            result[coordinate] = numerator[coordinate] / denominator[coordinate]
    return result


@wp.kernel
def smoothing_start(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    gradients: wp.array2d[wp.vec2d],
    areas: wp.array[wp.float64],
    edge_weights: wp.array[wp.vec3d],
    no_mask: wp.array[wp.bool],
    no_map: wp.array[wp.int32],
    chart: wp.array2d[wp.float64],
    width: wp.int32,
    slot: wp.int32,
    stride: wp.int32,
    out_inverse_diagonal: wp.array[wp.float64],
    out_residual: wp.array2d[wp.float64],
    out_direction: wp.array2d[wp.float64],
    out_partials: wp.array[wp.vec2d],
) -> None:
    # A smoothing pass's local step and the start of its conjugate gradients, warm-started from
    # the chart: the residual ``B - K U``, the Jacobi-preconditioned first direction, and the
    # preconditioned residual norm per coordinate into slot ``slot``. A vertex of no face has a
    # zero diagonal, preconditioned by 1: its residual stays zero.
    block, lane = wp.tid()
    offset, count = block_chunk(chart.shape[1], block, width)
    norm = wp.vec2d()
    for k in range(lane, count, wp.block_dim()):
        v = offset + k
        diagonal = stiffness_diagonal(faces, vertex_faces, vertex_offsets, gradients, areas, v)
        inverse = wp.where(diagonal > wp.float64(0.0), wp.float64(1.0) / diagonal, wp.float64(1.0))
        out_inverse_diagonal[v] = inverse
        rotated = rotation_row(
            faces, vertex_faces, vertex_offsets, gradients, areas, chart, no_mask, no_map, chart, v
        )
        residual = rotated - stiffness_row(
            faces, vertex_faces, vertex_offsets, edge_weights, chart, v
        )
        store_chart_point(out_residual, v, residual)
        store_chart_point(out_direction, v, inverse * residual)
        norm += inverse * wp.cw_mul(residual, residual)
    store_block_total(block, lane, norm, out_partials, slot, stride, last_ticket_vector)


@wp.kernel
def smoothing_apply(
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    edge_weights: wp.array[wp.vec3d],
    direction: wp.array2d[wp.float64],
    width: wp.int32,
    slot: wp.int32,
    stride: wp.int32,
    out_applied: wp.array2d[wp.float64],
    out_partials: wp.array[wp.vec2d],
) -> None:
    # ``K p`` and the curvature ``p . K p`` per coordinate into slot ``slot``.
    block, lane = wp.tid()
    offset, count = block_chunk(direction.shape[1], block, width)
    curvature = wp.vec2d()
    for k in range(lane, count, wp.block_dim()):
        v = offset + k
        applied = stiffness_row(faces, vertex_faces, vertex_offsets, edge_weights, direction, v)
        store_chart_point(out_applied, v, applied)
        curvature += wp.cw_mul(chart_point(direction, v), applied)
    store_block_total(block, lane, curvature, out_partials, slot, stride, last_ticket_vector)


@wp.kernel
def smoothing_update(
    direction: wp.array2d[wp.float64],
    applied: wp.array2d[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    width: wp.int32,
    norm_slot: wp.int32,
    curvature_slot: wp.int32,
    stride: wp.int32,
    out_chart: wp.array2d[wp.float64],
    out_residual: wp.array2d[wp.float64],
    out_partials: wp.array[wp.vec2d],
) -> None:
    # Step the chart along the direction by ``alpha = (r . z) / (p . K p)`` per coordinate, update
    # the residual, and its preconditioned norm into slot ``norm_slot + 1``.
    block, lane = wp.tid()
    offset, count = block_chunk(direction.shape[1], block, width)
    length = step_ratio(out_partials[norm_slot], out_partials[curvature_slot])
    norm = wp.vec2d()
    for k in range(lane, count, wp.block_dim()):
        v = offset + k
        store_chart_point(
            out_chart, v, chart_point(out_chart, v) + wp.cw_mul(length, chart_point(direction, v))
        )
        residual = chart_point(out_residual, v) - wp.cw_mul(length, chart_point(applied, v))
        store_chart_point(out_residual, v, residual)
        norm += inverse_diagonal[v] * wp.cw_mul(residual, residual)
    store_block_total(block, lane, norm, out_partials, norm_slot + 1, stride, last_ticket_vector)


@wp.kernel
def smoothing_direction(
    residual: wp.array2d[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    partials: wp.array[wp.vec2d],
    norm_slot: wp.int32,
    out_direction: wp.array2d[wp.float64],
) -> None:
    # The next conjugate direction, ``z + beta p`` with ``beta = (r' . z') / (r . z)``.
    v = wp.int32(wp.tid())
    ratio = step_ratio(partials[norm_slot + 1], partials[norm_slot])
    store_chart_point(
        out_direction,
        v,
        inverse_diagonal[v] * chart_point(residual, v)
        + wp.cw_mul(ratio, chart_point(out_direction, v)),
    )


@wp.kernel
def start_chart_areas(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    out_partials: wp.array[wp.vec3d],
) -> None:
    # A start chart's sums over the faces, lane-strided, into slot 0 of a ``_launch.block_partials``
    # buffer: ``(surface area, |UV area|, signed UV area)``. The first two scale the chart to the
    # mesh's units; the third's sign says whether it winds clockwise, which ARAP cannot undo.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(faces.shape[0] // 3, block)
    sums = wp.vec3d()
    for k in range(lane, count, wp.block_dim()):
        f = offset + k
        p0, p1, p2 = face_vertices_vec3d(vertices, faces, f)
        q0, q1, q2 = face_vertices(uv, faces, f)
        signed = wp.float64(0.5) * wp.float64(cross2(q1 - q0, q2 - q0))
        surface = wp.float64(0.5) * wp.length(wp.cross(p1 - p0, p2 - p0))
        sums += wp.vec3d(surface, wp.abs(signed), signed)
    store_block_total(block, lane, sums, out_partials, 0, 1, last_ticket_vector)


@wp.kernel
def scaled_start_chart(
    uv: wp.array[wp.vec2], areas: wp.array[wp.vec3d], out_chart: wp.array2d[wp.float64]
) -> None:
    # A start chart as a ``(2, n)`` ``float64`` chart, scaled by ``sqrt(surface / UV area)`` to the
    # mesh's units and mirrored in v if its faces wind clockwise on the whole.
    i = wp.int32(wp.tid())
    totals = areas[0]
    scale = wp.float64(1.0)
    if totals[1] > wp.float64(0.0):
        scale = wp.sqrt(totals[0] / totals[1])
    mirror = wp.where(totals[2] < wp.float64(0.0), wp.float64(-1.0), wp.float64(1.0))
    p = uv[i]
    out_chart[0, i] = scale * wp.float64(p[0])
    out_chart[1, i] = mirror * scale * wp.float64(p[1])


@wp.kernel
def chart_to_uv(chart: wp.array2d[wp.float64], out_uv: wp.array[wp.vec2]) -> None:
    # A ``(2, n)`` chart narrowed to ``float32`` UV.
    i = wp.int32(wp.tid())
    out_uv[i] = column_uv(chart, i)


def _declare_block_dims() -> None:
    """Declare the CUDA launch widths of this module's kernels besides the default (precompile)."""
    # ``low_distortion_parametrization``'s lane-strided folds launch ``TILE_1D``-lane blocks.
    declare_block_dims(__name__, (twc.TILE_1D,))


_declare_block_dims()
