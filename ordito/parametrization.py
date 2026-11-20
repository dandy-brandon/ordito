"""
Flatten a mesh into the plane: harmonic, Tutte, ARAP and LSCM parametrizations.

Six maps, differing in what they hold fixed and what they minimize. The first three pin the
boundary and solve for the interior:
[`harmonic`][ordito.parametrization.harmonic] minimizes Dirichlet energy with cotangent weights,
[`tutte`][ordito.parametrization.tutte] is the same solve with uniform weights (guaranteeing an
injective map for a convex boundary), and [`arap`][ordito.parametrization.arap] alternates local
rotation fits with a global solve to trade conformality for low area distortion.
[`lscm`][ordito.parametrization.lscm] instead pins only two vertices and lets the boundary find its
own shape, minimizing conformal rather than Dirichlet energy.
[`low_distortion_parametrization`][ordito.parametrization.low_distortion_parametrization]
pins nothing and needs no start: a free-boundary chart of low ARAP distortion in the mesh's own
units for an open surface of any topology, converged on a coarse mesh;
[`low_distortion_disk_parametrization`][ordito.parametrization.low_distortion_disk_parametrization]
is the same chart of a disk, refined from a harmonic, Tutte or LSCM map.

[`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle] supplies the boundary
condition the fixed-boundary three need, and
[`face_flipped_indices`][ordito.parametrization.face_flipped_indices] is the diagnostic that
says whether a result is actually injective. All but the two low-distortion charts port the
corresponding ``igl::`` routines.

The fixed-vertex maps (``harmonic``, ``tutte``, ``lscm``) take the mesh as an
[`ordito.mesh.Trimesh`][ordito.mesh.Trimesh] or as its ``vertices`` and ``faces``. Their solves
iterate, verified, and fall back to a sparse Cholesky factorization where the iteration fails;
given a `Trimesh`, a factorization is kept on it
([`Trimesh.fixed_vertex_solver`][ordito.mesh.Trimesh.fixed_vertex_solver]), and one built ahead
with [`FixedVertexSolver.factor`][ordito.parametrization.FixedVertexSolver.factor] -- or by a
call with ``solver="direct"`` -- turns every later call fixing the same vertices into a factored
solve with no iteration.
"""

from __future__ import annotations

import logging
import math
import weakref
from typing import Literal, cast, overload

import warp as wp

import ordito as od
import ordito.linalg as twl
import ordito.typing as odt
from ordito import _launch
from ordito._common import read_scalar, read_values, require_same_device
from ordito.constants import TILE_1D
from ordito.kernels import parametrization as kernel_parametrization
from ordito.kernels import reduce as kernel_reduce
from ordito.laplacian import cotmatrix, graph_laplacian, mass_matrix_entries
from ordito.mesh import Trimesh, mesh_arguments

_CG_TOLERANCE = 1e-8
# Vertices from which ``"auto"`` factors a ``k == 2`` harmonic / tutte system, or an ``lscm`` one,
# at once: their iterations grow faster with the mesh than a factorization does (the biharmonic
# operator's conditioning is the Laplacian's squared, and ``lscm`` pins only two vertices), and
# past this size the factorization is the cheaper one-shot solve too.
_FACTOR_FROM_VERTICES = 4_000
# Vertices from which the CPU device factors a ``k == 1`` harmonic / tutte system at once: its
# rounds cost far more against its factorization than on CUDA, where the iteration's verified
# budget (``linalg.CG_FACTOR_AFTER_POLYNOMIAL_ROUNDS``) is the only switch. A factored solve
# against the iteration on the CPU, on saddles: 1.05-1.54x to 4 624 vertices, 0.49-0.67x at 6 400,
# 0.30-0.40x at 17 689.
_FACTOR_K1_FROM_VERTICES_CPU = 5_000

# ``arap`` factors its global-step operator once its second solve's conjugate-gradient rounds,
# times the iterations still to run, reach this: the later warm-started solves then cost more than
# a factorization and its solves. On CUDA a factorization costs about 80-160 rounds from 4 k to
# 330 k vertices, so it is reached on graded meshes (saddle_graded, 112 rounds a solve) and long
# schedules on large ones (283 k vertices, 52), never on the patch benchmarks' well-conditioned
# meshes (at most 18). The CPU's rounds are dearer against its factorization: 10 iterations on
# those patch meshes factor there too (0.45-0.62x), and only a schedule's last solve or two does
# not pay (1.04-1.08x).
_ARAP_FACTOR_FROM_ROUNDS_CUDA = 300
_ARAP_FACTOR_FROM_ROUNDS_CPU = 16

# Loops up to this many vertices are mapped onto the circle by one block, a chunk of the scan at
# a time, in one launch; longer ones on CUDA by a length launch, a device-wide scan and a position
# launch (one block 0.39x to 2 048 vertices, 0.80x at 4 096, 1.5x at 8 192). The CPU device takes
# the block at every size, its one lane summing in loop order as the scan does (0.48-0.74x from 64
# to 65 536 vertices, byte-identical).
_CIRCLE_ONE_BLOCK_VERTICES = 4_096

# ``low_distortion_parametrization``: the factored iterations stop once one moves no vertex by
# more than this fraction of ``coarse_cell``; a clustering leaving fewer coarse faces than this is
# iterated directly; and the factored loop reads its largest move back every this many iterations
# (the iterations past the settled one are no-ops on the device, so the stop stays exact).
_TWO_LEVEL_SETTLED = 1e-3
_TWO_LEVEL_MIN_COARSE_FACES = 16
_TWO_LEVEL_READ_EVERY = 8
_TWO_LEVEL_FACE_KEYS = 9
_TWO_LEVEL_FLAT_FACE = 2e-9
# The largest grid-cell key and cluster count whose packed keys fit 63 bits: ``n_cells`` keys,
# and a coarse face's ``n_clusters ** 3``.
_MAX_PACKED_KEY = (1 << 63) - 1
_MAX_PACKED_CLUSTERS = (1 << 21) - 1
# ``low_distortion_disk_parametrization``'s starts, and the default ``coarse_cell`` in mean edge
# lengths (the disk chart's stop is a thousandth of it too).
_DISK_STARTS = ("harmonic", "tutte", "lscm")
_DEFAULT_CELL_EDGES = 5.0

_LOGGER = logging.getLogger(__name__)


def face_flipped_mask(vertices: wp.array[wp.vec2], faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    """
    Per-face flag: whether a triangle is inverted (negative 2D signed area) in the parametrization.

    For each triangle the 2D signed area of its three UV vertices is computed; a face is flagged
    ``True`` when that area is strictly negative, i.e. the triangle has folded over (flipped
    orientation) in the 2D domain. Mirrors libigl's ``flipped_triangles`` per-triangle test
    (determinant of the homogeneous ``3 x 3`` vertex matrix ``< 0``). Degenerate (zero-area)
    triangles are **not** flagged, matching the strict ``< 0`` comparison.
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices] is the index form of
    this mask.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` 2D vertex positions (the parametrization / UV coordinates).
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` flipped-triangle mask on ``vertices.device``. Empty for an empty mesh.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices]
    [`face_flip_mask`][ordito.validation.face_flip_mask]

    Notes
    -----
    Equivalent to the per-triangle predicate behind libigl ``flipped_triangles``: the 2D cross
    product ``(v1 - v0) x (v2 - v0)`` equals ``det([[x0, x1, x2], [y0, y1, y2], [1, 1, 1]])``, so a
    ``True`` entry corresponds exactly to a triangle libigl would list as flipped.

    This is a different flip from
    [`face_flip_mask`][ordito.validation.face_flip_mask], which is about the *3D winding*: this
    mask flags a triangle whose UV image has folded over, a property of the parametrization that
    says nothing about the surface, while that one flags a triangle whose index order must be
    reversed to agree with its patch. A mesh can be consistently wound and still have flipped UVs,
    and vice versa.

    Examples
    --------
    Two triangles sharing an edge, the second wound clockwise in the plane:

    ```python
    import warp as wp

    import ordito as od

    uv = wp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=wp.vec2)
    faces = wp.array([0, 1, 2, 1, 2, 3], dtype=wp.int32)
    print(od.parametrization.face_flipped_mask(uv, faces).numpy())  # [False  True]
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    out_mask = _launch.empty(n_faces, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_parametrization.face_flipped_mask,
        dim=n_faces,
        inputs=[vertices, faces, out_mask],
        device=device,
    )
    return out_mask


def face_flipped_indices(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Return the indices of triangles inverted (negative 2D signed area) in the parametrization.

    Convenience wrapper returning ``flatnonzero`` of
    [`face_flipped_mask`][ordito.parametrization.face_flipped_mask]: the indices into ``faces``
    of triangles whose 2D signed area is strictly negative (folded over in the UV domain). Matches
    libigl's ``flipped_triangles``, which returns the same list of flipped-triangle indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` 2D vertex positions (the parametrization / UV coordinates).
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.int32]
        ``(m,)`` ascending face indices of the ``m`` flipped triangles on ``vertices.device``.
        Empty when no
        triangle is flipped.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`face_flipped_mask`][ordito.parametrization.face_flipped_mask]
    [`flatnonzero`][ordito.array.flatnonzero]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    uv = wp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=wp.vec2)
    faces = wp.array([0, 1, 2, 1, 2, 3], dtype=wp.int32)
    print(od.parametrization.face_flipped_indices(uv, faces).numpy())  # [1]
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)
    # ``flatnonzero`` of the mask, with the flags written straight into the buffer they are scanned
    # in: no bool mask between them.
    inclusive = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_parametrization.face_flipped_flags,
        dim=n_faces,
        inputs=[vertices, faces, inclusive],
        device=device,
    )
    return od.array.flatnonzero_in_place(inclusive)


def map_vertices_to_circle(
    vertices: wp.array[wp.vec3], boundary: wp.array[wp.int32]
) -> wp.array[wp.vec2]:
    """
    Map an ordered boundary loop onto the unit circle by arc length.

    Places boundary vertex ``i`` at angle ``2*pi * len[i] / total``, where ``len[i]`` is the
    cumulative edge length from ``boundary[0]`` to ``boundary[i]`` along the loop and ``total`` is
    the full perimeter (closing over the wrap edge). The result is in loop order: row ``i`` is the
    position of ``boundary[i]``. Feed it as ``boundary_uv`` to
    [`tutte`][ordito.parametrization.tutte] or [`harmonic`][ordito.parametrization.harmonic] for a
    disk parametrization.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    boundary
        ``(n_boundary,)`` ordered boundary-loop vertex indices, e.g. from
        [`longest_boundary_loop`][ordito.boundary.longest_boundary_loop].

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_boundary,)`` unit-circle positions on ``vertices.device``, aligned with ``boundary``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``boundary`` are not all on one device.

    Notes
    -----
    The name and the arc-length placement follow ``igl::map_vertices_to_circle``.

    See Also
    --------
    [`longest_boundary_loop`][ordito.boundary.longest_boundary_loop]
    [`tutte`][ordito.parametrization.tutte]
    [`harmonic`][ordito.parametrization.harmonic]

    Examples
    --------
    The boundary of a 3 x 3 grid, eight vertices at equal spacing, lands at multiples of 45°:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.grid(count=(3, 3))
    boundary = od.boundary.longest_boundary_loop(vertices, faces)
    circle = od.parametrization.map_vertices_to_circle(vertices, boundary)
    print(circle.shape)  # (8,)
    print(np.allclose(np.linalg.norm(circle.numpy(), axis=1), 1.0))  # True
    print(circle.numpy()[1].round(4))  # [0.7071 0.7071]
    ```
    """
    require_same_device(vertices=vertices, boundary=boundary)
    device = vertices.device
    n_boundary = boundary.size
    out_uv = _launch.empty(n_boundary, dtype=wp.vec2, device=device)
    if n_boundary == 0:
        return out_uv

    if n_boundary <= _CIRCLE_ONE_BLOCK_VERTICES or not wp.get_device(device).is_cuda:
        _launch.launch_tiled(
            kernel_parametrization.circle_positions_block,
            dim=1,
            inputs=[boundary, vertices, out_uv],
            device=device,
        )
        return out_uv
    segment_lengths = _launch.empty(n_boundary, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_parametrization.boundary_edge_lengths,
        dim=n_boundary,
        inputs=[boundary, vertices, segment_lengths],
        device=device,
    )
    # Scanned in place: the lengths are read only through their running sum.
    _launch.array_scan(segment_lengths, out_array=segment_lengths, inclusive=True)
    _launch.launch(
        kernel_parametrization.circle_positions,
        dim=n_boundary,
        inputs=[boundary, vertices, segment_lengths, out_uv],
        device=device,
    )
    return out_uv


@overload
def harmonic(
    mesh: Trimesh,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    /,
    *,
    k: int = 1,
    solver: Literal["auto", "direct"] = "auto",
    nonnegative_weights: bool = False,
) -> wp.array[wp.vec2]: ...
@overload
def harmonic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    /,
    k: int = 1,
    *,
    solver: Literal["auto", "direct"] = "auto",
    nonnegative_weights: bool = False,
) -> wp.array[wp.vec2]: ...
def harmonic(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    boundary_indices: wp.array[wp.int32] | wp.array[wp.vec2] | None = None,
    boundary_uv: wp.array[wp.vec2] | None = None,
    k: int = 1,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    solver: Literal["auto", "direct"] = "auto",
    nonnegative_weights: bool = False,
) -> wp.array[wp.vec2]:
    """
    Harmonic parametrization with fixed boundary.

    Minimizes the ``k``-harmonic energy built from the cotangent Laplacian
    [`cotmatrix`][ordito.laplacian.cotmatrix] subject to the boundary vertices being pinned to
    ``boundary_uv``. For ``k == 1`` this is the harmonic map (each interior UV is the
    cotangent-weighted average of its neighbors); ``k == 2`` is the biharmonic map, and so on. The
    interior system is iterated, verified, or factored, as ``solver`` and Notes say.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    boundary_indices
        ``(n_boundary,)`` indices of the fixed (constrained) vertices.
    boundary_uv
        ``(n_boundary,)`` target UV positions for ``boundary_indices``, in the same order (e.g. from
        [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]).
    k
        Harmonic power (``>= 1``). ``k > 1`` additionally uses the barycentric lumped mass matrix
        [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries]. The operator is assembled in
        float64 (built native, never recast) so the ill-conditioned ``k > 1`` solve is accurate.
    solver
        ``"auto"`` (default) factors the system at once at ``k >= 3``, where the iteration does not
        converge, at ``k == 2`` on a mesh of 4 000 vertices or more, and on the CPU device at
        ``k == 1`` on one of 5 000 or more, where a factorization costs less than the iteration;
        otherwise it solves by conjugate gradient, verified, and factors the system only when the
        iteration's result fails the check. ``"direct"`` factors it at once (a sparse Cholesky);
        given a `Trimesh`, the factorization is kept on it, so every later call fixing the same
        vertices -- whatever their values -- is a factored solve with no iteration. Pays from about
        the second such call; a one-shot call is usually faster under ``"auto"``.
    nonnegative_weights
        Build the Laplacian from cotangent weights clamped at zero
        ([`cotmatrix`][ordito.laplacian.cotmatrix]'s ``nonnegative_weights``). At ``k == 1`` each
        free vertex is then a convex combination of its neighbors, so with the boundary on a
        convex polygon the map is an embedding (Tutte's and Floater's theorem) whatever the mesh's
        obtuse angles, where the signed weights can fold triangles; the map is no longer exactly
        harmonic. A `Trimesh` keeps this system's factorization apart from the signed one's.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``k < 1``, if there are interior vertices but ``boundary_indices`` is empty (the
        Dirichlet system would be singular), if ``boundary_indices`` and ``boundary_uv`` have
        different lengths, if ``boundary_indices`` repeats a vertex or holds one outside
        ``[0, n_vertices)``, or if ``solver`` is not ``"auto"`` or ``"direct"``.
    RuntimeError
        If ``vertices``, ``faces``, ``boundary_indices`` and ``boundary_uv`` are not all on one
        device.

    Warns
    -----
    UserWarning
        When no verified solution is reached (see
        [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]): the interior system cannot be
        factored -- too large, or not definite in ``float64``, which a high ``k`` on a strongly
        graded mesh reaches -- and conjugate gradient does not converge.

    See Also
    --------
    [`tutte`][ordito.parametrization.tutte]
    [`k_harmonic`][ordito.energies.k_harmonic]
        The *operator* this minimizes, as opposed to this *map* -- the two are the only two
        "harmonic" names in the package and they are not interchangeable.
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]

    Notes
    -----
    Matches ``igl::harmonic`` at ``k == 1``. Above, igl's default mass matrix is the Voronoi one
    where this uses the barycentric lumped mass; ``igl::harmonic`` given a barycentric
    ``massmatrix`` is the same map.

    Each ``k`` is solved its own way, since raising the operator to the ``k``-th power raises its
    condition number to it. ``k == 1`` runs conjugate gradient under the Jacobi-Chebyshev polynomial
    ([`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner]), ``k == 2`` under the
    square of that Laplacian's polynomial
    ([`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner]) -- on a
    mesh of fewer than 4 000 vertices; from 4 000 the ``k == 2`` system is factored at once, as
    ``k >= 3`` is below, its iteration growing faster with the mesh than the factorization. Both
    results are verified -- converged within
    [`CG_FACTOR_AFTER_POLYNOMIAL_ROUNDS`][ordito.linalg.CG_FACTOR_AFTER_POLYNOMIAL_ROUNDS] rounds,
    with a componentwise
    backward error under
    [`VERIFIED_BACKWARD_ERROR`][ordito.linalg.VERIFIED_BACKWARD_ERROR] -- and a result that is not
    is replaced by a [`sparse_cholesky`][ordito.cholesky.sparse_cholesky] solve: on a strongly
    graded mesh the ``k == 2`` system's condition number nears the reciprocal of ``float64``'s
    precision, where no iteration converges. ``k >= 3`` is factored from the start, the iteration
    (under the multigrid preconditioner
    [`multigrid_preconditioner`][ordito.linalg.multigrid_preconditioner]) taken only when the
    system cannot be factored; a `Trimesh` remembers that refusal, so its later calls iterate
    without attempting the factorization again. Given a `Trimesh`, a factorization built here is
    kept on it, and one prepared with
    [`FixedVertexSolver.factor`][ordito.parametrization.FixedVertexSolver.factor] serves every
    later call fixing the same vertices -- whatever their values -- with no iteration; the
    ``vertices, faces`` form drops it, and says so at ``INFO``.

    On such an ill-conditioned system the answer itself is determined only to the system's own
    precision: direct solvers under different orderings agree to about a percent of the UV range
    there, as this one does with them.

    Examples
    --------
    A disk-topology cap mapped onto the unit disk, with no flipped triangle:

    ```python
    import ordito as od

    vertices, faces = od.creation.sphere_cap(angle=1.0, subdivisions=3)
    boundary = od.boundary.longest_boundary_loop(vertices, faces)
    circle = od.parametrization.map_vertices_to_circle(vertices, boundary)
    uv = od.parametrization.harmonic(
        vertices=vertices, faces=faces, boundary_indices=boundary, boundary_uv=circle
    )
    print(uv.shape)  # (217,)
    print(od.parametrization.face_flipped_indices(uv, faces).shape)  # (0,)
    ```

    On a flat grid pinned to its own boundary positions, the harmonic map interpolates them
    linearly, so it reproduces the grid:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(8, 8))
    boundary = od.boundary.longest_boundary_loop(vertices, faces)
    boundary_uv = wp.array(vertices.numpy()[boundary.numpy(), :2], dtype=wp.vec2)
    uv = od.parametrization.harmonic(
        vertices=vertices, faces=faces, boundary_indices=boundary, boundary_uv=boundary_uv
    )
    print(np.allclose(uv.numpy(), vertices.numpy()[:, :2], atol=1e-5))  # True
    ```
    """
    bound, owned, arguments = mesh_arguments(
        "harmonic", mesh, vertices, faces, (boundary_indices, boundary_uv, k), 2
    )
    return _fixed_boundary_map(
        bound,
        owned,
        _HARMONIC_NONNEGATIVE if nonnegative_weights else "harmonic",
        cast("wp.array[wp.int32]", arguments[0]),
        cast("wp.array[wp.vec2]", arguments[1]),
        cast("int", arguments[2]),
        solver,
    )


@overload
def tutte(
    mesh: Trimesh,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    /,
    *,
    k: int = 1,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
@overload
def tutte(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    /,
    k: int = 1,
    *,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
def tutte(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    boundary_indices: wp.array[wp.int32] | wp.array[wp.vec2] | None = None,
    boundary_uv: wp.array[wp.vec2] | None = None,
    k: int = 1,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]:
    """
    Tutte embedding with fixed boundary (uniform-Laplacian parametrization).

    Identical to [`harmonic`][ordito.parametrization.harmonic] except the operator is the
    combinatorial [`graph_laplacian`][ordito.laplacian.graph_laplacian] instead of the
    cotangent one — for ``k == 1`` that Laplacian is the *only* difference. Because the uniform
    Laplacian's free-free block is a diagonally dominant M-matrix, the Tutte embedding of a mesh
    with a convex boundary is guaranteed bijective (fold-free), unlike the harmonic/conformal maps.
    For ``k > 1`` the mass matrix is the identity (matching libigl's ``speye`` graph-Laplacian
    variant). Solved, and its factorizations kept, as [`harmonic`][ordito.parametrization.harmonic]
    is (see its Notes).

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    boundary_indices
        ``(n_boundary,)`` indices of the fixed (constrained) vertices.
    boundary_uv
        ``(n_boundary,)`` target UV positions for ``boundary_indices``, in the same order (e.g. a
        convex loop from
        [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]).
    k
        Laplacian power (``>= 1``). ``k == 1`` is the classic Tutte embedding.
    solver
        ``"auto"`` (default) or ``"direct"``, as for
        [`harmonic`][ordito.parametrization.harmonic].

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``k < 1``, if there are interior vertices but ``boundary_indices`` is empty, if
        ``boundary_indices`` and ``boundary_uv`` have different lengths, if ``boundary_indices``
        repeats a vertex or holds one outside ``[0, n_vertices)``, or if ``solver`` is not
        ``"auto"`` or ``"direct"``.
    RuntimeError
        If ``vertices``, ``faces``, ``boundary_indices`` and ``boundary_uv`` are not all on one
        device.

    See Also
    --------
    [`harmonic`][ordito.parametrization.harmonic]
    [`graph_laplacian`][ordito.laplacian.graph_laplacian]
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]

    Examples
    --------
    Even a cap curved almost into a sphere embeds without a fold onto the convex unit disk:

    ```python
    import ordito as od

    vertices, faces = od.creation.sphere_cap(angle=2.5, subdivisions=3)
    boundary = od.boundary.longest_boundary_loop(vertices, faces)
    circle = od.parametrization.map_vertices_to_circle(vertices, boundary)
    uv = od.parametrization.tutte(
        vertices=vertices, faces=faces, boundary_indices=boundary, boundary_uv=circle
    )
    print(uv.shape)  # (217,)
    print(od.parametrization.face_flipped_indices(uv, faces).shape)  # (0,)
    ```
    """
    bound, owned, arguments = mesh_arguments(
        "tutte", mesh, vertices, faces, (boundary_indices, boundary_uv, k), 2
    )
    return _fixed_boundary_map(
        bound,
        owned,
        "tutte",
        cast("wp.array[wp.int32]", arguments[0]),
        cast("wp.array[wp.vec2]", arguments[1]),
        cast("int", arguments[2]),
        solver,
    )


def _fixed_boundary_map(
    mesh: Trimesh,
    owned: bool,
    method: str,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int,
    solver: str,
) -> wp.array[wp.vec2]:
    """``harmonic`` / ``tutte`` on a resolved mesh: validate, pin the boundary, solve, scatter."""
    vertices = mesh.vertices
    require_same_device(
        vertices=vertices,
        faces=mesh.faces,
        boundary_indices=boundary_indices,
        boundary_uv=boundary_uv,
    )
    if k < 1:
        raise ValueError(f"{method} power k must be >= 1, got {k}.")
    _require_solver_name(solver, method)
    device = vertices.device
    n_vertices = vertices.size
    if n_vertices == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device)
    # A mesh with interior vertices and no fixed boundary is a singular Dirichlet system; checked
    # before any operator is assembled.
    _require_fixed_vertices(
        boundary_indices.size,
        n_vertices,
        1,
        "harmonic / tutte require at least one fixed boundary vertex; the Dirichlet system is "
        "otherwise singular.",
    )
    fixed_mask, fixed_values = _scatter_constraints(
        n_vertices, boundary_indices, boundary_uv, device
    )
    fixed_values_2d = odt.as_array2d(fixed_values, wp.float64)
    sol, free_map = _solve_fixed_vertices(
        mesh, owned, method, k, fixed_mask, fixed_values_2d, solver, boundary_indices.size
    )
    out_uv = _launch.empty(n_vertices, dtype=wp.vec2, device=device)
    _launch.launch(
        kernel_parametrization.scatter_solution,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, sol, fixed_values_2d, out_uv],
        device=device,
    )
    return out_uv


def arap(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    fixed_indices: wp.array[wp.int32],
    fixed_uv: wp.array[wp.vec2],
    uv_init: wp.array[wp.vec2],
    max_iterations: int = 10,
    tolerance: float = 1e-7,
    *,
    pattern: od.laplacian.MeshOperatorPattern | None = None,
) -> wp.array[wp.vec2]:
    """
    As-rigid-as-possible (ARAP) parametrization with fixed vertices.

    Minimizes the ARAP energy of the 2D parametrization by local/global alternation, starting from
    ``uv_init`` and keeping ``fixed_indices`` pinned to ``fixed_uv`` at every iteration. The local
    step fits, per triangle, the closest rotation between the isometrically flattened rest triangle
    and its current UV image (closed-form 2D polar decomposition, reflections forbidden); the global
    step solves the cotangent-Laplacian Poisson system ``(-L)_uu U_u = (K R)_u - (-L)_ub bc`` for
    each UV column with conjugate gradient, or by a factorization of it once those solves prove
    long (see Notes).

    A good ``uv_init`` matters: ARAP is non-convex, so feed a fold-free initial map such as
    [`harmonic`][ordito.parametrization.harmonic] or [`tutte`][ordito.parametrization.tutte], with
    the boundary placed by
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]. Inspect the result
    for inverted triangles with
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    fixed_indices
        ``(n_fixed,)`` indices of the pinned (constrained) vertices — libigl's ``b``. At least one
        is required whenever the mesh has interior vertices (the ARAP global system is otherwise a
        singular, translation-invariant Poisson problem). Pinning the whole boundary loop reproduces
        the classic fixed-boundary ARAP disk parametrization.
    fixed_uv
        ``(n_fixed,)`` target UV positions for ``fixed_indices``, in the same order — libigl's
        ``bc``.
    uv_init
        ``(n_vertices,)`` initial UV coordinates (the warm start). Never mutated; the interior
        values seed the first local step and the conjugate-gradient warm start, and the pinned rows
        are overwritten with ``fixed_uv`` before the first iteration.
    max_iterations
        Number of local/global iterations (``>= 1``). libigl defaults to ``10``.
    tolerance
        Relative residual tolerance of each global step's solve (``> 0``). Looser than the
        ``1e-8`` of the single-solve maps: each solve is an inner step of a truncated outer
        iteration, and solving past that iteration's own error changes nothing.
    pattern
        A ``"cotmatrix"`` [`mesh_operator_pattern`][ordito.laplacian.mesh_operator_pattern] of
        ``faces`` ([`Trimesh.mesh_operator_pattern`][ordito.mesh.Trimesh.mesh_operator_pattern]),
        which spares the cotangent assembly its sort.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``. Empty for an empty mesh. Rows at
        ``fixed_indices`` equal ``fixed_uv`` exactly.

    Raises
    ------
    ValueError
        If ``max_iterations < 1``, if ``tolerance <= 0``, if there are interior vertices but
        ``fixed_indices`` is empty, if ``fixed_indices`` and ``fixed_uv`` have different lengths,
        if ``fixed_indices`` repeats a vertex or holds one outside ``[0, n_vertices)``, or if
        ``pattern`` was built for another operator or vertex count.
    RuntimeError
        If ``vertices``, ``faces``, ``fixed_indices``, ``fixed_uv``, ``uv_init`` and ``pattern``
        are not all on one device.

    See Also
    --------
    [`harmonic`][ordito.parametrization.harmonic]
    [`tutte`][ordito.parametrization.tutte]
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices]
    [`low_distortion_parametrization`][ordito.parametrization.low_distortion_parametrization]
        The same energy with no pins and no start.

    Notes
    -----
    Uses the *elements* ARAP energy (one rotation per triangle), libigl's default for the flat
    ``dim = 2`` case, with unclamped half-cotangent weights as libigl's. Each face's rotation is the
    proper rotation nearest its Jacobian -- the rotation ``igl::fit_rotations_planar`` fits to the
    weighted edge covariance -- and the right-hand side is gathered per vertex, the same vector as
    libigl's rotated rest edges. The local step shares its code with
    [`low_distortion_parametrization`][ordito.parametrization.low_distortion_parametrization].
    Everything between iterations is ``float64``.

    The global step's operator is the same at every iteration. When the second iteration's solve
    shows the solves are long -- a graded mesh, or a long schedule on a large one -- the remaining
    iterations solve by a [`sparse_cholesky`][ordito.cholesky.sparse_cholesky] factorization of it,
    refined to ``tolerance``, held until the call returns: device memory of the order of the
    operator's fill-in, within [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET].
    An operator that cannot be factored keeps iterating.

    Examples
    --------
    Started from a Tutte embedding, the result stays flip-free and keeps the pinned boundary
    exactly:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.sphere_cap(angle=1.0, subdivisions=3)
    boundary = od.boundary.longest_boundary_loop(vertices, faces)
    circle = od.parametrization.map_vertices_to_circle(vertices, boundary)
    uv_init = od.parametrization.tutte(
        vertices=vertices, faces=faces, boundary_indices=boundary, boundary_uv=circle
    )
    uv = od.parametrization.arap(vertices, faces, boundary, circle, uv_init)
    print(od.parametrization.face_flipped_indices(uv, faces).shape)  # (0,)
    print(np.array_equal(uv.numpy()[boundary.numpy()], circle.numpy()))  # True
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        fixed_indices=fixed_indices,
        fixed_uv=fixed_uv,
        uv_init=uv_init,
        pattern=None if pattern is None else pattern.offsets,
    )
    if max_iterations < 1:
        raise ValueError(f"arap max_iterations must be >= 1, got {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"arap tolerance must be > 0, got {tolerance}.")
    device = vertices.device
    n_vertices = vertices.size
    if n_vertices == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device)

    # Interior vertices with nothing pinned leave the ARAP global system translation-invariant
    # (singular) -- and since n_vertices > 0, an empty ``fixed_indices`` always means at least one
    # interior vertex exists. Checked before the cotangent/Laplacian build below (mirroring
    # harmonic / tutte), so a rejected call doesn't pay for the assembly first.
    _require_fixed_vertices(
        fixed_indices.size,
        n_vertices,
        1,
        "arap requires at least one fixed vertex when the mesh has interior vertices; the ARAP "
        "global system is otherwise singular (translation invariant).",
    )
    n_faces = faces.size // 3

    laplacian = cotmatrix(vertices, faces, dtype=wp.float64, pattern=pattern)

    # Partition vertices into pinned (fixed) and interior (free), and extract the global-step
    # operator with it: the interior block of Q = -L and the constant boundary term -(-L)_ub bc.
    # ``fixed_values`` is the (2, n_vertices) prescribed-UV buffer (row 0 = u, row 1 = v) shared
    # with the assembly / scatter kernels; ``interior_map`` compacts free vertices into the reduced
    # system.
    # ``future work``: libigl also supports rotation groups ``G`` (shared rotations across grouped
    # faces, replacing the per-face fit with a group-summed covariance) and ``with_dynamics`` (a
    # mass-matrix + timestep term added to Q and the right-hand side); both are out of scope here.
    fixed_mask, fixed_values = _scatter_constraints(n_vertices, fixed_indices, fixed_uv, device)
    fixed_values_2d = odt.as_array2d(fixed_values, wp.float64)
    q_uu, rhs_const, interior_map, n_interior = twl.partitioned_interior_system(
        laplacian, fixed_mask, fixed_values_2d, scale=-1.0
    )
    # The scatter skips an out-of-range index and a repeated one marks its vertex once, so either
    # leaves fewer vertices fixed than ``fixed_indices`` lists: the free count just read shows it.
    if n_vertices - n_interior != fixed_indices.size:
        raise ValueError(
            "arap: fixed_indices must be distinct vertex indices in [0, n_vertices); they fix "
            f"{n_vertices - n_interior} vertices where {fixed_indices.size} were given"
        )

    out_uv = _launch.empty(n_vertices, dtype=wp.vec2, device=device)
    if n_interior == 0:
        # Every vertex pinned: the prescribed positions are the whole answer, and the loop's
        # setup would serve no solve.
        empty_sol = _launch.zeros((2, 0), dtype=wp.float64, device=device)
        _launch.launch(
            kernel_parametrization.scatter_solution,
            dim=n_vertices,
            inputs=[fixed_mask, interior_map, empty_sol, fixed_values_2d, out_uv],
            device=device,
        )
        return out_uv

    # The loop's buffers, none allocated inside it: each face's gradients and area, the
    # warm-started solution per column ``sol`` (seeded from ``uv_init``'s free rows by the same
    # launch) and the global step's right-hand side ``b``, the last two in one allocation.
    vertex_faces, offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    gradients = odt.empty_2d((n_faces, 3), wp.vec2d, device=device)
    areas = _launch.empty(n_faces, dtype=wp.float64, device=device)
    solve_buffers = _launch.empty(4 * n_interior, dtype=wp.float64, device=device)
    sol = odt.as_array2d(
        _launch.view(solve_buffers, 0, 2 * n_interior).reshape((2, n_interior)), wp.float64
    )
    b = odt.as_array2d(
        _launch.view(solve_buffers, 2 * n_interior, 4 * n_interior).reshape((2, n_interior)),
        wp.float64,
    )
    _launch.launch(
        kernel_parametrization.arap_setup,
        dim=max(n_faces, n_vertices),
        inputs=[vertices, faces, fixed_mask, interior_map, uv_init],
        outputs=[gradients, areas, sol],
        device=device,
    )

    # One batched CG state for both UV columns, reading ``b`` and warm-started from ``sol``. The
    # global steps are short warm-started Jacobi-Chebyshev solves of an operator rebuilt per call,
    # so it factors only once they are found long (``_ARAP_FACTOR_FROM_ROUNDS_CUDA`` / ``_CPU``).
    solver = twl.spd_column_solver(
        q_uu, b, sol, tol=tolerance, maxiter=10 * n_interior, preconditioner="chebyshev",
        factorize=False,
    )  # fmt: skip

    held = None
    for iteration in range(max_iterations):
        # The local step and the global step's right-hand side at each free vertex.
        _launch.launch(
            kernel_parametrization.arap_rhs,
            dim=n_vertices,
            inputs=[faces, vertex_faces, offsets, gradients, areas, fixed_values_2d, fixed_mask,
                    interior_map, sol, rhs_const],
            outputs=[b],
            device=device,
        )  # fmt: skip
        # Both UV columns in one batched symmetric-PD solve, warm-started from the previous ``sol``;
        # by a factorization of the operator once the iteration is found long.
        if held is not None:
            held.solve(b, sol, tol=tolerance)
            continue
        rounds, _, _ = solver()
        if iteration == 1 and iteration + 1 < max_iterations:
            # The second solve's round count, read once a call: it predicts the later solves'
            # length (the first, from ``uv_init``, does not: a harmonic start's first solve can be
            # long and every later one short), which decides whether a factorization pays.
            remaining = max_iterations - iteration - 1
            if _round_count(rounds) * remaining >= (
                _ARAP_FACTOR_FROM_ROUNDS_CUDA
                if wp.get_device(device).is_cuda
                else _ARAP_FACTOR_FROM_ROUNDS_CPU
            ):
                held = _arap_factorization(q_uu, vertices, fixed_mask, interior_map)
    if held is not None:
        held.release()
    # The full UV field, the pinned rows at their constraints.
    _launch.launch(
        kernel_parametrization.scatter_solution,
        dim=n_vertices,
        inputs=[fixed_mask, interior_map, sol, fixed_values_2d, out_uv],
        device=device,
    )
    return out_uv


def _arap_factorization(
    q_uu: odt.BsrMatrix[wp.float64],
    vertices: wp.array[wp.vec3],
    fixed_mask: wp.array[wp.bool],
    interior_map: wp.array[wp.int32],
) -> twl.OperatorFactorization | None:
    """
    Factor ``arap``'s global-step operator for its remaining solves, or ``None`` if it cannot be.

    The factorization records its solve graph on its first solve, since every later iteration
    replays it.
    """
    held = twl.OperatorFactorization(q_uu, vertices, fixed_mask=fixed_mask, free_map=interior_map)
    if not held.factor() or held.factorization is None:
        return None
    held.factorization.records_at_once = True
    return held


def _round_count(rounds: object) -> int:
    """Return a solve's round count, a device array on CUDA and a host integer on the CPU."""
    if isinstance(rounds, wp.array):
        return int(read_scalar(cast("wp.array[wp.int32]", rounds), 0))
    return int(cast("int", rounds))


def _scatter_constraints(
    n_vertices: int, indices: wp.array[wp.int32], uv: wp.array[wp.vec2], device: wp.DeviceLike
) -> tuple[wp.array[wp.bool], wp.array[wp.float64]]:
    """
    Expand a list of pinned vertices into the dense mask and prescribed-UV buffers the solvers take.

    ``fixed_mask`` marks the constrained vertices; ``fixed_values`` is the ``(2, n_vertices)``
    prescribed-UV buffer (row 0 = u, row 1 = v) the assembly and scatter kernels read. Shared by
    ``_fixed_boundary_map`` and
    [`arap`][ordito.parametrization.arap]; an empty ``indices`` yields an all-``False`` mask, which
    each caller rejects on its own terms. Only the fixed entries of ``fixed_values`` are written:
    every reader (the extraction's right-hand side, the scatters) reads it behind the mask.

    Raises
    ------
    ValueError
        If ``indices`` and ``uv`` have different lengths -- the scatter kernel below indexes ``uv``
        at every position up to ``indices.shape[0]``, so a shorter ``uv`` is an out-of-bounds read.
    """
    n_fixed = indices.size
    if uv.size != n_fixed:
        raise ValueError(f"indices and uv must have the same length, got {n_fixed} and {uv.size}.")
    # One scatter marks the mask and writes the values, skipping an out-of-range index in both. A
    # skipped or repeated index fixes fewer vertices than ``indices`` lists, which each caller
    # rejects from the free count it reads anyway (``_require_fixed_count``, ``arap``).
    fixed_mask = _launch.zeros(n_vertices, dtype=wp.bool, device=device)
    fixed_values = _launch.empty((2, n_vertices), dtype=wp.float64, device=device)
    if n_fixed > 0:
        _launch.launch(
            kernel_parametrization.scatter_fixed_uv,
            dim=n_fixed,
            inputs=[indices, uv, fixed_mask, fixed_values],
            device=device,
        )
    return fixed_mask, fixed_values


def low_distortion_parametrization(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    coarse_cell: float | None = None,
    normal: wp.vec3 | None = None,
    iterations: int = 150,
    smoothing_passes: int = 3,
    smoothing_steps: int = 8,
    single_level_vertices: int = 6_000,
) -> wp.array[wp.vec2]:
    """
    Free-boundary chart of low ARAP distortion in the mesh's own units, for any topology.

    Flattens an open surface -- with handles, several boundary loops, several components -- into a
    near-isometric chart with a free boundary: no vertex is pinned, no boundary shape is imposed
    and no initial map is needed. It minimizes the element ARAP energy ``sum_f A_f |J_f - R_f|^2``
    (one proper rotation per face) in two levels, because a chart's large-scale layout needs dozens
    of local/global iterations while depending only on the coarse shape:

    1. The vertices are clustered on a grid of cells of side ``coarse_cell``. The clusters' means,
       joined by every face whose corners fall in three distinct clusters, form a coarse mesh. Each
       coarse face is wound as the fine face that spans it, so a flat sheet's coarse mesh is flat.
    2. ARAP is iterated on the coarse mesh from its projection onto the plane of ``normal``. Each
       global step is proximal, ``(K + epsilon I) U = B + epsilon U_prev``, with ``epsilon`` a
       millionth of the mean stiffness diagonal: it removes the translations of every piece the
       clustering disconnects and keeps each where it started, and changes nothing at
       convergence. The system is factored once.
    3. Each vertex takes its cluster's UV plus its offset from the cluster's mean, carried by the
       cluster's area-weighted mean Jacobian; a cluster no coarse face touches borrows the nearest
       covered cluster's.
    4. ``smoothing_passes`` ARAP passes on the full mesh repair the fine detail, each global step
       cut short after ``smoothing_steps`` Jacobi-preconditioned conjugate-gradient steps from the
       current chart. The full mesh's system is applied face by face, never assembled.

    A mesh of at most ``single_level_vertices`` vertices, or one whose clustering leaves fewer
    than 16 coarse faces or more clusters than half its vertices, is iterated directly as in
    step 2, from its own projection.
    [`low_distortion_disk_parametrization`][ordito.parametrization.low_distortion_disk_parametrization]
    starts a disk from a harmonic, Tutte or LSCM map instead.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer. Faces of no area are ignored.
    coarse_cell
        Side of the grid cells the vertices are clustered on, in the mesh's units: a few times
        the mesh's edge length, and smaller than the gaps between sheets that must not be welded
        together (two sheets closer than a cell share clusters, which folds the chart there).
        ``None`` takes five times the mean edge length
        ([`mean_edge_length`][ordito.edges.mean_edge_length]). The iterations also stop once one
        moves no vertex by more than a thousandth of it.
    normal
        Normal of the plane the start is projected onto (any length but zero). ``None`` takes the
        plane fitted to the longest boundary loop, the rim a scan crop or a disk is viewed through,
        or the vertices' best-fit plane on a mesh with no boundary.
    iterations
        ARAP iteration budget on the coarse mesh (or on the mesh itself when it is iterated
        directly). ``0`` keeps the projection.
    smoothing_passes
        ARAP passes on the full mesh after the coarse level.
    smoothing_steps
        Conjugate-gradient steps of each smoothing pass.
    single_level_vertices
        A mesh of at most this many vertices is iterated directly, without a coarse level.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``, in the mesh's units, with the
        faces mostly counter-clockwise; the rotation is unspecified. A vertex no face uses follows
        its cluster, or keeps its projection when the mesh is iterated directly. Empty for an
        empty mesh.

    Raises
    ------
    ValueError
        If ``vertices`` is not ``wp.vec3`` or ``faces`` not ``wp.int32`` with a multiple of three
        entries, if ``coarse_cell`` is given and not a positive finite number, if ``normal`` is
        zero or not finite, if ``iterations``, ``smoothing_passes``, ``smoothing_steps`` or
        ``single_level_vertices`` is negative, if a vertex coordinate is not finite, if no face has
        a positive area, or if ``coarse_cell`` is so small against the mesh that the grid's cells
        or clusters cannot be numbered.
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.

    See Also
    --------
    [`low_distortion_disk_parametrization`][ordito.parametrization.low_distortion_disk_parametrization]
    [`arap`][ordito.parametrization.arap]
        The pinned, warm-started ARAP map.
    [`lscm`][ordito.parametrization.lscm]
        A free-boundary conformal map.
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices]

    Notes
    -----
    The chart approximates a converged full-resolution element ARAP: within a few points of its
    mean distortion. It is not guaranteed free of inverted faces, and where the grid welds sheets
    that pass closer than a cell, or squashes a handle, it leaves more of them than a
    full-resolution solve would.

    The coarse level's sums and its solve run in ``float64``. Under
    [`is_deterministic`][ordito.config.is_deterministic] the chart repeats bit for bit; otherwise
    its last bits can move between calls on the CUDA device.

    Examples
    --------
    A flat sheet has an exact answer, an isometry, which the chart reproduces:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.grid(count=(41, 41))
    uv = od.parametrization.low_distortion_parametrization(vertices, faces)
    corners_np = vertices.numpy()[faces.numpy().reshape(-1, 3)]
    corners_uv = uv.numpy()[faces.numpy().reshape(-1, 3)]
    lengths_np = np.linalg.norm(corners_np - np.roll(corners_np, 1, axis=1), axis=2)
    lengths_uv = np.linalg.norm(corners_uv - np.roll(corners_uv, 1, axis=1), axis=2)
    print(uv.shape)  # (1681,)
    print(np.allclose(lengths_uv, lengths_np, atol=1e-5))  # True
    print(od.parametrization.face_flipped_indices(uv, faces).shape)  # (0,)
    ```
    """
    name = "low_distortion_parametrization"
    bounds = _low_distortion_bounds(
        name,
        vertices,
        faces,
        (
            ("iterations", iterations),
            ("smoothing_passes", smoothing_passes),
            ("smoothing_steps", smoothing_steps),
            ("single_level_vertices", single_level_vertices),
        ),
    )
    if coarse_cell is not None and not (math.isfinite(coarse_cell) and coarse_cell > 0.0):
        raise ValueError(f"{name} coarse_cell must be positive and finite, got {coarse_cell}")
    if normal is not None:
        length = math.sqrt(sum(float(c) ** 2 for c in normal))
        if not (math.isfinite(length) and length > 0.0):
            raise ValueError(f"{name} normal must be non-zero and finite, got {normal}")
    if bounds is None:
        return _launch.empty(0, dtype=wp.vec2, device=vertices.device)
    low, high = bounds
    if coarse_cell is None:
        coarse_cell = _DEFAULT_CELL_EDGES * od.edges.mean_edge_length(vertices, faces)
    plane_normal = _projection_normal(vertices, faces, normal)

    settled = _TWO_LEVEL_SETTLED * coarse_cell
    chart = None
    if vertices.size > single_level_vertices:
        chart = _two_level_chart(
            vertices,
            faces,
            low,
            high,
            coarse_cell,
            plane_normal,
            iterations,
            settled,
            smoothing_passes,
            smoothing_steps,
        )
    if chart is None:
        vertex_faces, offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=vertices.size)
        chart = _proximal_arap(
            od.array.astype(vertices, wp.vec3d),
            vertices,
            faces,
            vertex_faces,
            offsets,
            iterations,
            settled,
            normal=plane_normal,
        )
    return _chart_uv(chart)


def low_distortion_disk_parametrization(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    init: Literal["harmonic", "tutte", "lscm"] = "harmonic",
    iterations: int = 150,
) -> wp.array[wp.vec2]:
    """
    Free-boundary chart of low ARAP distortion of a disk, from a harmonic, Tutte or LSCM start.

    The disk counterpart of
    [`low_distortion_parametrization`][ordito.parametrization.low_distortion_parametrization]:
    the same pin-free element ARAP energy ``sum_f A_f |J_f - R_f|^2`` and free boundary, started
    from a classic map of the whole mesh instead of a coarse level, and iterated on the mesh
    itself, each global step a factored solve, until an iteration moves no vertex by more than
    ``5e-3`` mean edge lengths. A start's distortion is large-scale, which only whole solves
    repair.

    The start is [`harmonic`][ordito.parametrization.harmonic] or
    [`tutte`][ordito.parametrization.tutte] with the boundary loop placed on a circle by
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle], or
    [`lscm`][ordito.parametrization.lscm] with two vertices half the loop apart pinned; it is
    scaled to the mesh's area and mirrored if its faces wind clockwise on the whole, since ARAP
    fits proper rotations and cannot unmirror it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer of a disk: one connected component with one
        boundary loop and Euler characteristic 1. Faces of no area are ignored.
    init
        The start: ``"harmonic"`` (default), ``"tutte"`` or ``"lscm"``.
    iterations
        ARAP iteration budget. ``0`` keeps the scaled start.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``, in the mesh's units, with the
        faces mostly counter-clockwise; the rotation is unspecified. Empty for an empty mesh.

    Raises
    ------
    ValueError
        If ``vertices`` is not ``wp.vec3`` or ``faces`` not ``wp.int32`` with a multiple of three
        entries, if ``init`` is not one of the three, if ``iterations`` is negative, if a vertex
        coordinate is not finite, if no face has a positive area, or if the mesh is not a disk.
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.

    See Also
    --------
    [`low_distortion_parametrization`][ordito.parametrization.low_distortion_parametrization]
        Any topology, from a coarse level.
    [`arap`][ordito.parametrization.arap]
        The pinned, warm-started ARAP map.

    Notes
    -----
    The factored iterations' sums and solves run in ``float64``. Under
    [`is_deterministic`][ordito.config.is_deterministic] the chart repeats bit for bit; otherwise
    its last bits can move between calls on the CUDA device.

    Examples
    --------
    A curved cap from each start, its edges kept at the surface's mean length:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.sphere_cap(angle=1.0, subdivisions=3)
    corners = faces.numpy().reshape(-1, 3)


    def mean_edge(points: np.ndarray) -> str:
        triangles = points[corners]
        lengths = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
        return f"{lengths.mean():.3f}"


    charts = [
        od.parametrization.low_distortion_disk_parametrization(vertices, faces, init=init)
        for init in ("harmonic", "tutte", "lscm")
    ]
    print(mean_edge(vertices.numpy()))  # 0.134
    print([mean_edge(uv.numpy()) for uv in charts])  # ['0.134', '0.134', '0.134']
    print(od.parametrization.face_flipped_indices(charts[0], faces).shape)  # (0,)
    ```
    """
    name = "low_distortion_disk_parametrization"
    bounds = _low_distortion_bounds(name, vertices, faces, (("iterations", iterations),))
    if init not in _DISK_STARTS:
        raise ValueError(f"{name} init must be one of {_DISK_STARTS}, got {init!r}")
    if bounds is None:
        return _launch.empty(0, dtype=wp.vec2, device=vertices.device)
    loop = _disk_boundary(name, vertices, faces)
    settled = _TWO_LEVEL_SETTLED * _DEFAULT_CELL_EDGES * od.edges.mean_edge_length(vertices, faces)
    vertex_faces, offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=vertices.size)
    chart = _proximal_arap(
        od.array.astype(vertices, wp.vec3d),
        vertices,
        faces,
        vertex_faces,
        offsets,
        iterations,
        settled,
        start=_start_chart(vertices, faces, loop, init),
    )
    return _chart_uv(chart)


def _disk_boundary(
    name: str, vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Return a disk's boundary loop, after checking the mesh is one.

    Raises
    ------
    ValueError
        Unless the mesh is one connected component with one boundary loop and Euler
        characteristic 1.
    """
    n_vertices = vertices.size
    euler = od.measures.euler_characteristic(faces, n_vertices=n_vertices)
    loops, loop_offsets = od.boundary.boundary_loops_with_offsets(vertices, faces)
    n_loops = loop_offsets.size - 1
    labels = od.adjacency.face_connected_component_labels(faces, n_vertices=n_vertices)
    n_components = od.grouping.unique_1d(labels).size
    if euler != 1 or n_loops != 1 or n_components != 1:
        raise ValueError(
            f"{name} needs a disk: one connected component with one boundary loop and Euler "
            f"characteristic 1, got {n_components} components, {n_loops} boundary loops and "
            f"Euler characteristic {euler}"
        )
    return loops


def _low_distortion_bounds(
    name: str,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    counts: tuple[tuple[str, int], ...],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """
    Validate a low-distortion chart's mesh and counts; return the vertices' bounding box.

    ``None`` for a mesh with no vertices.

    Raises
    ------
    ValueError
        If the arrays have the wrong dtypes, a count is negative, a coordinate is not finite or no
        face has a positive area.
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    if vertices.dtype is not wp.vec3 or faces.dtype is not wp.int32 or faces.size % 3 != 0:
        raise ValueError(
            f"{name} takes wp.vec3 vertices and a flat wp.int32 faces buffer of 3 * n_faces "
            f"entries, got {vertices.dtype.__name__} vertices and {faces.size} "
            f"{faces.dtype.__name__} faces"
        )
    for argument, value in counts:
        if value < 0:
            raise ValueError(f"{name} {argument} must be >= 0, got {value}")
    device = vertices.device
    n_vertices = vertices.size
    if n_vertices == 0:
        return None
    n_faces = faces.size // 3
    bounds = _launch.full(8, math.inf, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_parametrization.two_level_bounds,
        dim=kernel_reduce.blocks_1d(max(n_vertices, n_faces)),
        inputs=[vertices, faces],
        outputs=[bounds],
        block_dim=TILE_1D,
        device=device,
    )
    # Read back: the validation decides whether to raise, and the grid's extent sizes its keys.
    low_x, low_y, low_z, high_x, high_y, high_z, non_finite, positive_face = read_values(
        bounds, 0, 8
    )
    if non_finite == 0.0:
        raise ValueError(f"{name}: every vertex coordinate must be finite")
    if positive_face != 0.0:
        raise ValueError(f"{name} needs at least one face of positive area")
    return (
        (float(low_x), float(low_y), float(low_z)),
        (-float(high_x), -float(high_y), -float(high_z)),
    )


def _projection_normal(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], normal: wp.vec3 | None
) -> wp.vec3d:
    """
    Return the unit normal of the start's projection plane, zero for the best-fit plane.

    ``normal`` normalized when given; else the normal of the plane fitted to the longest boundary
    loop, or zero (the vertices' best-fit plane) when there is none.
    """
    if normal is None:
        loop = od.boundary.longest_boundary_loop(vertices, faces)
        if loop.size < 3:
            return wp.vec3d(0.0, 0.0, 0.0)
        normal = od.points.fit_plane(cast("wp.array[wp.vec3]", od.array.gather(vertices, loop)))[0]
    x, y, z = float(normal[0]), float(normal[1]), float(normal[2])
    length = math.sqrt(x * x + y * y + z * z)
    return wp.vec3d(x / length, y / length, z / length)


def _chart_uv(chart: odt.Array2dFloat64) -> wp.array[wp.vec2]:
    """Narrow a ``(2, n)`` ``float64`` chart to ``float32`` UV."""
    n_vertices = int(chart.shape[1])
    out_uv = _launch.empty(n_vertices, dtype=wp.vec2, device=chart.device)
    _launch.launch(
        kernel_parametrization.chart_to_uv,
        dim=n_vertices,
        inputs=[chart],
        outputs=[out_uv],
        device=chart.device,
    )
    return out_uv


def _start_chart(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], loop: wp.array[wp.int32], init: str
) -> odt.Array2dFloat64:
    """
    Return the map ``low_distortion_disk_parametrization`` starts from, as a ``(2, n)`` chart.

    Placing the disk's boundary ``loop``; scaled to the mesh's area and mirrored if its faces wind
    clockwise on the whole.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    if init == "lscm":
        # Two vertices half the loop apart, at unit distance: the area scaling fixes the scale.
        slots = wp.array([0, loop.size // 2], dtype=wp.int32, device=device)
        pins = cast("wp.array[wp.int32]", od.array.gather(loop, slots))
        pins_uv = wp.array([[0.0, 0.0], [1.0, 0.0]], dtype=wp.vec2, device=device)
        uv = lscm(vertices, faces, pins, pins_uv)
    else:
        circle = map_vertices_to_circle(vertices, loop)
        boundary_map = harmonic if init == "harmonic" else tutte
        uv = boundary_map(vertices, faces, loop, circle)
    blocks = kernel_reduce.blocks_1d(n_faces)
    areas, _ = _launch.block_partials(blocks, 1, wp.vec3d, device)
    _launch.launch_tiled(
        kernel_parametrization.start_chart_areas,
        dim=blocks,
        inputs=[vertices, faces, uv],
        outputs=[areas],
        block_dim=TILE_1D,
        device=device,
    )
    chart = odt.as_array2d(
        _launch.empty((2, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    _launch.launch(
        kernel_parametrization.scaled_start_chart,
        dim=n_vertices,
        inputs=[uv, areas],
        outputs=[chart],
        device=device,
    )
    return chart


def _two_level_chart(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    low: tuple[float, float, float],
    high: tuple[float, float, float],
    coarse_cell: float,
    normal: wp.vec3d,
    iterations: int,
    settled: float,
    smoothing_passes: int,
    smoothing_steps: int,
) -> odt.Array2dFloat64 | None:
    """
    Return the chart through the coarse level, as ``(2, n_vertices)`` ``float64``.

    ``None`` when the clustering leaves too few coarse faces, or too many clusters, for a coarse
    level to pay: the caller then iterates the mesh directly.
    """
    coarse = _coarse_mesh(vertices, faces, low, high, coarse_cell)
    if coarse is None:
        return None
    cluster, centroids, centroids_f32, coarse_faces = coarse
    n_clusters = centroids.size
    coarse_vertex_faces, coarse_offsets = od.adjacency.vertex_face_adjacency(
        coarse_faces, n_vertices=n_clusters
    )
    coarse_chart = _proximal_arap(
        centroids,
        centroids_f32,
        coarse_faces,
        coarse_vertex_faces,
        coarse_offsets,
        iterations,
        settled,
        normal=normal,
    )
    chart = _prolonged_chart(
        vertices, cluster, centroids, centroids_f32, coarse_faces, coarse_vertex_faces,
        coarse_offsets, coarse_chart,
    )  # fmt: skip
    _smooth_chart(vertices, faces, chart, smoothing_passes, smoothing_steps)
    return chart


def _coarse_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    low: tuple[float, float, float],
    high: tuple[float, float, float],
    coarse_cell: float,
) -> tuple[wp.array[wp.int32], wp.array[wp.vec3d], wp.array[wp.vec3], wp.array[wp.int32]] | None:
    """
    Cluster the vertices on a grid and join the clusters' means by the faces spanning three.

    ``low`` and ``high`` are the corners of the vertices' bounding box; cells are anchored at
    ``low``. Returns ``(cluster, centroids, centroids_f32, coarse_faces)``: each vertex's cluster,
    numbered in sorted cell order, the clusters' means in ``float64`` and ``float32``, and the
    flat coarse faces, each triple of clusters once per orientation, wound as the fine face that
    spans it and in ascending key order, nearly collinear ones left out. ``None`` when there are
    more clusters than half the vertices, or fewer than ``_TWO_LEVEL_MIN_COARSE_FACES`` coarse
    faces: no coarse level pays.

    Raises
    ------
    ValueError
        If ``coarse_cell`` is so small that the grid's cells or the clusters cannot be numbered
        by a 63-bit key.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    # The grid's extent in cells, the reference's ``max(floor((x - min) / cell)) + 1`` per axis.
    extents = [math.floor((high[axis] - low[axis]) / coarse_cell) + 1 for axis in range(3)]
    n_cells = extents[0] * extents[1] * extents[2]
    if n_cells > _MAX_PACKED_KEY:
        raise ValueError(
            f"low_distortion_parametrization: coarse_cell {coarse_cell} splits the mesh's "
            "bounding box into "
            f"{n_cells} cells, more than a 63-bit key numbers"
        )
    cell_keys = _launch.empty(n_vertices, dtype=wp.int64, device=device)
    positions = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_parametrization.cluster_cell_keys,
        dim=n_vertices,
        inputs=[vertices, wp.vec3d(*low), wp.float64(coarse_cell), wp.int64(extents[1]),
                wp.int64(extents[2])],
        outputs=[cell_keys, positions],
        device=device,
    )  # fmt: skip
    unique_cells, cluster, counts = od.grouping.unique_1d(
        cell_keys, return_inverse=True, return_counts=True, max_value=n_cells - 1
    )
    n_clusters = unique_cells.size
    if 2 * n_clusters > n_vertices:
        return None
    if n_clusters > _MAX_PACKED_CLUSTERS:
        raise ValueError(
            f"low_distortion_parametrization: coarse_cell {coarse_cell} leaves {n_clusters} "
            "clusters, more than "
            "the coarse faces' 63-bit keys number"
        )
    sums = od.reduce.scatter_sum(positions, cluster, n_clusters)
    centroids = _launch.empty(n_clusters, dtype=wp.vec3d, device=device)
    centroids_f32 = _launch.empty(n_clusters, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_parametrization.cluster_centroids,
        dim=n_clusters,
        inputs=[sums, counts],
        outputs=[centroids, centroids_f32],
        device=device,
    )
    face_keys = _launch.empty(n_faces, dtype=wp.int64, device=device)
    _launch.launch(
        kernel_parametrization.coarse_face_keys,
        dim=n_faces,
        inputs=[vertices, faces, cluster, centroids, wp.int64(n_clusters)],
        outputs=[face_keys],
        device=device,
    )
    unique_faces = od.grouping.unique_1d(face_keys, max_value=n_clusters**3)
    kept = _launch.empty(unique_faces.size, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_parametrization.coarse_face_flags,
        dim=unique_faces.size,
        inputs=[unique_faces, centroids, wp.int64(n_clusters),
                wp.float64(_TWO_LEVEL_FLAT_FACE * coarse_cell * coarse_cell)],
        outputs=[kept],
        device=device,
    )  # fmt: skip
    kept = od.array.flatnonzero_in_place(kept)
    n_coarse_faces = kept.size
    if n_coarse_faces < _TWO_LEVEL_MIN_COARSE_FACES:
        return None
    coarse_faces = _launch.empty(3 * n_coarse_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_parametrization.coarse_faces_emit,
        dim=n_coarse_faces,
        inputs=[unique_faces, kept, wp.int64(n_clusters)],
        outputs=[coarse_faces],
        device=device,
    )
    return cluster, centroids, centroids_f32, coarse_faces


def _proximal_arap(
    points: wp.array[wp.vec3d],
    points_f32: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    iterations: int,
    settled: float,
    *,
    start: odt.Array2dFloat64 | None = None,
    normal: wp.vec3d | None = None,
) -> odt.Array2dFloat64:
    """
    Pin-free ARAP iterated from ``start``, each global step a factored solve.

    ``start`` is a ``(2, n_points)`` chart, iterated in place; ``None`` starts from the points'
    projection onto the plane of unit ``normal``, their best-fit plane when it is zero or
    ``None``.

    Returns the ``(2, n_points)`` chart. The stiffness matrix alone is singular (a translation of
    the chart, or of any piece no face joins to the rest, costs nothing); ``K + epsilon I``,
    matched on the right-hand side by ``epsilon`` times the previous iterate, is not, and holds a
    loose piece where it started. Each iteration's largest move is reduced on the device and read
    every ``_TWO_LEVEL_READ_EVERY`` iterations; the iterations after the one that settles change
    nothing (``kernels/parametrization.proximal_rhs``), so the stop is exact.
    """
    device = points.device
    n_points = points.size
    n_faces = faces.size // 3
    chart = start
    if chart is None:
        chart = odt.as_array2d(
            _launch.empty((2, n_points), dtype=wp.float64, device=device), wp.float64
        )
        _launch.launch_tiled(
            kernel_parametrization.plane_projection_block,
            dim=1,
            inputs=[points, faces, wp.vec3d(0.0, 0.0, 0.0) if normal is None else normal],
            outputs=[chart],
            device=device,
        )
    if iterations == 0:
        return chart

    n_keys = _TWO_LEVEL_FACE_KEYS * n_faces + n_points
    keys, order = od.array.csr_key_buffers(n_keys, device)
    gradients = odt.empty_2d((n_faces, 3), wp.vec2d, device=device)
    areas = _launch.empty(n_faces, dtype=wp.float64, device=device)
    setup_blocks = kernel_reduce.blocks_1d(max(n_faces, n_points))
    trace, _ = _launch.block_partials(setup_blocks, 1, wp.float64, device)
    _launch.launch_tiled(
        kernel_parametrization.proximal_system_setup,
        dim=setup_blocks,
        inputs=[points, faces],
        outputs=[gradients, areas, keys, order, trace],
        block_dim=TILE_1D,
        device=device,
    )
    offsets, columns, starts = od.array.csr_from_keys(keys, order, n_keys, n_points, n_points)
    values = _launch.empty(n_keys, dtype=wp.float64, device=device)
    anchor = _launch.empty(1, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_parametrization.proximal_system_values,
        dim=n_keys,
        inputs=[offsets, starts, order, gradients, areas, trace, wp.int32(n_points)],
        outputs=[values, anchor],
        device=device,
    )
    system = od.array.bsr_from_csr(n_points, n_points, offsets, columns, values)
    factorization = od.cholesky.sparse_cholesky(system, points_f32)
    # Every iteration solves into the same two buffers: recorded on the first solve, replayed.
    factorization.records_at_once = True

    rhs = odt.as_array2d(_launch.empty((2, n_points), dtype=wp.float64, device=device), wp.float64)
    previous = odt.as_array2d(
        _launch.empty((2, n_points), dtype=wp.float64, device=device), wp.float64
    )
    moves = _launch.zeros(iterations, dtype=wp.float64, device=device)
    move_width, move_blocks = kernel_reduce.spanned_blocks_1d(n_points, device)
    # Every vertex is solved: no pinned mask for the local step's reader.
    no_mask, no_map = _no_pins(device)
    rotations = _launch.empty(n_faces, dtype=wp.vec2d, device=device)
    for iteration in range(iterations):
        _launch.launch(
            kernel_parametrization.proximal_rotations,
            dim=n_faces,
            inputs=[faces, gradients, moves, wp.int32(iteration), wp.float64(settled), no_mask,
                    no_map, chart],
            outputs=[rotations],
            device=device,
        )  # fmt: skip
        _launch.launch(
            kernel_parametrization.proximal_rhs,
            dim=n_points,
            inputs=[faces, vertex_faces, vertex_offsets, gradients, areas, rotations, anchor,
                    moves, wp.int32(iteration), wp.float64(settled), chart],
            outputs=[rhs, previous],
            device=device,
        )  # fmt: skip
        factorization.solve(rhs, chart)
        _launch.launch_tiled(
            kernel_parametrization.proximal_moves,
            dim=move_blocks,
            inputs=[chart, previous, wp.int32(iteration), wp.int32(move_width)],
            outputs=[moves],
            block_dim=TILE_1D,
            device=device,
        )
        if (iteration + 1) % _TWO_LEVEL_READ_EVERY == 0 and iteration + 1 < iterations:
            # Read back: whether the loop has settled decides whether to issue more iterations.
            if float(read_scalar(moves, iteration)) <= settled:
                break
    return chart


def _prolonged_chart(
    vertices: wp.array[wp.vec3],
    cluster: wp.array[wp.int32],
    centroids: wp.array[wp.vec3d],
    centroids_f32: wp.array[wp.vec3],
    coarse_faces: wp.array[wp.int32],
    coarse_vertex_faces: wp.array[wp.int32],
    coarse_offsets: wp.array[wp.int32],
    coarse_chart: odt.Array2dFloat64,
) -> odt.Array2dFloat64:
    """
    Carry a coarse chart to the vertices through each cluster's mean Jacobian.

    Returns the ``(2, n_vertices)`` chart. A cluster no coarse face touches has no Jacobian: its
    vertices follow the nearest cluster that has one.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_clusters = centroids.size
    jacobians = _launch.empty(n_clusters, dtype=kernel_parametrization.mat23d, device=device)
    covered = _launch.empty(n_clusters, dtype=wp.bool, device=device)
    uncovered = _launch.empty(n_clusters, dtype=wp.bool, device=device)
    source = _launch.empty(n_clusters, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_parametrization.cluster_jacobians,
        dim=n_clusters,
        inputs=[centroids, coarse_faces, coarse_vertex_faces, coarse_offsets, coarse_chart],
        outputs=[jacobians, covered, uncovered, source],
        device=device,
    )
    uncovered_clusters = od.array.flatnonzero(uncovered)
    if uncovered_clusters.size > 0:
        covered_clusters = od.array.flatnonzero(covered)
        nearest, _ = od.neighbors.query_nearest(
            od.array.gather(centroids_f32, covered_clusters),
            od.array.gather(centroids_f32, uncovered_clusters),
            k=1,
        )
        _launch.launch(
            kernel_parametrization.borrow_cluster_sources,
            dim=uncovered_clusters.size,
            inputs=[uncovered_clusters, covered_clusters, nearest],
            outputs=[source],
            device=device,
        )
    chart = odt.as_array2d(
        _launch.empty((2, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    _launch.launch(
        kernel_parametrization.prolong_chart,
        dim=n_vertices,
        inputs=[vertices, cluster, source, centroids, jacobians, coarse_chart],
        outputs=[chart],
        device=device,
    )
    return chart


def _smooth_chart(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    chart: odt.Array2dFloat64,
    passes: int,
    steps: int,
) -> None:
    """
    Run ARAP passes on ``chart`` in place, each global step cut short after ``steps`` CG steps.

    The conjugate gradients start from the current chart and are preconditioned by the stiffness
    diagonal; u and v share the matrix and are solved side by side, each with its own step
    lengths, which stay on the device. The stiffness is applied face by face from edge weights.
    """
    if passes == 0:
        return
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    vertex_faces, offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    gradients = odt.empty_2d((n_faces, 3), wp.vec2d, device=device)
    areas = _launch.empty(n_faces, dtype=wp.float64, device=device)
    edge_weights = _launch.empty(n_faces, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_parametrization.smoothing_face_terms,
        dim=n_faces,
        inputs=[vertices, faces],
        outputs=[gradients, areas, edge_weights],
        device=device,
    )
    width, blocks = kernel_reduce.spanned_blocks_1d(n_vertices, device)
    # Per pass: the preconditioned residual norm at the start and after each step, then each
    # step's curvature.
    slots = 2 * steps + 1
    partials, stride = _launch.block_partials(blocks, passes * slots, wp.vec2d, device)
    inverse_diagonal = _launch.empty(n_vertices, dtype=wp.float64, device=device)
    residual, direction, applied = (
        odt.as_array2d(_launch.empty((2, n_vertices), dtype=wp.float64, device=device), wp.float64)
        for _ in range(3)
    )
    width_arg, stride_arg = wp.int32(width), wp.int32(stride)
    no_mask, no_map = _no_pins(device)
    for pass_index in range(passes):
        base = pass_index * slots
        _launch.launch_tiled(
            kernel_parametrization.smoothing_start,
            dim=blocks,
            inputs=[faces, vertex_faces, offsets, gradients, areas, edge_weights, no_mask, no_map,
                    chart,
                    width_arg, wp.int32(base), stride_arg],
            outputs=[inverse_diagonal, residual, direction, partials],
            block_dim=TILE_1D,
            device=device,
        )  # fmt: skip
        for step in range(steps):
            norm_slot = wp.int32(base + step)
            curvature_slot = wp.int32(base + steps + 1 + step)
            _launch.launch_tiled(
                kernel_parametrization.smoothing_apply,
                dim=blocks,
                inputs=[faces, vertex_faces, offsets, edge_weights, direction, width_arg,
                        curvature_slot, stride_arg],
                outputs=[applied, partials],
                block_dim=TILE_1D,
                device=device,
            )  # fmt: skip
            _launch.launch_tiled(
                kernel_parametrization.smoothing_update,
                dim=blocks,
                inputs=[direction, applied, inverse_diagonal, width_arg, norm_slot,
                        curvature_slot, stride_arg],
                outputs=[chart, residual, partials],
                block_dim=TILE_1D,
                device=device,
            )  # fmt: skip
            # The last step's direction is never used: the next pass restarts from its residual.
            if step + 1 < steps:
                _launch.launch(
                    kernel_parametrization.smoothing_direction,
                    dim=n_vertices,
                    inputs=[residual, inverse_diagonal, partials, norm_slot],
                    outputs=[direction],
                    device=device,
                )


def _no_pins(device: wp.DeviceLike) -> tuple[wp.array[wp.bool], wp.array[wp.int32]]:
    """Return the empty pinned mask and free map of a local step that solves every vertex."""
    return (
        _launch.empty(0, dtype=wp.bool, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
    )


@overload
def lscm(
    mesh: Trimesh,
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
    /,
    *,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
@overload
def lscm(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
    /,
    *,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
def lscm(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    pinned_indices: wp.array[wp.int32] | wp.array[wp.vec2] | None = None,
    pinned_uv: wp.array[wp.vec2] | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]:
    """
    Constrained least-squares conformal map.

    Computes the conformal (angle-preserving) parametrization that minimizes the LSCM (Levy)
    conformal energy subject to a set of pinned vertices, by solving a single quadratic program
    over the stacked ``[u; v]`` vector of ``2 * n_vertices`` unknowns with the LSCM Hessian
    [`lscm_hessian`][ordito.energies.lscm_hessian] as the operator. Unlike
    [`harmonic`][ordito.parametrization.harmonic] / [`tutte`][ordito.parametrization.tutte] (which
    pin the whole boundary and solve two independent columns), LSCM couples ``u`` and ``v`` through
    the boundary vector-area term, so it needs only a few pins — typically **two** — to fix the
    remaining similarity-transform (rotation + scale + translation) degree of freedom.

    On a mesh of fewer than 4 000 vertices the interior system is solved with conjugate gradient,
    verified as [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] verifies it, with a
    sparse Cholesky factorization taking over a solve that does not converge within its budget;
    from 4 000 vertices it is factored at once, since with only a few pins its iteration grows
    faster with the mesh than the factorization does. Closed meshes are
    valid input: the boundary vector-area matrix is then zero and the Hessian reduces to
    ``-repdiag(L, 2)``. Given a `Trimesh`,
    a factorization is kept on it; one prepared with
    [`FixedVertexSolver.factor`][ordito.parametrization.FixedVertexSolver.factor] (``"lscm"``, the
    pinned indices) answers every later call pinning the same vertices with no iteration.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    pinned_indices
        ``(n_pinned,)`` indices of the pinned (constrained) vertices — igl's ``b``. At least two are
        required (unless the mesh has fewer than two vertices) to remove the conformal map's
        similarity-transform null space.
    pinned_uv
        ``(n_pinned,)`` target UV positions for ``pinned_indices``, in the same order (igl's
        ``bc``).
    solver
        ``"auto"`` (default) solves by conjugate gradient, verified, and factors the system only
        when the iteration's result fails the check -- or at once on a mesh of 4 000 vertices or
        more, where a factorization costs less than the iteration. ``"direct"`` factors it at once
        (a sparse Cholesky); given a `Trimesh`, the factorization is kept on it, so every later
        call fixing the same vertices -- whatever their values -- is a factored solve with no
        iteration. Pays from about the second such call on a smaller mesh.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``. Empty for an empty mesh.

    Raises
    ------
    ValueError
        If fewer than two vertices are pinned (and the mesh has at least two vertices), if
        ``pinned_indices`` and ``pinned_uv`` have different lengths, if ``pinned_indices`` repeats
        a vertex or holds one outside ``[0, n_vertices)``, or if ``solver`` is not ``"auto"`` or
        ``"direct"``.
    RuntimeError
        If ``vertices``, ``faces``, ``pinned_indices`` and ``pinned_uv`` are not all on one device.

    Warns
    -----
    UserWarning
        When no verified solution is reached (see
        [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]).

    See Also
    --------
    [`lscm_hessian`][ordito.energies.lscm_hessian]
    [`vector_area_matrix`][ordito.energies.vector_area_matrix]
    [`harmonic`][ordito.parametrization.harmonic]
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices]

    Notes
    -----
    The unknowns are stacked ``[u; v]`` (all ``u`` DOFs then all ``v`` DOFs), matching igl's
    ``lscm``: pin ``i`` fixes DOF ``i`` (``u``) and DOF ``i + n_vertices`` (``v``). The returned
    ``Q`` of ``igl.lscm`` equals ``-repdiag(L, 2) - 2 A`` exactly (see
    [`lscm_hessian`][ordito.energies.lscm_hessian]).

    Examples
    --------
    A flat grid with two opposite corners pinned where they are: the identity map is conformal, so
    LSCM recovers it:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(6, 6))
    pinned = wp.array([0, 35], dtype=wp.int32)
    pinned_uv = wp.array(vertices.numpy()[[0, 35], :2], dtype=wp.vec2)
    uv = od.parametrization.lscm(
        vertices=vertices, faces=faces, pinned_indices=pinned, pinned_uv=pinned_uv
    )
    print(uv.shape)  # (36,)
    print(np.allclose(uv.numpy(), vertices.numpy()[:, :2], atol=1e-5))  # True
    ```
    """
    bound, owned, arguments = mesh_arguments(
        "lscm", mesh, vertices, faces, (pinned_indices, pinned_uv), 2
    )
    pinned_indices = cast("wp.array[wp.int32]", arguments[0])
    pinned_uv = cast("wp.array[wp.vec2]", arguments[1])
    _require_solver_name(solver, "lscm")
    require_same_device(
        vertices=bound.vertices,
        faces=bound.faces,
        pinned_indices=pinned_indices,
        pinned_uv=pinned_uv,
    )
    device = bound.device
    n = bound.n_vertices
    if n == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device)

    n_pinned = pinned_indices.size
    _require_fixed_vertices(
        n_pinned,
        n,
        2,
        "lscm requires at least two pinned vertices to remove the conformal map's "
        f"similarity-transform null space; got {n_pinned}.",
    )
    fixed_mask, fixed_values = _pin_stacked(n, pinned_indices, pinned_uv, device)
    sol, free_map = _solve_fixed_vertices(
        bound,
        owned,
        "lscm",
        1,
        fixed_mask,
        odt.as_array2d(fixed_values, wp.float64),
        solver,
        2 * n_pinned,
    )
    out_uv = _launch.empty(n, dtype=wp.vec2, device=device)
    _launch.launch(
        kernel_parametrization.scatter_solution_stacked,
        dim=n,
        inputs=[fixed_mask, free_map, sol[0], fixed_values[0], out_uv],
        device=device,
    )
    return out_uv


_FIXED_VERTEX_METHODS = ("harmonic", "tutte", "lscm")
# ``harmonic(nonnegative_weights=True)``'s system: a ``FixedVertexSolver`` key of its own, so its
# factorization is never taken for the signed system's, but not a method the public menu offers.
_HARMONIC_NONNEGATIVE = "harmonic_nonnegative"
_FIXED_VERTEX_SYSTEMS = (*_FIXED_VERTEX_METHODS, _HARMONIC_NONNEGATIVE)


class FixedVertexSolver:
    """
    Factorizations of the fixed-vertex maps' systems on one mesh, kept by the mesh.

    What [`Trimesh.fixed_vertex_solver`][ordito.mesh.Trimesh.fixed_vertex_solver] returns, and what
    [`harmonic`][ordito.parametrization.harmonic], [`tutte`][ordito.parametrization.tutte] and
    [`lscm`][ordito.parametrization.lscm] consult when given a `Trimesh`. Each map solves the
    system its operator reduces to once the fixed vertices are eliminated, and that system depends
    on *which* vertices are fixed -- not on the values they are fixed to -- so one factorization
    per ``(method, k)`` serves every later call that fixes the same vertices: such a call runs no
    iteration, only the factorization's solve.

    A factorization is kept here when [`factor`][ordito.parametrization.FixedVertexSolver.factor]
    builds it ahead of the calls, when a call asks for one (``solver="direct"``), or when a call
    built one anyway: the iteration's result failed
    its verification (see [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]), or the
    system is factored from the start -- at ``k >= 3``, for a ``k == 2`` ``harmonic`` / ``tutte``
    system or an ``lscm`` one on a mesh of 4 000 vertices or more, and on the CPU device for a
    ``k == 1`` one on a mesh of 5 000 or more. A call fixing other
    vertices than the kept factorization's solves as if none were kept, and replaces it with its
    own when it builds one. A system factored from the start that cannot be factored (over
    [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET], or not definite) is
    remembered as such until [`release`][ordito.parametrization.FixedVertexSolver.release], and
    later ``"auto"`` calls iterate without attempting it again.

    Notes
    -----
    A kept factorization holds device memory of the order of its system's fill-in for as long as
    this solver lives -- the mesh's lifetime -- reported by
    [`nbytes`][ordito.parametrization.FixedVertexSolver.nbytes] and returned by
    [`release`][ordito.parametrization.FixedVertexSolver.release] or
    [`Trimesh.release_factorizations`][ordito.mesh.Trimesh.release_factorizations]. The solver
    refers to its mesh weakly and is unusable once the mesh is gone.

    See Also
    --------
    [`Trimesh.fixed_vertex_solver`][ordito.mesh.Trimesh.fixed_vertex_solver]
    [`MinQuadWithFixedData`][ordito.linalg.MinQuadWithFixedData]

    Examples
    --------
    Factor once for a boundary, then every harmonic map of that mesh fixing it is a direct solve:

    ```python
    import ordito as od

    vertices, faces = od.creation.sphere_cap(angle=1.0, subdivisions=3)
    mesh = od.mesh.Trimesh(vertices, faces)
    boundary = od.boundary.longest_boundary_loop(vertices, faces)
    solver = mesh.fixed_vertex_solver()
    print(solver.factor("harmonic", boundary))  # True
    circle = od.parametrization.map_vertices_to_circle(vertices, boundary)
    uv = od.parametrization.harmonic(mesh, boundary_indices=boundary, boundary_uv=circle)
    print(uv.shape, solver.nbytes > 0)  # (217,) True
    solver.release()
    print(solver.nbytes)  # 0
    ```
    """

    def __init__(self, mesh: Trimesh) -> None:
        """Bind a solver to ``mesh``; nothing is assembled until a factorization is asked for."""
        self._mesh = weakref.ref(mesh)
        self._systems: dict[tuple[str, int], twl.MinQuadWithFixedData] = {}
        # The ``(method, k)`` whose system ``"auto"`` factored at once (``_factors_at_once``) and
        # the factorization refused -- over the memory budget, or not definite. Its later calls
        # iterate without paying for the refused attempt's analysis again; ``release`` forgets it.
        self._refused: set[tuple[str, int]] = set()

    @property
    def nbytes(self) -> int:
        """Device memory this solver's kept factorizations hold, in bytes."""
        return sum(system.nbytes for system in self._systems.values())

    def release(self) -> None:
        """Drop every kept factorization, and forget which systems could not be factored."""
        for system in self._systems.values():
            system.release()
        self._systems.clear()
        self._refused.clear()

    def factor(self, method: str, fixed_indices: wp.array[wp.int32], *, k: int = 1) -> bool:
        """
        Factor ``method``'s system with ``fixed_indices`` fixed, for every later call fixing them.

        Parameters
        ----------
        method
            ``"harmonic"``, ``"tutte"`` or ``"lscm"``: the map the factorization serves.
        fixed_indices
            ``(n_fixed,)`` the vertices the calls fix: ``boundary_indices`` of
            [`harmonic`][ordito.parametrization.harmonic] / [`tutte`][ordito.parametrization.tutte],
            ``pinned_indices`` of [`lscm`][ordito.parametrization.lscm]. Only the set matters,
            not the order.
        k
            The harmonic power the calls pass (``1`` for ``"lscm"``).

        Returns
        -------
        bool
            Whether the calls will run no iteration: ``False`` when the system cannot be factored
            (over [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET], or not
            definite in ``float64``), and they keep iterating.

        Raises
        ------
        ValueError
            If ``method`` is not one of the three, if ``k < 1``, if ``k != 1`` for ``"lscm"``, or
            if ``fixed_indices`` repeats a vertex or holds one outside ``[0, n_vertices)``.
        RuntimeError
            If ``fixed_indices`` is not on the mesh's device.

        Examples
        --------
        ```python
        import ordito as od

        vertices, faces = od.creation.sphere_cap(angle=1.0, subdivisions=3)
        mesh = od.mesh.Trimesh(vertices, faces)
        boundary = od.boundary.longest_boundary_loop(vertices, faces)
        print(mesh.fixed_vertex_solver().factor("tutte", boundary, k=2))  # True
        ```
        """
        _require_fixed_vertex_method(method, k, "FixedVertexSolver.factor")
        mesh = self._bound_mesh()
        require_same_device(vertices=mesh.vertices, fixed_indices=fixed_indices)
        n_vertices = mesh.n_vertices
        if n_vertices == 0:
            return True
        n_fixed = fixed_indices.size
        # The values are scattered into a buffer the factorization never reads.
        no_uv = _launch.empty(n_fixed, dtype=wp.vec2, device=mesh.device)
        if method == "lscm":
            fixed_mask, _ = _pin_stacked(n_vertices, fixed_indices, no_uv, mesh.device)
            n_fixed *= 2
        else:
            fixed_mask, _ = _scatter_constraints(n_vertices, fixed_indices, no_uv, mesh.device)
        system, _ = _fixed_vertex_system(mesh, method, k, fixed_mask, n_fixed, factor=True)
        if system.factored:
            self._keep(method, k, system)
        return system.factored or system.n_free == 0

    def solve(
        self,
        method: str,
        fixed_mask: wp.array[wp.bool],
        fixed_values: odt.Array2dFloat64,
        *,
        k: int = 1,
        solver: Literal["auto", "direct"] = "auto",
    ) -> tuple[odt.Array2dFloat, wp.array[wp.int32]]:
        """
        Solve ``method``'s reduced system: by the kept factorization if it fixes ``fixed_mask``.

        Otherwise the system is assembled and solved verified, as
        [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] solves it, and a factorization
        that solve builds is kept here, replacing the one ``(method, k)`` held.

        Parameters
        ----------
        method
            ``"harmonic"``, ``"tutte"`` or ``"lscm"``.
        fixed_mask
            ``(n_dofs,)`` the fixed degrees of freedom: ``(n_vertices,)`` for ``harmonic`` /
            ``tutte``, ``(2 * n_vertices,)`` (``u`` then ``v``) for ``lscm``.
        fixed_values
            ``(n_columns, n_dofs)`` prescribed values; only the fixed entries are read.
        k
            The harmonic power (``1`` for ``"lscm"``).
        solver
            ``"auto"`` iterates, verified, and factors only when the check fails -- or at once at
            ``k >= 3``, for ``k == 2`` and ``"lscm"`` on a mesh of 4 000 vertices or more, and on
            the CPU device for ``k == 1`` on one of 5 000 or more, unless that system was already
            refused; ``"direct"`` factors at once. A factorization
            either builds is kept.

        Returns
        -------
        solution : odt.Array2dFloat
            ``(n_columns, n_free)`` values of the free degrees of freedom.
        free_map : wp.array[wp.int32]
            ``(n_dofs,)`` compact index of each free degree of freedom.

        Raises
        ------
        ValueError
            If ``method`` is not one of the three, if ``k < 1``, if ``k != 1`` for ``"lscm"``, or
            if ``solver`` is not ``"auto"`` or ``"direct"``.
        RuntimeError
            If ``fixed_mask`` and ``fixed_values`` are not on the mesh's device.

        Warns
        -----
        UserWarning
            When no verified solution is reached (see
            [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]).

        Examples
        --------
        The harmonic system with the boundary of a flat grid fixed to its ``x`` coordinates: the
        free values are the interior ``x`` coordinates:

        ```python
        import numpy as np
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.grid(count=(5, 5))
        mesh = od.mesh.Trimesh(vertices, faces)
        boundary = od.boundary.longest_boundary_loop(vertices, faces).numpy()
        mask = np.zeros(vertices.shape[0], dtype=bool)
        mask[boundary] = True
        x = vertices.numpy()[:, 0].astype(np.float64)
        solution, free_map = mesh.fixed_vertex_solver().solve(
            "harmonic", wp.array(mask, dtype=wp.bool), wp.array(x[None, :], dtype=wp.float64)
        )
        print(solution.shape)  # (1, 9)
        print(np.allclose(solution.numpy()[0], x[~mask]))  # True
        ```
        """
        _require_fixed_vertex_method(method, k, "FixedVertexSolver.solve")
        return self._solve(method, fixed_mask, fixed_values, k, solver, None)

    def _solve(
        self,
        method: str,
        fixed_mask: wp.array[wp.bool],
        fixed_values: odt.Array2dFloat64,
        k: int,
        solver: Literal["auto", "direct"],
        n_fixed: int | None,
    ) -> tuple[odt.Array2dFloat, wp.array[wp.int32]]:
        """
        ``solve``, confirming the mask fixes ``n_fixed`` degrees of freedom when that is given.

        The maps pass the count their indices fix when distinct and in range: a repeated or
        out-of-range index fixes fewer, which the system's free count -- read when it is built --
        shows before anything is factored or solved.
        """
        _require_fixed_vertex_method(method, k, "FixedVertexSolver.solve", _FIXED_VERTEX_SYSTEMS)
        _require_solver_name(solver, "FixedVertexSolver.solve")
        mesh = self._bound_mesh()
        require_same_device(
            vertices=mesh.vertices, fixed_mask=fixed_mask, fixed_values=fixed_values
        )
        system = self._matching(method, k, fixed_mask)
        preconditioner: str | twl.SquaredLaplacianPreconditioner = "adaptive"
        built = system is None
        if system is None:
            at_once = _factors_at_once(method, k, mesh.vertices.size, mesh.device)
            refused = at_once and solver == "auto" and (method, k) in self._refused
            system, preconditioner = _fixed_vertex_system(
                mesh,
                method,
                k,
                fixed_mask,
                n_fixed,
                factor=solver == "direct" or (at_once and not refused),
            )
            if at_once and not refused and not system.factored and system.n_free > 0:
                self._refused.add((method, k))
                _LOGGER.debug("%s: the k=%d system cannot be factored; it iterates", method, k)
        else:
            _require_fixed_count(system.fixed_mask.size - system.n_free, n_fixed, method)
        solution, free_map, _ = system.solve(
            fixed_values, tol=_CG_TOLERANCE, preconditioner=preconditioner
        )
        if built and system.factored:
            self._keep(method, k, system)
            _LOGGER.debug("%s: kept the factorization of its k=%d system", method, k)
        return solution, free_map

    def _matching(
        self, method: str, k: int, fixed_mask: wp.array[wp.bool]
    ) -> twl.MinQuadWithFixedData | None:
        """Return the kept system of ``(method, k)`` when it fixes exactly ``fixed_mask``."""
        system = self._systems.get((method, k))
        if system is not None and system.matches(fixed_mask):
            return system
        return None

    def _keep(self, method: str, k: int, system: twl.MinQuadWithFixedData) -> None:
        """Keep ``system``'s factorization for ``(method, k)``, releasing the one it replaces."""
        previous = self._systems.get((method, k))
        if previous is not None and previous is not system:
            previous.release()
        self._systems[(method, k)] = system

    def _bound_mesh(self) -> Trimesh:
        """Return the mesh this solver belongs to."""
        mesh = self._mesh()
        if mesh is None:
            raise RuntimeError("FixedVertexSolver: its Trimesh no longer exists")
        return mesh


def _solve_fixed_vertices(
    mesh: Trimesh,
    owned: bool,
    method: str,
    k: int,
    fixed_mask: wp.array[wp.bool],
    fixed_values: odt.Array2dFloat64,
    solver: str,
    n_fixed: int,
) -> tuple[odt.Array2dFloat, wp.array[wp.int32]]:
    """
    Solve a fixed-vertex map's reduced system on the mesh's solver, or once for the call.

    ``n_fixed`` is the degrees of freedom the caller's indices fix when they are distinct and in
    range, which the system's free count confirms. A caller-owned mesh keeps whatever factorization
    the solve builds; a ``vertices, faces`` call drops it, saying so at ``INFO``.
    """
    solver_name = cast("Literal['auto', 'direct']", solver)
    if owned:
        return mesh.fixed_vertex_solver()._solve(  # pyright: ignore[reportPrivateUsage]
            method, fixed_mask, fixed_values, k, solver_name, n_fixed
        )
    return _solve_fixed_vertices_once(
        mesh, method, k, fixed_mask, fixed_values, solver_name, n_fixed
    )


def _solve_fixed_vertices_once(
    mesh: Trimesh,
    method: str,
    k: int,
    fixed_mask: wp.array[wp.bool],
    fixed_values: odt.Array2dFloat64,
    solver: Literal["auto", "direct"],
    n_fixed: int,
) -> tuple[odt.Array2dFloat, wp.array[wp.int32]]:
    """
    ``FixedVertexSolver.solve`` on a solver made for the call, which nothing keeps.

    The same system, factorization rule and solve as a fresh solver's, but with the right-hand side
    formed by the extraction's own pass, as
    [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] forms it: nothing outlives the call,
    so the system is never held for a later solve to re-form its right-hand side.
    """
    _require_fixed_vertex_method(method, k, "FixedVertexSolver.solve", _FIXED_VERTEX_SYSTEMS)
    _require_solver_name(solver, "FixedVertexSolver.solve")
    vertices = mesh.vertices
    require_same_device(vertices=vertices, fixed_mask=fixed_mask, fixed_values=fixed_values)
    device = fixed_mask.device
    at_once = _factors_at_once(method, k, vertices.size, vertices.device)
    q, laplacian, mass_diag = _fixed_vertex_operator(mesh, method, k)
    q_uu, rhs, free_map, n_free = twl.partitioned_interior_system(q, fixed_mask, fixed_values)
    _require_fixed_count(fixed_mask.size - n_free, n_fixed, method)
    n_rhs = int(fixed_values.shape[0])
    solution = odt.as_array2d(
        _launch.zeros((n_rhs, n_free), dtype=wp.float64, device=device), wp.float64
    )
    if n_free == 0 or n_rhs == 0:
        return solution, free_map
    held = twl.OperatorFactorization(q_uu, vertices, fixed_mask=fixed_mask, free_map=free_map)
    if solver == "direct" or at_once:
        held.factor()
        if at_once and held.factorization is None:
            _LOGGER.debug("%s: the k=%d system cannot be factored; it iterates", method, k)
    if held.factorization is not None:
        held.solve(rhs, solution, tol=_CG_TOLERANCE)
    else:
        preconditioner = _fixed_vertex_preconditioner(
            k, laplacian, mass_diag, fixed_mask, free_map, n_free, factored=False
        )
        twl.solve_spd_columns(
            q_uu,
            rhs,
            solution,
            tol=_CG_TOLERANCE,
            preconditioner=preconditioner,
            factorization=held,
        )
    if held.nbytes > 0:
        _LOGGER.info(
            "%s: built a sparse Cholesky factorization (%d bytes) and is discarding it; pass an "
            "ordito.mesh.Trimesh to keep it for later calls fixing the same vertices",
            method,
            held.nbytes,
        )
        held.release()
    return solution, free_map


def _require_solver_name(solver: str, caller: str) -> None:
    """Validate the fixed-vertex maps' ``solver`` menu argument."""
    if solver not in ("auto", "direct"):
        raise ValueError(f'{caller}: solver must be "auto" or "direct", got {solver!r}.')


def _require_fixed_vertex_method(
    method: str, k: int, caller: str, methods: tuple[str, ...] = _FIXED_VERTEX_METHODS
) -> None:
    """Validate a ``FixedVertexSolver`` menu argument (one of ``methods``) and its power."""
    if method not in methods:
        raise ValueError(f"{caller}: method must be one of {list(methods)}, got {method!r}.")
    if k < 1 or (method == "lscm" and k != 1):
        raise ValueError(f"{caller}: k={k} is not a power {method} takes.")


def _factors_at_once(method: str, k: int, n_vertices: int, device: wp.DeviceLike) -> bool:
    """Whether ``"auto"`` factors ``method``'s system before iterating (see ``solve``)."""
    if method == "lscm":
        return n_vertices >= _FACTOR_FROM_VERTICES
    if k == 1:
        return n_vertices >= _FACTOR_K1_FROM_VERTICES_CPU and not wp.get_device(device).is_cuda
    return k >= 3 or n_vertices >= _FACTOR_FROM_VERTICES


def _fixed_vertex_system(
    mesh: Trimesh,
    method: str,
    k: int,
    fixed_mask: wp.array[wp.bool],
    n_fixed: int | None,
    *,
    factor: bool,
) -> tuple[twl.MinQuadWithFixedData, str | twl.SquaredLaplacianPreconditioner]:
    """
    Assemble ``method``'s reduced system over ``fixed_mask``'s free set, and its preconditioner.

    The mask's fixed count is checked against ``n_fixed`` (``_require_fixed_count``) after the
    extraction, which reads the free count anyway, and before the factorization.
    """
    q, laplacian, mass_diag = _fixed_vertex_operator(mesh, method, k)
    system = _checked_system(q, fixed_mask, mesh.vertices, n_fixed, method, factor)
    preconditioner = _fixed_vertex_preconditioner(
        k, laplacian, mass_diag, fixed_mask, system.free_map, system.n_free, system.factored
    )
    return system, preconditioner


def _fixed_vertex_operator(
    mesh: Trimesh, method: str, k: int
) -> tuple[
    odt.BsrMatrix[wp.float64], odt.BsrMatrix[wp.float64] | None, wp.array[wp.float64] | None
]:
    """
    Assemble ``method``'s ``float64`` operator ``Q``, with what its preconditioner needs.

    ``harmonic`` / ``tutte`` minimize ``Q = -L`` at ``k == 1`` and ``Q = (-L) (M^-1 (-L))^(k-1)``
    above ([`k_harmonic`][ordito.energies.k_harmonic]; ``M`` the barycentric lumped mass, the
    identity for ``tutte``), ``lscm`` its Hessian; the operator is ``float64``, since ``k > 1``
    raises the Laplacian's condition number to the ``k``-th power. The Laplacian ``L`` and the mass
    diagonal (``None`` for ``tutte``) come back above ``k == 1``, and ``None`` at it.
    """
    vertices, faces = mesh.vertices, mesh.faces
    # The mesh's cached cotangent weights and sparsity serve the float64 operators too (both are
    # dtype-free), so a second system over one mesh pays only for its values. The clamped weights
    # of ``harmonic(nonnegative_weights=True)`` are not cached: its factorization is.
    nonnegative = method == _HARMONIC_NONNEGATIVE
    if nonnegative:
        method = "harmonic"
    if method == "lscm":
        q = od.energies.lscm_hessian(
            vertices, faces, cot_entries=mesh.cotmatrix_entries, pattern=mesh.mesh_operator_pattern
        )
        return q, None, None
    if method == "harmonic":
        cot_entries = (
            od.laplacian.cotmatrix_entries(vertices, faces, nonnegative_weights=True)
            if nonnegative
            else mesh.cotmatrix_entries
        )
        if k == 1 and faces.size > 0:
            # ``-L`` written by one row kernel over the mesh's pattern, so ``L`` is never formed.
            _, _, offsets, columns, run_start, keys, count, order = mesh.mesh_operator_pattern
            negated_values = _launch.empty(columns.size, dtype=wp.float64, device=vertices.device)
            _launch.launch(
                kernel_parametrization.negated_cotmatrix_rows,
                dim=vertices.size,
                inputs=[offsets, columns, run_start, keys, wp.int32(count), order, cot_entries],
                outputs=[negated_values],
                device=vertices.device,
            )
            q = od.array.bsr_from_csr(
                vertices.size, vertices.size, offsets, columns, negated_values
            )
            return q, None, None
        laplacian = cotmatrix(
            vertices, faces, cot_entries, dtype=wp.float64, pattern=mesh.mesh_operator_pattern
        )
        mass_diag = (
            mass_matrix_entries(vertices, faces, dtype=wp.float64, face_areas=mesh.face_areas)
            if k > 1
            else None
        )
    else:
        laplacian = graph_laplacian(vertices, faces, dtype=wp.float64)
        mass_diag = None
    if k == 1:
        # ``k_harmonic``'s ``-L`` (``tutte``'s, or a mesh with no faces), one negating map, over
        # ``L``'s own offsets and columns: ``L`` is built for this system alone, so they are shared
        # rather than copied.
        negated_values = _launch.empty_like(laplacian.values)
        _launch.map(wp.neg, laplacian.values, out=negated_values)
        q = od.array.bsr_from_csr(
            int(laplacian.nrow),
            int(laplacian.ncol),
            laplacian.offsets,
            laplacian.columns,
            negated_values,
        )
        return q, None, None
    return od.energies.k_harmonic(laplacian, mass_diag, k=k), laplacian, mass_diag


def _fixed_vertex_preconditioner(
    k: int,
    laplacian: odt.BsrMatrix[wp.float64] | None,
    mass_diag: wp.array[wp.float64] | None,
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    n_free: int,
    factored: bool,
) -> str | twl.SquaredLaplacianPreconditioner:
    """
    Return the preconditioner a fixed-vertex system's iteration runs under when not factored.

    The Jacobi-Chebyshev polynomial for a Laplacian (``k == 1``) or ``lscm`` -- with only the
    boundary or two vertices fixed their solves are long -- and for ``k == 2`` the square of the
    Laplacian's polynomial
    ([`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner]): the
    free block of ``L M^-1 L`` is spectrally close to ``L_ff D^-2 L_ff`` with ``D = sqrt(M_f)``,
    the terms it drops being the one ring of the fixed boundary. ``k >= 3`` is factored from the
    start: conjugate gradient needs thousands of rounds there and still leaves an error of 1e-5 to
    1e-2 of the range, so the multigrid ``"auto"`` route remains only for a system the
    factorization refuses.
    """
    if k == 1:
        return "chebyshev"
    if k >= 3 or factored or n_free == 0:
        return "auto"
    assert laplacian is not None
    device = fixed_mask.device
    n_vertices = fixed_mask.size
    no_values = odt.as_array2d(
        _launch.empty((0, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    # ``-L``'s free block, negated as it is extracted.
    l_ff, _ = twl.assemble_interior_system(
        laplacian, fixed_mask, free_map, no_values, n_free, scale=-1.0
    )
    if mass_diag is None:
        roots = _launch.full(n_free, 1.0, dtype=wp.float64, device=device)
    else:
        roots = _launch.empty(n_free, dtype=wp.float64, device=device)
        _launch.launch(
            kernel_parametrization.free_mass_roots,
            dim=n_vertices,
            inputs=[fixed_mask, free_map, mass_diag, roots],
            device=device,
        )
    return twl.squared_laplacian_preconditioner(l_ff, roots)


def _checked_system(
    q: odt.BsrMatrix[wp.float64],
    fixed_mask: wp.array[wp.bool],
    vertices: wp.array[wp.vec3],
    n_fixed: int | None,
    method: str,
    factor: bool,
) -> twl.MinQuadWithFixedData:
    """Build ``q``'s reduced system over ``fixed_mask``, check its fixed count, then factor it."""
    system = twl.MinQuadWithFixedData(q, fixed_mask, vertices, factor=False)
    _require_fixed_count(system.fixed_mask.size - system.n_free, n_fixed, method)
    if factor and system.factorization is not None:
        system.factorization.factor()
    return system


def _require_fixed_count(fixed: int, n_fixed: int | None, method: str) -> None:
    """
    Raise ``ValueError`` unless a system fixes the ``n_fixed`` degrees of freedom asked for.

    The scatter kernels mark one degree of freedom (two for ``lscm``) per distinct in-range index
    and skip an out-of-range one, so a repeated or out-of-range index leaves fewer fixed than the
    caller's count. ``fixed`` is the mask's size less the free count the extraction read: no read
    of its own.
    """
    if n_fixed is None:
        return
    if fixed != n_fixed:
        raise ValueError(
            f"{method}: the fixed vertex indices must be distinct and in [0, n_vertices); they fix "
            f"{fixed} degrees of freedom where {n_fixed} were given"
        )


def _pin_stacked(
    n_vertices: int, indices: wp.array[wp.int32], uv: wp.array[wp.vec2], device: wp.DeviceLike
) -> tuple[wp.array[wp.bool], wp.array[wp.float64]]:
    """
    ``lscm``'s pins over the stacked ``[u; v]`` unknowns: pin ``i`` fixes DOF ``i`` and ``i + n``.

    Only the fixed entries of the values are written, as ``_scatter_constraints`` writes them.

    Raises
    ------
    ValueError
        If ``indices`` and ``uv`` have different lengths -- the scatter kernel indexes ``uv`` at
        every position up to ``indices.shape[0]``, so a shorter ``uv`` is an out-of-bounds read.
    """
    n_pinned = indices.size
    if uv.size != n_pinned:
        raise ValueError(
            f"pinned_indices and pinned_uv must have the same length, got {n_pinned} and {uv.size}."
        )
    fixed_mask = _launch.zeros(2 * n_vertices, dtype=wp.bool, device=device)
    fixed_values = _launch.empty((1, 2 * n_vertices), dtype=wp.float64, device=device)
    if n_pinned > 0:
        _launch.launch(
            kernel_parametrization.scatter_pinned_stacked,
            dim=n_pinned,
            inputs=[indices, uv, wp.int32(n_vertices), fixed_mask, fixed_values],
            device=device,
        )
    return fixed_mask, fixed_values


def _require_fixed_vertices(n_fixed: int, n_vertices: int, min_required: int, message: str) -> None:
    """
    Raise ``ValueError(message)`` unless at least ``min_required`` vertices are fixed.

    Waived when the mesh has fewer than ``min_required`` vertices, since it is then impossible to
    pin that many distinct ones. The shared arithmetic behind three otherwise-differently-worded
    guards: harmonic / tutte's "at least one fixed boundary vertex" (``min_required=1``, via
    ``_fixed_boundary_map``), arap's "at least one
    fixed vertex" (``min_required=1``) and lscm's "at least two pinned vertices"
    (``min_required=2``). Each caller keeps its own message, since the *reason* the count is
    required differs (a singular Dirichlet system, translation invariance, a similarity-transform
    null space) even though the arithmetic is identical.
    """
    if n_vertices >= min_required and n_fixed < min_required:
        raise ValueError(message)
