"""Ray and point-in-mesh queries on ``wp.Mesh``."""

from __future__ import annotations

import math

import warp as wp

from ordito import _launch
from ordito._common import offsets_buffer, read_scalar, require_same_device, scan_count
from ordito._solid import FLOAT32_EPSILON, positions_box, solid_from_mesh
from ordito.config import is_deterministic
from ordito.constants import TOLERANCE_PLANAR
from ordito.halfedge import halfedge_twins
from ordito.kernels import proximity as kernel_proximity
from ordito.kernels import ray as kernel_ray
from ordito.kernels.algorithms import solid as kernel_solid


def intersects_location(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Return world-space locations where rays hit the mesh surface (first hit per ray).

    Uses ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are unitized before
    querying. Returns only rays that hit within ``max_t`` as sparse ``(m,)`` arrays,
    carrying the same set of hits as the dense output of
    [`intersects_first`][ordito.ray.intersects_first].

    With [`is_deterministic`][ordito.config.is_deterministic] off (the default) the returned
    arrays are prefix views of buffers sized for every ray, so they keep that storage alive for
    as long as they are held, and **row order is unspecified**: each hitting ray claims its
    output slot from an atomic counter, so rows arrive in the order the rays completed rather than
    in ray order, and two runs over the same input may order them differently. With it on, the
    rows come back in ascending ray order, from a scan of the hit flags.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, the search is unbounded, which finds the same first hit as any bound at least
        as long as the diagonal of the box enclosing the mesh and the ray origins.

    Returns
    -------
    locations
        ``(m,)`` intersection points.
    index_ray
        ``(m,)`` index of the ray that produced each hit.
    index_tri
        ``(m,)`` face indices for each hit.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.

    Examples
    --------
    Two rays along ``+x`` towards a unit cube: the first enters through the ``x = -0.5`` face, the
    second passes above the cube and is left out of the result:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[-2.0, 0.1, 0.2], [-2.0, 3.0, 0.0]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    locations, index_ray, index_tri = od.ray.intersects_location(mesh, origins, directions)
    print(index_ray.numpy().tolist())  # [0]
    print(locations.numpy()[0].round(3))  # [-0.5  0.1  0.2]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    device = ray_origins.device
    if n == 0:
        empty_int = _launch.empty(0, dtype=wp.int32, device=device)
        return _launch.empty(0, dtype=wp.vec3, device=device), empty_int, empty_int

    if max_t is None:
        max_t = math.inf

    if is_deterministic():
        return _intersects_location_ordered(mesh, ray_origins, ray_directions, max_t)
    # Both index buffers and, behind them, the hit counter: one allocation, the counter zeroed.
    indices = _launch.empty(2 * n + 1, dtype=wp.int32, device=device)
    index_ray = _launch.view(indices, 0, n)
    index_tri = _launch.view(indices, n, 2 * n)
    counter = _launch.view(indices, 2 * n, 2 * n + 1)
    _launch.zero_(counter)
    locations = _launch.empty(n, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_ray.first_hit_append,
        dim=n,
        inputs=[
            wp.uint64(mesh.id),
            ray_origins,
            ray_directions,
            wp.float32(max_t),
            index_ray,
            index_tri,
            locations,
            counter,
        ],
        device=device,
    )
    # Prefix views of the per-ray buffers rather than exact-size copies: three allocations and
    # three copies fewer, at the price of the returned arrays keeping ``n``-long storage alive.
    n_hits = int(read_scalar(counter, 0))
    if n_hits == 0:
        empty_int = _launch.empty(0, dtype=wp.int32, device=device)
        return _launch.empty(0, dtype=wp.vec3, device=device), empty_int, empty_int
    return (
        _launch.view(locations, 0, n_hits),
        _launch.view(indices, 0, n_hits),
        _launch.view(indices, n, n + n_hits),
    )


def _intersects_location_ordered(
    mesh: wp.Mesh, ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3], max_t: float
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    [`intersects_location`][ordito.ray.intersects_location]'s hits in ascending ray order.

    The deterministic setting's form: every ray's hit is stored at its own index, the hit flags
    are scanned in place and the hits compacted at their rank, so the order does not follow which
    ray's thread finished first.
    """
    device = ray_origins.device
    n = ray_origins.size
    faces, inclusive, hit_locations = _launch.carve_typed(
        ((wp.int32, n), (wp.int32, n), (wp.vec3, n)), device
    )
    _launch.launch(
        kernel_ray.first_hit_flags,
        dim=n,
        inputs=[wp.uint64(mesh.id), ray_origins, ray_directions, wp.float32(max_t)],
        outputs=[faces, hit_locations, inclusive],
        device=device,
    )
    # The hit count sizes the outputs.
    n_hits = scan_count(inclusive)
    locations = _launch.empty(n_hits, dtype=wp.vec3, device=device)
    index_ray = _launch.empty(n_hits, dtype=wp.int32, device=device)
    index_tri = _launch.empty(n_hits, dtype=wp.int32, device=device)
    if n_hits > 0:
        _launch.launch(
            kernel_ray.compact_scanned_hits,
            dim=n,
            inputs=[faces, hit_locations, inclusive],
            outputs=[index_ray, index_tri, locations],
            device=device,
        )
    return locations, index_ray, index_tri


def intersects_first(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> wp.array[wp.int32]:
    """
    Find the index of the first triangle each ray hits.

    Uses ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are unitized before
    querying. The search along each ray is unbounded unless ``max_t`` is given.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, the search is unbounded, which finds the same first hit as any bound at least
        as long as the diagonal of the box enclosing the mesh and the ray origins.

    Returns
    -------
    wp.array[wp.int32]
        ``(n,)`` face indices; ``-1`` when a ray misses within ``max_t``.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.

    Examples
    --------
    The same two rays as in [`intersects_location`][ordito.ray.intersects_location], with one row
    per ray and ``-1`` for the miss:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[-2.0, 0.1, 0.2], [-2.0, 3.0, 0.0]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.ray.intersects_first(mesh, origins, directions).numpy().tolist())  # [0, -1]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    if n == 0:
        return _launch.empty(0, dtype=wp.int32, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_triangle_index = _launch.empty(n, dtype=wp.int32, device=ray_origins.device)
    # ``first_hit`` returns ``(face, location)`` and ``wp.map`` wants one ``out=`` per returned
    # value, so the location is written and dropped. Kept rather than given a face-only
    # ``@wp.func`` of its own: that would generate a second ``map_*`` module to save one
    # allocation and an (n,) vec3 store, well under a percent of a call this size. Not carved
    # with the output: the returned faces would keep the scratch alive, four times their bytes.
    locations_scratch = _launch.empty(n, dtype=wp.vec3, device=ray_origins.device)
    _launch.map(
        kernel_ray.first_hit,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        out=[out_triangle_index, locations_scratch],
    )
    return out_triangle_index


def first_hits(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
    both_ways: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.int32], wp.array[wp.vec3]]:
    """
    Every ray's first hit: its location, distance, triangle and barycentric coordinates.

    One row per ray, in ray order: the dense form of
    [`intersects_location`][ordito.ray.intersects_location], carrying what
    [`intersects_first`][ordito.ray.intersects_first] finds plus where along the ray and where in
    the triangle the hit lies. Uses ``wp.mesh_query_ray`` on the mesh BVH; ray directions are
    unitized before querying, so a distance is a length in the mesh's units.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
    max_t
        Optional maximum distance along each ray, in each direction cast. When ``None``, the
        search is unbounded, which finds the same first hit as any bound at least as long as the
        diagonal of the box enclosing the mesh and the ray origins.
    both_ways
        When ``True``, also cast each ray backwards from its origin and keep whichever hit is
        nearer, the forward one on a tie: the surface point nearest the origin along the line.

    Returns
    -------
    locations
        ``(n,)`` hit positions, ``origin + distance * unit_direction``; the origin itself where
        the ray misses.
    distances
        ``(n,)`` distance from the origin to the hit; negative for a backward hit (only with
        ``both_ways=True``), ``inf`` where the ray misses.
    triangle_id
        ``(n,)`` index of the triangle hit, or ``-1`` where the ray misses.
    barycentric
        ``(n,)`` weights of each hit on its triangle's three corners, in the order ``mesh``'s
        indices list them, so the hit is their weighted sum; ``(0, 0, 0)`` on a miss.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.

    See Also
    --------
    [`intersects_first`][ordito.ray.intersects_first]
        The triangle alone.
    [`intersects_location`][ordito.ray.intersects_location]
        The hits alone, compacted.
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
        The nearest surface point in any direction.

    Examples
    --------
    From inside a unit cube along ``+x``, a ray reaches the face at ``x = 0.5``; cast both ways
    from nearer the ``x = -0.5`` face, the backward hit wins with a negative distance. A ray that
    passes the cube misses:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[0.0, 0.1, 0.2], [-0.3, 0.1, 0.2], [0.0, 3.0, 0.0]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    locations, distances, triangle_ids, weights = od.ray.first_hits(
        mesh, origins, directions, both_ways=True
    )
    print([round(float(d), 3) for d in distances.numpy()])  # [0.5, -0.2, inf]
    print(locations.numpy()[1].round(3))  # [-0.5  0.1  0.2]
    print(triangle_ids.numpy()[2], weights.numpy()[2].tolist())  # -1 [0.0, 0.0, 0.0]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    device = ray_origins.device
    # One allocation each rather than one carved buffer: a caller keeping some of the outputs can
    # drop the others and free their memory, which a view of a shared buffer would hold alive.
    out_locations = _launch.empty(n, dtype=wp.vec3, device=device)
    out_distances = _launch.empty(n, dtype=wp.float32, device=device)
    out_faces = _launch.empty(n, dtype=wp.int32, device=device)
    out_weights = _launch.empty(n, dtype=wp.vec3, device=device)
    if n == 0:
        return out_locations, out_distances, out_faces, out_weights
    if max_t is None:
        max_t = math.inf
    _launch.launch(
        kernel_ray.first_hits,
        dim=n,
        inputs=[
            wp.uint64(mesh.id),
            ray_origins,
            ray_directions,
            wp.float32(max_t),
            both_ways,
            out_locations,
            out_distances,
            out_faces,
            out_weights,
        ],
        device=device,
    )
    return out_locations, out_distances, out_faces, out_weights


def intersects_any(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> wp.array[wp.bool]:
    """
    Check whether each ray hits the mesh surface.

    Uses ``wp.mesh_query_ray_anyhit`` on the mesh BVH. Ray directions are unitized
    before querying. The search along each ray is unbounded unless ``max_t`` is given.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, the search is unbounded, which finds the same first hit as any bound at least
        as long as the diagonal of the box enclosing the mesh and the ray origins.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` hit flags; ``True`` when a ray hits within ``max_t``.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.

    Examples
    --------
    A ray towards a unit cube, one passing above it and one pointing away. The cube is ``1.5``
    away along the first ray, so ``max_t=1.0`` stops short of it:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[-2.0, 0.1, 0.2], [-2.0, 3.0, 0.0], [-2.0, 0.1, 0.2]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.ray.intersects_any(mesh, origins, directions).numpy().tolist())  # [True, False, False]
    hits = od.ray.intersects_any(mesh, origins, directions, max_t=1.0)
    print(hits.numpy().tolist())  # [False, False, False]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    if n == 0:
        return _launch.empty(0, dtype=wp.bool, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_hit = _launch.empty(n, dtype=wp.bool, device=ray_origins.device)
    _launch.map(
        kernel_ray.any_hit,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        out=out_hit,
    )
    return out_hit


def longest_ray(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
    planar_tol: float = TOLERANCE_PLANAR,
) -> wp.array[wp.float32]:
    """
    Find the length of the longest unobstructed ray segment along each direction.

    Uses iterative ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are
    unitized before querying. For each ray, returns the distance to the first mesh
    intersection strictly beyond ``planar_tol`` (to ignore degenerate on-surface
    hits), or ``inf`` when no such intersection exists within ``max_t``.

    Equivalent to [`trimesh.proximity.longest_ray`][].

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions.
    ray_directions
        ``(n,)`` ray direction vectors (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, the search is unbounded, which finds the same first hit as any bound at least
        as long as the diagonal of the box enclosing the mesh and the ray origins.
    planar_tol
        Ignore intersections closer than this distance from the ray origin.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` unobstructed ray lengths; ``inf`` when a ray misses within ``max_t``.

    Raises
    ------
    ValueError
        If ``ray_origins`` and ``ray_directions`` do not have the same shape.
    RuntimeError
        If ``mesh``, ``ray_origins`` and ``ray_directions`` are not all on one device.

    Examples
    --------
    From the centre of a unit cube the free path along ``+x`` ends on the face at ``x = 0.5``; from
    outside, pointing away, it never ends:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = wp.Mesh(points=vertices, indices=faces)
    origins = wp.array([[0.0, 0.1, 0.2], [2.0, 0.1, 0.2]], dtype=wp.vec3)
    directions = wp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.ray.longest_ray(mesh, origins, directions).numpy().tolist())  # [0.5, inf]
    ```
    """
    require_same_device(mesh=mesh, ray_origins=ray_origins, ray_directions=ray_directions)
    _validate_ray_inputs(ray_origins, ray_directions)
    n = ray_origins.size
    if n == 0:
        return _launch.empty(0, dtype=wp.float32, device=ray_origins.device)
    if max_t is None:
        max_t = math.inf

    out_distances = _launch.empty(n, dtype=wp.float32, device=ray_origins.device)
    _launch.map(
        kernel_ray.longest_ray_distance,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        wp.float32(planar_tol),
        out=out_distances,
    )
    return out_distances


def _validate_ray_inputs(ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3]) -> None:
    if ray_origins.shape != ray_directions.shape:
        raise ValueError(
            "ray_origins and ray_directions must have the same shape: "
            f"{tuple(ray_origins.shape)} vs {tuple(ray_directions.shape)}"
        )


def contains_points(
    mesh: wp.Mesh, points: wp.array[wp.vec3], *, n_sample: int = 5, perturbation_scale: float = 0.1
) -> wp.array[wp.bool]:
    """
    Test whether query points lie inside a closed mesh (ray parity sign).

    Casts the perturbed rays of ``wp.mesh_query_point_sign_parity``'s parity vote on the mesh
    BVH, the same vote that signs
    [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh], and stops casting once
    the majority is decided. Points outside the mesh axis-aligned bounding box are rejected
    without a parity test.

    Ordito mesh queries use **Warp's SDF sign convention** (outside positive,
    inside negative). A point is classified as inside when its signed distance
    would be negative; behavior on the on-surface tolerance band is undefined.
    Boolean results still match [`trimesh.Trimesh.contains`][].

    !!! note "Non-watertight meshes"

        Ray parity has no principled answer on an open or holed surface. For those, use the
        generalized winding-number sign of
        [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] instead
        (``sign_mode="winding"``, negative inside; see the second example below).

        That mode needs a ``wp.Mesh`` built with ``support_winding_number=True``; without the flag
        Warp's winding query silently falls back to ray parity.
        [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] builds such a mesh
        itself, or accepts one passed as ``mesh=`` after checking the flag.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    points
        ``(n,)`` query positions.
    n_sample
        Number of perturbed rays for parity voting (higher is more robust).
    perturbation_scale
        Uniform perturbation scale applied to the base ray direction.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` flags; ``True`` when the point is classified as inside the mesh.

    Raises
    ------
    RuntimeError
        If ``mesh`` and ``points`` are not all on one device.

    See Also
    --------
    [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]

    Examples
    --------
    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    mesh = wp.Mesh(points=vertices, indices=faces)
    points = wp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [2.0, 0.0, 0.0]], dtype=wp.vec3)
    print(od.ray.contains_points(mesh, points).numpy().tolist())  # [True, True, False]
    ```

    On an open surface, here a sphere with a hole around its ``-z`` pole, the winding-number sign
    still separates the inside from the outside:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.sphere_cap(angle=2.5)
    print(od.validation.is_watertight(vertices, faces))  # False
    points = wp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=wp.vec3)
    signed = od.proximity.signed_distance_on_mesh(vertices, faces, points, sign_mode="winding")
    print((signed.numpy() < 0.0).tolist())  # [True, False]
    ```
    """
    require_same_device(mesh=mesh, points=points)
    n = points.size
    if n == 0:
        return _launch.empty(0, dtype=wp.bool, device=points.device)
    # The kernel reads the mesh's box off its BVH root for the box rejection and for the query
    # cap, its diagonal, so the call reads nothing back.
    out_contains = _launch.empty(n, dtype=wp.bool, device=points.device)
    _launch.launch(
        kernel_proximity.contains_points_sign_parity,
        dim=n,
        inputs=[mesh.id, points, wp.int32(n_sample), wp.float32(perturbation_scale), out_contains],
        device=points.device,
    )
    return out_contains


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
    _, parameters, points, _, offsets = od.ray.segment_hits(vertices, faces, origins, endpoints)
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
    solid = solid_from_mesh(vertices, faces, 0, "")
    offsets, _ = offsets_buffer(n_segments, device)
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
        kernel_ray.segment_edge_mesh,
        dim=n_segments,
        inputs=[origins, endpoints],
        outputs=[ray_positions, ray_faces, ray_mates, ray_normals],
        device=device,
    )
    segment_box_lower, segment_box_upper = positions_box(ray_positions)
    scale = max(solid.scale, *(abs(x) for x in (*segment_box_lower, *segment_box_upper)))
    pad = max(scale, 1.0) * 4.0 * FLOAT32_EPSILON
    # The face tree for the segments' ray walks: boxes padded past the walk's ``float32`` rounding.
    lowers = _launch.empty(solid.n_faces, dtype=wp.vec3, device=device)
    uppers = _launch.empty(solid.n_faces, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_solid.operand_boxes,
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
    candidate_offsets, candidate_counts = offsets_buffer(n_segments, device)
    _launch.launch(
        kernel_ray.count_segment_candidates,
        dim=n_segments,
        inputs=[ray_positions, bvh.id],
        outputs=[candidate_counts],
        device=device,
    )
    n_candidates = scan_count(candidate_counts)
    if n_candidates == 0:
        return (
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.float64, device=device),
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.vec3, device=device),
            offsets,
        )
    segments, candidate_faces, hit_offsets, candidate_parameters, candidate_points = (
        _launch.carve_typed(
            [
                (wp.int32, n_candidates),
                (wp.int32, n_candidates),
                (wp.int32, n_candidates + 1),
                (wp.float64, n_candidates),
                (wp.vec3d, n_candidates),
            ],
            device,
        )
    )
    _launch.launch(
        kernel_ray.write_segment_candidates,
        dim=n_segments,
        inputs=[ray_positions, bvh.id, candidate_offsets],
        outputs=[segments, candidate_faces],
        device=device,
    )
    _launch.launch(
        kernel_ray.decide_segment_candidates,
        dim=n_candidates,
        inputs=[
            ray_faces,
            ray_positions,
            ray_normals,
            ray_mates,
            solid.faces,
            solid.positions,
            solid.vertex_normals,
            solid.face_normals,
            mates,
            segments,
            candidate_faces,
        ],
        outputs=[hit_offsets, candidate_parameters, candidate_points],
        device=device,
    )
    n_hits = scan_count(_launch.view(hit_offsets, 1, n_candidates + 1))
    hit_faces = _launch.empty(n_hits, dtype=wp.int32, device=device)
    parameters = _launch.empty(n_hits, dtype=wp.float64, device=device)
    points = _launch.empty(n_hits, dtype=wp.vec3, device=device)
    normals = _launch.empty(n_hits, dtype=wp.vec3, device=device)
    assert solid.face_source is not None
    _launch.launch(
        kernel_ray.write_segment_hits,
        dim=n_segments + 1,
        inputs=[
            candidate_offsets,
            hit_offsets,
            candidate_faces,
            candidate_parameters,
            candidate_points,
            solid.face_normals,
            solid.face_source,
        ],
        outputs=[offsets, hit_faces, parameters, points, normals],
        device=device,
    )
    return hit_faces, parameters, points, normals, offsets
