---
head_title: "potpourri3d on the GPU: migrating from potpourri3d"
description: >-
  GPU equivalents of potpourri3d's heat method: geodesic distance, vector heat, parallel transport, log maps and signed heat distance, in ordito.
---

# Migrating from potpourri3d

potpourri3d (the Python binding for geometry-central) is the reference for the heat-method
family — vector heat, parallel transport, log maps, and signed heat — and ordito's
[`ordito.heat`][ordito.heat] module mirrors that scope. The main structural difference:
potpourri3d builds a stateful `*Solver` object per mesh and reuses it across calls; in ordito
the state lives on the mesh: pass an [`ordito.Trimesh`][ordito.mesh.Trimesh] to the heat
functions and it keeps their operators and factorizations (its
[`heat_solver`][ordito.mesh.Trimesh.heat_solver]) for every later call.

```python
import potpourri3d as pp3d  # before
import ordito as od  # after
```

## Distance and vector heat

| potpourri3d | ordito |
|---|---|
| `pp3d.MeshHeatMethodDistanceSolver(V, F)` then `.compute_distance(source)` | `mesh = od.Trimesh(vertices, faces)` then [`heat.heat_geodesic(mesh, sources)`][ordito.heat.heat_geodesic] |
| `pp3d.MeshVectorHeatSolver(V, F)` then `.extend_scalar(sources, values)` | `mesh = od.Trimesh(vertices, faces)` then [`heat.extend_scalar(mesh, sources, values)`][ordito.heat.extend_scalar] |
| `.transport_tangent_vector(source, vector)` | [`heat.transport_tangent_vectors`][ordito.heat.transport_tangent_vectors] |
| `.compute_log_map(source)` | [`heat.log_map`][ordito.heat.log_map] |
| `pp3d.SignedHeatSolver(V, F)` then `.compute_distance(curve)` | [`heat.heat_signed_distance`][ordito.heat.heat_signed_distance] |
| `pp3d.edges(V, F)` | [`edges.faces_to_edges`][ordito.edges.faces_to_edges] / [`edges.edges_unique`][ordito.edges.edges_unique] |
| `pp3d.GeodesicTracer(V, F).trace_geodesic_from_vertex(...)` | [`geodesic_walk`][ordito.geodesic_walk] (straightest geodesics: walk a given distance in a given direction, batched over many walks) |
| `pp3d.PointCloudHeatSolver(P)` then `.compute_distance(source)`, `.extend_scalar`, `.transport_tangent_vector`, `.compute_log_map`, `.compute_signed_distance` | [`heat.PointCloudHeatSolver(points)`][ordito.heat.PointCloudHeatSolver] then [`.distance`][ordito.heat.PointCloudHeatSolver.distance], [`.extend_scalar`][ordito.heat.PointCloudHeatSolver.extend_scalar], [`.transport_tangent_vectors`][ordito.heat.PointCloudHeatSolver.transport_tangent_vectors], [`.log_map`][ordito.heat.PointCloudHeatSolver.log_map], [`.signed_distance`][ordito.heat.PointCloudHeatSolver.signed_distance] (a stateful solver here too, as on a cloud there is no mesh to keep the state) |
| `pp3d.PointCloudLocalTriangulation(P).get_local_triangulation()` | [`points.local_triangulation(points, neighbor_idx)`][ordito.points.local_triangulation] |
| Tangent frames used internally by the solvers | [`tangent_space.vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] / [`face_tangent_frames`][ordito.tangent_space.face_tangent_frames] (public and directly inspectable in ordito) |

## What's different, not just renamed

- **The solver state lives on the mesh, not in a solver object.** Where potpourri3d builds a
  solver per mesh and calls methods on it, ordito's functions take a `Trimesh` that keeps the
  operators and factorizations (see
  [Geodesic distance with the heat method](../examples/geodesics.md#g1));
  the `vertices, faces` form keeps nothing between calls.
- **The defaults discretize differently.** potpourri3d's solvers default to `use_robust=True`,
  which mollifies the mesh and builds an intrinsic Delaunay Laplacian; ordito defaults to the
  plain cotangent Laplacian (`use_robust=False`). To compare the two outputs, construct the
  potpourri3d solver with `use_robust=False`, or pass `use_robust=True` to
  [`heat.heat_geodesic`][ordito.heat.heat_geodesic].
- **Tangent-space quantities are gauge-dependent in both libraries.** A tangent frame from
  [`tangent_space.vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] agrees with
  potpourri3d's own frames only up to a rotation about the normal — compare transport results
  through a gauge-invariant combination (e.g. the angle between two transported vectors), never
  their raw 2D tangent components, exactly as you would with potpourri3d's own output.
- **Signed heat distance defaults to a vertex discretization.** potpourri3d's
  `SignedHeatSolver` diffuses on the mesh's edges; ordito reproduces that exactly under
  `method="potpourri3d"`, but its default `method="vertex_connection"` diffuses on the vertices,
  for a longer time.

    !!! warning "potpourri3d's signed heat method is wrong far from the curve on fine meshes"
        Its default diffusion time is so short that, on a mesh of a few hundred thousand faces,
        the diffused field falls below the smallest `float64` before it has crossed the surface,
        and the distance beyond that point is wrong by a large fraction of its range;
        `pp3d.SignedHeatSolver` returns a wrong field there (its direct solve loses the far
        field even sooner). `method="potpourri3d"` keeps geometry-central's time where it holds
        and lengthens it on meshes too fine for it, so the two agree on small meshes and ordito
        stays right on large ones; it warns if the field still falls out of range. The default
        method is more accurate at every resolution measured.

- **Point-cloud normals fix the stars' winding.** `PointCloudHeatSolver` estimates each
  tangent plane from the neighbours when no `normals` are given, which fixes the plane but not
  its sign, so its stars, and the fields, differ from potpourri3d's by a fraction of a per cent;
  pass oriented normals (a scanner's, or [`points.orient_normals`][ordito.points.orient_normals]'
  output) to get potpourri3d's answer. Tangent vectors in and out are in geometry-central's
  frames, as the class docstring spells out.
- **Isolines come back as points, not mesh elements.** potpourri3d's `marching_triangles`
  reports each isoline point as a geometry-central element index plus barycentric coordinates;
  ordito's [`intersection.marching_triangles`][ordito.intersection.marching_triangles] returns 3D
  positions directly.
