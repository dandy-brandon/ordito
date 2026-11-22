---
head_title: "manifold3d on the GPU: migrating from Manifold"
description: >-
  GPU equivalents of manifold3d in ordito: exact mesh Booleans, batch Booleans, plane cuts, ray casts, level sets, smooth refinement, simplification and polygon triangulation, with the same meshes as output.
---

# Migrating from manifold3d

[`ordito.boolean`][ordito.boolean] is a port of [Manifold](https://github.com/elalish/manifold)'s
Boolean, and the functions below port its other operations. They are not look-alikes: each runs
Manifold's own exact predicates, symbolic perturbation, triangulator and topology cleanup on the
GPU, and the test suite compares its output with `manifold3d`'s array for array, so on the same
`float32` inputs you get the same mesh back: for the Booleans and plane cuts the same vertices
and triangles in the same order, and elsewhere the same mesh up to vertex numbering and face order.
Like building a `Manifold` from a `Mesh`, each function first cleans its input (drops triangles
that name a vertex twice, splits pinched vertices, collapses edges shorter than the precision,
swaps away degenerate triangles, drops unused vertices), so a raw scan gives the same answer in
both.

```python
import manifold3d as m3  # before
import ordito as od  # after
```

The structural difference: a `manifold3d.Manifold` is an immutable solid object, built once from a
`Mesh` and evaluated lazily, and its operations are methods and operators (`a + b`, `a - b`,
`a ^ b`). ordito has no solid object. Each function takes `(vertices, faces)` arrays, validates
that they form a closed, consistently wound mesh, computes the answer at once and returns new
arrays, which are again a valid solid and can be passed straight on.

```python
import manifold3d as m3
import numpy as np
import warp as wp

import ordito as od

# A Manifold from ordito arrays, and back.
vertices, faces = od.creation.icosphere(subdivisions=3)
triangles = faces.numpy().reshape(-1, 3).astype(np.uint32)
solid_m3 = m3.Manifold(m3.Mesh(vert_properties=vertices.numpy(), tri_verts=triangles))
mesh_m3 = solid_m3.to_mesh()
vertices_od = wp.array(np.asarray(mesh_m3.vert_properties)[:, :3], dtype=wp.vec3)
faces_od = wp.array(np.asarray(mesh_m3.tri_verts).ravel(), dtype=wp.int32)
print(od.equality.meshes_allclose(vertices, faces, vertices_od, faces_od))  # True
```

## Booleans and cuts

| manifold3d | ordito |
|---|---|
| `a + b` | [`boolean.union(*a, *b)`][ordito.boolean.union] |
| `a ^ b` | [`boolean.intersection`][ordito.boolean.intersection] |
| `a - b` | [`boolean.difference`][ordito.boolean.difference] |
| `Manifold.batch_boolean(parts, OpType.Add)` | [`boolean.union_all(parts)`][ordito.boolean.union_all] |
| `Manifold.batch_boolean(parts, OpType.Intersect)` | [`boolean.intersection_all`][ordito.boolean.intersection_all] |
| `Manifold.batch_boolean(parts, OpType.Subtract)` | [`boolean.difference_all`][ordito.boolean.difference_all] |
| `a.split(b)` | [`boolean.split_by_mesh(*a, *b)`][ordito.boolean.split_by_mesh] |
| `a.split_by_plane(normal, offset)` | [`boolean.split_by_plane(*a, normal, offset)`][ordito.boolean.split_by_plane] |
| `a.trim_by_plane(normal, offset)` | [`boolean.trim_by_plane(*a, normal, offset)`][ordito.boolean.trim_by_plane] |
| `a.slice(height)` | [`intersection.marching_triangles`][ordito.intersection.marching_triangles] of the vertices' heights (the section's segments, rather than a `CrossSection`) |
| `a.decompose()` | [`combine.split`][ordito.combine.split] |

## Building and refining solids

| manifold3d | ordito |
|---|---|
| `Manifold.level_set(sdf, bounds, edge_length, level, tolerance)` | [`levelset.function_level_set(sdf, bounds, edge_length, level, tolerance)`][ordito.levelset.function_level_set] (`sdf` is a `@wp.func` of a `wp.vec3d`, not a Python callable) |
| `a.smooth_out(min_sharp_angle, min_smoothness).refine(n)` | [`remesh.subdivide_smooth(*a, n, min_sharp_angle=..., min_smoothness=...)`][ordito.remesh.subdivide_smooth] |
| `.refine_to_length(length)`, `.refine_to_tolerance(tolerance)` after `smooth_out` | [`remesh.subdivide_smooth(*a, length=...)`][ordito.remesh.subdivide_smooth], `tolerance=...` |
| `a.refine(2)` (no `smooth_out`) | [`remesh.subdivide`][ordito.remesh.subdivide] |
| `a.simplify(tolerance)` | [`remesh.simplify(*a, tolerance)`][ordito.remesh.simplify] |
| `Manifold.hull_points(points)`, `a.hull()` | [`points.convex_hull`][ordito.points.convex_hull] (triangles indexing the input points; compact them with [`repair.remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices]) |
| `CrossSection.extrude(height)` of a polygon | [`polyline.triangulate_polygons`][ordito.polyline.triangulate_polygons], then [`creation.extrude_triangulation`][ordito.creation.extrude_triangulation] |
| `CrossSection.revolve(segments)` | [`creation.revolve`][ordito.creation.revolve] (a similar purpose; the tessellation differs) |
| `m3.triangulate(polygons, epsilon)` | [`polyline.triangulate_polygons(polygons, epsilon)`][ordito.polyline.triangulate_polygons] (the same triangles in the same order), or [`triangulate_polygons_from_offsets`][ordito.polyline.triangulate_polygons_from_offsets] for packed loops |

## Queries

| manifold3d | ordito |
|---|---|
| `a.ray_cast(origin, endpoint)`, one segment a call | [`ray.segment_hits(*a, origins, endpoints)`][ordito.ray.segment_hits] (every segment at once; each one's crossings packed with offsets) |
| `a.min_gap(b, search_length)` | [`proximity.mesh_to_mesh_distance`][ordito.proximity.mesh_to_mesh_distance] |
| `a.volume()`, `a.surface_area()` | [`measures.volume`][ordito.measures.volume], `Trimesh.area` |
| `a.genus()` | `(2 - `[`measures.euler_characteristic`][ordito.measures.euler_characteristic]`(faces)) // 2` for one component |
| `a.calculate_normals(...)` | [`triangles.corner_normals`][ordito.triangles.corner_normals] with crease edges |

## What's different, not just renamed

- **Validation is eager, and an invalid input raises.** Manifold records a bad input in
  `status()` and returns an empty solid; ordito raises `ValueError` naming what is wrong (not
  closed, an edge shared by more than two faces, inconsistent winding).
  [`repair.make_solid`][ordito.repair.make_solid] turns a broken scan into a valid input.
- **Inputs are welded by index.** A `manifold3d.Mesh` with `merge_from_vert` / `merge_to_vert`
  can describe a solid whose vertices are duplicated along seams; ordito reads the faces'
  indices as they are, so weld first with
  [`repair.remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices].
- **No vertex properties or face ids ride along.** Manifold carries `vert_properties` beyond the
  position and a per-face `run_original_id` through a Boolean. In ordito a result names its
  vertices as the inputs' vertices it keeps, then the crossings; to carry colours, UVs or labels
  through other edits use ``return_provenance=True`` where the function offers it.
- **No lazy evaluation.** Each call computes its answer before returning, so time a Boolean as it
  returns rather than after a `num_vert()` or `to_mesh()` call that forces Manifold's.
- **`collapse_collinear=False`** keeps the crossing vertices Manifold collapses where flat faces
  cross along straight lines: the vertex set of an exact-arithmetic Boolean such as libigl's
  CGAL one (what PyMeshLab's `generate_boolean_*` filters return). On curved inputs both give the
  same mesh.

See the [Solids and Booleans](../examples/solids.md) gallery for worked examples, and the
[boolean benchmarks](../benchmarks/boolean.md) for timings against manifold3d, MeshLib,
PyMeshLab and trimesh.
