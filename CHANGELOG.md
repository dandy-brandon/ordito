# Changelog

All notable changes to ordito are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and ordito adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a pre-1.0 caveat: while the
version is `0.x`, a **minor** bump may change a public signature, add or remove a public module,
or raise the `warp-lang` floor, and a **patch** bump is limited to bug fixes and documentation.

ordito is pre-1.0. The public API is safe to build on — it is covered by an extensive regression
suite and a nine-library parity gate — but until 1.0 a public signature may move a positional
argument to a keyword or gain a required parameter between minor versions. 1.0 will be the point
at which public function signatures become a commitment: module, name and positional argument
order stable, with change limited to additive keyword arguments.

Entries are written by hand rather than generated from commit messages: the git history is mostly
internal engineering (kernel-shape refactors, measurement rounds, review passes) that is not a
user-facing change, and a generated log would bury the handful of entries that are.

Every level-2 heading below is a released version, and `release.yml` cuts the GitHub Release body
from the one matching its tag — so prose sections do not belong between them.

## [Unreleased]

## [0.3.1] - 2026-11-17

### Fixed

- `reconstruction.screened_poisson` (`method="octree"`, and `method="adaptive"` with a band) no
  longer fails with CUDA error 700 on Blackwell GPUs of compute capability 12.0 (RTX 50 series,
  RTX PRO 6000) under Warp's CUDA 12 build (`warp-lang` `+cu12`), whose compiler mis-builds the
  iso-value sum for that architecture. Other GPUs, and Warp's CUDA 13 build, were not affected.
- `sparse_cholesky` (and the dense hole-fill solve) no longer fails to build its kernels when a
  CUDA 12 and a CUDA 13 build of Warp share one kernel cache: it clears the stale linked objects
  and rebuilds once, with a warning.
- Three kernels read a value before setting it (an unrolled loop index, a module constant):
  release builds happened to give the right answer, debug builds (`wp.config.mode = "debug"`)
  did not — `screened_poisson(method="octree")` read out of bounds there, and the sparse Cholesky
  panel updates and exact group sums read garbage bounds. Results of release builds are unchanged.
- The `warp-lang` requirement is now `>=1.19`. 0.3.0 declared `>=1.18` but uses Warp 1.19
  features and does not build on 1.18; this corrects the metadata rather than dropping a
  supported version.

## [0.3.0] - 2026-11-16

### Added

- `interpolation.Provenance`: where each output vertex and face of a topology edit came from, so
  vertex and face attributes (UVs, colours, labels, per-corner values) carry through it.
  `return_provenance=True` returns one from every function that rebuilds connectivity — the
  subdivisions, decimations, `isotropic_remesh` and the region refiners, `split_edges`, the
  `selection.submesh_*` family, `delete_region_keep_boundary`, the face-dropping and
  vertex-merging repairs, `make_solid`, `fix_degeneracies`, `fix_self_intersections`, the hole
  fills, extensions, bridges, stitches and joins, `seams.cut_along_edges`, `bounds.crop_mesh`,
  `combine.split`, and the `intersection` splits, slices and clips. Edits chain with
  `Provenance.then`; `vertex_values` / `face_values` carry an attribute through. Built from
  `Provenance.from_rows`, `from_edge_points`, `from_clusters` and `identity`, over the new
  `interpolation.transfer_through_index`, `operator_from_index`, `operator_representatives` and
  `compose_operators`.
- `interpolation.interpolate_barycentric`, and `sample.sample_surface(return_barycentric=True)`
  to read a vertex field at the sampled points.
- `repair.split_vertices_by_corner_values`: unweld vertices whose corners carry different values
  (UV seams); `repair.remove_duplicated_vertices(vertex_values=)` welds only vertices whose values
  also agree.
- `io.load_mesh_data` reads OBJ files itself and returns per-corner UVs (`corner_uv`); PLY
  texture-coordinate face lists are read too.
- `precompile`: `python -m ordito.precompile` (or `od.precompile.precompile`) builds every kernel
  ordito launches into Warp's kernel cache ahead of time, for any CUDA architecture without a GPU
  (`--arch`) and for the CPU, so a first call compiles nothing.
- `levelset.marching_cubes(method="openvdb")`: OpenVDB's `volumeToMesh` at adaptivity 0 (dual
  marching cubes), equal to MeshLib's `gridToMesh` vertex for vertex and face for face.
- `levelset.offset_mesh(method=)`: `"meshlab"` is MeshLab's offset; `"meshlib"` the exact signed
  distance on OpenVDB's lattice, extracted as above.
- `levelset.signed_distance_level_set` takes `wp.vec3d` bounds for a `float64` lattice, and
  classic `marching_cubes` a `float64` field: `resample_uniform(method="igl")` now matches
  `igl.offset_surface` to ~2e-7.
- `method="meshlib"` on `holes.fill_smooth`, `refill_region`, `stitch_smooth`,
  `bridge_edges_smooth` and `smoothing.smooth_region_boundary`: MeshLib's `fillHoleNicely`,
  `patchMesh`, `stitchHolesNicely`, `makeSmoothBridge` and `smoothRegionBoundary`.
- `heat.heat_geodesic(method="potpourri3d")`: geometry-central's heat method with its intrinsic
  tufted cover; `geodesic_walk.geodesic_path(method="potpourri3d")`: geometry-central's edge-flip
  geodesics (the default stays `"heat"`).
- Method aliases named for the library a default already reproduces:
  `filter_laplacian(method="trimesh")`, `filter_humphrey(method="trimesh")`,
  `icp(method="trimesh" | "open3d" | "pytorch3d")`, `marching_cubes(method="igl" | "pyvista")` and
  `principal_curvature(method="igl")`.
- `voxels.space_filling_codes` / `cells_from_space_filling_codes` (Hilbert and Morton codes of
  integer cells) and `points.space_filling_order`.
- `reduce.scatter_min` / `scatter_max` / `scatter_mean` and `segment_min` / `segment_max` /
  `segment_mean`.
- `neighbors.mutual_nearest`: reciprocal nearest-neighbour pairs between two clouds.
- `equality.triangles_allclose`: two meshes' triangles equal up to face order and rotation, with
  no vertex matching.
- `laplacian.cotmatrix_entries_and_pattern`; `array.gather(out=)`;
  `halfedge.halfedge_twins(defect_counts=)`; `graph.shortest_path_envelope(order=)`;
  `linalg.OperatorFactorization.refused`.

### Changed

- **Breaking:** the `vertices` module is now `normals` (`vertex_normals`, `mean_vertex_normals`,
  `weighted_vertex_normals`), and `vertex_defects` moved to `curvature`.
- **Breaking:** `return_index`, `return_operator`, `return_inverse` and `split_edges(index=)` are
  replaced by `return_provenance` (on `quadric_decimate`, `subdivide_to_size`, `subdivide_loop`,
  `split_edges`, `submesh_from_face_indices` / `_mask` and `remove_unreferenced_vertices`).
- **Breaking:** `halfedge.key_ordered_halfedges` and the `key_order` keyword of
  `boundary.oriented_boundary_edges` are removed. Boundary, oriented-boundary, region-boundary
  and crease edge rows come out in halfedge order, `resolve_duplicated_faces(method="cancel")`
  keeps faces in input order, and `cluster_decimate` numbers cells by arrival (by first
  occurrence under the deterministic setting). These outputs are sets: compare them through
  `ordito.equality`.
- Vertex normals (and everything built on them) repeat bit for bit on CUDA under the deterministic
  setting.
- The texture rasterizer gives the same images on CPU and CUDA.
- `smoothing.filter_normals(method="meshlib")` (the Mumford-Shah denoiser) follows MeshLib 3.2;
  `heat.fast_marching_distance(method="meshlib")` follows MeshLib 3.2's surface-distance front;
  `registration.procrustes` weights its cross-covariance as trimesh 5.2.1 does;
  `creation.parametric_surface("klein")` is the closed Klein bottle, as pyvista 0.49 builds it.
- A settled heat diffusion that fails its accuracy check and cannot be factored within the memory
  budget now warns that its far field is unreliable, instead of returning silently.
- Faster across the library — host overhead, launch counts, hub vertices of very high valence,
  tree searches on scrambled point orders, and the CPU device — and tested on `warp-lang` 1.19.

### Fixed

- `heat` `method="meshlab"` / `"igl"`: the default diffusion time is floored on large meshes, where
  the far-field distance stopped growing.
- `smoothing.smooth_region_fixed_rim` no longer collapses a free component that touches no fixed
  vertex.
- `intersection.clip_mesh_with_field(cap=True)`: sections passing a point twice, and faces the
  cut collapses, are capped correctly.
- Closest points on slivers (`icp(method="pyvista")`, the `float64` level-set band,
  `max_tangent_sphere`) no longer jump to a face's first corner.
- `linalg` multigrid no longer crashes on a fragmented free set, and carries the near-nullspace
  through its levels (fewer CG iterations on deep hierarchies).
- `energies.hessian_energy` on fans of more than 21 faces, and `smoothing.smooth_region` past the
  `int32` range, no longer write out of bounds.
- `remesh.refine_region_to_density` is reproducible on CUDA under the deterministic setting.
- `boundary.boundary_loops(pinches="gap")` keeps dead-end walks in their fan.

## [0.2.0] - 2026-11-03

### Added

- `heat.fast_marching_distance`: geodesic distance by a front marched across the faces, as
  MeshLib's `computeSurfaceDistances` computes it with no cap on re-opened vertices, solved in
  parallel rounds. `method="meshlab"` is MeshLab's geodesic front instead (vcg's face unfolding,
  in `float64`), equal to `compute_scalar_by_geodesic_distance_from_given_point_per_vertex` from one
  source.
- `curvature.vertex_curvatures`: pointwise mean and Gaussian curvature (Meyer et al.), as MeshLab's
  `compute_scalar_by_discrete_curvature_per_vertex` forms them.
- `curvature.principal_curvature(method="pyvista")`: VTK's 1-ring extreme curvatures, equal to
  `pyvista.PolyData.curvature("maximum")` / `("minimum")`.
- `curvature.principal_curvature(method="meyer")`: the principal pair implied by Meyer et al.'s
  pointwise mean and Gaussian curvature.
- `smoothing.filter_windowed_sinc`: the windowed-sinc low-pass filter of Taubin, Zhang and Golub,
  equal to `pyvista.PolyData.smooth_taubin` (VTK's `vtkWindowedSincPolyDataFilter`).
- `repair.resolve_duplicated_faces(method=)`: `"igl"` drops a group no cancellation resolves, as
  libigl does; `"open3d"` and `"meshlab"` keep the first face of each duplicate group, as
  `remove_duplicated_triangles` and `meshing_remove_duplicate_faces` do. The default still raises.
- `reconstruction.resample_uniform(method="igl")`: `igl.offset_surface`'s lattice.
- `reconstruction.resample_uniform(method="meshlab")`: MeshLab's uniform resampling, equal to
  `generate_resampled_uniform_mesh` mesh for mesh on an ordinary mesh: its lattice, its
  point-face distance and sign rule, and Lewiner's topologically controlled marching cubes
  (MC33). Where faces wound against each other coincide (duplicated or seamed faces), a few
  vertices around the affected lattice nodes can differ.
- `levelset.marching_cubes(method="lewiner")`: Lewiner et al.'s topologically controlled marching
  cubes (MC33), its ambiguous faces and cell interiors resolved by the field, on a `float32` or
  `float64` lattice.
- `array.bit_ranks`: the running set-bit count of a `uint32` bitmask's words, the rank table of
  a flag stored as one bit (`out=` writes it into a caller's buffer).
- `texture.rasterize_attribute` / `rasterize_discrete_attribute(subpixel_bits=)`: snap vertices
  to a sub-pixel grid as a GPU rasterizer does; `8` reproduces OpenGL.

### Changed

- `smoothing.filter_two_step(method="meshlab")` runs in `float64`, MeshLab's precision.
- `halfedge.key_ordered_halfedges` takes its flags as a bitmask and the bitmask's word ranks
  (`words, word_ranks`, from `array.bit_ranks`), a bit a halfedge and each class row padded to
  whole words, in place of an `int32` inclusive scan: a 32nd of the memory.
- `reconstruction.screened_poisson(method="octree")` runs to depth 10, matching open3d's
  reconstruction there, and is faster and leaner at every depth: its sweeps visit only bricks
  holding an active node, its index tables are rank-packed bitsets, and it shares its solve buffers
  with the dense and adaptive methods.
- Lower peak memory across `reconstruction`, `voxels`, `levelset`, `visibility`, `intersection`,
  `linalg` and `smoothing`; `triangulate_point_cloud` and `ball_pivoting` size their buffers for
  what a run holds.
- `energies.hessian_energy` / `curved_hessian_energy` build their sparsity pattern first, with far
  lower peak memory.
- On the CPU device, iterative loops (CG, Poisson V-cycle, fan flood, ear clipping and others) run
  as one launch, with identical results.

### Fixed

- `energies.hessian_energy` / `curved_hessian_energy` refuse a triplet count that overflows `int32`
  instead of faulting on very large meshes.
- `reconstruction.ball_pivoting` seeding reads only the failures of earlier waves.
- `sample.sample_surface_poisson_disk` is reproducible under the deterministic setting.
- The lazy submodule import no longer keeps the calling frame's arrays alive.
- `heat`: the vector heat solvers and `heat_signed_distance(method="potpourri3d")` lengthen their
  default time on fine meshes.
- `cholesky`: a solve into a caller's buffer no longer replays a stale recording.

## [0.1.0] - 2026-10-27

The first public release.

### Added

- GPU geometry processing on NVIDIA Warp: 52 public modules covering mesh creation, topology and
  validation, measures and curvature, repair, remeshing and smoothing, spatial queries and
  distances, point clouds and reconstruction, registration, Laplacians and sparse solvers
  (CG, multigrid, Cholesky), parametrization, geodesics and heat methods, and polylines.
- An optional `Trimesh` object API with cached derived quantities, over the same functions.
- Opt-in bit-reproducible results on CUDA (`ordito.config.set_deterministic`).
- Inline type information (`py.typed`).

### Notes

- Requires Python 3.11+ and `warp-lang` 1.18+; the only other runtime dependency is `numpy`.
  Mesh I/O needs the optional `io` extra (meshio).
- Runs on CUDA when available and on Warp's CPU backend otherwise; see
  [Platform support](https://lucagrementieri.github.io/ordito/getting-started/#platform-support).

[Unreleased]: https://github.com/lucagrementieri/ordito/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/lucagrementieri/ordito/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/lucagrementieri/ordito/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lucagrementieri/ordito/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lucagrementieri/ordito/releases/tag/v0.1.0
