---
head_title: "Point cloud processing on the GPU in Python"
description: >-
  Down-sample, sample, estimate and orient normals, smooth, cluster, remove outliers, find boundary and visible points, and compute exact convex hulls of point clouds on the GPU in Python.
---

# Point clouds

Down-sampling, surface sampling, normal estimation and orientation, outlier removal,
interpolation, exact convex hulls, smoothing, DBSCAN clustering, boundary and visible points, and
local frames.

## Down-sampling: voxel grid, farthest point and blue noise {#pc1}

Three ways to thin the dragon's 437 000 vertices to a few thousand points.
[`voxel_down_sample`][ordito.voxels.voxel_down_sample] averages the points in each occupied
cell of a grid; [`farthest_point_sample`][ordito.points.farthest_point_sample] greedily picks
the point farthest from all picked so far, for an exact count with even coverage; and
[`sample_surface_blue_noise`][ordito.sample.sample_surface_blue_noise] draws new points on
the mesh surface, no two closer than a radius.

```python
import ordito as od
from examples import data

vertices, faces = data.load("dragon", device)

voxel = od.voxels.voxel_down_sample(vertices, 0.004)
farthest = vertices.numpy()[od.points.farthest_point_sample(vertices, voxel.shape[0]).numpy()]
blue, _ = od.sample.sample_surface_blue_noise(vertices, faces, radius=0.0027, seed=1)
print(f"{vertices.shape[0]} input points")
print(f"voxel grid: {voxel.shape[0]}, farthest point: {farthest.shape[0]}")
print(f"blue noise: {blue.shape[0]}")
```

```text title="Output"
437645 input points
voxel grid: 6269, farthest point: 6269
blue noise: 6186
```

![Down-sampling: voxel grid, farthest point and blue noise](../assets/examples/pc1.webp)

Modelled on: [Open3D: voxel down-sampling](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) · [PyVista: farthest point sampling](https://docs.pyvista.org/examples/01-filter/farthest_point_sampling) · [pytorch3d: sample_farthest_points](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html)
{: .ordito-credits }

## Surface sampling: uniform, Poisson disk and blue noise {#pc2}

[`sample_surface`][ordito.sample.sample_surface] draws points uniformly by area, which leaves
clumps and gaps; [`sample_surface_poisson_disk`][ordito.sample.sample_surface_poisson_disk]
draws five times as many and eliminates the most crowded ones until the requested count
remains, so the same number of points covers the surface more evenly;
[`sample_surface_blue_noise`][ordito.sample.sample_surface_blue_noise] instead takes a
radius and guarantees that no two samples are closer than it. The histogram of each sample's
distance to its nearest neighbour
([`nearest_neighbor_distance`][ordito.neighbors.nearest_neighbor_distance]) shows the
difference: the uniform draw spreads down to zero, the elimination pulls most gaps towards
the typical spacing, and blue noise has a hard floor at its radius.

```python
import numpy as np

import ordito as od
from examples import data

vertices, faces = data.load("bunny", device)
uniform, _ = od.sample.sample_surface(vertices, faces, 4000, seed=1)
poisson, _ = od.sample.sample_surface_poisson_disk(vertices, faces, 4000, seed=1)
blue, _ = od.sample.sample_surface_blue_noise(vertices, faces, radius=0.003, seed=1)

gap_uniform = od.neighbors.nearest_neighbor_distance(uniform).numpy()
gap_poisson = od.neighbors.nearest_neighbor_distance(poisson).numpy()
gap_blue = od.neighbors.nearest_neighbor_distance(blue).numpy()
for name, gap in (("uniform", gap_uniform), ("Poisson", gap_poisson), ("blue", gap_blue)):
    median, smallest = np.median(gap), gap.min()
    print(f"{name:>8}: {len(gap)} samples, gap median {median:.5f}, min {smallest:.5f}")
```

```text title="Output"
 uniform: 4000 samples, gap median 0.00177, min 0.00003
 Poisson: 4000 samples, gap median 0.00315, min 0.00280
    blue: 4016 samples, gap median 0.00316, min 0.00300
```

![Surface sampling: uniform, Poisson disk and blue noise](../assets/examples/pc2.webp)

The Poisson-disk histogram keeps a small tail of near-coincident pairs, about 2 % of the
samples closer than a tenth of the typical spacing. Weighted sample elimination should remove
the most crowded points first, so that tail is unexpected; use the blue-noise sampler when a
hard minimum spacing matters.

Modelled on: [Open3D: mesh sampling](https://www.open3d.org/docs/release/tutorial/geometry/mesh.html) · [libigl 810](https://libigl.github.io/tutorial/#blue-noise-sampling) · [PyMeshLab: Poisson-disk sampling](https://pymeshlab.readthedocs.io/en/latest/filter_list.html)
{: .ordito-credits }

## Normal estimation {#pc3}

A bare point cloud has no normals; [`estimate_normals`][ordito.points.estimate_normals] fits
a plane to each point's neighbourhood (here its 16 nearest neighbours from
[`query_nearest`][ordito.neighbors.query_nearest]) and takes the plane's normal. The sign is
a convention, not a measurement. By default it follows the neighbours (pytorch3d's rule: each
normal points to the side its neighbourhood bulges toward), which agrees locally but not
across the cloud; here ``orientation="centroid"`` turns each normal away from the cloud's
centroid instead, which is right on most of the bunny and wrong where the surface folds back
towards the centre (red, compared with the exact normals of the surface the cloud was sampled
from).

```python
import numpy as np

import ordito as od
from examples import data

points = data.load_points("bunny_cloud", device)
neighbours, _ = od.neighbors.query_nearest(points, points, k=16)
normals = od.points.estimate_normals(points, neighbours, orientation="centroid")

# Compare with the exact normals of the surface the cloud was sampled from.
cosine = np.sum(normals.numpy() * data.cloud_normals("bunny_cloud"), axis=1)
angle = np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))
print(f"median angle to the true normal (ignoring sign): {np.median(angle):.2f} degrees")
print(f"flipped relative to the true normal: {(cosine < 0).mean():.1%} of points")
```

```text title="Output"
median angle to the true normal (ignoring sign): 4.15 degrees
flipped relative to the true normal: 9.9% of points
```

![Normal estimation](../assets/examples/pc3.webp)

Modelled on: [Open3D: vertex normal estimation](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) · [pytorch3d: estimate_pointcloud_normals](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html)
{: .ordito-credits }

## Outlier removal: statistical, radius and probabilistic {#pc4}

A jittered scan of the bunny with 900 stray points scattered through its bounding box. Three
detectors, all from neighbourhood statistics:

- [`statistical_outlier_mask`][ordito.points.statistical_outlier_mask] flags points whose
  mean distance to their `k` nearest neighbours is far above the cloud's average;
- [`radius_outlier_mask`][ordito.points.radius_outlier_mask] flags points with too few
  neighbours within a fixed radius;
- [`outlier_probability`][ordito.points.outlier_probability] scores each point by how
  stretched its neighbourhood is relative to its neighbours' (Local Outlier Probability).

```python
import numpy as np

import ordito as od
from examples import data

points = data.load_points("noisy_cloud", device)
neighbours, distances = od.neighbors.query_nearest(points, points, k=20)

statistical = od.points.statistical_outlier_mask(distances, std_ratio=2.0).numpy()
radius = od.points.radius_outlier_mask(points, radius=0.004, min_neighbors=4).numpy()
probability = od.points.outlier_probability(neighbours, distances).numpy()

stray = np.arange(points.shape[0]) >= 30_000  # the input's last 900 points are the strays
for name, flagged in (
    ("statistical", statistical),
    ("radius", radius),
    ("LoOP > 0.8", probability > 0.8),
):
    print(f"{name:>11}: flags {flagged.sum()}, of which {(flagged & stray).sum()} strays")
```

```text title="Output"
statistical: flags 657, of which 657 strays
     radius: flags 765, of which 765 strays
 LoOP > 0.8: flags 402, of which 402 strays
```

![Outlier removal: statistical, radius and probabilistic](../assets/examples/pc4.webp)

Modelled on: [Open3D: point cloud outlier removal](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html) · [PyMeshLab: select outliers](https://pymeshlab.readthedocs.io/en/latest/filter_list.html)
{: .ordito-credits }

## Cloud-to-cloud distance {#pc5}

Two scans of the same bunny, the second pushed outwards by a bump on its flank and a gentle
ripple. [`query_nearest`][ordito.neighbors.query_nearest] with `k=1` finds, for every point
of the second cloud, its nearest point in the first and the distance to it: a per-point
deviation map with no mesh in sight. For one cloud against itself,
[`nearest_neighbor_distance`][ordito.neighbors.nearest_neighbor_distance] gives each point's
spacing to its closest other point, the noise floor such a comparison sits on.

```python
import numpy as np

import ordito as od
from examples import data

reference = data.load_points("bunny_cloud", device)
displaced = data.load_points("bunny_cloud_displaced", device)

_, distance = od.neighbors.query_nearest(reference, displaced, k=1)
spacing = od.neighbors.nearest_neighbor_distance(reference)
print(f"deviation: median {np.median(distance.numpy()):.5f}, max {distance.numpy().max():.5f}")
print(f"spacing of the reference cloud: median {np.median(spacing.numpy()):.5f}")
```

```text title="Output"
deviation: median 0.00022, max 0.00287
spacing of the reference cloud: median 0.00065
```

![Cloud-to-cloud distance](../assets/examples/pc5.webp)

Modelled on: [Open3D: point cloud distance](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) · [PyVista: distance between point clouds](https://docs.pyvista.org/examples/01-filter/distance_between_surfaces)
{: .ordito-credits }

## Interpolating sparse samples onto a surface {#pc6}

A field measured at only 60 probe points, spread over the bunny by
[`farthest_point_sample`][ordito.points.farthest_point_sample], is carried onto every vertex
by [`interpolate_from_points`][ordito.interpolation.interpolate_from_points]: a
Gaussian-weighted mean of the nearby probes. With a radius footprint, vertices farther than
the radius from every probe get the `null_value` (grey); with `k` nearest probes every vertex
gets a value, however far its probes are.

```python
import numpy as np
import warp as wp

import ordito as od
from examples import data

vertices, _ = data.load("bunny", device)
probes = vertices.numpy()[od.points.farthest_point_sample(vertices, 60).numpy()]
readings = np.sin(40.0 * probes[:, 0]) + np.cos(40.0 * probes[:, 1])  # the "measurement"

source = wp.array(probes, dtype=wp.vec3, device=device)
values = wp.array(readings, dtype=wp.float32, device=device)
by_radius = od.interpolation.interpolate_from_points(
    source, values, vertices, radius=0.015, null_value=float("nan")
)
by_neighbours = od.interpolation.interpolate_from_points(
    source, values, vertices, radius=0.015, k=4
)
print(f"vertices with no probe within the radius: {np.isnan(by_radius.numpy()).sum()}")
```

```text title="Output"
vertices with no probe within the radius: 9266
```

![Interpolating sparse samples onto a surface](../assets/examples/pc6.webp)

Modelled on: [PyVista: interpolate](https://docs.pyvista.org/examples/01-filter/interpolate)
{: .ordito-credits }

## Convex-hull points and half-space tests {#pc7}

Which points of the cloud lie on its convex hull?
[`convex_subset_mask`][ordito.points.convex_subset_mask] marks the points that are extreme
along some sampled direction: every one is on the hull, and more directions find more of
them. [`convex_superset_mask`][ordito.points.convex_superset_mask] answers the other way
round, discarding only points provably inside, so every hull vertex survives.
[`half_space_mask`][ordito.points.half_space_mask] is the building block: the points strictly
on one side of a plane. Both masks approximate the vertex set without building a surface; the
hull's triangles, exactly, come from [`convex_hull`][ordito.points.convex_hull]
([next example](#pc8)).

```python
import warp as wp

import ordito as od
from examples import data

points = data.load_points("bunny_cloud", device)
on_hull = od.points.convex_subset_mask(points, n_directions=4096).numpy()
candidates = od.points.convex_superset_mask(points).numpy()
upper = od.points.half_space_mask(
    points, wp.vec3(0.3, 1.0, 0.0), plane_origin=wp.vec3(0.0, 0.11, 0.0)
).numpy()
print(f"{on_hull.sum()} points certified on the hull (subset)")
print(f"{candidates.sum()} points that may be on it (superset), of {points.shape[0]}")
print(f"{upper.sum()} points above the plane")
```

```text title="Output"
1104 points certified on the hull (subset)
6340 points that may be on it (superset), of 30000
9513 points above the plane
```

![Convex-hull points and half-space tests](../assets/examples/pc7.webp)

Checked against an exact hull (`scipy.spatial.ConvexHull`, 1564 vertices on this cloud): all
1104 points of the subset are hull vertices, and every hull vertex is among the superset's
candidates.

Modelled on: [Open3D: convex hull](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) · [trimesh: convex](https://trimesh.org/trimesh.convex.html)
{: .ordito-credits }

## Exact convex hull {#pc8}

[`convex_hull`][ordito.points.convex_hull] returns the convex hull of a cloud as triangles
that index the input points, the layout of [`scipy.spatial.ConvexHull`][]'s `simplices`. Each
triangle is wound counter-clockwise seen from outside, and a hull of `V` vertices has `2V - 4`
of them. Every decision the algorithm takes is an exact orientation sign, with the same answer
on the CPU and the CUDA device and no tolerance to tune, so the hull is the true hull of the
`float32` points: its vertex set equals qhull's on both clouds below, the bunny sampled at
30 000 points and the dragon scan's 437 645 vertices.
[`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices] turns the
triangles into a compact mesh, outward-facing and ready for a [`Trimesh`][ordito.mesh.Trimesh].

```python
import numpy as np
import scipy.spatial

import ordito as od
from examples import data

clouds = {"bunny cloud": data.load_points("bunny_cloud", device)}
clouds["dragon scan"], _ = data.load("dragon", device)
hulls = {}
for name, points in clouds.items():
    faces = od.points.convex_hull(points)
    hull_vertices = np.unique(faces.numpy())
    qhull = scipy.spatial.ConvexHull(points.numpy())
    print(
        f"{name}: {points.shape[0]} points, hull of {hull_vertices.size} vertices "
        f"and {faces.shape[0] // 3} triangles, same vertices as qhull: "
        f"{np.array_equal(hull_vertices, np.sort(qhull.vertices))}"
    )
    hulls[name] = od.repair.remove_unreferenced_vertices(points, faces)[:2]
```

```text title="Output"
bunny cloud: 30000 points, hull of 1564 vertices and 3124 triangles, same vertices as qhull: True
dragon scan: 437645 points, hull of 2536 vertices and 5068 triangles, same vertices as qhull: True
```

![Exact convex hull](../assets/examples/pc8.webp)

**How it works.** The algorithm is ffHull (Gao, Cao, Tan and Huang, *Flip-Flop: Convex Hull
Construction via Star-Shaped Polyhedron in 3D*, I3D 2013). Incremental algorithms such as
quickhull add one point at a time. ffHull keeps the surface *star-shaped* instead of convex,
every face seeing one fixed kernel point inside, and that weaker invariant can be kept by
thousands of local edits at once. The surface starts as a tetrahedron. Each round, every face
with points beyond it splits into three at the farthest of them, all at once, and the points
the new surface encloses leave the computation for good, so the cost follows the points near
the hull rather than the size of the cloud. The figure runs the same growth in the plane (the
cross is the kernel point; blue points are still outside, grey ones dropped):

![ffHull's growth in the plane](../assets/examples/pc8-growth.svg)

In the plane the growth ends convex. In space it ends with *reflex* edges, dents where two faces
meet inward. Flip-Flop removes them with edge flips, again in parallel: a 2-2 flip swaps the
shared diagonal of two triangles (red) for the other one (blue) and only grows the volume;
where a dent cannot be flipped away, one of its vertices is provably not extreme and a 3-1 flip
deletes it. ordito also flips during growth, which drops the points under the dents early (2.4x
fewer flip rounds on the dragon, 3x on the bunny).

![A reflex edge before and after a 2-2 flip](../assets/examples/pc8-flip.webp)

Timings against qhull, the hull behind SciPy, trimesh, Open3D and MeshLab, are on the
[points benchmarks](../benchmarks/points.md#convex_hull) page.

Modelled on: [Open3D: convex hull](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) · [trimesh: convex](https://trimesh.org/trimesh.convex.html)
{: .ordito-credits }

## Convex hulls of degenerate clouds {#pc9}

The clouds a tolerance-based hull gets wrong are the ones made of exact coincidences. A
9 x 9 x 9 lattice has every point of a face in one plane and every point of an edge on one
line: [`convex_hull`][ordito.points.convex_hull] keeps the 8 corners and nothing else. An
octahedron's 6 corners, with 2 000 more points placed *exactly* on its faces (multiples of
1/64, which `float32` stores without rounding), keep the corners alone. A disc 2e-6 thick, six
orders of magnitude thinner than it is wide, is still a solid: its hull runs round the rim and
over the few interior points that stand proud of their neighbours by a fraction of a micron.
Each hull's vertex set is checked against `scipy.spatial.ConvexHull`.

```python
import numpy as np
import scipy.spatial
import warp as wp

import ordito as od

rng = np.random.default_rng(0)
axis = np.linspace(-1.0, 1.0, 9)
lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

# Points on the octahedron |x| + |y| + |z| = 1 with coordinates in multiples of 1/64.
steps = rng.integers(0, 65, size=(4000, 2))
steps = steps[steps.sum(axis=1) <= 64][:2000]
signs = rng.choice([-1.0, 1.0], size=(steps.shape[0], 3))
on_faces = signs * np.column_stack([steps, 64 - steps.sum(axis=1)]) / 64.0
octahedron = np.vstack([np.eye(3), -np.eye(3), on_faces])

disc = rng.uniform(-1.0, 1.0, size=(6000, 2))
disc = disc[np.hypot(disc[:, 0], disc[:, 1]) < 1.0]
slab = np.column_stack([disc, rng.uniform(-1e-6, 1e-6, size=disc.shape[0])])

clouds = {"lattice": lattice, "octahedron": octahedron, "slab": slab}
hulls = {}
for name, cloud in clouds.items():
    points = wp.array(cloud, dtype=wp.vec3, device=device)
    faces = od.points.convex_hull(points)
    hull_vertices = np.unique(faces.numpy())
    qhull = scipy.spatial.ConvexHull(points.numpy())
    print(
        f"{name}: {points.shape[0]} points, {hull_vertices.size} hull vertices, "
        f"same as qhull: {np.array_equal(hull_vertices, np.sort(qhull.vertices))}"
    )
    hulls[name] = (points.numpy(), faces.numpy().reshape(-1, 3))
```

```text title="Output"
lattice: 729 points, 8 hull vertices, same as qhull: True
octahedron: 2006 points, 6 hull vertices, same as qhull: True
slab: 4750 points, 179 hull vertices, same as qhull: True
```

![Convex hulls of degenerate clouds](../assets/examples/pc9.webp)

The rules for degenerate input:

- **Coplanar points on a facet** (the lattice's faces): the facet is triangulated arbitrarily,
  and the points inside it are not vertices.
- **Collinear points along a hull edge**: only the two ends are vertices.
- **Coincident points**: one of them is kept as the vertex.
- **A cloud a fraction of a micron thick**: still a solid, hulled like any other.
- **Fewer than 4 points, or no 4 spanning a tetrahedron** (an exactly flat cloud): `ValueError`,
  as there is no volume to bound.

Modelled on: [trimesh: convex](https://trimesh.org/trimesh.convex.html)
{: .ordito-credits }

## Orienting and smoothing normals {#pc10}

[`estimate_normals`][ordito.points.estimate_normals] fits each normal's direction but
can only guess its sign locally, so some normals point inward ([PC3](#pc3)).
[`orient_normals`][ordito.points.orient_normals] makes the signs agree across the whole
cloud: it builds the minimum spanning tree of the neighbour graph, weighting each edge by how
far its two normals are from parallel, and flips every normal to agree with its parent's,
Hoppe's propagation as Open3D's `orient_normals_consistent_tangent_plane` runs it. On a
closed surface every normal then faces one side.
[`smooth_normals`][ordito.points.smooth_normals]
averages each normal over its neighbours, sign-aligned to its own, which takes out the
noise of the plane fits. The cloud below is the bunny sample of [PC3](#pc3) jittered by
scanner-like noise, scored against the exact normals of the surface it was sampled from.

```python
import numpy as np
import warp as wp

import ordito as od
from examples import data

clean = data.arrays("bunny_cloud")[0]
truth = data.cloud_normals("bunny_cloud")  # the exact normals of the sampled surface
jitter = np.random.default_rng(3).normal(0.0, 0.0008, clean.shape)  # a noisy scan
points = wp.array(clean + jitter, dtype=wp.vec3, device=device)
neighbours, _ = od.neighbors.query_nearest(points, points, k=16)

def report(name: str, normals: np.ndarray) -> None:
    cosine = np.sum(normals * truth, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))
    outward = (cosine > 0).mean()
    print(f"{name:>9}: outward {outward:6.1%}, median angle {np.median(angle):.2f} degrees")

estimated = od.points.estimate_normals(points, neighbours)
oriented = od.points.orient_normals(points, estimated, neighbours)
smoothed = od.points.smooth_normals(oriented, neighbours, iterations=2)
report("estimated", estimated.numpy())
report("oriented", oriented.numpy())
report("smoothed", smoothed.numpy())
```

```text title="Output"
estimated: outward  46.1%, median angle 13.68 degrees
 oriented: outward  99.7%, median angle 13.68 degrees
 smoothed: outward  99.7%, median angle 9.01 degrees
```

![Orienting and smoothing normals](../assets/examples/pc10.webp)

Modelled on: [Open3D: normal orientation](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) · [PCL: normal estimation](https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html)
{: .ordito-credits }

## Smoothing a noisy cloud {#pc11}

Four ways to take the noise out of a scan without a mesh, scored by each point's distance to
the surface the clean cloud was sampled from.
[`smooth_points`][ordito.points.smooth_points] moves each point towards the mean of its
neighbours (Laplacian steps, which also shrink the cloud) or, given `mu`, alternates a
step out (Taubin's filter, which does not);
[`smooth_points_mls`][ordito.points.smooth_points_mls] projects each point onto the plane
fitted to its neighbourhood, a moving least squares step that also returns normals; and
[`smooth_points_bilateral`][ordito.points.smooth_points_bilateral] averages with weights that
fall off along the normal, so it spares creases. All four are Open3D's point-cloud filters,
run over a fixed neighbour table.

```python
import numpy as np
import warp as wp

import ordito as od
from examples import data

clean = data.arrays("bunny_cloud")[0]
rng = np.random.default_rng(3)
noisy = wp.array(clean + rng.normal(0.0, 0.0008, clean.shape), dtype=wp.vec3, device=device)
bunny = data.load("bunny", device)
neighbours, _ = od.neighbors.query_nearest(noisy, noisy, k=16)
normals = od.points.estimate_normals(noisy, neighbours)

smoothed = {
    "noisy": noisy,
    "Laplacian": od.points.smooth_points(noisy, neighbours, iterations=5),
    "Taubin": od.points.smooth_points(noisy, neighbours, iterations=5, mu=-0.53),
    "MLS": od.points.smooth_points_mls(noisy, neighbours)[0],
    "bilateral": od.points.smooth_points_bilateral(
        noisy, normals, neighbours, sigma_s=0.003, sigma_r=0.001
    ),
}
for name, points in smoothed.items():
    _, distance, _ = od.proximity.closest_point_on_mesh(*bunny, points)
    print(f"{name:>9}: median distance to the surface {np.median(distance.numpy()):.6f}")
```

```text title="Output"
    noisy: median distance to the surface 0.000536
Laplacian: median distance to the surface 0.000217
   Taubin: median distance to the surface 0.000246
      MLS: median distance to the surface 0.000205
bilateral: median distance to the surface 0.000272
```

![Smoothing a noisy cloud](../assets/examples/pc11.webp)

Modelled on: [Open3D: point cloud filters](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html) · [PCL: moving least squares](https://pcl.readthedocs.io/projects/tutorials/en/latest/resampling.html)
{: .ordito-credits }

## DBSCAN clustering {#pc12}

[`cluster_labels`][ordito.points.cluster_labels] labels each point with its DBSCAN cluster:
a point with at least `min_points` others within `radius` is a core point, core points
within reach of each other share a cluster, the points they reach join it, and the rest are
noise (`-1`). The labels are Open3D's `cluster_dbscan` labels, cluster numbers included, not
only the same partition. Here a scan of six separate parts, with 400 stray points in their
bounding box. The parts come out as five clusters: the box and the torus pass within 2 mm
of each other, closer than the radius, and DBSCAN rightly joins them, while the sphere and
the cone, 4 cm apart, stay separate. Most strays come out as noise; the rest landed within
the radius of a part, where they are border points of its cluster.

```python
import open3d as o3d
import warp as wp

import ordito as od
from examples import data

samples, _ = data.sample_points("parts", 120_000, seed=4)
rng = np.random.default_rng(5)
strays = rng.uniform(samples.min(axis=0), samples.max(axis=0), (400, 3)).astype(np.float32)
cloud = np.vstack([samples, strays])
points = wp.array(cloud, dtype=wp.vec3, device=device)

labels = od.points.cluster_labels(points, radius=0.025, min_points=8).numpy()
print(f"{labels.max() + 1} clusters, {np.sum(labels == -1)} noise points")
print(f"strays labelled noise: {np.mean(labels[len(samples) :] == -1):.0%}")

reference = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud.astype(np.float64)))
labels_o3d = np.asarray(reference.cluster_dbscan(eps=0.025, min_points=8))
print(f"same labels as Open3D: {np.array_equal(labels, labels_o3d)}")
```

```text title="Output"
5 clusters, 339 noise points
strays labelled noise: 85%
same labels as Open3D: True
```

![DBSCAN clustering](../assets/examples/pc12.webp)

Modelled on: [Open3D: DBSCAN clustering](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) · [scikit-learn: DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
{: .ordito-credits }

## Boundary points of a scan {#pc13}

[`point_boundary_mask`][ordito.points.point_boundary_mask] finds the rim of a sampled
surface without meshing it: each point's neighbours are projected into its tangent plane,
and a point with an empty wedge wider than `max_gap` around it is on the boundary, PCL's
boundary estimation as Open3D's `compute_boundary_points` runs it. The cloud below samples
the bunny with its holes cut open; the flagged points trace the holes' rims, and nearly all
of them lie within a few spacings of a rim of the mesh the cloud came from. The
neighbourhood must be wide enough that random sampling gaps do not look like an edge: with
16 neighbours many interior points of this area-uniform sample have an empty wedge.

```python
import numpy as np
import warp as wp

import ordito as od
from examples import data

samples, sample_normals = data.sample_points("holey_bunny", 60_000, seed=6)
points = wp.array(samples, dtype=wp.vec3, device=device)
normals = wp.array(sample_normals, dtype=wp.vec3, device=device)
neighbours, _ = od.neighbors.query_nearest(points, points, k=40)
on_rim = od.points.point_boundary_mask(points, normals, neighbours).numpy()
print(f"{on_rim.sum()} of {len(samples)} points flagged as boundary")

# Check against the rims of the mesh the cloud was sampled from.
vertices, faces = data.load("holey_bunny", device)
loops = od.boundary.boundary_loops(vertices, faces)
rim = np.vstack([vertices.numpy()[loop.numpy()] for loop in loops])
_, distance = od.neighbors.query_nearest(wp.array(rim, dtype=wp.vec3, device=device), points)
spacing = np.median(od.neighbors.nearest_neighbor_distance(points).numpy())
near = distance.numpy()[on_rim] < 6 * spacing
print(
    f"{len(loops)} rims in the mesh; "
    f"{near.mean():.1%} of flagged points within 6 spacings of one"
)
```

```text title="Output"
801 of 60000 points flagged as boundary
11 rims in the mesh; 92.6% of flagged points within 6 spacings of one
```

![Boundary points of a scan](../assets/examples/pc13.webp)

Modelled on: [Open3D: boundary points](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html) · [PCL: boundary estimation](https://pointclouds.org/documentation/classpcl_1_1_boundary_estimation.html)
{: .ordito-credits }

## Points visible from a viewpoint {#pc14}

[`point_visible_mask`][ordito.points.point_visible_mask] keeps the points a camera would
see, from the points alone: Katz, Tal and Basri's hidden point removal flips the cloud
about a large sphere centred on the camera, and the points whose images lie on the convex
hull of the flipped cloud are the visible ones. The hull is
[`convex_hull`][ordito.points.convex_hull]'s, exact. To check it, each point is also tested
by casting a ray from the camera at the bunny mesh the cloud was sampled from: a point is
visible when the first hit is (within a small distance) the point itself.

```python
import warp as wp

import ordito as od

points = data.load_points("bunny_cloud", device)
cloud = points.numpy()
diameter = float(np.linalg.norm(cloud.max(axis=0) - cloud.min(axis=0)))
camera = cloud.mean(axis=0) + np.array([0.0, 0.05, 0.4])
visible = od.points.point_visible_mask(points, wp.vec3(*camera), radius=100 * diameter).numpy()
print(f"{visible.sum()} of {len(cloud)} points visible")

# Reference: a ray from the camera to each point, against the bunny mesh.
vertices, faces = data.load("bunny", device)
mesh = wp.Mesh(points=vertices, indices=faces)
origins = wp.array(np.repeat([camera], len(cloud), axis=0), dtype=wp.vec3, device=device)
directions = wp.array(cloud - camera, dtype=wp.vec3, device=device)
hit, index, _ = od.ray.intersects_location(mesh, origins, directions)
first = np.full(len(cloud), np.inf)
reach = np.linalg.norm(hit.numpy() - camera, axis=1)
np.minimum.at(first, index.numpy(), reach)
seen = first > np.linalg.norm(cloud - camera, axis=1) - 0.001
print(f"agreement with ray casting: {np.mean(visible == seen):.1%}")
```

```text title="Output"
8720 of 30000 points visible
agreement with ray casting: 87.5%
```

![Points visible from a viewpoint](../assets/examples/pc14.webp)

The flip radius sets how generous the test is: larger radii count more grazing points as
visible. Katz et al. suggest about a hundred times the cloud's size, used here.

Modelled on: [Open3D: hidden point removal](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)
{: .ordito-credits }

## Local frames and surface variation {#pc15}

[`point_frames`][ordito.points.point_frames] gives every point the principal axes of its
neighbourhood, widest first, as the rows of a rotation, together with the spreads along
them: the eigen-decomposition of [`point_covariances`][ordito.points.point_covariances], with
the axes' signs fixed by the SHOT rule as pytorch3d's
`estimate_pointcloud_local_coord_frames` fixes them. The smallest spread over their sum, the
*surface variation*, is near zero on flat patches and rises at creases and corners, a
curvature proxy that needs no mesh: here it picks out the sharp edges of a machined part.

```python
import numpy as np
import warp as wp

import ordito as od
from examples import data

samples, _ = data.sample_points("cad_part", 50_000, seed=7)
points = wp.array(samples, dtype=wp.vec3, device=device)
neighbours, _ = od.neighbors.query_nearest(points, points, k=24)
frames, spreads = od.points.point_frames(points, neighbours)

spread = spreads.numpy()
variation = spread[:, 2] / spread.sum(axis=1)
normals = frames.numpy()[:, 2]  # the last row is the fitted normal
print(f"surface variation: median {np.median(variation):.4f}, max {variation.max():.3f}")
print(f"points on a crease (variation > 0.02): {np.mean(variation > 0.02):.1%}")
```

```text title="Output"
surface variation: median 0.0000, max 0.201
points on a crease (variation > 0.02): 13.7%
```

![Local frames and surface variation](../assets/examples/pc15.webp)

Modelled on: [pytorch3d: local coordinate frames](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html) · [Open3D: covariances](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)
{: .ordito-credits }
