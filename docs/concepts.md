---
head_title: "Concepts: Warp arrays in, Warp arrays out"
description: >-
  The ideas behind ordito's API: free functions over Warp arrays, flat face buffers, devices that follow the data, budgeted host syncs and measured defaults.
---

# Concepts

Five ideas shape every function in ordito. Each is a constraint the whole library is built
around, so knowing them up front explains most signatures before you read them.

## Arrays in, arrays out

The core API is free functions over [`warp.array`](https://nvidia.github.io/warp/modules/runtime.html#arrays)
buffers. There is no mandatory mesh object and no hidden state:

```python
import warp as wp

import ordito as od

vertices, faces = od.creation.icosphere(subdivisions=3)
normals, areas = od.triangles.face_normals_and_areas(vertices, faces)
print(vertices.shape, faces.shape, areas.shape)  # (642,) (3840,) (1280,)
```

A mesh is always the same two arrays: a `wp.array[wp.vec3]` of vertex positions and a **flat**
`wp.array[wp.int32]` of length `3 * n_faces`, where triangle `f` is the three indices
`faces[3 * f]`, `faces[3 * f + 1]`, `faces[3 * f + 2]`. Every function in the package uses this
one layout, so the output of one is always a valid input to the next. To move between the flat
layout and the `(n_faces, 3)` one at the host boundary:

```python
faces_rows = faces.numpy().reshape(-1, 3)  # (n_faces, 3), the layout trimesh and libigl use
faces_flat = wp.array(faces_rows.reshape(-1), dtype=wp.int32, device=vertices.device)
```

The optional [`Trimesh`][ordito.mesh.Trimesh] class is a thin wrapper around exactly this pair,
which computes derived quantities (normals, adjacency, boundary loops) the first time you ask for
them and caches them. It exists for convenience: each of its properties is also a public function
you can call on the raw arrays.

**Why not NumPy?** On a GPU, a `wp.array` lives in GPU memory, and `.numpy()` copies it back to
the host, waiting for the GPU to finish first. If every function took and returned NumPy arrays, a
pipeline would pay that copy at every step. With Warp arrays it pays only where you actually need
values on the host: to print them, save them, or hand them to another library.
`wp.array(numpy_array, dtype=..., device=...)` and `warp_array.numpy()` are the two directions
across that boundary.

## The device follows the data

There is no global device switch. Every kernel launch and every allocation inherits the device of
the function's input arrays:

```python
cpu_vertices, cpu_faces = od.creation.icosphere(subdivisions=2, device="cpu")
_, cpu_areas = od.triangles.face_normals_and_areas(cpu_vertices, cpu_faces)
# The kernel ran on the CPU because its inputs live there.
print(cpu_areas.device)  # cpu

if wp.is_cuda_available():
    gpu_vertices, gpu_faces = od.creation.icosphere(subdivisions=2, device="cuda:0")
    _, gpu_areas = od.triangles.face_normals_and_areas(gpu_vertices, gpu_faces)
    print(gpu_areas.device)  # cuda:0
```

Passing arrays from two different devices to one function raises a `RuntimeError` naming them;
move one with `array.to(device)` first.

A function with no input array to follow (a primitive constructor like `od.creation.icosphere`)
takes a `device=` keyword and defaults to Warp's current device: CUDA when a GPU is present, the
CPU otherwise. `wp.set_device(...)`, or a `with wp.ScopedDevice(...):` block, changes that
default, as in any other Warp program.

This is what makes chaining functions cheap: a pipeline of ten calls with no `device=` anywhere
runs entirely on one device, because each output lives where its inputs did.

## Host syncs are budgeted

Reading a value back to the host (`.numpy()`, or a function returning a Python `int`, `float` or
`bool`) makes the CPU wait until the GPU has finished all the work queued before it. A few of
those per call are cheap; one per iteration of a tight loop can dominate it. ordito's functions
avoid readbacks they do not need. Where one is unavoidable, usually because the size of an output
depends on a value only the device knows (how many faces survived a filter), the docstring says
so.

Some functions do a readback only to *infer* something you may already know, and take it as a
keyword instead:

```python
# Without n_vertices, the function reads back the largest index in faces to size its output.
vertex_faces, offsets = od.adjacency.vertex_face_adjacency(faces)

# You already know the vertex count: pass it and skip that readback.
vertex_faces, offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=vertices.shape[0])
```

The answer is the same either way. A value you pass is trusted, not checked, so it must be a true
bound (here, larger than every index in `faces`), as each such docstring states.

## Solids are checked, and decided exactly

Most functions accept any triangle soup. The ones that treat a mesh as a *solid*, the Booleans
of [`ordito.boolean`][ordito.boolean], [`subdivide_smooth`][ordito.remesh.subdivide_smooth],
[`simplify`][ordito.remesh.simplify] and [`segment_hits`][ordito.ray.segment_hits], require it to
be closed (every edge shared by exactly two faces) and consistently wound, check that before
doing anything else, and raise `ValueError` if it is not: an answer about the inside of a surface
with a hole in it would be a guess. [`make_solid`][ordito.repair.make_solid] repairs a scan into a
valid input. In return, every inside/outside decision is an exact sign computed from the
`float32` coordinates, and inputs that touch, share a plane or coincide are resolved by one
consistent rule, so the result is again a closed solid whatever the inputs' relative position,
on the CPU and on CUDA alike:

```python
import ordito as od

# Two unit cubes sharing a face: the worst case for a tolerance, routine for exact signs.
left = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
right = od.creation.box(bounds=[[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
vertices, faces = od.boolean.union(*left, *right)
print(f"{od.measures.volume(vertices, faces):.4f}")  # 2.0000
print(od.validation.is_watertight(vertices, faces))  # True

# A cube with one triangle missing is not a solid: the Boolean refuses it.
try:
    od.boolean.union(left[0], left[1][3:], *right)
except ValueError:
    print("not a closed solid")  # not a closed solid
```

## Measured, not assumed

Every public module has a test file and a benchmark file. Wherever an established library
computes the same quantity (trimesh, libigl, Open3D, PyMeshLab, PyVista, MeshLib, PyMeshFix,
potpourri3d, manifold3d, PyTorch3D, SciPy or NumPy), a test compares ordito's output with it
value by value, not just its shape. Where libraries define a quantity differently (Chamfer
distance squared or not, a signed distance negative inside or outside), the docstring says which
convention ordito follows and how to get the other. Performance changes land with a before/after measurement;
[Benchmarks](benchmarks/index.md) shows how to reproduce any published number.

## Where to next

- **[Examples](examples/index.md)**: these five ideas applied to real tasks.
- **[Reproducible results](reproducibility.md)**: why a GPU sum can change its last bits
  from call to call, and the one setting that stops it.
- **[Migrating from another library](migrating-from/index.md)**: how these conventions map onto
  the API you already know.
