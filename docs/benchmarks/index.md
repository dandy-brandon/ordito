---
head_title: "Benchmarks: GPU vs trimesh, libigl, Open3D, MeshLab"
description: >-
  How fast is ordito? GPU timings against trimesh, libigl, Open3D, MeshLab, PyVista, MeshLib, potpourri3d, manifold3d and PyTorch3D on real 3D scans up to 28 M triangles.
---

# Benchmarks

ordito runs on the GPU, and the libraries most mesh-processing code reaches for do not. This page
compares the two on the same inputs: ordito on CUDA against twelve reference libraries — trimesh,
libigl, Open3D, SciPy, NumPy, potpourri3d (geometry-central), MeshLab (through PyMeshLab), PyVista
(VTK), MeshLib, PyMeshFix, manifold3d (Manifold) and PyTorch3D, the one other library with GPU
kernels.

Every comparison times the same quantity on both sides, and the correctness suite checks that the
two answers agree. A time is the median of repeated calls from Python, including the wait for the
GPU to finish, so it is what a caller measures. The inputs are real 3D scans from the Stanford
repository (from 16 thousand to 28 million triangles) and synthetic meshes that each stress one
cost driver: many components, long boundaries, needle triangles, high valence.

<!-- bench:provenance -->
*Measured on GPU: NVIDIA GeForce RTX 5090 · CPU: Intel Xeon w7-2495X*
<!-- /bench:provenance -->

## Summary

How long each library takes relative to ordito, over every function both implement: ordito is 1x,
and a library at 30x takes thirty times as long. Each function counts once: its ratio is first
averaged over the inputs it was measured on, then the functions are averaged. Both averages are
geometric means, the usual way to combine ratios: a 10x win and a 10x loss cancel out.

<!-- bench:summary -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="282" viewBox="0 0 640 282" role="img" aria-label="Time of each library relative to ordito"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="264"/><text class="bench-tick" x="196.0" y="277" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="264"/><text class="bench-tick" x="274.0" y="277" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="264"/><text class="bench-tick" x="352.0" y="277" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="264"/><text class="bench-tick" x="430.0" y="277" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="264"/><text class="bench-tick" x="508.0" y="277" text-anchor="middle">1000x</text><g><title>ordito (CUDA): the reference, 1x</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x</text></g><g><title>manifold3d: 6x ordito&#x27;s time, geometric mean over 22 functions; slower than ordito in 19</title><rect class="bench-reference" x="196" y="29.0" width="138.6" height="10"/><text class="bench-label" x="166" y="38.0" text-anchor="end">manifold3d</text><text class="bench-value" x="339.6" y="38.0">6x  (22)</text></g><g><title>MeshLib: 7.3x ordito&#x27;s time, geometric mean over 145 functions; slower than ordito in 119</title><rect class="bench-reference" x="196" y="49.0" width="145.2" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="47.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">MeshLib</text><text class="bench-value" x="346.2" y="58.0">7.3x  (145)</text></g><g><title>PyTorch3D (CUDA): 11x ordito&#x27;s time, geometric mean over 42 functions; slower than ordito in 41</title><rect class="bench-reference" x="196" y="69.0" width="159.5" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="65.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="360.5" y="78.0">11x  (42)</text></g><g><title>NumPy: 16x ordito&#x27;s time, geometric mean over 43 functions; slower than ordito in 39</title><rect class="bench-reference" x="196" y="89.0" width="171.3" height="10"/><image class="bench-logo bench-logo-numpy" href="../assets/logos/numpy.png" x="172.6" y="85.0" width="16.9" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">NumPy</text><text class="bench-value" x="372.3" y="98.0">16x  (43)</text></g><g><title>Open3D: 34x ordito&#x27;s time, geometric mean over 98 functions; slower than ordito in 93</title><rect class="bench-reference" x="196" y="109.0" width="197.1" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="106.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="118.0" text-anchor="end">Open3D</text><text class="bench-value" x="398.1" y="118.0">34x  (98)</text></g><g><title>SciPy: 51x ordito&#x27;s time, geometric mean over 24 functions; slower than ordito in 24</title><rect class="bench-reference" x="196" y="129.0" width="210.9" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="125.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="138.0" text-anchor="end">SciPy</text><text class="bench-value" x="411.9" y="138.0">51x  (24)</text></g><g><title>potpourri3d: 65x ordito&#x27;s time, geometric mean over 31 functions; slower than ordito in 31</title><rect class="bench-reference" x="196" y="149.0" width="219.7" height="10"/><text class="bench-label" x="166" y="158.0" text-anchor="end">potpourri3d</text><text class="bench-value" x="420.7" y="158.0">65x  (31)</text></g><g><title>libigl: 85x ordito&#x27;s time, geometric mean over 118 functions; slower than ordito in 117</title><rect class="bench-reference" x="196" y="169.0" width="228.5" height="10"/><text class="bench-label" x="166" y="178.0" text-anchor="end">libigl</text><text class="bench-value" x="429.5" y="178.0">85x  (118)</text></g><g><title>PyVista: 96x ordito&#x27;s time, geometric mean over 81 functions; slower than ordito in 79</title><rect class="bench-reference" x="196" y="189.0" width="232.6" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="191.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="198.0" text-anchor="end">PyVista</text><text class="bench-value" x="433.6" y="198.0">96x  (81)</text></g><g><title>trimesh: 102x ordito&#x27;s time, geometric mean over 144 functions; slower than ordito in 140</title><rect class="bench-reference" x="196" y="209.0" width="234.6" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="210.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="218.0" text-anchor="end">trimesh</text><text class="bench-value" x="435.6" y="218.0">102x  (144)</text></g><g><title>MeshLab: 116x ordito&#x27;s time, geometric mean over 104 functions; slower than ordito in 104</title><rect class="bench-reference" x="196" y="229.0" width="239.0" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="225.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="238.0" text-anchor="end">MeshLab</text><text class="bench-value" x="440.0" y="238.0">116x  (104)</text></g><g><title>PyMeshFix: 494x ordito&#x27;s time, geometric mean over 4 functions; slower than ordito in 4</title><rect class="bench-reference" x="196" y="249.0" width="288.1" height="10"/><text class="bench-label" x="166" y="258.0" text-anchor="end">PyMeshFix</text><text class="bench-value" x="489.1" y="258.0">494x  (4)</text></g></svg>
<figcaption>Time relative to ordito (CUDA), geometric mean over every function both implement; in parentheses, the number of functions. Log scale. Runs stopped at the time limit are left out.</figcaption>
</figure>

??? note "Values"

    | Library | Time relative to ordito | Functions compared | ordito faster in |
    |---|---|---|---|
    | manifold3d | 6x | 22 | 19 |
    | MeshLib | 7.3x | 145 | 119 |
    | PyTorch3D (CUDA) | 11x | 42 | 41 |
    | NumPy | 16x | 43 | 39 |
    | Open3D | 34x | 98 | 93 |
    | SciPy | 51x | 24 | 24 |
    | potpourri3d | 65x | 31 | 31 |
    | libigl | 85x | 118 | 117 |
    | PyVista | 96x | 81 | 79 |
    | trimesh | 102x | 144 | 140 |
    | MeshLab | 116x | 104 | 104 |
    | PyMeshFix | 494x | 4 | 4 |
<!-- /bench:summary -->

Libraries are compared on different sets of functions, so each bar measures one library against
ordito, not the libraries against each other. The libraries closest to ordito are the ones whose
functions are mostly fast, compiled code on small inputs; the gaps grow with the input.

## By module

ordito's speedup per module, against the fastest library measured on each input — the toughest
comparison available for each function. Each module has its own page, with a chart for every
function.

<!-- bench:modules -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="962" viewBox="0 0 640 962" role="img" aria-label="ordito&#x27;s speedup per module"><line class="bench-baseline" x1="196.0" y1="0" x2="196.0" y2="944"/><text class="bench-tick" x="196.0" y="957" text-anchor="middle">1x</text><line class="bench-grid" x1="274.0" y1="0" x2="274.0" y2="944"/><text class="bench-tick" x="274.0" y="957" text-anchor="middle">10x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="944"/><text class="bench-tick" x="352.0" y="957" text-anchor="middle">100x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="944"/><text class="bench-tick" x="430.0" y="957" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="944"/><text class="bench-tick" x="508.0" y="957" text-anchor="middle">10000x</text><g><title>texture: ordito 2696x faster than the fastest library, geometric mean over 1 functions; faster in 1</title><rect class="bench-subject" x="196" y="9.0" width="267.6" height="10"/><text class="bench-label" x="190" y="18.0" text-anchor="end">texture</text><text class="bench-value" x="468.6" y="18.0">2696x  (1)</text></g><g><title>adjacency: ordito 189x faster than the fastest library, geometric mean over 7 functions; faster in 7</title><rect class="bench-subject" x="196" y="29.0" width="177.6" height="10"/><text class="bench-label" x="190" y="38.0" text-anchor="end">adjacency</text><text class="bench-value" x="378.6" y="38.0">189x  (7)</text></g><g><title>triangles: ordito 182x faster than the fastest library, geometric mean over 10 functions; faster in 10</title><rect class="bench-subject" x="196" y="49.0" width="176.3" height="10"/><text class="bench-label" x="190" y="58.0" text-anchor="end">triangles</text><text class="bench-value" x="377.3" y="58.0">182x  (10)</text></g><g><title>tangent_space: ordito 138x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="69.0" width="167.0" height="10"/><text class="bench-label" x="190" y="78.0" text-anchor="end">tangent_space</text><text class="bench-value" x="368.0" y="78.0">138x  (2)</text></g><g><title>grouping: ordito 86x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="89.0" width="151.0" height="10"/><text class="bench-label" x="190" y="98.0" text-anchor="end">grouping</text><text class="bench-value" x="352.0" y="98.0">86x  (4)</text></g><g><title>curvature: ordito 82x faster than the fastest library, geometric mean over 10 functions; faster in 10</title><rect class="bench-subject" x="196" y="109.0" width="149.4" height="10"/><text class="bench-label" x="190" y="118.0" text-anchor="end">curvature</text><text class="bench-value" x="350.4" y="118.0">82x  (10)</text></g><g><title>edges: ordito 79x faster than the fastest library, geometric mean over 14 functions; faster in 14</title><rect class="bench-subject" x="196" y="129.0" width="147.9" height="10"/><text class="bench-label" x="190" y="138.0" text-anchor="end">edges</text><text class="bench-value" x="348.9" y="138.0">79x  (14)</text></g><g><title>validation: ordito 69x faster than the fastest library, geometric mean over 14 functions; faster in 14</title><rect class="bench-subject" x="196" y="149.0" width="143.4" height="10"/><text class="bench-label" x="190" y="158.0" text-anchor="end">validation</text><text class="bench-value" x="344.4" y="158.0">69x  (14)</text></g><g><title>sample: ordito 66x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="169.0" width="142.0" height="10"/><text class="bench-label" x="190" y="178.0" text-anchor="end">sample</text><text class="bench-value" x="343.0" y="178.0">66x  (4)</text></g><g><title>smoothing: ordito 58x faster than the fastest library, geometric mean over 25 functions; faster in 24</title><rect class="bench-subject" x="196" y="189.0" width="137.8" height="10"/><text class="bench-label" x="190" y="198.0" text-anchor="end">smoothing</text><text class="bench-value" x="338.8" y="198.0">58x  (25)</text></g><g><title>seams: ordito 52x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="209.0" width="133.7" height="10"/><text class="bench-label" x="190" y="218.0" text-anchor="end">seams</text><text class="bench-value" x="334.7" y="218.0">52x  (4)</text></g><g><title>graph: ordito 51x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="229.0" width="133.2" height="10"/><text class="bench-label" x="190" y="238.0" text-anchor="end">graph</text><text class="bench-value" x="334.2" y="238.0">51x  (6)</text></g><g><title>normals: ordito 51x faster than the fastest library, geometric mean over 3 functions; faster in 3</title><rect class="bench-subject" x="196" y="249.0" width="133.2" height="10"/><text class="bench-label" x="190" y="258.0" text-anchor="end">normals</text><text class="bench-value" x="334.2" y="258.0">51x  (3)</text></g><g><title>laplacian: ordito 45x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="269.0" width="129.2" height="10"/><text class="bench-label" x="190" y="278.0" text-anchor="end">laplacian</text><text class="bench-value" x="330.2" y="278.0">45x  (9)</text></g><g><title>repair: ordito 45x faster than the fastest library, geometric mean over 22 functions; faster in 20</title><rect class="bench-subject" x="196" y="289.0" width="128.7" height="10"/><text class="bench-label" x="190" y="298.0" text-anchor="end">repair</text><text class="bench-value" x="329.7" y="298.0">45x  (22)</text></g><g><title>measures: ordito 36x faster than the fastest library, geometric mean over 3 functions; faster in 3</title><rect class="bench-subject" x="196" y="309.0" width="121.8" height="10"/><text class="bench-label" x="190" y="318.0" text-anchor="end">measures</text><text class="bench-value" x="322.8" y="318.0">36x  (3)</text></g><g><title>energies: ordito 35x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="329.0" width="120.6" height="10"/><text class="bench-label" x="190" y="338.0" text-anchor="end">energies</text><text class="bench-value" x="321.6" y="338.0">35x  (9)</text></g><g><title>points: ordito 30x faster than the fastest library, geometric mean over 33 functions; faster in 33</title><rect class="bench-subject" x="196" y="349.0" width="115.2" height="10"/><text class="bench-label" x="190" y="358.0" text-anchor="end">points</text><text class="bench-value" x="316.2" y="358.0">30x  (33)</text></g><g><title>remesh: ordito 28x faster than the fastest library, geometric mean over 15 functions; faster in 15</title><rect class="bench-subject" x="196" y="369.0" width="113.4" height="10"/><text class="bench-label" x="190" y="378.0" text-anchor="end">remesh</text><text class="bench-value" x="314.4" y="378.0">28x  (15)</text></g><g><title>neighbors: ordito 26x faster than the fastest library, geometric mean over 15 functions; faster in 15</title><rect class="bench-subject" x="196" y="389.0" width="110.4" height="10"/><text class="bench-label" x="190" y="398.0" text-anchor="end">neighbors</text><text class="bench-value" x="311.4" y="398.0">26x  (15)</text></g><g><title>mesh: ordito 26x faster than the fastest library, geometric mean over 14 functions; faster in 14</title><rect class="bench-subject" x="196" y="409.0" width="109.8" height="10"/><text class="bench-label" x="190" y="418.0" text-anchor="end">mesh</text><text class="bench-value" x="310.8" y="418.0">26x  (14)</text></g><g><title>metrics: ordito 25x faster than the fastest library, geometric mean over 8 functions; faster in 8</title><rect class="bench-subject" x="196" y="429.0" width="109.5" height="10"/><text class="bench-label" x="190" y="438.0" text-anchor="end">metrics</text><text class="bench-value" x="310.5" y="438.0">25x  (8)</text></g><g><title>heat: ordito 25x faster than the fastest library, geometric mean over 19 functions; faster in 19</title><rect class="bench-subject" x="196" y="449.0" width="109.0" height="10"/><text class="bench-label" x="190" y="458.0" text-anchor="end">heat</text><text class="bench-value" x="310.0" y="458.0">25x  (19)</text></g><g><title>geodesic_walk: ordito 23x faster than the fastest library, geometric mean over 7 functions; faster in 6</title><rect class="bench-subject" x="196" y="469.0" width="106.4" height="10"/><text class="bench-label" x="190" y="478.0" text-anchor="end">geodesic_walk</text><text class="bench-value" x="307.4" y="478.0">23x  (7)</text></g><g><title>interpolation: ordito 21x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="489.0" width="102.7" height="10"/><text class="bench-label" x="190" y="498.0" text-anchor="end">interpolation</text><text class="bench-value" x="303.7" y="498.0">21x  (6)</text></g><g><title>levelset: ordito 21x faster than the fastest library, geometric mean over 8 functions; faster in 8</title><rect class="bench-subject" x="196" y="509.0" width="102.4" height="10"/><text class="bench-label" x="190" y="518.0" text-anchor="end">levelset</text><text class="bench-value" x="303.4" y="518.0">21x  (8)</text></g><g><title>visibility: ordito 20x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="529.0" width="102.2" height="10"/><text class="bench-label" x="190" y="538.0" text-anchor="end">visibility</text><text class="bench-value" x="303.2" y="538.0">20x  (6)</text></g><g><title>array: ordito 20x faster than the fastest library, geometric mean over 21 functions; faster in 20</title><rect class="bench-subject" x="196" y="549.0" width="100.9" height="10"/><text class="bench-label" x="190" y="558.0" text-anchor="end">array</text><text class="bench-value" x="301.9" y="558.0">20x  (21)</text></g><g><title>parametrization: ordito 19x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="569.0" width="99.9" height="10"/><text class="bench-label" x="190" y="578.0" text-anchor="end">parametrization</text><text class="bench-value" x="300.9" y="578.0">19x  (9)</text></g><g><title>creation: ordito 19x faster than the fastest library, geometric mean over 22 functions; faster in 18</title><rect class="bench-subject" x="196" y="589.0" width="99.2" height="10"/><text class="bench-label" x="190" y="598.0" text-anchor="end">creation</text><text class="bench-value" x="300.2" y="598.0">19x  (22)</text></g><g><title>registration: ordito 14x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="609.0" width="88.9" height="10"/><text class="bench-label" x="190" y="618.0" text-anchor="end">registration</text><text class="bench-value" x="289.9" y="618.0">14x  (9)</text></g><g><title>combine: ordito 12x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="629.0" width="84.9" height="10"/><text class="bench-label" x="190" y="638.0" text-anchor="end">combine</text><text class="bench-value" x="285.9" y="638.0">12x  (2)</text></g><g><title>selection: ordito 9.8x faster than the fastest library, geometric mean over 9 functions; faster in 6</title><rect class="bench-subject" x="196" y="649.0" width="77.3" height="10"/><text class="bench-label" x="190" y="658.0" text-anchor="end">selection</text><text class="bench-value" x="278.3" y="658.0">9.8x  (9)</text></g><g><title>proximity: ordito 8.6x faster than the fastest library, geometric mean over 6 functions; faster in 4</title><rect class="bench-subject" x="196" y="669.0" width="72.8" height="10"/><text class="bench-label" x="190" y="678.0" text-anchor="end">proximity</text><text class="bench-value" x="273.8" y="678.0">8.6x  (6)</text></g><g><title>intersection: ordito 8.5x faster than the fastest library, geometric mean over 12 functions; faster in 8</title><rect class="bench-subject" x="196" y="689.0" width="72.5" height="10"/><text class="bench-label" x="190" y="698.0" text-anchor="end">intersection</text><text class="bench-value" x="273.5" y="698.0">8.5x  (12)</text></g><g><title>voxels: ordito 7.4x faster than the fastest library, geometric mean over 15 functions; faster in 12</title><rect class="bench-subject" x="196" y="709.0" width="67.6" height="10"/><text class="bench-label" x="190" y="718.0" text-anchor="end">voxels</text><text class="bench-value" x="268.6" y="718.0">7.4x  (15)</text></g><g><title>reduce: ordito 7.3x faster than the fastest library, geometric mean over 22 functions; faster in 18</title><rect class="bench-subject" x="196" y="729.0" width="67.5" height="10"/><text class="bench-label" x="190" y="738.0" text-anchor="end">reduce</text><text class="bench-value" x="268.5" y="738.0">7.3x  (22)</text></g><g><title>ray: ordito 6.8x faster than the fastest library, geometric mean over 7 functions; faster in 6</title><rect class="bench-subject" x="196" y="749.0" width="64.9" height="10"/><text class="bench-label" x="190" y="758.0" text-anchor="end">ray</text><text class="bench-value" x="265.9" y="758.0">6.8x  (7)</text></g><g><title>reconstruction: ordito 6.7x faster than the fastest library, geometric mean over 7 functions; faster in 7</title><rect class="bench-subject" x="196" y="769.0" width="64.5" height="10"/><text class="bench-label" x="190" y="778.0" text-anchor="end">reconstruction</text><text class="bench-value" x="265.5" y="778.0">6.7x  (7)</text></g><g><title>holes: ordito 6.2x faster than the fastest library, geometric mean over 16 functions; faster in 12</title><rect class="bench-subject" x="196" y="789.0" width="62.0" height="10"/><text class="bench-label" x="190" y="798.0" text-anchor="end">holes</text><text class="bench-value" x="263.0" y="798.0">6.2x  (16)</text></g><g><title>boundary: ordito 4.5x faster than the fastest library, geometric mean over 7 functions; faster in 3</title><rect class="bench-subject" x="196" y="809.0" width="50.6" height="10"/><text class="bench-label" x="190" y="818.0" text-anchor="end">boundary</text><text class="bench-value" x="251.6" y="818.0">4.5x  (7)</text></g><g><title>homology: ordito 4.3x faster than the fastest library, geometric mean over 1 functions; faster in 1</title><rect class="bench-subject" x="196" y="829.0" width="49.7" height="10"/><text class="bench-label" x="190" y="838.0" text-anchor="end">homology</text><text class="bench-value" x="250.7" y="838.0">4.3x  (1)</text></g><g><title>transform: ordito 3.6x faster than the fastest library, geometric mean over 5 functions; faster in 3</title><rect class="bench-subject" x="196" y="849.0" width="42.9" height="10"/><text class="bench-label" x="190" y="858.0" text-anchor="end">transform</text><text class="bench-value" x="243.9" y="858.0">3.6x  (5)</text></g><g><title>linalg: ordito 3.3x faster than the fastest library, geometric mean over 1 functions; faster in 1</title><rect class="bench-subject" x="196" y="869.0" width="40.3" height="10"/><text class="bench-label" x="190" y="878.0" text-anchor="end">linalg</text><text class="bench-value" x="241.3" y="878.0">3.3x  (1)</text></g><g><title>polyline: ordito 2.3x faster than the fastest library, geometric mean over 11 functions; faster in 6</title><rect class="bench-subject" x="196" y="889.0" width="28.9" height="10"/><text class="bench-label" x="190" y="898.0" text-anchor="end">polyline</text><text class="bench-value" x="229.9" y="898.0">2.3x  (11)</text></g><g><title>boolean: ordito 2.3x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="909.0" width="27.8" height="10"/><text class="bench-label" x="190" y="918.0" text-anchor="end">boolean</text><text class="bench-value" x="228.8" y="918.0">2.3x  (9)</text></g><g><title>bounds: ordito 1.4x faster than the fastest library, geometric mean over 9 functions; faster in 7</title><rect class="bench-subject" x="196" y="929.0" width="11.5" height="10"/><text class="bench-label" x="190" y="938.0" text-anchor="end">bounds</text><text class="bench-value" x="212.5" y="938.0">1.4x  (9)</text></g></svg>
<figcaption>ordito (CUDA)'s speedup over the fastest library measured on each input, geometric mean per module; in parentheses, the number of functions. Log scale.</figcaption>
</figure>

??? note "Values"

    | Module | ordito's speedup | Functions | ordito faster in |
    |---|---|---|---|
    | [`texture`](texture.md) | 2696x | 1 | 1 |
    | [`adjacency`](adjacency.md) | 189x | 7 | 7 |
    | [`triangles`](triangles.md) | 182x | 10 | 10 |
    | [`tangent_space`](tangent_space.md) | 138x | 2 | 2 |
    | [`grouping`](grouping.md) | 86x | 4 | 4 |
    | [`curvature`](curvature.md) | 82x | 10 | 10 |
    | [`edges`](edges.md) | 79x | 14 | 14 |
    | [`validation`](validation.md) | 69x | 14 | 14 |
    | [`sample`](sample.md) | 66x | 4 | 4 |
    | [`smoothing`](smoothing.md) | 58x | 25 | 24 |
    | [`seams`](seams.md) | 52x | 4 | 4 |
    | [`graph`](graph.md) | 51x | 6 | 6 |
    | [`normals`](normals.md) | 51x | 3 | 3 |
    | [`laplacian`](laplacian.md) | 45x | 9 | 9 |
    | [`repair`](repair.md) | 45x | 22 | 20 |
    | [`measures`](measures.md) | 36x | 3 | 3 |
    | [`energies`](energies.md) | 35x | 9 | 9 |
    | [`points`](points.md) | 30x | 33 | 33 |
    | [`remesh`](remesh.md) | 28x | 15 | 15 |
    | [`neighbors`](neighbors.md) | 26x | 15 | 15 |
    | [`mesh`](mesh.md) | 26x | 14 | 14 |
    | [`metrics`](metrics.md) | 25x | 8 | 8 |
    | [`heat`](heat.md) | 25x | 19 | 19 |
    | [`geodesic_walk`](geodesic_walk.md) | 23x | 7 | 6 |
    | [`interpolation`](interpolation.md) | 21x | 6 | 6 |
    | [`levelset`](levelset.md) | 21x | 8 | 8 |
    | [`visibility`](visibility.md) | 20x | 6 | 6 |
    | [`array`](array.md) | 20x | 21 | 20 |
    | [`parametrization`](parametrization.md) | 19x | 9 | 9 |
    | [`creation`](creation.md) | 19x | 22 | 18 |
    | [`registration`](registration.md) | 14x | 9 | 9 |
    | [`combine`](combine.md) | 12x | 2 | 2 |
    | [`selection`](selection.md) | 9.8x | 9 | 6 |
    | [`proximity`](proximity.md) | 8.6x | 6 | 4 |
    | [`intersection`](intersection.md) | 8.5x | 12 | 8 |
    | [`voxels`](voxels.md) | 7.4x | 15 | 12 |
    | [`reduce`](reduce.md) | 7.3x | 22 | 18 |
    | [`ray`](ray.md) | 6.8x | 7 | 6 |
    | [`reconstruction`](reconstruction.md) | 6.7x | 7 | 7 |
    | [`holes`](holes.md) | 6.2x | 16 | 12 |
    | [`boundary`](boundary.md) | 4.5x | 7 | 3 |
    | [`homology`](homology.md) | 4.3x | 1 | 1 |
    | [`transform`](transform.md) | 3.6x | 5 | 3 |
    | [`linalg`](linalg.md) | 3.3x | 1 | 1 |
    | [`polyline`](polyline.md) | 2.3x | 11 | 6 |
    | [`boolean`](boolean.md) | 2.3x | 9 | 9 |
    | [`bounds`](bounds.md) | 1.4x | 9 | 7 |
<!-- /bench:modules -->

## Highlights

The comparisons with the widest field of libraries. Each chart shows one input, with ordito at 1x
and every other library's time relative to it; each module's page has every function and every
input.

### Signed distance to a mesh

Signed distance from a point set to a triangle mesh, the widest field among the spatial queries.

<!-- bench:cell test_proximity/signed_distance_on_mesh -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="162" viewBox="0 0 640 162" role="img" aria-label="signed_distance_on_mesh on dragon (871 414 faces) · sign_mode=parity"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="144"/><text class="bench-tick" x="196.0" y="157" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="144"/><text class="bench-tick" x="258.4" y="157" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="144"/><text class="bench-tick" x="320.8" y="157" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="144"/><text class="bench-tick" x="383.2" y="157" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="144"/><text class="bench-tick" x="445.6" y="157" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="144"/><text class="bench-tick" x="508.0" y="157" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 4.69 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 4.69 ms</text></g><g><title>MeshLib: 34.05 ms, 7.3x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="116.1" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="317.1" y="38.0">7.3x · 34.05 ms</text></g><g><title>Open3D: 270.03 ms, 58x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="172.2" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="373.2" y="58.0">58x · 270.03 ms</text></g><g><title>PyVista: 1.025 s, 219x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="208.4" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="409.4" y="78.0">219x · 1.025 s</text></g><g><title>libigl: 1.124 s, 240x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="210.9" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="411.9" y="98.0">240x · 1.124 s</text></g><g><title>MeshLab: 7.289 s, 1554x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="261.6" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="105.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">MeshLab</text><text class="bench-value" x="462.6" y="118.0">1554x · 7.289 s</text></g><g><title>trimesh: 26.022 s, 5548x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="129.0" width="296.0" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="130.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="138.0" text-anchor="end">trimesh</text><text class="bench-value" x="497.0" y="138.0">5548x · 26.022 s</text></g></svg>
<figcaption>dragon (871 414 faces) · sign_mode=parity · time relative to ordito, log scale · [all proximity benchmarks](proximity.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 4.69 ms | 1x |
    | MeshLib | 34.05 ms | 7.3x |
    | Open3D | 270.03 ms | 58x |
    | PyVista | 1.025 s | 219x |
    | libigl | 1.124 s | 240x |
    | MeshLab | 7.289 s | 1554x |
    | trimesh | 26.022 s | 5548x |
<!-- /bench:cell test_proximity/signed_distance_on_mesh -->

### Nearest neighbours, k = 7

Seven nearest neighbours of every query point, against CPU trees and PyTorch3D's GPU kernel.

<!-- bench:cell test_neighbors/query_nearest_bvh_k7 -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="query_nearest_bvh_k7 on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="104"/><text class="bench-tick" x="300.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="104"/><text class="bench-tick" x="404.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">100x</text><g><title>ordito (CUDA): 1.85 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 1.85 ms</text></g><g><title>PyTorch3D (CUDA): 44.53 ms, 24x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="247.6" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="25.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="448.6" y="38.0">24x · 44.53 ms</text></g><g><title>Open3D: 114.34 ms, 62x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="290.2" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="491.2" y="58.0">62x · 114.34 ms</text></g><g><title>SciPy: 144.94 ms, 78x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="301.0" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="65.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="78.0" text-anchor="end">SciPy</text><text class="bench-value" x="502.0" y="78.0">78x · 144.94 ms</text></g><g><title>libigl: out of memory (predicted)</title><path class="bench-failed" d="M198,90.0L206,98.0M206,90.0L198,98.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="212.0" y="98.0">out of memory (predicted)</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · a red cross is a run that left no time · [all neighbors benchmarks](neighbors.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.85 ms | 1x |
    | PyTorch3D (CUDA) | 44.53 ms | 24x |
    | Open3D | 114.34 ms | 62x |
    | SciPy | 144.94 ms | 78x |
    | libigl | out of memory (predicted) | |
<!-- /bench:cell test_neighbors/query_nearest_bvh_k7 -->

### Nearest neighbour, k = 1

The same query with a single neighbour. Every library with a tree, ordito included, builds it
over the cloud inside the timed call, and the build is most of the work: about two thirds of
ordito's time.

<!-- bench:cell test_neighbors/query_nearest_bvh_k1 -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="query_nearest_bvh_k1 on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="124"/><text class="bench-tick" x="274.0" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="124"/><text class="bench-tick" x="352.0" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="124"/><text class="bench-tick" x="430.0" y="137" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 628.0 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 628.0 µs</text></g><g><title>MeshLib: 14.00 ms, 22x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="183.2" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="384.2" y="38.0">22x · 14.00 ms</text></g><g><title>PyTorch3D (CUDA): 33.72 ms, 54x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="212.9" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="45.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="413.9" y="58.0">54x · 33.72 ms</text></g><g><title>Open3D: 113.52 ms, 181x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="254.1" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="455.1" y="78.0">181x · 113.52 ms</text></g><g><title>SciPy: 133.10 ms, 212x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="259.4" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="85.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="98.0" text-anchor="end">SciPy</text><text class="bench-value" x="460.4" y="98.0">212x · 133.10 ms</text></g><g><title>libigl: out of memory (predicted)</title><path class="bench-failed" d="M198,110.0L206,118.0M206,110.0L198,118.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">libigl</text><text class="bench-value" x="212.0" y="118.0">out of memory (predicted)</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · a red cross is a run that left no time · [all neighbors benchmarks](neighbors.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 628.0 µs | 1x |
    | MeshLib | 14.00 ms | 22x |
    | PyTorch3D (CUDA) | 33.72 ms | 54x |
    | Open3D | 113.52 ms | 181x |
    | SciPy | 133.10 ms | 212x |
    | libigl | out of memory (predicted) | |
<!-- /bench:cell test_neighbors/query_nearest_bvh_k1 -->

### Quadric decimation

Edge-collapse simplification under the quadric error metric, against five independent
decimators.

<!-- bench:cell test_remesh/quadric_decimate -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="quadric_decimate on saddle (34 848 faces) · target_ratio=0.5"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="124"/><text class="bench-tick" x="300.0" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="124"/><text class="bench-tick" x="404.0" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">100x</text><g><title>ordito (CUDA): 2.81 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 2.81 ms</text></g><g><title>MeshLib: 16.16 ms, 5.7x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="183.0" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="384.0" y="38.0">5.7x · 16.16 ms</text></g><g><title>PyVista: 44.68 ms, 16x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="228.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="51.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyVista</text><text class="bench-value" x="429.9" y="58.0">16x · 44.68 ms</text></g><g><title>Open3D: 45.03 ms, 16x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="229.3" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="430.3" y="78.0">16x · 45.03 ms</text></g><g><title>libigl: 52.29 ms, 19x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="236.0" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="437.0" y="98.0">19x · 52.29 ms</text></g><g><title>MeshLab: 254.90 ms, 91x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="307.6" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="105.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">MeshLab</text><text class="bench-value" x="508.6" y="118.0">91x · 254.90 ms</text></g></svg>
<figcaption>saddle (34 848 faces) · target_ratio=0.5 · time relative to ordito, log scale · [all remesh benchmarks](remesh.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 2.81 ms | 1x |
    | MeshLib | 16.16 ms | 5.7x |
    | PyVista | 44.68 ms | 16x |
    | Open3D | 45.03 ms | 16x |
    | libigl | 52.29 ms | 19x |
    | MeshLab | 254.90 ms | 91x |
<!-- /bench:cell test_remesh/quadric_decimate -->

### Mesh Booleans

Union of two closed spheres of 328 thousand faces each, rotated and shifted so their surfaces
cross along a long curve. manifold3d is Manifold itself, the library `ordito.boolean` ports, run on
all CPU cores; trimesh calls the same engine, and MeshLab is libigl's exact CGAL Boolean. Every
library returns a closed solid, and ordito returns Manifold's very mesh.

<!-- bench:cell test_boolean/boolean_union -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="boolean_union on sphere_large (327 680 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 7.88 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 7.88 ms</text></g><g><title>manifold3d: 31.20 ms, 4x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="124.6" height="10"/><text class="bench-label" x="166" y="38.0" text-anchor="end">manifold3d</text><text class="bench-value" x="325.6" y="38.0">4x · 31.20 ms</text></g><g><title>MeshLib: 33.32 ms, 4.2x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="126.8" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="47.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">MeshLib</text><text class="bench-value" x="327.8" y="58.0">4.2x · 33.32 ms</text></g><g><title>trimesh: 139.86 ms, 18x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="175.4" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="70.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="78.0" text-anchor="end">trimesh</text><text class="bench-value" x="376.4" y="78.0">18x · 139.86 ms</text></g><g><title>MeshLab: 2.529 s, 321x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="273.5" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="474.5" y="98.0">321x · 2.529 s</text></g></svg>
<figcaption>sphere_large (327 680 faces) · time relative to ordito, log scale · [all boolean benchmarks](boolean.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 7.88 ms | 1x |
    | manifold3d | 31.20 ms | 4x |
    | MeshLib | 33.32 ms | 4.2x |
    | trimesh | 139.86 ms | 18x |
    | MeshLab | 2.529 s | 321x |
<!-- /bench:cell test_boolean/boolean_union -->

### Minimal oriented bounding box

Every library here searches for the minimum-volume box, not the PCA box, so the bars compare
the same optimisation.

<!-- bench:cell test_bounds/oriented_bounding_box -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="oriented_bounding_box on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 1.22 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 1.22 ms</text></g><g><title>PyVista: 19.65 ms, 16x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="137.7" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="31.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyVista</text><text class="bench-value" x="338.7" y="38.0">16x · 19.65 ms</text></g><g><title>trimesh: 154.05 ms, 126x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="193.5" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="394.5" y="58.0">126x · 154.05 ms</text></g><g><title>Open3D: 2.035 s, 1667x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="263.4" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="464.4" y="78.0">1667x · 2.035 s</text></g><g><title>libigl: 2.378 s, 1947x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="267.7" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="468.7" y="98.0">1947x · 2.378 s</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all bounds benchmarks](bounds.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.22 ms | 1x |
    | PyVista | 19.65 ms | 16x |
    | trimesh | 154.05 ms | 126x |
    | Open3D | 2.035 s | 1667x |
    | libigl | 2.378 s | 1947x |
<!-- /bench:cell test_bounds/oriented_bounding_box -->

### Convex hull

<!-- bench:cell test_points/convex_hull -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="182" viewBox="0 0 640 182" role="img" aria-label="convex_hull on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="164"/><text class="bench-tick" x="196.0" y="177" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="164"/><text class="bench-tick" x="274.0" y="177" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="164"/><text class="bench-tick" x="352.0" y="177" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="164"/><text class="bench-tick" x="430.0" y="177" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="164"/><text class="bench-tick" x="508.0" y="177" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 1.82 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 1.82 ms</text></g><g><title>MeshLib: 21.56 ms, 12x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="161.7" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="362.7" y="38.0">12x · 21.56 ms</text></g><g><title>manifold3d: 24.63 ms, 14x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="166.2" height="10"/><text class="bench-label" x="166" y="58.0" text-anchor="end">manifold3d</text><text class="bench-value" x="367.2" y="58.0">14x · 24.63 ms</text></g><g><title>PyVista: 32.92 ms, 18x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="176.0" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="377.0" y="78.0">18x · 32.92 ms</text></g><g><title>Open3D: 48.34 ms, 27x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="189.0" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="86.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="98.0" text-anchor="end">Open3D</text><text class="bench-value" x="390.0" y="98.0">27x · 48.34 ms</text></g><g><title>SciPy: 59.10 ms, 32x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="195.9" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="105.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="118.0" text-anchor="end">SciPy</text><text class="bench-value" x="396.9" y="118.0">32x · 59.10 ms</text></g><g><title>trimesh: 77.18 ms, 42x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="129.0" width="204.9" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="130.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="138.0" text-anchor="end">trimesh</text><text class="bench-value" x="405.9" y="138.0">42x · 77.18 ms</text></g><g><title>MeshLab: 399.80 ms, 219x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="149.0" width="260.6" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="145.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="158.0" text-anchor="end">MeshLab</text><text class="bench-value" x="461.6" y="158.0">219x · 399.80 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all points benchmarks](points.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.82 ms | 1x |
    | MeshLib | 21.56 ms | 12x |
    | manifold3d | 24.63 ms | 14x |
    | PyVista | 32.92 ms | 18x |
    | Open3D | 48.34 ms | 27x |
    | SciPy | 59.10 ms | 32x |
    | trimesh | 77.18 ms | 42x |
    | MeshLab | 399.80 ms | 219x |
<!-- /bench:cell test_points/convex_hull -->

### Orienting point-cloud normals

Consistent normal orientation by minimum-spanning-tree propagation, on the dragon scan's vertices
as a bare cloud: the step before Poisson or ball-pivoting reconstruction. Open3D's
`orient_normals_consistent_tangent_plane` runs the same propagation; on lucy
it ran past the ten-minute limit.

<!-- bench:cell test_points/orient_normals -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="62" viewBox="0 0 640 62" role="img" aria-label="orient_normals on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="44"/><text class="bench-tick" x="196.0" y="57" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="44"/><text class="bench-tick" x="258.4" y="57" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="44"/><text class="bench-tick" x="320.8" y="57" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="44"/><text class="bench-tick" x="383.2" y="57" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="44"/><text class="bench-tick" x="445.6" y="57" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="44"/><text class="bench-tick" x="508.0" y="57" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 2.21 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 2.21 ms</text></g><g><title>Open3D: 15.214 s, 6880x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="301.9" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="26.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="38.0" text-anchor="end">Open3D</text><text class="bench-value" x="502.9" y="38.0">6880x · 15.214 s</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all points benchmarks](points.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 2.21 ms | 1x |
    | Open3D | 15.214 s | 6880x |
<!-- /bench:cell test_points/orient_normals -->

### Blue-noise sampling

Poisson-disk sampling of a surface: four different algorithms for the same distribution.

<!-- bench:cell test_sample/blue_noise -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="blue_noise on dragon (871 414 faces) · radius_scale=1.0"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 1.20 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 1.20 ms</text></g><g><title>MeshLib: 12.42 ms, 10x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="157.2" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="358.2" y="38.0">10x · 12.42 ms</text></g><g><title>Open3D: 44.05 ms, 37x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="200.1" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="401.1" y="58.0">37x · 44.05 ms</text></g><g><title>libigl: 255.55 ms, 213x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="259.6" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="460.6" y="78.0">213x · 255.55 ms</text></g><g><title>MeshLab: 449.87 ms, 375x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="278.8" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="479.8" y="98.0">375x · 449.87 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · radius_scale=1.0 · time relative to ordito, log scale · [all sample benchmarks](sample.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.20 ms | 1x |
    | MeshLib | 12.42 ms | 10x |
    | Open3D | 44.05 ms | 37x |
    | libigl | 255.55 ms | 213x |
    | MeshLab | 449.87 ms | 375x |
<!-- /bench:cell test_sample/blue_noise -->

### Geodesic distance

Geodesic distance from a source vertex by the heat method, including building the solver.

<!-- bench:cell test_heat/heat_geodesic -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="heat_geodesic on sphere_med (81 920 faces) · setup=full"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="104"/><text class="bench-tick" x="300.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="104"/><text class="bench-tick" x="404.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">100x</text><g><title>ordito (CUDA): 6.60 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 6.60 ms</text></g><g><title>PyVista: 24.76 ms, 3.8x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="163.7" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="31.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyVista</text><text class="bench-value" x="364.7" y="38.0">3.8x · 24.76 ms</text></g><g><title>potpourri3d: 520.51 ms, 79x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="301.3" height="10"/><text class="bench-label" x="166" y="58.0" text-anchor="end">potpourri3d</text><text class="bench-value" x="502.3" y="58.0">79x · 520.51 ms</text></g><g><title>MeshLab: 568.74 ms, 86x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="305.3" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="65.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">MeshLab</text><text class="bench-value" x="506.3" y="78.0">86x · 568.74 ms</text></g><g><title>libigl: 585.77 ms, 89x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="306.6" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="507.6" y="98.0">89x · 585.77 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · setup=full · time relative to ordito, log scale · [all heat benchmarks](heat.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 6.60 ms | 1x |
    | PyVista | 24.76 ms | 3.8x |
    | potpourri3d | 520.51 ms | 79x |
    | MeshLab | 568.74 ms | 86x |
    | libigl | 585.77 ms | 89x |
<!-- /bench:cell test_heat/heat_geodesic -->

### Geodesic distance on a point cloud

The heat method on a bare point cloud, geometry-central's point-cloud solver: local
triangulations, their tufted cover, and two sparse solves, built and solved in one call.

<!-- bench:cell test_heat/point_cloud_heat_distance -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="62" viewBox="0 0 640 62" role="img" aria-label="point_cloud_heat_distance on sphere_large (327 680 faces) · setup=full"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="44"/><text class="bench-tick" x="196.0" y="57" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="44"/><text class="bench-tick" x="274.0" y="57" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="44"/><text class="bench-tick" x="352.0" y="57" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="44"/><text class="bench-tick" x="430.0" y="57" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="44"/><text class="bench-tick" x="508.0" y="57" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 30.74 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 30.74 ms</text></g><g><title>potpourri3d: 9.267 s, 301x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="271.4" height="10"/><text class="bench-label" x="166" y="38.0" text-anchor="end">potpourri3d</text><text class="bench-value" x="472.4" y="38.0">301x · 9.267 s</text></g></svg>
<figcaption>sphere_large (327 680 faces) · setup=full · time relative to ordito, log scale · [all heat benchmarks](heat.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 30.74 ms | 1x |
    | potpourri3d | 9.267 s | 301x |
<!-- /bench:cell test_heat/point_cloud_heat_distance -->

### Repair: merging duplicate vertices

<!-- bench:cell test_repair/remove_duplicated_vertices -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="182" viewBox="0 0 640 182" role="img" aria-label="remove_duplicated_vertices on sphere_med (81 920 faces) · epsilon=0.0"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="164"/><text class="bench-tick" x="196.0" y="177" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="164"/><text class="bench-tick" x="274.0" y="177" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="164"/><text class="bench-tick" x="352.0" y="177" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="164"/><text class="bench-tick" x="430.0" y="177" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="164"/><text class="bench-tick" x="508.0" y="177" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 233.3 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 233.3 µs</text></g><g><title>Open3D: 15.09 ms, 65x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="219.2" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="26.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="38.0" text-anchor="end">Open3D</text><text class="bench-value" x="420.2" y="38.0">65x · 15.09 ms</text></g><g><title>MeshLib: 24.05 ms, 103x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="235.0" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="47.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">MeshLib</text><text class="bench-value" x="436.0" y="58.0">103x · 24.05 ms</text></g><g><title>PyVista: 24.66 ms, 106x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="235.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="436.9" y="78.0">106x · 24.66 ms</text></g><g><title>libigl: 30.24 ms, 130x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="242.8" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="443.8" y="98.0">130x · 30.24 ms</text></g><g><title>trimesh: 47.65 ms, 204x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="258.2" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="110.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="118.0" text-anchor="end">trimesh</text><text class="bench-value" x="459.2" y="118.0">204x · 47.65 ms</text></g><g><title>manifold3d: 78.92 ms, 338x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="129.0" width="275.3" height="10"/><text class="bench-label" x="166" y="138.0" text-anchor="end">manifold3d</text><text class="bench-value" x="476.3" y="138.0">338x · 78.92 ms</text></g><g><title>MeshLab: 210.88 ms, 904x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="149.0" width="308.6" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="145.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="158.0" text-anchor="end">MeshLab</text><text class="bench-value" x="509.6" y="158.0">904x · 210.88 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · epsilon=0.0 · time relative to ordito, log scale · [all repair benchmarks](repair.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 233.3 µs | 1x |
    | Open3D | 15.09 ms | 65x |
    | MeshLib | 24.05 ms | 103x |
    | PyVista | 24.66 ms | 106x |
    | libigl | 30.24 ms | 130x |
    | trimesh | 47.65 ms | 204x |
    | manifold3d | 78.92 ms | 338x |
    | MeshLab | 210.88 ms | 904x |
<!-- /bench:cell test_repair/remove_duplicated_vertices -->

### Repair: consistent winding

Re-orienting every face to agree with its neighbours, a whole-mesh propagation.

<!-- bench:cell test_repair/make_winding_consistent -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="162" viewBox="0 0 640 162" role="img" aria-label="make_winding_consistent on sphere_med (81 920 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="144"/><text class="bench-tick" x="196.0" y="157" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="144"/><text class="bench-tick" x="274.0" y="157" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="144"/><text class="bench-tick" x="352.0" y="157" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="144"/><text class="bench-tick" x="430.0" y="157" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="144"/><text class="bench-tick" x="508.0" y="157" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 140.7 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 140.7 µs</text></g><g><title>MeshLib: 6.03 ms, 43x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="205.3" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="406.3" y="38.0">43x · 6.03 ms</text></g><g><title>PyVista: 13.81 ms, 98x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="233.4" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="51.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyVista</text><text class="bench-value" x="434.4" y="58.0">98x · 13.81 ms</text></g><g><title>trimesh: 18.19 ms, 129x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="242.7" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="70.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="78.0" text-anchor="end">trimesh</text><text class="bench-value" x="443.7" y="78.0">129x · 18.19 ms</text></g><g><title>libigl: 35.13 ms, 250x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="265.0" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="466.0" y="98.0">250x · 35.13 ms</text></g><g><title>MeshLab: 50.13 ms, 356x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="277.0" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="105.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">MeshLab</text><text class="bench-value" x="478.0" y="118.0">356x · 50.13 ms</text></g><g><title>Open3D: 90.67 ms, 644x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="129.0" width="297.1" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="126.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="138.0" text-anchor="end">Open3D</text><text class="bench-value" x="498.1" y="138.0">644x · 90.67 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · time relative to ordito, log scale · [all repair benchmarks](repair.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 140.7 µs | 1x |
    | MeshLib | 6.03 ms | 43x |
    | PyVista | 13.81 ms | 98x |
    | trimesh | 18.19 ms | 129x |
    | libigl | 35.13 ms | 250x |
    | MeshLab | 50.13 ms | 356x |
    | Open3D | 90.67 ms | 644x |
<!-- /bench:cell test_repair/make_winding_consistent -->

### Connected components

<!-- bench:cell test_adjacency/face_connected_component_labels -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="face_connected_component_labels on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 239.2 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 239.2 µs</text></g><g><title>MeshLib: 21.42 ms, 90x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="184.2" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="385.2" y="38.0">90x · 21.42 ms</text></g><g><title>PyVista: 269.72 ms, 1128x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="252.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="51.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyVista</text><text class="bench-value" x="453.9" y="58.0">1128x · 269.72 ms</text></g><g><title>libigl: 405.56 ms, 1696x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="263.9" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="464.9" y="78.0">1696x · 405.56 ms</text></g><g><title>Open3D: 942.26 ms, 3940x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="286.8" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="86.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="98.0" text-anchor="end">Open3D</text><text class="bench-value" x="487.8" y="98.0">3940x · 942.26 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all adjacency benchmarks](adjacency.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 239.2 µs | 1x |
    | MeshLib | 21.42 ms | 90x |
    | PyVista | 269.72 ms | 1128x |
    | libigl | 405.56 ms | 1696x |
    | Open3D | 942.26 ms | 3940x |
<!-- /bench:cell test_adjacency/face_connected_component_labels -->

### Unique edges

The undirected edge list, the first step of most topology code.

<!-- bench:cell test_edges/edges_unique -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="edges_unique on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 455.7 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 455.7 µs</text></g><g><title>PyTorch3D (CUDA): 3.57 ms, 7.8x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="147.7" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="25.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="348.7" y="38.0">7.8x · 3.57 ms</text></g><g><title>trimesh: 232.28 ms, 510x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="289.2" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="490.2" y="58.0">510x · 232.28 ms</text></g><g><title>PyVista: 250.28 ms, 549x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="291.7" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="492.7" y="78.0">549x · 250.28 ms</text></g><g><title>libigl: 327.63 ms, 719x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="300.8" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="501.8" y="98.0">719x · 327.63 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all edges benchmarks](edges.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 455.7 µs | 1x |
    | PyTorch3D (CUDA) | 3.57 ms | 7.8x |
    | trimesh | 232.28 ms | 510x |
    | PyVista | 250.28 ms | 549x |
    | libigl | 327.63 ms | 719x |
<!-- /bench:cell test_edges/edges_unique -->

### Taubin smoothing

<!-- bench:cell test_smoothing/filter_taubin -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="102" viewBox="0 0 640 102" role="img" aria-label="filter_taubin on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="84"/><text class="bench-tick" x="196.0" y="97" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="84"/><text class="bench-tick" x="258.4" y="97" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="84"/><text class="bench-tick" x="320.8" y="97" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="84"/><text class="bench-tick" x="383.2" y="97" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="84"/><text class="bench-tick" x="445.6" y="97" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="84"/><text class="bench-tick" x="508.0" y="97" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 575.4 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 575.4 µs</text></g><g><title>trimesh: 192.15 ms, 334x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="219.9" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="30.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="38.0" text-anchor="end">trimesh</text><text class="bench-value" x="420.9" y="38.0">334x · 192.15 ms</text></g><g><title>Open3D: 298.67 ms, 519x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="231.8" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="432.8" y="58.0">519x · 298.67 ms</text></g><g><title>MeshLab: 836.49 ms, 1454x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="259.7" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="65.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">MeshLab</text><text class="bench-value" x="460.7" y="78.0">1454x · 836.49 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all smoothing benchmarks](smoothing.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 575.4 µs | 1x |
    | trimesh | 192.15 ms | 334x |
    | Open3D | 298.67 ms | 519x |
    | MeshLab | 836.49 ms | 1454x |
<!-- /bench:cell test_smoothing/filter_taubin -->

### Voxelizing a mesh

<!-- bench:cell test_voxels/voxelize_mesh -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="voxelize_mesh on dragon (871 414 faces) · divisor=64"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 419.3 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 419.3 µs</text></g><g><title>MeshLib: 106.97 ms, 255x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="212.6" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="413.6" y="38.0">255x · 106.97 ms</text></g><g><title>Open3D: 156.97 ms, 374x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="223.0" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="424.0" y="58.0">374x · 156.97 ms</text></g><g><title>PyVista: 264.49 ms, 631x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="237.1" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="438.1" y="78.0">631x · 264.49 ms</text></g><g><title>trimesh: 656.12 ms, 1565x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="261.7" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="90.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="98.0" text-anchor="end">trimesh</text><text class="bench-value" x="462.7" y="98.0">1565x · 656.12 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · divisor=64 · time relative to ordito, log scale · [all voxels benchmarks](voxels.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 419.3 µs | 1x |
    | MeshLib | 106.97 ms | 255x |
    | Open3D | 156.97 ms | 374x |
    | PyVista | 264.49 ms | 631x |
    | trimesh | 656.12 ms | 1565x |
<!-- /bench:cell test_voxels/voxelize_mesh -->

### Boundary edges, a loss

Listing the edges on a mesh's boundary is a single pass that serial C++ finishes in less time than
a GPU round trip takes. ordito loses this one, as it does most operations whose whole cost is a few
tens of microseconds; the module pages show the rest.

<!-- bench:cell test_boundary/boundary_edges -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="boundary_edges on rim_long (131 072 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.01x</text><line class="bench-grid" x1="258.4" y1="0" x2="258.4" y2="124"/><text class="bench-tick" x="258.4" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="320.8" y1="0" x2="320.8" y2="124"/><text class="bench-tick" x="320.8" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="124"/><text class="bench-tick" x="383.2" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="124"/><text class="bench-tick" x="445.6" y="137" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">1000x</text><g><title>MeshLib: 47.0 µs, 0.31x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="9.0" width="93.3" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="7.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end">MeshLib</text><text class="bench-value" x="294.3" y="18.0">0.31x · 47.0 µs</text></g><g><title>ordito (CUDA): 150.5 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="29.0" width="124.8" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="25.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="325.8" y="38.0">1x · 150.5 µs</text></g><g><title>trimesh: 18.34 ms, 122x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="255.0" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="456.0" y="58.0">122x · 18.34 ms</text></g><g><title>MeshLab: 41.15 ms, 273x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="276.9" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="65.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">MeshLab</text><text class="bench-value" x="477.9" y="78.0">273x · 41.15 ms</text></g><g><title>libigl: 41.38 ms, 275x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="277.0" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="478.0" y="98.0">275x · 41.38 ms</text></g><g><title>PyVista: 54.34 ms, 361x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="284.4" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="111.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">PyVista</text><text class="bench-value" x="485.4" y="118.0">361x · 54.34 ms</text></g></svg>
<figcaption>rim_long (131 072 faces) · time relative to ordito, log scale · [all boundary benchmarks](boundary.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | MeshLib | 47.0 µs | 0.31x |
    | ordito (CUDA) | 150.5 µs | 1x |
    | trimesh | 18.34 ms | 122x |
    | MeshLab | 41.15 ms | 273x |
    | libigl | 41.38 ms | 275x |
    | PyVista | 54.34 ms | 361x |
<!-- /bench:cell test_boundary/boundary_edges -->

## Reproduce

The suite lives in [`benchmarks/`](https://github.com/lucagrementieri/ordito/tree/main/benchmarks);
its README describes the inputs and every comparison in detail. It needs Linux, an NVIDIA GPU and
the CUDA toolkit:

```bash
git clone https://github.com/lucagrementieri/ordito
cd ordito
uv sync --group bench --no-install-package pytorch3d
uv sync --group bench

mkdir -p results
for f in benchmarks/test_*.py; do
  uv run pytest "$f" --device=cuda --bench-all-libs \
    --benchmark-json="results/$(basename "$f" .py).json" -q
done

uv run python -m benchmarks.tools.render_docs results
```

The last command redraws every chart on this page and on the module pages from your own run.
Absolute times depend on the hardware; the ratios are what carries over.

A ratio holds only for the library, the input and the hardware it was measured on: besides its
size, a mesh's number of components, boundary length and triangle shapes move it. A few functions
have little parallel work at realistic sizes (a sequential pass over one small boundary loop), and
a fast single-threaded C++ implementation beats them; the module pages show those cases beside
the others.
