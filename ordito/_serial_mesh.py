"""
Serial half-edge mesh editing on the host, for the ``method="meshlib"`` hole patching options.

The reference patching that [`holes`][ordito.holes] and
[`smooth_region_boundary`][ordito.smoothing.smooth_region_boundary] reproduce is a chain of
serial decisions -- an interval DP over a rim, a priority queue of edge splits whose ties fall to
edge ids, a fan of Delaunay flips around each new vertex -- whose answer depends on the order the
reference numbers and orients its edges. This module keeps a half-edge structure numbered exactly
as that order: undirected edge ``k`` is the half-edges ``2 k`` and ``2 k + 1``, each record holds
``next`` / ``prev`` around its origin, its origin vertex and its left face, and every operation
(the builder, ``splice``, an edge split or flip, a face deletion) allocates and relinks records in
the same order, so the ids a later tie reads are the ones the reference would read. Arithmetic is
``float32`` in the reference's operation order wherever it rounds.

It is host Python: a split or flip is a dependent pointer walk, so only the builder's corner
numbering and each DP diagonal are batched (NumPy), and the device receives the finished mesh.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np
import warp as wp

# The builder partitions a triangle list into vertex ranges of at most this many triangles a
# part, and at most this many parts, when it holds more than one part's worth.
_TRIANGLES_PER_PART = 32768
_MAX_PARTS = 64


class _VertexRows:
    """
    Each vertex's weights over input vertices: its own vertex for the first ones, unless replaced.

    Rows are read and replaced whole, never edited in place.
    """

    def __init__(self, n_identity: int, n_rows: int) -> None:
        self.n_identity = n_identity
        self.n_rows = n_rows
        self.replaced: dict[int, dict[int, float]] = {}

    def __len__(self) -> int:
        return self.n_rows

    def __getitem__(self, v: int) -> dict[int, float]:
        row = self.replaced.get(v)
        if row is not None:
            return row
        return {v: 1.0} if v < self.n_identity else {}

    def __setitem__(self, v: int, row: dict[int, float]) -> None:
        self.replaced[v] = row

    def append(self, row: dict[int, float]) -> None:
        """Add one row."""
        self.replaced[self.n_rows] = row
        self.n_rows += 1

    def extend_to(self, n_rows: int) -> None:
        """Add empty rows up to ``n_rows``."""
        self.n_rows = max(self.n_rows, n_rows)

    def table(self, kept: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(len(kept), width)`` sources (``-1`` padded) and weights of rows ``kept``."""
        replaced = {r: self.replaced[v] for r, v in enumerate(kept.tolist()) if v in self.replaced}
        width = max(1, max((len(row) for row in replaced.values()), default=1))
        sources = np.full((kept.size, width), -1, dtype=np.int32)
        weights = np.zeros((kept.size, width), dtype=np.float32)
        own = kept < self.n_identity
        sources[own, 0] = kept[own]
        weights[own, 0] = 1.0
        for r, row in replaced.items():
            sources[r] = -1
            weights[r] = 0.0
            for k, (v, w) in enumerate(sorted(row.items())):
                sources[r, k] = v
                weights[r, k] = w
        return sources, weights


class Topology:
    """
    A half-edge mesh connectivity whose element ids follow the serial reference allocation.

    Attributes
    ----------
    nxt, prv
        Per half-edge, the next / previous half-edge counter-clockwise around its origin.
    org, lft
        Per half-edge, the origin vertex and the left face, ``-1`` when none.
    edge_per_vertex, edge_per_face
        Per vertex / face, one half-edge with that origin / left face, ``-1`` when invalid.
    """

    __slots__ = (
        "_arrays",
        "edge_per_face",
        "edge_per_vertex",
        "face_source",
        "lft",
        "nxt",
        "org",
        "prv",
        "vertex_rows",
    )

    def __init__(self) -> None:
        # A field is held in one form at a time: here as an array (after the builder or a native
        # kernel), or as its list once Python code reads a field (``_HeldTopology``).
        self._arrays: dict[str, np.ndarray] = {}
        self.nxt: list[int] = []
        self.prv: list[int] = []
        self.org: list[int] = []
        self.lft: list[int] = []
        self.edge_per_vertex: list[int] = []
        self.edge_per_face: list[int] = []
        # Provenance, when tracked: each vertex's weights over input vertices and each face's
        # input face (``-1`` for none).
        self.vertex_rows: _VertexRows | None = None
        self.face_source: list[int] | None = None

    def reader(self) -> _Reader:
        """Return the fields for a read-only walk, converting none."""
        return _Reader(self)

    def array(self, name: str) -> np.ndarray:
        """Field ``name`` as an integer array to read; a list field stays a list."""
        held = self._arrays.get(name)
        if held is not None:
            return held
        return np.asarray(getattr(self, name), dtype=np.int64)

    def set_array(self, name: str, values: np.ndarray) -> None:
        """Hold field ``name`` as ``values`` until Python code next reads a field."""
        self._arrays[name] = values
        try:
            object.__delattr__(self, name)
        except AttributeError:
            pass
        # The hooks live on a subclass so that a topology holding only lists keeps plain slot
        # loads: a Python ``__getattr__`` stops CPython specializing every attribute load of its
        # class (the list walks ran 1.06-1.10x slower).
        object.__setattr__(self, "__class__", _HeldTopology)

    def size(self, name: str) -> int:
        """Length of field ``name``, whichever form holds it."""
        held = self._arrays.get(name)
        return held.shape[0] if held is not None else len(getattr(self, name))

    def vertex_count(self) -> int:
        """Return the number of vertex ids, invalid ones included."""
        return self.size("edge_per_vertex")

    def face_count(self) -> int:
        """Return the number of face ids, invalid ones included."""
        return self.size("edge_per_face")

    def track(self, n_input_vertices: int) -> None:
        """Start recording where every vertex and face came from, the current ones their own."""
        n_verts = self.vertex_count()
        self.vertex_rows = _VertexRows(n_verts, max(n_verts, n_input_vertices))
        self.face_source = list(range(self.face_count()))

    # -- element allocation ------------------------------------------------------------------

    def make_edge(self) -> int:
        """Append one lone undirected edge and return its even half-edge."""
        e = len(self.nxt)
        self.nxt.extend((e, e + 1))
        self.prv.extend((e, e + 1))
        self.org.extend((-1, -1))
        self.lft.extend((-1, -1))
        return e

    def add_vert(self) -> int:
        """Append one vertex id, invalid until an edge takes it as origin."""
        self.edge_per_vertex.append(-1)
        if self.vertex_rows is not None:
            self.vertex_rows.append({})
        return len(self.edge_per_vertex) - 1

    def add_face(self) -> int:
        """Append one face id, invalid until an edge takes it as left face."""
        self.edge_per_face.append(-1)
        if self.face_source is not None:
            self.face_source.append(-1)
        return len(self.edge_per_face) - 1

    def edge_count(self) -> int:
        """Return the number of half-edges, lone ones included."""
        return self.size("nxt")

    # -- navigation --------------------------------------------------------------------------

    def org_ring(self, e: int) -> Iterator[int]:
        """Half-edges leaving ``org(e)``, counter-clockwise from ``e``."""
        i = e
        while True:
            yield i
            i = self.nxt[i]
            if i == e:
                return

    def left_ring(self, e: int) -> Iterator[int]:
        """Half-edges around ``left(e)`` from ``e``, each the previous one's successor there."""
        i = e
        while True:
            yield i
            i = self.prv[i ^ 1]
            if i == e:
                return

    def is_left_tri(self, e: int) -> bool:
        """Report whether the left of ``e`` closes after exactly three half-edges."""
        a = self.prv[e ^ 1]
        if a == e:
            return False
        b = self.prv[a ^ 1]
        if b == e:
            return False
        return self.prv[b ^ 1] == e

    def find_edge(self, o: int, d: int) -> int:
        """Find a half-edge from ``o`` to ``d`` around ``o`` from its stored edge, else ``-1``."""
        e0 = self.edge_per_vertex[o] if o < self.vertex_count() else -1
        if e0 < 0:
            return -1
        e = e0
        while True:
            if self.org[e ^ 1] == d:
                return e
            e = self.nxt[e]
            if e == e0:
                return -1

    def tri_verts(self, e: int) -> tuple[int, int, int]:
        """Return the left triangle's vertices of ``e``: ``org(e)``, ``dest(e)``, the third."""
        return self.org[e], self.org[e ^ 1], self.org[self.prv[e ^ 1] ^ 1]

    # -- linking -----------------------------------------------------------------------------

    def _set_org_ring(self, a: int, v: int) -> None:
        for i in self.org_ring(a):
            self.org[i] = v

    def _set_left_ring(self, a: int, f: int) -> None:
        for i in self.left_ring(a):
            self.lft[i] = f

    def set_org(self, a: int, v: int) -> None:
        """Set the origin of the ring of ``a`` to ``v``, updating both vertices' stored edges."""
        old = self.org[a]
        if v == old:
            return
        self._set_org_ring(a, v)
        if old >= 0:
            self.edge_per_vertex[old] = -1
        if v >= 0:
            self.edge_per_vertex[v] = a

    def set_left(self, a: int, f: int) -> None:
        """Set the left face of the ring of ``a`` to ``f``, updating both faces' stored edges."""
        old = self.lft[a]
        if f == old:
            return
        self._set_left_ring(a, f)
        if old >= 0:
            self.edge_per_face[old] = -1
        if f >= 0:
            self.edge_per_face[f] = a

    def _same_org_ring(self, a: int, b: int) -> bool:
        return any(i == b for i in self.org_ring(a))

    def _same_left_ring(self, a: int, b: int) -> bool:
        return any(i == b for i in self.left_ring(a))

    def splice(self, a: int, b: int) -> None:
        """Join or split the origin rings (and left rings) of ``a`` and ``b``."""
        if a == b:
            return
        nxt, prv, org, lft = self.nxt, self.prv, self.org, self.lft
        a_next = nxt[a]
        b_next = nxt[b]
        same_org = org[a] == org[b]
        same_left = lft[a] == lft[b]
        if not same_org:
            if org[a] >= 0:
                self._set_org_ring(b, org[a])
            elif org[b] >= 0:
                self._set_org_ring(a, org[b])
        if not same_left:
            if lft[a] >= 0:
                self._set_left_ring(b, lft[a])
            elif lft[b] >= 0:
                self._set_left_ring(a, lft[b])
        nxt[a], nxt[b] = b_next, a_next
        prv[a_next], prv[b_next] = prv[b_next], prv[a_next]
        if same_org and org[b] >= 0:
            self._set_org_ring(b, -1)
            v = org[a]
            if not self._same_org_ring(self.edge_per_vertex[v], a):
                self.edge_per_vertex[v] = a
        if same_left and lft[b] >= 0:
            self._set_left_ring(b, -1)
            f = lft[a]
            if not self._same_left_ring(self.edge_per_face[f], a):
                self.edge_per_face[f] = a

    # -- editing -----------------------------------------------------------------------------

    def split_edge(self, e: int, region: set[int] | None = None) -> int:
        """
        Split ``e`` at a new vertex and its two triangles each in two.

        Afterwards ``e`` runs from the new vertex to its old destination and the returned
        half-edge from its old origin to the new vertex; a new face cut from one in ``region``
        joins it.
        """
        lf = self.lft[e]
        if lf >= 0:
            self._set_left_ring(e, -1)
        rf = self.lft[e ^ 1]
        if rf >= 0:
            self._set_left_ring(e ^ 1, -1)
        ends = (self.org[e], self.org[e ^ 1])
        e_prev = self.prv[e]
        e_next = self.nxt[e]
        v0 = -1
        if e_prev != e:
            self.splice(e_prev, e)
        else:
            v0 = self.org[e]
            self.set_org(e, -1)
        e0 = self.make_edge()
        self.splice(e, e0 ^ 1)
        if e_prev != e:
            self.splice(e_prev, e0)
        else:
            self.set_org(e0, v0)
        if lf >= 0 and e_next != e:
            el = self.make_edge()
            self.splice(e, el)
            self.splice(e_next ^ 1, el ^ 1)
            new_face = self.add_face()
            self.set_left(el, new_face)
            if region is not None and lf in region:
                region.add(new_face)
            if self.face_source is not None:
                self.face_source[new_face] = self.face_source[lf]
        if rf >= 0 and e_prev != e:
            er = self.make_edge()
            self.splice(e0 ^ 1, er)
            self.splice(self.prv[e_prev ^ 1], er ^ 1)
            new_face = self.add_face()
            self.set_left(er ^ 1, new_face)
            if region is not None and rf in region:
                region.add(new_face)
            if self.face_source is not None:
                self.face_source[new_face] = self.face_source[rf]
        self._set_left_ring(e, lf)
        self._set_left_ring(e ^ 1, rf)
        if lf >= 0:
            self.edge_per_face[lf] = e
        if rf >= 0:
            self.edge_per_face[rf] = e ^ 1
        new_vert = self.add_vert()
        self.set_org(e, new_vert)
        if self.vertex_rows is not None:
            row: dict[int, float] = {}
            for end in ends:
                for v, w in self.vertex_rows[end].items():
                    row[v] = row.get(v, 0.0) + 0.5 * w
            self.vertex_rows[new_vert] = row
        return e0

    # -- export ------------------------------------------------------------------------------

    def triangles(self) -> np.ndarray:
        """``(n_faces, 3)`` vertex triples of every face id, ``-1`` rows for invalid ones."""
        edges = self.array("edge_per_face").astype(np.int64)
        out = np.full((edges.size, 3), -1, dtype=np.int32)
        valid = edges >= 0
        e = edges[valid]
        org = self.array("org")
        prv = self.array("prv").astype(np.int64)
        out[valid] = np.stack((org[e], org[e ^ 1], org[prv[e ^ 1] ^ 1]), axis=1)
        return out


class _HeldTopology(Topology):
    """A [`Topology`][ordito._serial_mesh.Topology] holding some fields as arrays."""

    __slots__ = ()

    def __getattr__(self, name: str) -> list[int]:
        # Reached only for a field held as an array: every held field becomes its list, which
        # Python code edits, and the topology a plain one again.
        arrays = object.__getattribute__(self, "_arrays")
        if name not in arrays:
            raise AttributeError(name)
        for field, values in arrays.items():
            object.__setattr__(self, field, values.tolist())
        arrays.clear()
        object.__setattr__(self, "__class__", Topology)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        # Assigning a field replaces its held array.
        arrays = object.__getattribute__(self, "_arrays")
        arrays.pop(name, None)
        object.__setattr__(self, name, value)
        if not arrays:
            object.__setattr__(self, "__class__", Topology)


class _Reader:
    """
    A topology's fields for a read-only walk: its lists, or views of the arrays it holds.

    Indexing a view returns a Python ``int``, as a list does, so a short walk on a large mesh
    reads a held field where it lies instead of converting every field to lists.
    """

    __slots__ = ("edge_per_vertex", "lft", "nxt", "org", "prv")
    edge_per_vertex: Sequence[int]
    lft: Sequence[int]
    nxt: Sequence[int]
    org: Sequence[int]
    prv: Sequence[int]

    def __init__(self, topology: Topology) -> None:
        arrays = topology._arrays  # pyright: ignore[reportPrivateUsage]
        for name in self.__slots__:
            held = arrays.get(name)
            setattr(self, name, getattr(topology, name) if held is None else held.data)

    def org_ring(self, e: int) -> Iterator[int]:
        """Half-edges leaving ``org(e)``, counter-clockwise from ``e``."""
        i = e
        while True:
            yield i
            i = self.nxt[i]
            if i == e:
                return


# A walk that only reads takes either.
_Walk = Topology | _Reader


# ---------------------------------------------------------------------------------------------
# Building from a triangle list
# ---------------------------------------------------------------------------------------------


def build(faces: np.ndarray, n_vertices: int) -> Topology:
    """
    Build the connectivity of ``(n_faces, 3)`` triangles in the reference numbering.

    From ``_MIN_CORNER_BUILD`` triangles, a manifold triangle list (every edge in at most two
    triangles, opposite ways, and every vertex one fan or at most two open ones) is numbered by
    corners: an edge belongs to the lower of its two corners and edges are numbered in corner
    order. Anything else is built face by face (``_build_face_by_face``).
    """
    tris_np = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if tris_np.shape[0] == 0:
        return Topology()
    if tris_np.shape[0] >= _MIN_CORNER_BUILD:
        built = _build_by_corners(tris_np)
        if built is not None:
            return built
    return _build_face_by_face(tris_np, n_vertices)


# Below this many triangles the face-at-a-time build is the reference's choice.
_MIN_CORNER_BUILD = 8192


def _build_by_corners(tris: np.ndarray) -> Topology | None:
    """Build a manifold triangle list numbered by corners; ``None`` when it is not manifold."""
    n_faces = tris.shape[0]
    bad = (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
    bad |= (tris < 0).any(axis=1)
    if bad.all():
        return None
    n_verts = int(tris[~bad].max()) + 1
    corner_v = tris.reshape(-1)
    corner_n = np.roll(tris, -1, axis=1).reshape(-1)
    live = np.repeat(~bad, 3)
    corners = np.flatnonzero(live)
    # A corner's half-edge runs from its vertex to the next one; its twin runs back.
    keys = corner_v[corners] * n_verts + corner_n[corners]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        return None  # an edge used twice the same way, or by three or more triangles
    back = corner_n[corners] * n_verts + corner_v[corners]
    where = np.searchsorted(sorted_keys, back)
    where = np.minimum(where, sorted_keys.size - 1)
    found = sorted_keys[where] == back
    twin = np.arange(3 * n_faces, dtype=np.int64)  # a skipped triangle's corners are their own
    twin[corners] = np.where(found, corners[order[where]], -1)

    # Corners by vertex, each vertex's corners in corner order.
    by_vertex = np.argsort(corner_v[corners], kind="stable")
    vc = corners[by_vertex]
    starts = np.searchsorted(corner_v[vc], np.arange(n_verts + 1))
    prev_corner = vc - vc % 3 + (vc + 2) % 3
    # Each vertex's fans: the corner after ``c`` around its vertex is the one whose half-edge
    # runs back along the edge entering ``c``, i.e. the twin of ``c``'s previous corner.
    step = np.full(3 * n_faces, -1, dtype=np.int64)
    step[vc] = twin[prev_corner]
    is_start = np.zeros(3 * n_faces, dtype=bool)
    is_start[vc] = twin[vc] < 0  # the half-edge leaving c has no twin: c opens a fan
    next_fan_start = np.full(3 * n_faces, -1, dtype=np.int64)
    min_corner = np.full(n_verts, -1, dtype=np.int64)
    has_corners = starts[1:] > starts[:-1]
    min_corner[has_corners] = vc[starts[:-1][has_corners]]
    boundary_vertex = np.zeros(n_verts, dtype=bool)
    boundary_vertex[corner_v[vc[is_start[vc]]]] = True

    for v in np.flatnonzero(boundary_vertex):
        ring = vc[starts[v] : starts[v + 1]]
        fan_starts = [int(c) for c in ring if is_start[c]]
        count = 0
        lasts: list[int] = []
        for c in fan_starts:
            n = 1
            i = c
            while step[i] >= 0 and n <= ring.size:
                i = int(step[i])
                n += 1
            count += n
            lasts.append(i)
        if count != ring.size or len(fan_starts) > 2:
            return None
        for j, last in enumerate(lasts):
            next_fan_start[last] = fan_starts[(j + 1) % len(fan_starts)]
    interior = has_corners & ~boundary_vertex
    if interior.any() and not _single_cycles(vc, starts, step, interior):
        return None

    owns = (twin < 0) | (np.arange(3 * n_faces) < twin)
    owns &= live
    corner_edge = np.full(3 * n_faces, -1, dtype=np.int64)
    corner_edge[owns] = 2 * np.arange(int(owns.sum()), dtype=np.int64)
    edge_of = np.where(owns, corner_edge, np.where(live, corner_edge[np.maximum(twin, 0)] ^ 1, -1))

    n_half = 2 * int(owns.sum())
    topology = Topology()
    nxt = np.full(n_half, -1, dtype=np.int64)
    prv = np.full(n_half, -1, dtype=np.int64)
    org = np.full(n_half, -1, dtype=np.int64)
    lft = np.full(n_half, -1, dtype=np.int64)
    c = corners
    e = edge_of[c]
    face = c // 3
    org[e] = corner_v[c]
    lft[e] = face
    h = edge_of[prev_corner_of(c)] ^ 1
    nxt[e] = h
    prv[h] = e
    open_side = twin[prev_corner_of(c)] < 0
    hb = h[open_side]
    org[hb] = corner_v[c[open_side]]
    lft[hb] = -1
    s = corner_edge[next_fan_start[c[open_side]]]
    nxt[hb] = s
    prv[s] = hb
    topology.set_array("nxt", nxt)
    topology.set_array("prv", prv)
    topology.set_array("org", org)
    topology.set_array("lft", lft)
    epv = np.full(n_verts, -1, dtype=np.int64)
    epv[has_corners] = edge_of[min_corner[has_corners]]
    topology.set_array("edge_per_vertex", epv)
    epf = np.where(bad, -1, edge_of[3 * np.arange(n_faces)])
    topology.set_array("edge_per_face", epf)
    return topology


def prev_corner_of(c: np.ndarray) -> np.ndarray:
    """Return the previous corner of each corner's triangle."""
    return c - c % 3 + (c + 2) % 3


def _single_cycles(
    vc: np.ndarray, starts: np.ndarray, step: np.ndarray, interior: np.ndarray
) -> bool:
    """
    Report whether every ``interior`` vertex's corners form one cycle under ``step``.

    Pointer jumping: each corner carries the least corner reached so far, doubling its reach a
    round; a vertex is one cycle exactly when every corner of it ends at the vertex's least corner.
    """
    corner_vertex = np.repeat(np.arange(starts.size - 1), np.diff(starts))
    inner = interior[corner_vertex]
    corners = vc[inner]
    if corners.size == 0:
        return True
    if np.any(step[corners] < 0):
        return False
    reach = step.copy()
    least = np.arange(step.size, dtype=np.int64)
    longest = int(np.diff(starts)[interior].max())
    hops = 1
    while hops < longest:
        least[corners] = np.minimum(least[corners], least[reach[corners]])
        reach[corners] = reach[reach[corners]]
        hops *= 2
    least[corners] = np.minimum(least[corners], least[reach[corners]])
    first = vc[starts[:-1][interior]]
    return bool(np.all(least[corners] == np.repeat(first, np.diff(starts)[interior])))


def _build_face_by_face(tris_np: np.ndarray, n_vertices: int) -> Topology:
    """
    Link triangles one at a time, in passes, in the reference order.

    Each triangle is linked in when every vertex it shares is reachable through an open edge or a
    single gap; one that cannot be placed yet waits for a later pass. Above one part's worth of
    triangles the list is first split into vertex-range parts built on their own and joined.
    """
    tris = [(int(r[0]), int(r[1]), int(r[2])) for r in tris_np]
    n = len(tris)
    n_parts = min((n + _TRIANGLES_PER_PART - 1) // _TRIANGLES_PER_PART, _MAX_PARTS)
    if n_parts > 1:
        return _build_partitioned(tris, n_parts)
    topology = Topology()
    topology.edge_per_vertex = [-1] * max(n_vertices, max(max(t) for t in tris) + 1)
    topology.edge_per_face = [-1] * n
    _add_triangles(topology, tris, range(n), shift=0)
    return topology


def _build_partitioned(tris: list[tuple[int, int, int]], n_parts: int) -> Topology:
    """Build by parts of whole vertex ranges, then link the triangles between them."""
    max_vert = max(max(t) for t in tris)
    verts_in_part = (max_vert + n_parts) // n_parts
    border = [
        not (t[0] // verts_in_part == t[1] // verts_in_part == t[2] // verts_in_part) for t in tris
    ]
    parts: list[tuple[Topology, list[int], list[int], list[int]]] = []
    for part in range(n_parts):
        begin, end = part * verts_in_part, (part + 1) * verts_in_part
        local: list[tuple[int, int, int]] = []
        fmap: list[int] = []
        vmap = [-1] * verts_in_part
        for f, t in enumerate(tris):
            if border[f] or not begin <= t[0] < end:
                continue
            lt = (t[0] % verts_in_part, t[1] % verts_in_part, t[2] % verts_in_part)
            local.append(lt)
            fmap.append(f)
            for lv, gv in zip(lt, t, strict=True):
                vmap[lv] = gv
        sub = Topology()
        rem: list[int] = []
        if local:
            sub.edge_per_vertex = [-1] * (max(max(t) for t in local) + 1)
            sub.edge_per_face = [-1] * len(local)
            rem = _add_triangles(sub, local, range(len(local)), shift=0)
        parts.append((sub, fmap, vmap, rem))

    topology = Topology()
    topology.edge_per_vertex = [-1] * (max_vert + 1)
    topology.edge_per_face = [-1] * len(tris)
    region = list(border)
    for sub, fmap, vmap, rem in parts:
        offset = topology.edge_count()
        for v, e in enumerate(sub.edge_per_vertex):
            if e >= 0:
                topology.edge_per_vertex[vmap[v]] = e + offset
        for f, e in enumerate(sub.edge_per_face):
            if e >= 0:
                topology.edge_per_face[fmap[f]] = e + offset
        topology.nxt.extend(i + offset for i in sub.nxt)
        topology.prv.extend(i + offset for i in sub.prv)
        topology.org.extend(vmap[v] if v >= 0 else -1 for v in sub.org)
        topology.lft.extend(fmap[f] if f >= 0 else -1 for f in sub.lft)
        for f in rem:
            region[fmap[f]] = True
    _add_triangles(topology, tris, [f for f in range(len(tris)) if region[f]], shift=0)
    return topology


def _add_triangles(
    topology: Topology, tris: Sequence[tuple[int, int, int]], order: Sequence[int], shift: int
) -> list[int]:
    """
    Link triangles ``order`` in passes until a pass places none; return the ones never placed.

    A triangle whose placement is ambiguous now waits for a later pass; a degenerate or
    non-manifold one is dropped.
    """
    active = list(order)
    bad: list[int] = []
    while True:
        placed = 0
        waiting: list[int] = []
        for f in active:
            result = _add_face(topology, f + shift, tris[f])
            if result == _TRY_LATER:
                waiting.append(f)
            elif result == _ADDED:
                placed += 1
            else:
                bad.append(f)
        active = waiting
        if placed == 0:
            break
    return sorted(active + bad)


_ADDED, _TRY_LATER, _FAILED = 0, 1, 2


def _open_edge(topology: Topology, o: int, d: int) -> int:
    """Find a half-edge from ``o`` to ``d`` with no left face, else ``-1``."""
    e0 = topology.edge_per_vertex[o]
    if e0 < 0:
        return -1
    e = e0
    while True:
        if topology.org[e ^ 1] == d and topology.lft[e] < 0:
            return e
        e = topology.nxt[e]
        if e == e0:
            return -1


def _single_gap(topology: Topology, a: int) -> int:
    """Find the one half-edge leaving ``a`` with no left face; ``-1`` for none or several."""
    e0 = topology.edge_per_vertex[a]
    if e0 < 0:
        return -1
    eh = e0
    while topology.lft[eh] >= 0:
        eh = topology.nxt[eh]
        if eh == e0:
            return -1
    e = topology.nxt[eh]
    while e != e0:
        if topology.lft[e] < 0:
            return -1
        e = topology.nxt[e]
    return eh


def _add_face(topology: Topology, face: int, verts: tuple[int, int, int]) -> int:
    """Link one triangle into the open edges and gaps around its vertices, if unambiguous."""
    if len(set(verts)) < 3:
        return _FAILED
    epv = topology.edge_per_vertex
    simple = [False, False, False]
    edges = [-1, -1, -1]
    gaps = [-1, -1, -1]
    for i in range(3):
        if epv[verts[i]] < 0:
            simple[i] = True
            continue
        i1 = (i + 1) % 3
        edges[i] = _open_edge(topology, verts[i], verts[i1])
        if edges[i] >= 0:
            simple[i] = True
            simple[i1] = True
    for i in range(3):
        i2 = (i + 2) % 3
        if edges[i] >= 0 and edges[i2] >= 0 and topology.nxt[edges[i]] != edges[i2] ^ 1:
            return _FAILED
        if not simple[i]:
            gaps[i] = _single_gap(topology, verts[i])
            if gaps[i] < 0:
                return _TRY_LATER
    for i in range(3):
        if edges[i] < 0:
            edges[i] = topology.make_edge()
    for i in range(3):
        i2 = (i + 2) % 3
        if topology.org[edges[i]] == verts[i] and topology.org[edges[i2] ^ 1] == verts[i]:
            continue
        if gaps[i] >= 0 and gaps[i] != edges[i]:
            topology.splice(gaps[i], edges[i])
        topology.splice(edges[i], topology.prv[edges[i2] ^ 1])
        topology.set_org(edges[i], verts[i])
    topology.set_left(edges[0], face)
    return _ADDED


# ---------------------------------------------------------------------------------------------
# Positions and single-precision vector arithmetic in the reference's operation order
# ---------------------------------------------------------------------------------------------

F32 = np.float32
_DBL_MAX = float(np.finfo(np.float64).max)
_FLT_MAX = F32(np.finfo(np.float32).max)
BAD_METRIC = 1e10


class Points:
    """A growable ``(n, 3)`` ``float32`` position table, indexed by vertex id."""

    __slots__ = ("data", "size")

    def __init__(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        self.size = points.shape[0]
        self.data = np.empty((max(16, 2 * self.size), 3), dtype=np.float32)
        self.data[: self.size] = points

    def set(self, v: int, p: np.ndarray) -> None:
        """Store position ``p`` at vertex ``v``, growing the table to hold it."""
        if v >= self.data.shape[0]:
            grown = np.empty((2 * v + 16, 3), dtype=np.float32)
            grown[: self.size] = self.data[: self.size]
            self.data = grown
        self.size = max(self.size, v + 1)
        self.data[v] = p

    def array(self) -> np.ndarray:
        """Return the used ``(n, 3)`` prefix."""
        return self.data[: self.size]


def _lsq(v: np.ndarray) -> np.ndarray:
    return v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2]


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def _normalized(v: np.ndarray) -> np.ndarray:
    length = np.sqrt(_lsq(v))
    inv = np.where(length > 0, np.ones_like(length) / np.where(length > 0, length, 1), 0)
    return v * inv[..., None]


def _aspect_ratio(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Circumradius over twice the inradius, in the inputs' precision."""
    bc = np.sqrt(_lsq(c - b))
    ca = np.sqrt(_lsq(a - c))
    ab = np.sqrt(_lsq(b - a))
    two = bc.dtype.type(2)
    half = (bc + ca + ab) / two
    den = bc.dtype.type(8) * (half - bc) * (half - ca) * (half - ab)
    big = np.finfo(bc.dtype).max
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = bc * ca * ab / np.where(den > 0, den, 1)
    return np.where(den > 0, ratio, big)


def _circum_diameter_sq(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the squared circumcircle diameter (a side's square when another is zero)."""
    ab = _lsq(b - a)
    ca = _lsq(a - c)
    bc = _lsq(c - b)
    f = _lsq(_cross(b - a, c - a))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        general = np.where(f > 0, ab * ca * bc / np.where(f > 0, f, 1), np.inf)
    out = np.where(bc <= 0, ab, general)
    out = np.where(ca <= 0, bc, out)
    return np.where(ab <= 0, ca, out).astype(ab.dtype)


def _dihedral(left: np.ndarray, right: np.ndarray, edge: np.ndarray) -> np.ndarray:
    sin = _dot(_normalized(edge), _cross(left, right))
    return np.arctan2(sin, _dot(left, right))


# ---------------------------------------------------------------------------------------------
# Fill and stitch metrics, vectorized over vertex index arrays
# ---------------------------------------------------------------------------------------------


Term = Callable[..., np.ndarray]


class Metric:
    """A triangle term, an edge term over ``(a, b, left, right)`` and how terms combine."""

    def __init__(
        self,
        triangle: Term | None = None,
        edge: Term | None = None,
        use_max: bool = False,
        kernel_id: int = -1,
        normal: np.ndarray | None = None,
        inverse_area: float = 0.0,
        band_id: int = -1,
        up: tuple[float, float, float] | None = None,
    ) -> None:
        self.triangle_term = triangle
        self.edge_term = edge
        self.use_max = use_max
        # The id of the same metric in ``kernels/holes.serial_fill_dp_diagonal``, -1 when it has
        # none (a metric with transcendental terms, which the devices round differently).
        self.kernel_id = kernel_id
        self.normal = np.zeros(3) if normal is None else normal
        self.inverse_area = inverse_area
        # The id of the same band metric in ``kernels/holes.serial_band_search``, -1 for none.
        self.band_id = band_id
        self.up = up

    @property
    def has_triangle(self) -> bool:
        """Report whether the metric has a triangle term."""
        return self.triangle_term is not None

    @property
    def has_edge(self) -> bool:
        """Report whether the metric has an edge term."""
        return self.edge_term is not None

    def combine(self, a: Any, b: Any) -> Any:
        """Add (or take the larger of) two accumulated weights; an unreachable one stays so."""
        with np.errstate(over="ignore"):
            return np.maximum(a, b) if self.use_max else a + b

    def triangle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Evaluate the triangle term of each ``(a, b, c)``."""
        assert self.triangle_term is not None
        return self.triangle_term(a, b, c)

    def edge(self, a: np.ndarray, b: np.ndarray, lft: np.ndarray, rgt: np.ndarray) -> np.ndarray:
        """Evaluate the term of each edge ``a b`` between triangles reaching ``lft``, ``rgt``."""
        assert self.edge_term is not None
        return self.edge_term(a, b, lft, rgt)


def _f64(points: np.ndarray, i: np.ndarray) -> np.ndarray:
    return points[i].astype(np.float64)


def _circumscribed(p: np.ndarray) -> Term:
    def term(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.sqrt(_circum_diameter_sq(p[a], p[b], p[c])).astype(np.float64)

    return term


def _plane(p: np.ndarray, n: np.ndarray, normalized: bool) -> Term:
    def term(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ap, bp, cp = _f64(p, a), _f64(p, b), _f64(p, c)
        face = _cross(bp - ap, cp - ap)
        diameter = np.sqrt(_circum_diameter_sq(ap, bp, cp))
        dot = _dot(np.broadcast_to(n, face.shape), face)
        if not normalized:
            return np.where(dot < 0.0, BAD_METRIC, diameter)
        area_sq = _lsq(face)
        ratio = _aspect_ratio(ap, bp, cp)
        bad = (area_sq == 0.0) | (dot < 0.0) | (dot * dot * 4.0 < area_sq) | (ratio > BAD_METRIC)
        with np.errstate(invalid="ignore", over="ignore"):
            return np.where(bad, BAD_METRIC, diameter * ratio)

    return term


def _min_area(p: np.ndarray) -> Term:
    def term(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ap, bp, cp = _f64(p, a), _f64(p, b), _f64(p, c)
        return np.sqrt(_lsq(_cross(bp - ap, cp - ap)))

    return term


def _min_tri_angle(p: np.ndarray) -> Term:
    def term(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        pa, pb, pc = p[a], p[b], p[c]
        ab = np.sqrt(_lsq(pb - pa))
        ca = np.sqrt(_lsq(pa - pc))
        bc = np.sqrt(_lsq(pc - pb))
        f = np.sqrt(_lsq(_cross(pb - pa, pc - pa)))
        degenerate = (ab <= 0) | (ca <= 0) | (bc <= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sin = f * np.minimum(np.minimum(ab, ca), bc) / (ab * ca * bc)
        sin = np.where(degenerate, F32(0), sin).astype(np.float64)
        return np.exp(25.0 * (0.86602540378443864676372317075294 - sin))

    return term


def _edge_length(p: np.ndarray) -> Term:
    def term(a: np.ndarray, b: np.ndarray, _lft: np.ndarray, _rgt: np.ndarray) -> np.ndarray:
        return np.sqrt(_lsq(p[b] - p[a])).astype(np.float64)

    return term


def _dihedral_terms(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, lft: np.ndarray, rgt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ab = p[b] - p[a]
    return _cross(p[lft] - p[a], ab), _cross(ab, p[rgt] - p[a]), ab


def _universal_edge(p: np.ndarray) -> Term:
    def term(a: np.ndarray, b: np.ndarray, lft: np.ndarray, rgt: np.ndarray) -> np.ndarray:
        norm_l, norm_r, ab = _dihedral_terms(p, a, b, lft, rgt)
        area = np.sqrt(_lsq(norm_l)) + np.sqrt(_lsq(norm_r))
        angle = np.abs(_dihedral(norm_l, norm_r, ab))
        with np.errstate(over="ignore"):
            return (np.sqrt(area) * np.exp(F32(5) * angle)).astype(np.float64)

    return term


def _max_dihedral_edge(p: np.ndarray) -> Term:
    def term(a: np.ndarray, b: np.ndarray, lft: np.ndarray, rgt: np.ndarray) -> np.ndarray:
        norm_l, norm_r, ab = _dihedral_terms(p, a, b, lft, rgt)
        return np.abs(_dihedral(norm_l, norm_r, ab)).astype(np.float64)

    return term


def _complex_fill(p: np.ndarray, inverse_area: float) -> tuple[Term, Term]:
    def triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ratio = _aspect_ratio(p[a], p[b], p[c]).astype(np.float64)
        area = np.sqrt(_lsq(_cross(p[b] - p[a], p[c] - p[a]))).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            value = ratio + 1e2 * area * inverse_area
        return np.where(ratio > BAD_METRIC, BAD_METRIC, value)

    def edge(a: np.ndarray, b: np.ndarray, lft: np.ndarray, rgt: np.ndarray) -> np.ndarray:
        ab = p[b] - p[a]
        norm_a = _cross(p[rgt] - p[b], -ab)
        norm_c = _cross(p[lft] - p[a], ab)
        sa = np.sqrt(_lsq(norm_a))
        sc = np.sqrt(_lsq(norm_c))
        den = sa * sc
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            cos = _dot(norm_a, norm_c) / (sa * sc)
            x = (F32(1) - cos) / (F32(1) + cos)
            x = x * x
            value = (x * x).astype(np.float64)
        return np.where((den == 0) | (cos <= F32(-1)), BAD_METRIC, value)

    return triangle, edge


def _complex_stitch(p: np.ndarray) -> tuple[Term, Term]:
    def triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return ((_aspect_ratio(p[a], p[b], p[c]) - F32(1)) * F32(1e-2)).astype(np.float64)

    def edge(a: np.ndarray, b: np.ndarray, lft: np.ndarray, rgt: np.ndarray) -> np.ndarray:
        norm_l, norm_r, _ = _dihedral_terms(p, a, b, lft, rgt)
        cos = _dot(_normalized(norm_l), _normalized(norm_r)).astype(np.float64)
        return (1.0 - cos) * 1e4

    return triangle, edge


def _edge_length_stitch(p: np.ndarray) -> Term:
    def term(a: np.ndarray, _b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.sqrt(_lsq(p[c] - p[a])).astype(np.float64)

    return term


def _vertical_stitch(p: np.ndarray, up: tuple[float, float, float]) -> Term:
    u = np.asarray(up, dtype=np.float32).astype(np.float64)
    length = float(np.sqrt(_lsq(u)))
    unit = u * (1.0 / length) if length > 0 else np.zeros(3)

    def term(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ap, bp, cp = _f64(p, a), _f64(p, b), _f64(p, c)
        ab, ac, bc = bp - ap, cp - ap, cp - bp
        norm = _cross(ab, ac)
        parallel = np.abs(_dot(np.broadcast_to(unit, norm.shape), norm))
        sides = _lsq(ab) + _lsq(ac) + _lsq(bc)
        return _lsq(norm) + 100.0 * parallel * parallel + sides * sides * 0.5

    return term


def fill_metric(name: str, topology: Topology, points: np.ndarray, rim_edge: int) -> Metric:
    """Return the fill metric ``name`` for the hole left of ``rim_edge``."""
    if name == "circumscribed":
        return Metric(triangle=_circumscribed(points), kernel_id=0)
    if name in ("plane", "plane_normalized"):
        n = np.zeros(3)
        for e in topology.left_ring(rim_edge):
            o = points[topology.org[e]].astype(np.float64)
            d = points[topology.org[e ^ 1]].astype(np.float64)
            n = n + _cross(o, d)
        length = float(np.sqrt(_lsq(n)))
        n = n * (1.0 / length) if length > 0 else np.zeros(3)
        return Metric(
            triangle=_plane(points, n, name == "plane_normalized"),
            kernel_id=2 if name == "plane_normalized" else 1,
            normal=n,
        )
    if name == "min_area":
        return Metric(triangle=_min_area(points), kernel_id=3)
    if name == "min_tri_angle":
        return Metric(triangle=_min_tri_angle(points))
    if name == "edge_length":
        return Metric(edge=_edge_length(points), kernel_id=4)
    if name == "universal":
        return Metric(triangle=_circumscribed(points), edge=_universal_edge(points))
    if name == "max_dihedral":
        return Metric(edge=_max_dihedral_edge(points), use_max=True)
    if name == "complex_fill":
        longest = F32(0)
        for e in topology.left_ring(rim_edge):
            v = points[topology.org[e ^ 1]] - points[topology.org[e]]
            longest = max(longest, F32(_lsq(v)))
        inverse = F32(1) / longest if longest > 0 else F32(1)
        triangle, edge = _complex_fill(points, float(inverse))
        return Metric(triangle=triangle, edge=edge, kernel_id=5, inverse_area=float(inverse))
    raise ValueError(name)


def stitch_metric(
    name: str, points: np.ndarray, up: tuple[float, float, float] | None = None
) -> Metric:
    """Return the band metric ``name``."""
    if name == "complex_stitch":
        triangle, edge = _complex_stitch(points)
        return Metric(triangle=triangle, edge=edge, band_id=0)
    if name == "edge_length_stitch":
        return Metric(triangle=_edge_length_stitch(points), band_id=1)
    if name == "vertical":
        return Metric(
            triangle=_vertical_stitch(points, up if up is not None else (0.0, 0.0, 1.0)),
            band_id=2,
            up=up,
        )
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------
# Filling one hole: the interval DP over its rim, then linking the chosen chords
# ---------------------------------------------------------------------------------------------


def _apex_offsets(steps: int, max_subdivisions: int) -> list[int]:
    """Offsets from an interval's first inner rim position of the apexes the DP tries."""
    steps -= 1
    if steps <= max_subdivisions:
        return list(range(steps))
    quarter = max_subdivisions // 4
    out = list(range(quarter))
    big = (steps - max_subdivisions // 2) // (max_subdivisions // 2)
    count = max_subdivisions // 2
    if big < 2:
        big = 2
        count = quarter
    start = big // 2 + quarter - 1
    out.extend(start + i * big for i in range(count))
    out.extend(steps - i - 1 for i in range(quarter - 1, -1, -1))
    return out


def hole_representatives(topology: Topology) -> list[int]:
    """Return the lowest half-edge of every hole (no left face), in increasing order."""
    seen = [False] * topology.edge_count()
    out: list[int] = []
    for e in range(topology.edge_count()):
        if seen[e] or topology.lft[e] >= 0 or topology.org[e] < 0:
            continue
        for i in topology.left_ring(e):
            seen[i] = True
        out.append(e)
    return out


def hole_loop(topology: Topology, a0: int) -> list[int]:
    """Return the rim half-edges of the hole left of ``a0``, from ``a0`` in hole order."""
    loop = [a0]
    a = topology.prv[a0 ^ 1]
    while a != a0:
        loop.append(a)
        a = topology.prv[a ^ 1]
    return loop


def plan_fill(
    topology: Topology,
    a0: int,
    metric: Metric,
    smooth_rim: bool,
    avoid_existing: bool,
    max_subdivisions: int = 20,
    points: np.ndarray | None = None,
    device: wp.DeviceLike = None,
) -> tuple[list[tuple[int, int]], int]:
    """
    Plan the minimum-weight triangulation of the hole left of ``a0``.

    Returns the chords to create, each as two edge codes (a half-edge id, or ``-(2 i + 1)`` for
    the chord made by item ``i``) whose origins it joins, and the number of triangles; a plan with
    no chords and more than one triangle asks for a cone. With ``points`` and a ``device``, a metric
    the kernel has fills its DP there (``kernels/holes.serial_fill_dp_diagonal``), a launch a
    diagonal; otherwise each diagonal is one NumPy batch.
    """
    loop = hole_loop(topology, a0)
    n = len(loop)
    if n <= 3:
        return [], 1 if n == 3 else n
    org = topology.org
    vert = np.array([org[e] for e in loop], dtype=np.int64)
    # The vertex across each rim edge's existing triangle, ``-1`` where it has none.
    across = np.array(
        [org[topology.prv[e] ^ 1] if topology.lft[e ^ 1] >= 0 else -1 for e in loop], dtype=np.int64
    )
    weight = np.zeros((n, n), dtype=np.float64)
    apex = np.full((n, n), -1, dtype=np.int64)
    linked: list[set[int]] = []
    if avoid_existing:
        for e in loop:
            linked.append({org[i ^ 1] for i in topology.org_ring(e)})
    native = metric.kernel_id >= 0 and points is not None and device is not None
    if native:
        assert points is not None
        weight, apex = _fill_dp_on_device(
            points, vert, across, linked, metric, smooth_rim, max_subdivisions, device
        )
    for steps in range(2, 2 if native else n - 1):
        offsets = np.asarray(_apex_offsets(steps, max_subdivisions), dtype=np.int64)
        rows = np.arange(n, dtype=np.int64)
        ends = (rows + steps) % n
        blocked = np.zeros(n, dtype=bool)
        if avoid_existing:
            blocked = np.array([int(vert[c]) in linked[i] for i, c in enumerate(ends)])
        i_idx = np.repeat(rows, offsets.size)
        c_idx = np.repeat(ends, offsets.size)
        v_idx = (i_idx + 1 + np.tile(offsets, n)) % n
        w = metric.combine(weight[i_idx, v_idx], weight[v_idx, c_idx])
        va, vv, vc = vert[i_idx], vert[v_idx], vert[c_idx]
        with np.errstate(over="ignore", invalid="ignore"):
            if metric.has_triangle:
                w = metric.combine(w, metric.triangle(va, vv, vc))
            if metric.has_edge:
                left = _apex_vertex(apex[i_idx, v_idx], vert, across[i_idx], smooth_rim)
                right = _apex_vertex(apex[v_idx, c_idx], vert, across[v_idx], smooth_rim)
                has = left >= 0
                w = np.where(
                    has, metric.combine(w, metric.edge(va, vv, np.maximum(left, 0), vc)), w
                )
                has = right >= 0
                w = np.where(
                    has, metric.combine(w, metric.edge(vv, vc, np.maximum(right, 0), va)), w
                )
        w = np.where(va == vc, np.inf, w)
        w = np.where(np.isnan(w), np.inf, w).reshape(n, offsets.size)
        best = np.argmin(w, axis=1)
        best_w = w[rows, best]
        take = (best_w < _DBL_MAX) & ~blocked
        weight[rows, ends] = np.where(take, best_w, _DBL_MAX)
        apex[rows, ends] = np.where(take, v_idx.reshape(n, offsets.size)[rows, best], -1)

    fin_w = _DBL_MAX
    fin = -1
    for i in range(n):
        c = (i + 2) % n
        w = metric.combine(weight[i, c], weight[c, i])
        if metric.has_edge:
            lv = int(vert[apex[i, c]]) if apex[i, c] >= 0 else -1
            if lv < 0 and smooth_rim:
                lv = int(across[i])
            rv = int(vert[apex[c, i]]) if apex[c, i] >= 0 else -1
            if rv < 0 and smooth_rim:
                rv = int(across[c])
            if lv >= 0 and rv >= 0:
                term = metric.edge(
                    np.array([vert[i]]), np.array([vert[c]]), np.array([lv]), np.array([rv])
                )[0]
                with np.errstate(over="ignore", invalid="ignore"):
                    w = metric.combine(w, term)
        if w < fin_w:
            fin_w = float(w)
            fin = i
    if fin < 0:
        return [], n
    c = (fin + 2) % n
    # The traceback queue: each entry is an interval (a, b) with its apex, and the code of the
    # edge that closes it.
    items: list[tuple[int, int]] = []
    queue: list[tuple[int, int, int, int]] = [(c, (fin + 1) % n, fin, loop[(fin + 1) % n])]
    head = 0
    n_tris = 0
    while head < len(queue):
        a, b, prev_a, code = queue[head]
        head += 1
        if prev_a < 0 or len(queue) > 2 * n:
            raise RuntimeError("the hole triangulation's apex table is inconsistent")
        dist_a = (a - prev_a + n) % n
        dist_b = (b - prev_a + n) % n
        if 2 <= dist_a <= n - 2:
            new_code = -(2 * len(items) + 1)
            items.append((loop[prev_a], loop[a]))
            queue.append((a, prev_a, int(apex[a, prev_a]), new_code))
        if 2 <= dist_b <= n - 2:
            new_code = -(2 * len(items) + 1)
            items.append((code, loop[prev_a]))
            queue.append((prev_a, b, int(apex[prev_a, b]), new_code))
        n_tris += 1
    return items, n_tris


def _fill_dp_on_device(
    points: np.ndarray,
    vert: np.ndarray,
    across: np.ndarray,
    linked: list[set[int]],
    metric: Metric,
    smooth_rim: bool,
    max_subdivisions: int,
    device: wp.DeviceLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the DP's weight and apex tables on ``device``: a launch a diagonal, one on the CPU."""
    from ordito._common import read_array
    from ordito.kernels import holes as kernel_holes

    n = vert.size
    positions: dict[int, list[int]] = {}
    for i, v in enumerate(vert.tolist()):
        positions.setdefault(v, []).append(i)
    rows = [
        sorted(j for d in linked[i] for j in positions.get(d, ())) if linked else []
        for i in range(n)
    ]
    linked_offsets = np.zeros(n + 1, dtype=np.int32)
    linked_offsets[1:] = np.cumsum([len(r) for r in rows])
    linked_flat = np.array([j for r in rows for j in r] or [0], dtype=np.int32)
    per_step = [_apex_offsets(steps, max_subdivisions) for steps in range(2, n - 1)]
    starts = np.zeros(len(per_step) + 1, dtype=np.int64)
    starts[1:] = np.cumsum([len(o) for o in per_step])
    flat = np.array([k for o in per_step for k in o] or [0], dtype=np.int32)
    from ordito import _launch

    dp = kernel_holes.FillDp()
    dp.points = wp.array(points, dtype=wp.vec3, device=device)
    dp.vert = wp.array(vert.astype(np.int32), dtype=wp.int32, device=device)
    dp.across = wp.array(across.astype(np.int32), dtype=wp.int32, device=device)
    dp.linked_offsets = wp.array(linked_offsets, dtype=wp.int32, device=device)
    dp.linked = wp.array(linked_flat, dtype=wp.int32, device=device)
    dp.offsets = wp.array(flat, dtype=wp.int32, device=device)
    dp.metric = metric.kernel_id
    dp.has_triangle = int(metric.has_triangle)
    dp.has_edge = int(metric.has_edge)
    dp.smooth_rim = int(smooth_rim)
    dp.normal = wp.vec3d(*(float(x) for x in metric.normal))
    dp.inverse_area = float(metric.inverse_area)
    weight = wp.zeros((n, n), dtype=wp.float64, device=device)
    apex = wp.full((n, n), -1, dtype=wp.int32, device=device)
    dp.weight = weight
    dp.apex = apex
    resolved = wp.get_device(device)
    if resolved.is_cpu:
        _launch.launch(
            kernel_holes.serial_fill_dp_in_turn,
            dim=1,
            inputs=[dp, wp.array(starts.astype(np.int32), dtype=wp.int32, device=resolved)],
            device=resolved,
        )
    else:
        start_list = starts.tolist()
        for k, steps in enumerate(range(2, n - 1)):
            _launch.launch(
                kernel_holes.serial_fill_dp_diagonal,
                dim=n,
                inputs=[dp, start_list[k], start_list[k + 1] - start_list[k], steps],
                device=resolved,
            )
    # The tables feed the host traceback that turns them into chords.
    return read_array(weight), read_array(apex).astype(np.int64)


def _apex_vertex(
    apexes: np.ndarray, vert: np.ndarray, across: np.ndarray, smooth_rim: bool
) -> np.ndarray:
    """Return each apex's vertex, else (when ``smooth_rim``) the vertex across the rim edge."""
    out = np.where(apexes >= 0, vert[np.maximum(apexes, 0)], -1)
    return np.where(apexes >= 0, out, across) if smooth_rim else out


def _plan_edge(items: list[list[int]], code: int) -> int:
    if code >= 0:
        return code
    c = -(code + 1)
    e = items[c >> 1][0]
    return e ^ 1 if c & 1 else e


def _new_edge(topology: Topology, a: int, b: int) -> int:
    e = topology.make_edge()
    topology.splice(a, e)
    topology.splice(b, e ^ 1)
    return e


def execute_fill(
    topology: Topology,
    points: Points,
    a0: int,
    plan: tuple[list[tuple[int, int]], int],
    new_faces: list[int],
) -> None:
    """Create the chords and faces of ``plan`` in the hole left of ``a0``."""
    chords, n_tris = plan
    if not chords:
        if topology.is_left_tri(a0):
            f = topology.add_face()
            new_faces.append(f)
            topology.set_left(a0, f)
        elif n_tris >= 2:
            cone_fill(topology, points, a0, new_faces)
        return
    items = [[c1, c2] for c1, c2 in chords]
    for item in items:
        a = _plan_edge(items, item[0])
        b = _plan_edge(items, item[1])
        item[0] = _new_edge(topology, a, b)
    for item in items:
        e = item[0]
        for side in (e, e ^ 1):
            if topology.lft[side] >= 0:
                continue
            f = topology.add_face()
            new_faces.append(f)
            topology.set_left(side, f)


def cone_fill(topology: Topology, points: Points, a: int, new_faces: list[int]) -> int:
    """Close the hole left of ``a`` with a fan around a new vertex at its rim's centroid."""
    total = np.zeros(3)
    degree = 0
    for e in topology.left_ring(a):
        total = total + points.data[topology.org[e]].astype(np.float64)
        degree += 1
    center = (total * (1.0 / float(degree))).astype(np.float32)
    center_vert = topology.add_vert()
    if topology.vertex_rows is not None:
        row: dict[int, float] = {}
        for e in topology.left_ring(a):
            for v, w in topology.vertex_rows[topology.org[e]].items():
                row[v] = row.get(v, 0.0) + w / degree
        topology.vertex_rows[center_vert] = row
    points.set(center_vert, center)
    e0 = topology.make_edge()
    topology.splice(a, e0)
    last = e0
    bd = topology.prv[a ^ 1]
    for _ in range(1, degree):
        bd1 = topology.prv[bd ^ 1]
        ei = topology.make_edge()
        topology.splice(bd, ei)
        topology.splice(last ^ 1, ei ^ 1)
        f = topology.add_face()
        new_faces.append(f)
        topology.set_left(ei, f)
        last = ei
        bd = bd1
    f = topology.add_face()
    new_faces.append(f)
    topology.set_left(e0, f)
    topology.set_org(e0 ^ 1, center_vert)
    return center_vert


def fill_hole(
    topology: Topology,
    points: Points,
    a0: int,
    metric: str,
    smooth_rim: bool,
    avoid_existing: bool,
    device: wp.DeviceLike = None,
) -> list[int]:
    """Triangulate the hole left of ``a0`` over its rim; return the new faces."""
    new_faces: list[int] = []
    if topology.lft[a0] >= 0:
        return new_faces
    count = len(hole_loop(topology, a0))
    if count < 2:
        return new_faces
    if count == 2:
        a1 = topology.nxt[a0]
        a2 = topology.prv[a1 ^ 1]
        topology.splice(a0, a1)
        topology.splice(a2, a1 ^ 1)
        return new_faces
    arr = points.array()
    plan = plan_fill(
        topology,
        a0,
        fill_metric(metric, topology, arr, a0),
        smooth_rim,
        avoid_existing,
        points=arr,
        device=device,
    )
    execute_fill(topology, points, a0, plan, new_faces)
    return new_faces


# ---------------------------------------------------------------------------------------------
# Refining a region: longest-edge splits from a priority queue, each followed by Delaunay flips
# around the new vertex
# ---------------------------------------------------------------------------------------------

_PI_F = F32(np.pi)


def _in_region(topology: _Walk, e: int, region: set[int]) -> bool:
    f = topology.lft[e]
    return f >= 0 and f in region


class _NativeMesh:
    """A topology's lists copied into capacity-sized arrays a native kernel edits in place."""

    def __init__(
        self,
        topology: Topology,
        points: Points | None,
        *,
        extra_half: int = 0,
        extra_verts: int = 0,
        extra_faces: int = 0,
        region_flags: np.ndarray | None = None,
        heap_cap: int = 1,
        log_rows: int = 1,
    ) -> None:
        from ordito.kernels import holes as kernel_holes

        n_half = topology.edge_count()
        n_verts = topology.vertex_count()
        n_faces = topology.face_count()

        def padded(name: str, size: int) -> np.ndarray:
            values = topology.array(name)
            out = np.full(size, -1, dtype=np.int32)
            out[: values.shape[0]] = values
            return out

        half_cap = n_half + extra_half
        vert_cap = n_verts + extra_verts
        face_cap = n_faces + extra_faces
        self.nxt = padded("nxt", half_cap)
        self.prv = padded("prv", half_cap)
        self.org = padded("org", half_cap)
        self.lft = padded("lft", half_cap)
        self.epv = padded("edge_per_vertex", vert_cap)
        self.epf = padded("edge_per_face", face_cap)
        self.region = (
            np.zeros(face_cap, dtype=np.int32) if region_flags is None else region_flags.copy()
        )
        self.positions = np.zeros((vert_cap if points is not None else 1, 3), dtype=np.float32)
        if points is not None:
            self.positions[: points.size] = points.array()
        self.counts = np.array([n_half, n_verts, n_faces, 0, 0, 0], dtype=np.int32)
        self.log = np.full((log_rows, 7), -1, dtype=np.int32)
        mesh = kernel_holes.SerialMesh()
        mesh.nxt = wp.array(self.nxt, dtype=wp.int32, device="cpu", copy=False)
        mesh.prv = wp.array(self.prv, dtype=wp.int32, device="cpu", copy=False)
        mesh.org = wp.array(self.org, dtype=wp.int32, device="cpu", copy=False)
        mesh.lft = wp.array(self.lft, dtype=wp.int32, device="cpu", copy=False)
        mesh.edge_per_vertex = wp.array(self.epv, dtype=wp.int32, device="cpu", copy=False)
        mesh.edge_per_face = wp.array(self.epf, dtype=wp.int32, device="cpu", copy=False)
        mesh.region = wp.array(self.region, dtype=wp.int32, device="cpu", copy=False)
        mesh.points = wp.array(self.positions, dtype=wp.vec3, device="cpu", copy=False)
        mesh.counts = wp.array(self.counts, dtype=wp.int32, device="cpu", copy=False)
        mesh.heap_len = wp.array(np.empty(heap_cap, dtype=np.float32), device="cpu", copy=False)
        mesh.heap_edge = wp.array(np.empty(heap_cap, dtype=np.int32), device="cpu", copy=False)
        mesh.log = wp.array(self.log, dtype=wp.int32, device="cpu", copy=False)
        self.mesh = mesh

    def store(self, topology: Topology) -> None:
        """Hand the edited arrays back to ``topology``, cut to their counts."""
        n_half, n_verts, n_faces = (int(x) for x in self.counts[:3])
        topology.set_array("nxt", self.nxt[:n_half].copy())
        topology.set_array("prv", self.prv[:n_half].copy())
        topology.set_array("org", self.org[:n_half].copy())
        topology.set_array("lft", self.lft[:n_half].copy())
        topology.set_array("edge_per_vertex", self.epv[:n_verts].copy())
        topology.set_array("edge_per_face", self.epf[:n_faces].copy())


def subdivide(
    topology: Topology,
    points: Points,
    region: set[int],
    max_edge: float,
    max_splits: int,
    max_angle_change: float,
    new_verts: list[int],
) -> int:
    """
    Split the longest edges of ``region``, then Delaunay-flip around each new vertex.

    Edges inner to or bounding the region are split longest first until all are shorter than
    ``max_edge`` or ``max_splits`` splits are done. New faces join ``region``; every new vertex
    is appended to ``new_verts``. The queue runs natively, in one thread on the CPU device
    (``kernels/holes.serial_subdivide``), over copies of the lists it then writes back.
    """
    from ordito.kernels import holes as kernel_holes

    if max_splits <= 0:
        return 0
    n_faces = topology.face_count()
    flags = np.zeros(n_faces + 2 * max_splits, dtype=np.int32)
    flags[[f for f in region if topology.edge_per_face[f] >= 0]] = 1
    heap_cap = topology.edge_count() // 2 + 32 * max_splits
    while True:
        native = _NativeMesh(
            topology,
            points,
            extra_half=6 * max_splits,
            extra_verts=max_splits,
            extra_faces=2 * max_splits,
            region_flags=flags,
            heap_cap=heap_cap,
            log_rows=max_splits,
        )
        wp.launch(
            kernel_holes.serial_subdivide,
            dim=1,
            inputs=[native.mesh, F32(max_edge) * F32(max_edge), max_splits, F32(max_angle_change)],
            device="cpu",
        )
        if native.counts[5] == 0:
            break
        heap_cap *= 4  # the queue outgrew its capacity: rerun from the untouched lists

    native.store(topology)
    n_verts, n_faces, splits = int(native.counts[1]), int(native.counts[2]), int(native.counts[4])
    points.data = native.positions
    points.size = n_verts
    region.clear()
    region.update(np.flatnonzero(native.region[:n_faces]).tolist())
    log = native.log[:splits]
    new_verts.extend(log[:, 2].tolist())
    if topology.face_source is not None:
        source = topology.face_source
        source.extend([-1] * (n_faces - len(source)))
        for _, _, _, face_l, parent_l, face_r, parent_r in log.tolist():
            if face_l >= 0:
                source[face_l] = source[parent_l]
            if face_r >= 0:
                source[face_r] = source[parent_r]
    if topology.vertex_rows is not None:
        rows = topology.vertex_rows
        rows.extend_to(n_verts)
        for o, d, vert, *_ in log.tolist():
            row: dict[int, float] = {}
            for end in (o, d):
                for v, w in rows[end].items():
                    row[v] = row.get(v, 0.0) + 0.5 * w
            rows[vert] = row
    return splits


# ---------------------------------------------------------------------------------------------
# Smoothing a patch: a fixed-rim umbrella solve, then a least-squares cotangent fairing
# ---------------------------------------------------------------------------------------------


def _vertex_on_hole(topology: _Walk, v: int) -> bool:
    e0 = topology.edge_per_vertex[v]
    return e0 >= 0 and any(topology.lft[e] < 0 for e in topology.org_ring(e0))


def _neighbours(topology: _Walk, v: int) -> Iterator[int]:
    e0 = topology.edge_per_vertex[v]
    if e0 >= 0:
        for e in topology.org_ring(e0):
            yield topology.org[e ^ 1]


def expand_verts(topology: _Walk, verts: set[int], hops: int) -> set[int]:
    """``verts`` and every vertex within ``hops`` edges of them."""
    out = set(verts)
    frontier = set(verts)
    for _ in range(hops):
        grown = {d for v in frontier for d in _neighbours(topology, v)} - out
        out |= grown
        frontier = grown
    return out


def shrink_verts(topology: _Walk, verts: set[int], hops: int) -> set[int]:
    """``verts`` less every vertex within ``hops`` edges of a valid vertex outside them."""
    valid = {v for v, e in enumerate(topology.edge_per_vertex) if e >= 0}
    return valid - expand_verts(topology, valid - verts, hops)


def _exclude_full_components(topology: _Walk, verts: set[int]) -> set[int]:
    out = set(verts)
    seen: set[int] = set()
    for v in sorted(verts):
        if v in seen:
            continue
        component = {v}
        stack = [v]
        while stack:
            x = stack.pop()
            for d in _neighbours(topology, x):
                if d not in component:
                    component.add(d)
                    stack.append(d)
        seen |= component
        if component <= verts:
            out -= component
    return out


# Below this many unknowns a dense host factorization beats a device one's fixed cost.
_DENSE_SOLVE_MAX = 400
# Up to this many a dense factorization on a CUDA device beats the sparse one's analysis
# (2026-11-15: 0.32-0.41x at 500 unknowns, 0.84-0.86x at 1 500, 1.4-1.8x at 2 500).
_DENSE_DEVICE_SOLVE_MAX = 1536


class _HeldFactor:
    """A sparse factorization a loop of solves over one pattern refactors instead of rebuilding."""

    def __init__(self) -> None:
        self.factor: Any = None
        self.keys = np.empty(0, dtype=np.int64)


def _solve_spd(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    n: int,
    rhs: np.ndarray,
    device: wp.DeviceLike,
    reuse: _HeldFactor | None = None,
) -> np.ndarray:
    """
    Solve the symmetric positive definite system given by COO entries, for ``(n, k)`` columns.

    A direct solve in ``float64``: dense on the host for a small system or without a device, dense
    on a CUDA ``device`` up to ``_DENSE_DEVICE_SOLVE_MAX`` unknowns (unless ``reuse`` asks for a
    factorization kept across a loop), otherwise a sparse Cholesky factorization on ``device``,
    refined, and refactored in place when ``reuse`` holds one of the same pattern.
    """
    if device is None or n <= _DENSE_SOLVE_MAX:
        a = np.zeros((n, n))
        np.add.at(a, (rows, cols), vals)
        return np.linalg.solve(a, rhs)
    import ordito as od
    import ordito.typing as odt
    from ordito._common import read_array

    keys = rows.astype(np.int64) * n + cols
    unique, inverse = np.unique(keys, return_inverse=True)
    values = np.bincount(inverse, weights=vals, minlength=unique.size)
    resolved = wp.get_device(device)
    if resolved.is_cuda and reuse is None and n <= _DENSE_DEVICE_SOLVE_MAX and rhs.shape[1] <= 3:
        return _dense_solve_on_device(unique, values, n, rhs, resolved)
    offsets = np.searchsorted(unique // n, np.arange(n + 1)).astype(np.int32)
    matrix = od.array.bsr_from_csr(
        n,
        n,
        wp.array(offsets, dtype=wp.int32, device=device),
        wp.array((unique % n).astype(np.int32), dtype=wp.int32, device=device),
        wp.array(values, dtype=wp.float64, device=device),
        nnz=int(unique.size),
    )
    if reuse is not None and reuse.factor is not None and np.array_equal(reuse.keys, unique):
        factor = reuse.factor
        factor.refactor(matrix)
    else:
        factor = od.cholesky.sparse_cholesky(matrix)
        if reuse is not None:
            reuse.factor, reuse.keys = factor, unique
    columns = np.ascontiguousarray(rhs.T)
    rhs_wp = odt.as_array2d(wp.array(columns, dtype=wp.float64, device=device), wp.float64)
    solution = odt.as_array2d(wp.zeros(columns.shape, dtype=wp.float64, device=device), wp.float64)
    factor.solve(rhs_wp, solution, tol=1e-15)
    # The answer feeds the host pipeline's next serial step.
    return read_array(solution).T


class _Lists:
    """
    Reads of a topology's half-edge lists for one solve, by whichever way costs less.

    A solve touching a large share of the half-edges converts each list it reads once; a solve
    confined to a small region reads only the entries it touches.
    """

    # Converting a list costs about a quarter of what reading its entries one by one does.
    _WHOLE_FROM_SHARE = 0.25

    def __init__(self, topology: Topology, n_touched: int) -> None:
        self.topology = topology
        self.whole = n_touched >= self._WHOLE_FROM_SHARE * topology.edge_count()
        self.arrays: dict[str, np.ndarray] = {}

    def take(self, name: str, index: np.ndarray) -> np.ndarray:
        """``getattr(topology, name)[index]``."""
        held = self.topology._arrays.get(name)  # pyright: ignore[reportPrivateUsage]
        if held is not None:
            return held[index].astype(np.int64)
        values: list[int] = getattr(self.topology, name)
        if not self.whole:
            return np.array(list(map(values.__getitem__, index.tolist())), dtype=np.int64)
        array = self.arrays.get(name)
        if array is None:
            array = self.arrays[name] = np.asarray(values, dtype=np.int64)
        return array[index]


def _dense_solve_on_device(
    keys: np.ndarray, values: np.ndarray, n: int, rhs: np.ndarray, device: wp.Device
) -> np.ndarray:
    """Solve the system of summed ``values`` at sorted ``keys``: a dense Cholesky on ``device``."""
    from ordito import _launch
    from ordito._common import read_array
    from ordito.kernels import holes as kernel_holes

    tile = kernel_holes.SERIAL_DENSE_TILE
    m = -(-n // tile)
    padded = m * tile
    matrix = wp.zeros((padded, padded), dtype=wp.float64, device=device)
    inverse = _launch.empty((padded, 2 * tile), dtype=wp.float64, device=device)
    _launch.launch(
        kernel_holes.serial_dense_scatter,
        dim=keys.size + padded - n,
        inputs=[
            wp.array((keys // n).astype(np.int32), dtype=wp.int32, device=device),
            wp.array((keys % n).astype(np.int32), dtype=wp.int32, device=device),
            wp.array(values, dtype=wp.float64, device=device),
            n,
        ],
        outputs=[matrix],
        device=device,
    )
    for k in range(m):
        k0 = k * tile
        below = m - k - 1
        _launch.launch_tiled(
            kernel_holes.serial_dense_panel,
            dim=below + 1,
            inputs=[matrix, k0],
            outputs=[inverse],
            device=device,
            block_dim=kernel_holes.SERIAL_DENSE_BLOCK,
        )
        if below:
            _launch.launch_tiled(
                kernel_holes.serial_dense_update,
                dim=below * (below + 1) // 2,
                inputs=[matrix, k0],
                device=device,
                block_dim=kernel_holes.SERIAL_DENSE_BLOCK,
            )
    columns = np.zeros((padded, 3))
    columns[:n, : rhs.shape[1]] = rhs
    staged = _launch.empty(3 * padded, dtype=wp.vec3d, device=device)
    x = staged[2 * padded :]
    _launch.launch_tiled(
        kernel_holes.serial_dense_solve,
        dim=m,
        inputs=[
            matrix,
            inverse,
            wp.array(columns, dtype=wp.vec3d, device=device),
            wp.zeros(2 * m, dtype=wp.int32, device=device),
            staged[:padded],
            staged[padded : 2 * padded],
        ],
        outputs=[x],
        device=device,
        block_dim=kernel_holes.SERIAL_DENSE_SOLVE_BLOCK,
    )
    # The answer feeds the host pipeline's next serial step.
    return read_array(x)[:n, : rhs.shape[1]]


class _Rings:
    """
    Every listed vertex's ring of half-edges, walked in step.

    ``rows`` / ``edges`` list each ring's half-edges from the vertex's stored edge onward, rows in
    the order given and each row in ring order; ``step`` is an entry's position in its ring, so a
    sum taken step by step adds each row's terms in ring order.
    """

    def __init__(self, topology: Topology, verts: np.ndarray) -> None:
        # About seven half-edges leave a vertex, and a cotangent reads four lists of each.
        self.lists = lists = _Lists(topology, 7 * 4 * verts.size)
        start = lists.take("edge_per_vertex", verts)
        rows_by_step: list[np.ndarray] = []
        edges_by_step: list[np.ndarray] = []
        row = np.arange(verts.size)
        cur = start
        while row.size:
            rows_by_step.append(row)
            edges_by_step.append(cur)
            cur = lists.take("nxt", cur)
            more = cur != start[row]
            row, cur = row[more], cur[more]
        step = np.concatenate([np.full(r.size, k) for k, r in enumerate(rows_by_step)])
        rows = np.concatenate(rows_by_step)
        edges = np.concatenate(edges_by_step)
        order = np.lexsort((step, rows))
        self.rows, self.edges, self.step = rows[order], edges[order], step[order]
        self.dest = lists.take("org", self.edges ^ 1)
        self.counts = np.bincount(self.rows, minlength=verts.size)

    def sum_in_ring_order(self, terms: np.ndarray, n_rows: int) -> np.ndarray:
        """Sum each row's ``terms`` (entries in row order) one ring step at a time."""
        out = np.zeros((n_rows, *terms.shape[1:]))
        for k in range(int(self.step.max()) + 1 if self.step.size else 0):
            at = self.step == k
            np.add.at(out, self.rows[at], terms[at])
        return out


def umbrella_fixed_rim(
    topology: Topology, points: Points, verts: list[int], device: wp.DeviceLike = None
) -> None:
    """Place each vertex of ``verts`` at the mean of its neighbours, the others held fixed."""
    if not verts:
        return
    verts_np = np.asarray(verts, dtype=np.int64)
    rings = _Rings(topology, verts_np)
    data = points.data
    index = np.full(topology.vertex_count(), -1, dtype=np.int64)
    index[verts_np] = np.arange(verts_np.size)
    col = index[rings.dest]
    free_nbr = col >= 0
    fixed_terms = np.where(free_nbr[:, None], 0.0, data[rings.dest].astype(np.float64))
    rhs = rings.sum_in_ring_order(fixed_terms, verts_np.size)
    rows = np.concatenate((rings.rows[free_nbr], np.arange(verts_np.size)))
    cols = np.concatenate((col[free_nbr], np.arange(verts_np.size)))
    vals = np.concatenate((np.full(int(free_nbr.sum()), -1.0), rings.counts.astype(np.float64)))
    x = _solve_spd(rows, cols, vals, verts_np.size, rhs, device)
    data[verts_np] = x.astype(np.float32)


def _left_cotans(lists: _Lists, data: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each half-edge's cotangent of the angle opposite it in its left triangle, 0 with none."""
    has = lists.take("lft", edges) >= 0
    a = lists.take("org", edges)
    b = lists.take("org", edges ^ 1)
    c = lists.take("org", lists.take("prv", edges ^ 1) ^ 1)
    u = data[a] - data[c]
    w = data[b] - data[c]
    nom = _dot(u, w)
    den = np.sqrt(_lsq(_cross(u, w)))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        capped = np.abs(nom) >= _FLT_MAX * den
        out = np.where(capped, _FLT_MAX * np.sign(nom).astype(np.float32), nom / den)
    return np.where(has, out, F32(0)).astype(np.float32)


def _laplacian_least_squares(
    topology: Topology,
    data: np.ndarray,
    free_list: list[int],
    values: np.ndarray,
    cotan: bool,
    device: wp.DeviceLike,
    reuse: _HeldFactor | None = None,
) -> np.ndarray:
    """
    Solve for the free values in the least-squares sense, the first ring's ``values`` held fixed.

    Every free vertex and every vertex of their first ring is asked to sit at the weighted mean of
    its neighbours. ``values`` is ``(n_vertices, k)``; weights are clamped cotangents (or ones),
    each row scaled by the reciprocal of its weight sum, and a row whose weights sum to zero is
    dropped.
    """
    n_free = len(free_list)
    free_np = np.asarray(free_list, dtype=np.int64)
    free_rings = _Rings(topology, free_np)
    is_free = np.zeros(topology.vertex_count(), dtype=bool)
    is_free[free_np] = True
    first_layer = np.unique(free_rings.dest[~is_free[free_rings.dest]])
    order = np.concatenate((free_np, first_layer))
    rings = _Rings(topology, order)
    edges = rings.edges
    if cotan:
        both = _left_cotans(rings.lists, data, edges) + _left_cotans(rings.lists, data, edges ^ 1)
        weights = np.clip(both, F32(-1), F32(10)).astype(np.float64)
    else:
        weights = np.ones(edges.size)
    dest = rings.dest
    totals_np = rings.sum_in_ring_order(weights, order.size)
    keep_row = totals_np != 0
    scale = np.where(keep_row, 1.0 / np.where(keep_row, totals_np, 1.0), 0.0)
    row_of_entry = rings.rows
    coeff = -weights * scale[row_of_entry]
    kept_rows = np.flatnonzero(keep_row)
    row_id = np.full(order.size, -1, dtype=np.int64)
    row_id[kept_rows] = np.arange(kept_rows.size)
    index = np.full(topology.vertex_count(), -1, dtype=np.int64)
    index[free_np] = np.arange(n_free)
    free_col = index[dest]
    entry_ok = keep_row[row_of_entry]
    is_free_nbr = (free_col >= 0) & entry_ok
    m_rows = np.concatenate(
        (row_id[np.arange(n_free)][keep_row[:n_free]], row_id[row_of_entry[is_free_nbr]])
    )
    m_cols = np.concatenate((np.flatnonzero(keep_row[:n_free]), free_col[is_free_nbr]))
    m_vals = np.concatenate((np.ones(int(keep_row[:n_free].sum())), coeff[is_free_nbr]))
    k_cols = values.shape[1]
    rhs = np.zeros((len(order), k_cols))
    layer = np.arange(n_free, len(order))
    rhs[layer] = -values[first_layer] if first_layer.size else 0.0
    fixed_nbr = (free_col < 0) & entry_ok
    np.add.at(rhs, row_of_entry[fixed_nbr], -coeff[fixed_nbr, None] * values[dest[fixed_nbr]])
    rhs = rhs[kept_rows]
    # The normal equations M^T M x = M^T b, from M's entries grouped by row.
    order_m = np.argsort(m_rows, kind="stable")
    m_rows, m_cols, m_vals = m_rows[order_m], m_cols[order_m], m_vals[order_m]
    row_starts = np.searchsorted(m_rows, np.arange(kept_rows.size + 1))
    lengths = np.diff(row_starts)
    pair_rows = np.repeat(np.arange(kept_rows.size), lengths * lengths)
    first = np.repeat(row_starts[:-1], lengths * lengths)
    within = np.arange(pair_rows.size) - np.repeat(
        np.cumsum(lengths * lengths) - lengths * lengths, lengths * lengths
    )
    li = np.repeat(lengths, lengths * lengths)
    ia = first + within // li
    ib = first + within % li
    del pair_rows
    a_rows, a_cols, a_vals = m_cols[ia], m_cols[ib], m_vals[ia] * m_vals[ib]
    b = np.zeros((n_free, k_cols))
    np.add.at(b, m_cols, m_vals[:, None] * rhs[m_rows])
    return _solve_spd(a_rows, a_cols, a_vals, n_free, b, device, reuse)


def cotan_fairing(
    topology: Topology, points: Points, verts: list[int], cotan: bool, device: wp.DeviceLike = None
) -> None:
    """
    Fair ``verts`` in the least-squares sense, their first ring held fixed.

    Every vertex of ``verts`` and of their first ring is asked to sit at the weighted mean of its
    neighbours.
    """
    if not verts:
        return
    data = points.data
    free_list = sorted(verts)
    x = _laplacian_least_squares(
        topology, data, free_list, data[: points.size].astype(np.float64), cotan, device
    )
    data[np.array(free_list)] = x.astype(np.float32)


def smooth_patch(
    topology: Topology,
    points: Points,
    new_verts: list[int],
    new_faces: set[int],
    smooth_rim: bool,
    natural: bool,
    cotan: bool,
    device: wp.DeviceLike = None,
) -> None:
    """Smooth the refined patch's new interior vertices, and its collar when ``natural``."""
    walk = topology.reader()
    verts = sorted(v for v in set(new_verts) if not _vertex_on_hole(walk, v))
    umbrella_fixed_rim(topology, points, verts, device)
    if not smooth_rim:
        return
    cotan_fairing(topology, points, verts, cotan, device)
    if not natural:
        return
    incident: set[int] = set()
    for f in new_faces:
        e0 = topology.edge_per_face[f]
        if e0 < 0:
            continue
        for e in topology.left_ring(e0):
            g = topology.lft[e ^ 1]
            if g >= 0 and g not in new_faces:
                incident.add(topology.org[e])
                incident.add(topology.org[e ^ 1])
    incident = expand_verts(topology, incident, 5)
    incident = shrink_verts(topology, incident, 2)
    incident = _exclude_full_components(topology, incident)
    if incident:
        verts = sorted(v for v in incident if not _vertex_on_hole(topology, v))
        umbrella_fixed_rim(topology, points, verts, device)
        cotan_fairing(topology, points, verts, cotan, device)


# ---------------------------------------------------------------------------------------------
# Stitching two holes with a band: a best-first search over rim position pairs
# ---------------------------------------------------------------------------------------------


class _StdHeap:
    """
    A max-heap on the first tuple field whose sift order is the standard library's.

    Equal keys pop in the order ``std::priority_queue`` pops them, which the band search's result
    can depend on; the items are ``(weight, a, b, prev_a, prev_b)`` and the largest-priority item
    is the lightest.
    """

    def __init__(self) -> None:
        self.c: list[tuple[float, int, int, int, int]] = []

    def push(self, value: tuple[float, int, int, int, int]) -> None:
        """Insert ``value``."""
        c = self.c
        c.append(value)
        self._sift_up(len(c) - 1, value)

    def _sift_up(self, hole: int, value: tuple[float, int, int, int, int]) -> None:
        c = self.c
        parent = (hole - 1) // 2
        while hole > 0 and c[parent][0] > value[0]:
            c[hole] = c[parent]
            hole = parent
            parent = (hole - 1) // 2
        c[hole] = value

    def pop(self) -> tuple[float, int, int, int, int]:
        """Remove and return the lightest item."""
        c = self.c
        top = c[0]
        last = len(c) - 1
        value = c[last]
        c[last] = c[0]
        length = last
        hole = 0
        second = 0
        while second < (length - 1) // 2:
            second = 2 * (second + 1)
            if c[second][0] > c[second - 1][0]:
                second -= 1
            c[hole] = c[second]
            hole = second
        if (length & 1) == 0 and second == (length - 2) // 2:
            second = 2 * (second + 1)
            c[hole] = c[second - 1]
            hole = second - 1
        self._sift_up(hole, value)
        c.pop()
        return top

    def __bool__(self) -> bool:
        return bool(self.c)


def stitch_holes(topology: Topology, points: Points, a0: int, b0: int, metric: Metric) -> list[int]:
    """Join the holes left of ``a0`` and ``b0`` with a band of triangles; return its faces."""
    new_faces: list[int] = []
    if topology.lft[a0] >= 0 or topology.lft[b0] >= 0:
        return new_faces
    if a0 < b0:
        a0, b0 = b0, a0
    data = points.data
    org = topology.org
    a_loop = hole_loop(topology, a0)
    b_loop = hole_loop(topology, b0)
    best = _DBL_MAX
    ac = bc = -1
    a_pos = data[[org[e] for e in a_loop]].astype(np.float64)
    b_pos = data[[org[e] for e in b_loop]].astype(np.float64)
    for i, a in enumerate(a_loop):
        diff = a_pos[i] - b_pos
        dist = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
        j = int(np.argmin(dist))
        if dist[j] < best:
            best = float(dist[j])
            ac, bc = a, b_loop[j]
    na, nb = len(a_loop), len(b_loop)
    a_map: list[int] = []
    a = ac
    for _ in range(na):
        a_map.append(a)
        a = topology.prv[a ^ 1]
    b_map: list[int] = []
    b = bc
    for _ in range(nb):
        b_map.append(b)
        b = topology.nxt[b] ^ 1

    start = float(np.sqrt(best))
    native = None
    if metric.band_id >= 0:
        closing = (
            _band_closing_terms(topology, a_map, b_map, metric)
            if metric.has_edge
            else np.zeros((2, 2))
        )
        native = _band_search_native(
            topology, data, a_map, b_map, start, closing, metric.band_id, metric.up
        )
    if native is not None:
        prev_a, prev_b = (t.tolist() for t in native)
    else:
        terms = _band_terms(topology, a_map, b_map, metric)
        prev_a, prev_b = _band_search(
            na, nb, start, terms, metric, _StdHeap(), topology, a_map, b_map
        )

    def has_prev(x: int, y: int) -> bool:
        return prev_a[x][y] != -1 and prev_b[x][y] != -1

    e1 = topology.make_edge()
    topology.splice(ac, e1)
    topology.splice(bc, e1 ^ 1)
    last_a = na
    x, y = na, nb
    while has_prev(x, y) and has_prev(prev_a[x][y], prev_b[x][y]):
        x, y = prev_a[x][y], prev_b[x][y]
        e2 = topology.make_edge()
        sp = a_map[x % na]
        sym_sp = topology.prv[e1 ^ 1]
        if x == last_a:
            sp = e1
            sym_sp = topology.prv[sym_sp ^ 1]
        topology.splice(sp, e2)
        topology.splice(sym_sp, e2 ^ 1)
        f = topology.add_face()
        new_faces.append(f)
        topology.set_left(e1, f)
        last_a = x
        e1 = e2
    f = topology.add_face()
    new_faces.append(f)
    topology.set_left(e1, f)
    return new_faces


class _BandTerms(NamedTuple):
    """
    Every metric term the band search can add, by state ``(a, b)``, as ``(.., na + 1, nb + 1)``.

    ``tri[k]`` is the triangle made by advancing on loop ``k`` (0 for A, 1 for B);
    ``closing[2 k + j]`` the edge term against the triangle the state was reached by, a step on
    loop ``j``; ``opening[k]`` the edge term against the existing triangle across the rim edge
    passed, where ``has_opening[k]``. A metric with no triangle (edge) term has ``tri`` (the
    others) ``None``.
    """

    tri: np.ndarray | None
    closing: np.ndarray | None
    opening: np.ndarray | None
    has_opening: np.ndarray | None


def _band_terms(
    topology: Topology, a_map: list[int], b_map: list[int], metric: Metric
) -> _BandTerms:
    na, nb = len(a_map), len(b_map)
    org = topology.org
    av = np.array([org[e] for e in a_map], dtype=np.int64)
    bv = np.array([org[e] for e in b_map], dtype=np.int64)
    ga, gb = np.meshgrid(np.arange(na + 1), np.arange(nb + 1), indexing="ij")
    a0, b0 = av[ga % na], bv[gb % nb]
    a1, b1 = av[(ga + 1) % na], bv[(gb + 1) % nb]
    tri = closing = opening = has_opening = None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if metric.has_triangle:
            tri = np.stack((metric.triangle(a0, b0, a1), metric.triangle(a0, b0, b1)))
        if metric.has_edge:
            from_a = av[(ga - 1) % na]
            from_b = bv[(gb - 1) % nb]
            closing = np.stack(
                (
                    metric.edge(a0, b0, from_a, a1),
                    metric.edge(a0, b0, from_b, a1),
                    metric.edge(a0, b0, from_a, b1),
                    metric.edge(a0, b0, from_b, b1),
                )
            )

            def across(loop: list[int], idx: np.ndarray) -> np.ndarray:
                vert = np.array(
                    [org[topology.prv[e] ^ 1] if topology.lft[e ^ 1] >= 0 else -1 for e in loop],
                    dtype=np.int64,
                )
                return vert[idx]

            op_a = across(a_map, ga % na)
            op_b = across(b_map, (gb + 1) % nb)
            opening = np.stack(
                (
                    metric.edge(a1, a0, np.maximum(op_a, 0), b0),
                    metric.edge(b0, b1, np.maximum(op_b, 0), a0),
                )
            )
            has_opening = np.stack((op_a >= 0, op_b >= 0))
    return _BandTerms(tri, closing, opening, has_opening)


def _band_closing_terms(
    topology: Topology, a_map: list[int], b_map: list[int], metric: Metric
) -> np.ndarray:
    """
    ``(2, 2)`` band-closing edge terms: a last step on loop ``k``, after a first step on A or not.

    The term ``_band_closing`` adds when the band reaches ``(na, nb)``; its vertices depend only on
    the loop of the last step and on whether the band's first step advanced on A.
    """
    org = topology.org
    na, nb = len(a_map), len(b_map)
    out = np.zeros((2, 2))
    for k, last_from in enumerate((org[a_map[(na - 1) % na]], org[b_map[(nb - 1) % nb]])):
        for j, last in enumerate((org[a_map[1 % na]], org[b_map[1 % nb]])):
            out[k, j] = _scalar(metric.edge, org[a_map[0]], org[b_map[0]], last_from, last)
    return out


def _band_search_native(
    topology: Topology,
    points: np.ndarray,
    a_map: list[int],
    b_map: list[int],
    start: float,
    close_terms: np.ndarray,
    metric_id: int,
    up: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Run the band search natively (``kernels/holes.serial_band_search``), in one CPU thread.

    The kernel forms each metric term as the search reaches its state. Returns each state's
    predecessor tables, or ``None`` when the queue outgrew its capacity.
    """
    from ordito.kernels import holes as kernel_holes

    org = topology.org
    na, nb = len(a_map), len(b_map)
    shape = (na + 1, nb + 1)

    def across(loop: list[int]) -> np.ndarray:
        return np.array(
            [org[topology.prv[e] ^ 1] if topology.lft[e ^ 1] >= 0 else -1 for e in loop],
            dtype=np.int32,
        )

    q = kernel_holes.BandSearch()
    q.points = wp.array(points, dtype=wp.vec3, device="cpu")
    q.a_vert = wp.array(np.array([org[e] for e in a_map], dtype=np.int32), device="cpu")
    q.b_vert = wp.array(np.array([org[e] for e in b_map], dtype=np.int32), device="cpu")
    q.a_across = wp.array(across(a_map), device="cpu")
    q.b_across = wp.array(across(b_map), device="cpu")
    q.metric = metric_id
    u = np.asarray(up if up is not None else (0.0, 0.0, 1.0), dtype=np.float32).astype(np.float64)
    length = float(np.sqrt(_lsq(u)))
    unit = u * (1.0 / length) if length > 0 else np.zeros(3)
    q.up = wp.vec3d(*(float(x) for x in unit))
    q.close_term = wp.array(close_terms, dtype=wp.float64, device="cpu")
    weight = np.full(shape, _DBL_MAX)
    prev_a = np.full(shape, -1, dtype=np.int32)
    prev_b = np.full(shape, -1, dtype=np.int32)
    q.weight = wp.array(weight, dtype=wp.float64, device="cpu", copy=False)
    q.prev_a = wp.array(prev_a, dtype=wp.int32, device="cpu", copy=False)
    q.prev_b = wp.array(prev_b, dtype=wp.int32, device="cpu", copy=False)
    capacity = 4 * (na + 1) * (nb + 1) + 16
    q.heap_w = wp.array(np.empty(capacity), dtype=wp.float64, device="cpu", copy=False)
    heap_s = np.empty((capacity, 4), dtype=np.int32)
    q.heap_s = wp.array(heap_s, dtype=wp.int32, device="cpu", copy=False)
    status = np.zeros(2, dtype=np.int32)
    q.status = wp.array(status, dtype=wp.int32, device="cpu", copy=False)
    wp.launch(kernel_holes.serial_band_search, dim=1, inputs=[q, float(start)], device="cpu")
    if status[1] != 0:
        return None
    return prev_a, prev_b


def _band_search(
    na: int,
    nb: int,
    start: float,
    terms: _BandTerms,
    metric: Metric,
    heap: _StdHeap,
    topology: Topology,
    a_map: list[int],
    b_map: list[int],
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Search the band best first from ``(0, 0)`` to ``(na, nb)``: each state's predecessor.

    A state is updated only by a strictly lighter path, and the queue holds snapshots.
    """
    weight = [[_DBL_MAX] * (nb + 1) for _ in range(na + 1)]
    prev_a = [[-1] * (nb + 1) for _ in range(na + 1)]
    prev_b = [[-1] * (nb + 1) for _ in range(na + 1)]
    weight[0][0] = start
    tri = terms.tri.tolist() if terms.tri is not None else None
    closing = None
    opening = None
    if terms.closing is not None and terms.opening is not None and terms.has_opening is not None:
        closing = [[terms.closing[2 * k + j].tolist() for j in (0, 1)] for k in (0, 1)]
        opening = [
            [
                [t if ok else None for t, ok in zip(row_t, row_ok, strict=True)]
                for row_t, row_ok in zip(
                    terms.opening[k].tolist(), terms.has_opening[k].tolist(), strict=True
                )
            ]
            for k in (0, 1)
        ]
    use_max = metric.use_max
    heap.push((start, 0, 0, -1, -1))
    while heap:
        w, a, b, pa, pb = heap.pop()
        if a == na and b == nb:
            break
        came = -1
        if pa != -1 and pb != -1:
            if pa % na != a % na:
                came = 0
            elif pb % nb != b % nb:
                came = 1
        for k in (0, 1):
            xa, xb = (a + 1, b) if k == 0 else (a, b + 1)
            if xa > na or xb > nb or (xa == na and xb == 0) or (xb == nb and xa == 0):
                continue
            total = w
            if tri is not None:
                t = tri[k][a][b]
                total = max(total, t) if use_max else total + t
            if closing is not None and opening is not None:
                if came >= 0:
                    t = closing[k][came][a][b]
                    total = max(total, t) if use_max else total + t
                t = opening[k][a][b]
                if t is not None:  # the rim edge passed has a triangle across it
                    total = max(total, t) if use_max else total + t
            if total >= weight[xa][xb]:
                continue
            weight[xa][xb] = total
            prev_a[xa][xb] = a
            prev_b[xa][xb] = b
            if xa == na and xb == nb and metric.has_edge:
                total = _band_closing(
                    weight, prev_a, prev_b, topology, a_map, b_map, metric, k, a, b
                )
                weight[xa][xb] = total
            heap.push((total, xa, xb, a, b))
    return prev_a, prev_b


def _band_closing(
    weight: list[list[float]],
    prev_a: list[list[int]],
    prev_b: list[list[int]],
    topology: Topology,
    a_map: list[int],
    b_map: list[int],
    metric: Metric,
    k: int,
    a: int,
    b: int,
) -> float:
    """Add the edge term where the band closes, between its last and its first triangle."""
    na, nb = len(a_map), len(b_map)
    org = topology.org
    # The last step's vertex on the loop it advanced from: A's after advancing on A, else B's.
    b_vert = org[a_map[a % na]] if k == 0 else org[b_map[b % nb]]
    spin = (-1, -1)
    x, y = a, b
    while prev_a[x][y] != -1 and prev_b[x][y] != -1:
        spin = (x, y)
        x, y = prev_a[x][y], prev_b[x][y]
    last = org[a_map[1]] if spin[0] == 1 else org[b_map[1]]
    term = _scalar(metric.edge, org[a_map[0]], org[b_map[0]], b_vert, last)
    w = weight[na][nb]
    return float(max(w, term) if metric.use_max else w + term)


def _scalar(fn: Callable[..., np.ndarray], *verts: int) -> float:
    return float(fn(*(np.array([v]) for v in verts))[0])


# ---------------------------------------------------------------------------------------------
# A smooth bridge between two rim edges: a quadrangle split along a fitted spline
# ---------------------------------------------------------------------------------------------


def _is_left_quad(topology: Topology, e: int) -> bool:
    a = topology.prv[e ^ 1]
    if a == e:
        return False
    b = topology.prv[a ^ 1]
    if b == e:
        return False
    c = topology.prv[b ^ 1]
    if c == e:
        return False
    return topology.prv[c ^ 1] == e


def _quad_bridge(topology: Topology, a: int, b: int, new_faces: list[int]) -> tuple[int, int]:
    """Join rim edges ``a`` and ``b`` with one face; return the two new rim edges (``-1`` none)."""
    na = nb = -1
    if a == b:
        return na, nb
    swapped = False
    if topology.prv[b ^ 1] == a:
        swapped = True
        a, b = b, a
    if topology.prv[a ^ 1] == b:
        if not topology.is_left_tri(a):
            b_dest = topology.org[b ^ 1]
            i = topology.nxt[a]
            while i != a:
                if topology.org[i ^ 1] == b_dest:
                    return na, nb
                i = topology.nxt[i]
            e = topology.make_edge()
            topology.splice(a, e)
            topology.splice(topology.prv[b ^ 1], e ^ 1)
            if swapped:
                nb = e
            else:
                na = e
        f = topology.add_face()
        topology.set_left(a, f)
        new_faces.append(f)
        return na, nb
    b_org, b_dest = topology.org[b], topology.org[b ^ 1]
    for start in (a, a ^ 1):
        for e in topology.org_ring(start):
            if topology.org[e ^ 1] in (b_org, b_dest):
                return na, nb
    c = topology.make_edge()
    e = topology.make_edge()
    topology.splice(topology.prv[a ^ 1], c)
    topology.splice(a, e ^ 1)
    na = e ^ 1
    topology.splice(topology.prv[b ^ 1], e)
    topology.splice(b, c ^ 1)
    nb = c ^ 1
    f = topology.add_face()
    topology.set_left(a, f)
    new_faces.append(f)
    return na, nb


def _split_quad(topology: Topology, a: int, new_faces: list[int]) -> None:
    d = topology.make_edge()
    topology.splice(topology.prv[a ^ 1], d)
    topology.splice(topology.nxt[a] ^ 1, d ^ 1)
    f = topology.add_face()
    topology.set_left(d ^ 1, f)
    new_faces.append(f)


def _resample(
    contour: list[np.ndarray], marks: list[bool], step: np.float32, normals: list[np.ndarray]
) -> tuple[list[np.ndarray], list[bool], list[np.ndarray]]:
    out = [contour[0]]
    out_marks = [True]
    out_normals = [normals[0]]
    i = 0
    while i + 1 < len(contour):
        i1 = i
        distance = F32(0)
        i1 += 1
        while i1 < len(contour):
            distance = distance + np.sqrt(_lsq(contour[i1 - 1] - contour[i1]))
            if marks[i1]:
                break
            i1 += 1
        mid = int(distance / step)
        if mid > 0:
            sub = distance / F32(mid + 1)
            remaining = sub
            i2 = i
            p = contour[i2]
            n = normals[i2]
            while i2 < i1:
                seg = np.sqrt(_lsq(contour[i2 + 1] - p))
                if seg <= remaining:
                    remaining = remaining - seg
                    i2 += 1
                    p = contour[i2]
                    n = normals[i2]
                    continue
                t = remaining / seg
                p = (F32(1) - t) * p + t * contour[i2 + 1]
                out.append(p)
                out_marks.append(False)
                n = _normalized((F32(1) - t) * n + t * normals[i2 + 1])
                out_normals.append(n)
                remaining = sub
        i = i1
        out.append(contour[i])
        out_marks.append(True)
        out_normals.append(normals[i])
    return out, out_marks, out_normals


def _fair_polyline(
    contour: list[np.ndarray], marks: list[bool], stability: float
) -> list[np.ndarray]:
    """Least squares: zero second differences, marked points held with weight ``stability``."""
    size = len(contour)
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for i in range(size - 2):
        row = np.zeros(size)
        row[i], row[i + 1], row[i + 2] = -0.5, 1.0, -0.5
        rows.append(row)
        rhs.append(np.zeros(3))
    for i in range(size):
        if marks[i]:
            row = np.zeros(size)
            row[i] = stability
            rows.append(row)
            rhs.append((F32(stability) * contour[i]).astype(np.float64))
    c = np.array(rows)
    r = np.array(rhs)
    x = np.linalg.solve(c.T @ c, c.T @ r)
    return [x[i].astype(np.float32) for i in range(size)]


def smooth_bridge(topology: Topology, points: Points, a: int, b: int, step: float) -> list[int]:
    """Bridge rim edges ``a`` and ``b`` with a strip split along a spline; return its faces."""
    new_faces: list[int] = []
    na, nb = _quad_bridge(topology, a, b, new_faces)
    if na < 0 and nb < 0:
        return new_faces
    data = points.data
    step_f = F32(step)

    def center(e: int) -> np.ndarray:
        return F32(0.5) * data[topology.org[e ^ 1]] + F32(0.5) * data[topology.org[e]]

    def edge_vector(e: int) -> np.ndarray:
        return data[topology.org[e ^ 1]] - data[topology.org[e]]

    def left_tangent(e: int) -> np.ndarray:
        x, y, z = topology.tri_verts(e)
        normal = _normalized(_cross(data[y] - data[x], data[z] - data[x]))
        return _cross(normal, _normalized(edge_vector(e)))

    center_a, center_b = center(a), center(b)
    tangent_a, tangent_b = left_tangent(a ^ 1), left_tangent(b ^ 1)
    dir_a = _normalized(edge_vector(a))
    dir_b = _normalized(edge_vector(b ^ 1))
    tangent_step = F32(0.99) * step_f
    control = [
        center_a + tangent_step * tangent_a,
        center_a,
        center_b,
        center_b + tangent_step * tangent_b,
    ]
    normals = [dir_a, dir_a, dir_b, dir_b]
    contour = list(control)
    marks = [True] * 4
    for it in range(3):
        if it > 0:
            k = 0
            for j, m in enumerate(marks):
                if m:
                    contour[j] = control[k]
                    k += 1
        contour, marks, normals = _resample(contour, marks, step_f, normals)
        contour = _fair_polyline(contour, marks, 10.0)

    ca = na
    region = set(new_faces)
    mid = len(normals) - 4
    if mid > 0:
        len_a = np.sqrt(_lsq(edge_vector(a)))
        len_b = np.sqrt(_lsq(edge_vector(b)))
        for i in range(mid):
            u = F32(i + 1) / F32(mid + 1)
            length = (F32(1) - u) * len_a + u * len_b
            if ca >= 0:
                p = contour[i + 2] - (F32(0.5) * length) * normals[i + 2]
                e = topology.split_edge(ca, region)
                points.set(topology.org[ca], p)
                if topology.vertex_rows is not None:
                    topology.vertex_rows[topology.org[ca]] = {}
                if i == 0:
                    na = e
            if nb >= 0:
                p = contour[i + 2] + (F32(0.5) * length) * normals[i + 2]
                topology.split_edge(nb ^ 1, region)
                points.set(topology.org[nb ^ 1], p)
                if topology.vertex_rows is not None:
                    topology.vertex_rows[topology.org[nb ^ 1]] = {}
    new_faces = sorted(region)
    if ca >= 0 and _is_left_quad(topology, ca ^ 1):
        _split_quad(topology, topology.prv[ca], new_faces)
    return new_faces


# ---------------------------------------------------------------------------------------------
# Straightening a region's rim: band flips, then a harmonic field's zero set
# ---------------------------------------------------------------------------------------------


def smooth_region_rim(
    topology: Topology,
    points: Points,
    region: list[bool],
    iterations: int,
    device: wp.DeviceLike = None,
) -> None:
    """Flip the rim band of the face ``region``, then slide its vertices onto a level set."""
    n_verts = topology.vertex_count()
    if not any(region) or iterations <= 0:
        return
    from ordito.kernels import holes as kernel_holes

    tris = topology.triangles()
    region_np = np.asarray(region, dtype=bool)
    valid = (tris >= 0).all(axis=1)
    inside = np.zeros(n_verts, dtype=bool)
    outside = np.zeros(n_verts, dtype=bool)
    inside[tris[valid & region_np].reshape(-1)] = True
    outside[tris[valid & ~region_np].reshape(-1)] = True
    field = np.where(inside, F32(-1), F32(1)).astype(np.float32)
    free_mask = inside & outside
    # A component whose every vertex is free would leave the system singular: fix it. A component
    # holding a fixed vertex is one where some free vertex has a fixed neighbour, so the search
    # stays inside the free set.
    walk = topology.reader()
    seen: set[int] = set()
    for v in np.flatnonzero(free_mask).tolist():
        if v in seen:
            continue
        group = [v]
        seen.add(v)
        touches_fixed = False
        k = 0
        while k < len(group):
            for d in _neighbours(walk, group[k]):
                if not free_mask[d]:
                    touches_fixed = True
                elif d not in seen:
                    seen.add(d)
                    group.append(d)
            k += 1
        if not touches_fixed:
            free_mask[group] = False
    free_list = np.flatnonzero(free_mask).astype(np.int32)
    native = _NativeMesh(topology, points, region_flags=region_np.astype(np.int32))
    free_wp = wp.array(free_mask.astype(np.int32), dtype=wp.int32, device="cpu")
    free_list_wp = wp.array(free_list, dtype=wp.int32, device="cpu")
    wp.launch(
        kernel_holes.serial_rim_flips,
        dim=1,
        inputs=[native.mesh, free_wp, free_list_wp],
        device="cpu",
    )
    native.store(topology)
    points.data = native.positions
    data = points.data
    field_wp = wp.array(field, dtype=wp.float32, device="cpu", copy=False)
    moved = np.empty((free_list.size, 3), dtype=np.float32)
    moved_wp = wp.array(moved, dtype=wp.vec3, device="cpu", copy=False)
    free_python = free_list.tolist()
    # Every iteration solves over the same pattern: only the cotangents move.
    reuse = _HeldFactor()
    for _ in range(iterations):
        _harmonic_field(topology, data, free_python, field, device, reuse)
        if free_list.size:
            wp.launch(
                kernel_holes.rim_slide,
                dim=free_list.size,
                inputs=[native.mesh, field_wp, free_list_wp],
                outputs=[moved_wp],
                device="cpu",
            )
        data[free_list] = data[free_list] + F32(0.75) * (moved - data[free_list])


def _harmonic_field(
    topology: Topology,
    data: np.ndarray,
    free_list: list[int],
    field: np.ndarray,
    device: wp.DeviceLike = None,
    reuse: _HeldFactor | None = None,
) -> None:
    """Least-squares cotangent-harmonic values of ``field`` at ``free_list``, the rest fixed."""
    if not free_list:
        return
    x = _laplacian_least_squares(
        topology, data, free_list, field.astype(np.float64)[:, None], True, device, reuse
    )
    field[np.array(free_list)] = x[:, 0].astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Replacing a face region: its boundary loops, the deletion, a fill of each loop
# ---------------------------------------------------------------------------------------------


def _left_bd(topology: _Walk, e: int, region: set[int]) -> bool:
    return _in_region(topology, e, region) and not _in_region(topology, e ^ 1, region)


def region_boundary_loops(topology: Topology, region: set[int]) -> list[list[int]]:
    """Return the region's left boundary loops (leftmost turns), cut at repeated vertices."""
    reported: set[int] = set()
    loops: list[list[int]] = []
    # The left boundary edges, found in one vectorized pass: a left face in the region, the right
    # face not. Only these start a loop, in increasing id order as the full scan met them.
    lft = topology.array("lft").astype(np.int64)
    in_region = np.zeros(topology.face_count() + 1, dtype=bool)
    in_region[np.fromiter(region, dtype=np.int64, count=len(region))] = True
    left_in = in_region[lft]  # an empty face (-1) reads the trailing False slot
    starts = np.flatnonzero(left_in & ~left_in[np.arange(lft.size) ^ 1])
    walk = topology.reader()
    for e in starts.tolist():
        if e in reported:
            continue
        loop = []
        x = e
        while True:
            loop.append(x)
            reported.add(x)
            x = walk.prv[x ^ 1]
            while not _left_bd(walk, x, region):
                x = walk.prv[x]
            if x == e:
                break
        loops.append(loop)
    out: list[list[int]] = []
    for loop in loops:
        while True:
            seen: dict[int, int] = {}
            split = False
            for i, x in enumerate(loop):
                v = walk.org[x]
                if v in seen:
                    out.append(loop[seen[v] : i])
                    del loop[seen[v] : i]
                    split = True
                    break
                seen[v] = i
            if not split:
                out.append(loop)
                break
    return out


def delete_region_keep_rims(topology: Topology, region: set[int]) -> list[list[int]]:
    """Delete the faces of ``region`` keeping its rims; return the rims with a face outside."""
    loops = region_boundary_loops(topology, region)
    kept: list[list[int]] = []
    keep_edges: set[int] = set()
    lft = topology.reader().lft
    for loop in loops:
        if all(lft[x ^ 1] < 0 for x in loop):
            continue
        keep_edges.update(x >> 1 for x in loop)
        kept.append(loop)
    from ordito.kernels import holes as kernel_holes

    faces = np.array(sorted(region), dtype=np.int32)
    if faces.size:
        keep = np.zeros(topology.edge_count() // 2, dtype=np.int32)
        keep[np.fromiter(keep_edges, dtype=np.int64, count=len(keep_edges))] = 1
        native = _NativeMesh(topology, None)
        wp.launch(
            kernel_holes.serial_delete_faces,
            dim=1,
            inputs=[
                native.mesh,
                wp.array(faces, dtype=wp.int32, device="cpu"),
                wp.array(keep, dtype=wp.int32, device="cpu"),
            ],
            device="cpu",
        )
        native.store(topology)
    return kept


# ---------------------------------------------------------------------------------------------
# Whole operations, from host arrays to host arrays
# ---------------------------------------------------------------------------------------------


class Result(NamedTuple):
    """An edited mesh on the host: positions, triangles, the patch and where things came from."""

    vertices: np.ndarray
    faces: np.ndarray
    patch: np.ndarray
    sources: np.ndarray | None
    weights: np.ndarray | None
    face_source: np.ndarray | None


def _result(
    topology: Topology, points: Points, patch: set[int], n_input: int, was_valid: np.ndarray
) -> Result:
    """
    Export the edited mesh: faces and vertices in id order.

    A vertex the edit deleted is dropped; an input vertex no face ever used is kept.
    """
    keep = topology.array("edge_per_vertex") >= 0
    n_total = keep.size
    keep[:n_input] |= ~was_valid
    remap = np.cumsum(keep) - 1
    tris = topology.triangles()
    valid_face = (tris >= 0).all(axis=1)
    faces = remap[tris[valid_face]].astype(np.int32)
    vertices = points.array()[:n_total][keep].copy()
    in_patch = np.zeros(valid_face.size, dtype=bool)
    in_patch[np.fromiter(patch, dtype=np.int64, count=len(patch))] = True
    patch_mask = in_patch[valid_face]
    sources = weights = face_source = None
    if topology.vertex_rows is not None and topology.face_source is not None:
        sources, weights = topology.vertex_rows.table(np.flatnonzero(keep))
        face_source = np.array(topology.face_source, dtype=np.int32)[valid_face]
    return Result(vertices, faces, patch_mask, sources, weights, face_source)


def _start(
    vertices: np.ndarray, faces: np.ndarray, track: bool
) -> tuple[Topology, Points, np.ndarray]:
    """
    Build the half-edge structure of the input and its position table.

    Raises
    ------
    ValueError
        If a face cannot be linked: a repeated corner, an edge shared by three or more faces or
        by two wound the same way, or a vertex whose faces form several separate fans.
    """
    n = vertices.shape[0]
    topology = build(faces, n)
    epf = topology.array("edge_per_face")
    unplaced = int(np.count_nonzero(epf < 0)) + faces.shape[0] - epf.shape[0]
    if unplaced:
        raise ValueError(
            f"method='meshlib' needs an oriented manifold mesh: {unplaced} face(s) cannot be "
            "linked (a repeated corner, an edge in three or more faces or wound the same way "
            "twice, or a vertex whose faces form separate fans)"
        )
    epv = topology.array("edge_per_vertex")
    if epv.shape[0] < n:
        epv = np.concatenate((epv, np.full(n - epv.shape[0], -1, dtype=epv.dtype)))
        topology.set_array("edge_per_vertex", epv)
    if track:
        topology.track(n)
    was_valid = epv[:n] >= 0
    return topology, Points(vertices), was_valid


def _refine_and_smooth(
    topology: Topology,
    points: Points,
    new_faces: list[int],
    max_edge: float,
    max_splits: int,
    max_angle: float,
    smooth_curvature: bool,
    smooth_rim: bool,
    natural: bool,
    cotan: bool,
    device: wp.DeviceLike,
) -> set[int]:
    region = set(new_faces)
    new_verts: list[int] = []
    subdivide(topology, points, region, max_edge, max_splits, max_angle, new_verts)
    if smooth_curvature:
        smooth_patch(topology, points, new_verts, region, smooth_rim, natural, cotan, device)
    return region


def _loop_length(topology: Topology, data: np.ndarray, loop: list[int]) -> float:
    org = topology.reader().org
    return sum(float(np.sqrt(_lsq(data[org[x ^ 1]] - data[org[x]]))) for x in loop)


def fill_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    metric: str,
    triangulate_only: bool,
    max_edge: float | None,
    max_splits: int,
    max_angle: float,
    smooth_curvature: bool,
    natural: bool,
    cotan: bool,
    avoid_existing: bool,
    smooth_rim: bool,
    preserve_largest: bool,
    track: bool,
    device: wp.DeviceLike = None,
) -> Result:
    """Fill every hole one at a time in rim id order: triangulate, refine, smooth."""
    topology, points, was_valid = _start(vertices, faces, track)
    holes = hole_representatives(topology)
    loops = [hole_loop(topology, h) for h in holes]
    if preserve_largest:
        fillable = [i for i, loop in enumerate(loops) if len(loop) >= 3]
        if fillable:
            lengths = [_loop_length(topology, points.data, loops[i]) for i in fillable]
            drop = fillable[int(np.argmax(lengths))]
            holes = holes[:drop] + holes[drop + 1 :]
            loops = loops[:drop] + loops[drop + 1 :]
    target = max_edge
    if target is None:
        edges = [x for loop in loops if len(loop) >= 3 for x in loop]
        target = _loop_length(topology, points.data, edges) / len(edges) if edges else 0.0
    patch: set[int] = set()
    for a0 in holes:
        new_faces = fill_hole(topology, points, a0, metric, smooth_rim, avoid_existing, device)
        if triangulate_only or not new_faces:
            patch.update(new_faces)
            continue
        patch |= _refine_and_smooth(
            topology,
            points,
            new_faces,
            target,
            max_splits,
            max_angle,
            smooth_curvature,
            smooth_rim,
            natural,
            cotan,
            device,
        )
    return _result(topology, points, patch, vertices.shape[0], was_valid)


def refill(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_mask: np.ndarray,
    *,
    metric: str,
    triangulate_only: bool,
    max_edge: float | None,
    smooth_curvature: bool,
    track: bool,
    device: wp.DeviceLike = None,
) -> Result:
    """Delete the face region, then fill each rim it opened from that rim's first edge."""
    topology, points, was_valid = _start(vertices, faces, track)
    loops = delete_region_keep_rims(topology, set(np.flatnonzero(region_mask).tolist()))
    patch: set[int] = set()
    for loop in loops:
        target = max_edge
        if target is None or target <= 0:
            target = float(F32(_loop_length(topology, points.data, loop) / len(loop)) * F32(1.5))
        if topology.reader().lft[loop[0]] >= 0:
            continue
        new_faces = fill_hole(topology, points, loop[0], metric, True, True, device)
        if triangulate_only or not new_faces:
            patch.update(new_faces)
            continue
        patch |= _refine_and_smooth(
            topology,
            points,
            new_faces,
            target,
            1000,
            float(F32(30) * _PI_F / F32(180)),
            smooth_curvature,
            True,
            False,
            True,
            device,
        )
    return _result(topology, points, patch, vertices.shape[0], was_valid)


def stitch(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_vertices_a: int,
    *,
    metric: str,
    up: tuple[float, float, float] | None,
    triangulate_only: bool,
    max_edge: float | None,
    max_splits: int,
    max_angle: float,
    smooth_curvature: bool,
    natural: bool,
    cotan: bool,
    track: bool,
    device: wp.DeviceLike = None,
) -> Result:
    """Band the one hole of the first ``n_vertices_a`` vertices to the hole of the rest."""
    topology, points, was_valid = _start(vertices, faces, track)
    holes = hole_representatives(topology)
    a0 = next(h for h in holes if topology.org[h] < n_vertices_a)
    b0 = next(h for h in holes if topology.org[h] >= n_vertices_a)
    target = max_edge
    if target is None:
        edges = hole_loop(topology, a0) + hole_loop(topology, b0)
        target = _loop_length(topology, points.data, edges) / len(edges)
    new_faces = stitch_holes(topology, points, a0, b0, stitch_metric(metric, points.array(), up))
    patch = set(new_faces)
    if not triangulate_only and new_faces:
        patch = _refine_and_smooth(
            topology,
            points,
            new_faces,
            target,
            max_splits,
            max_angle,
            smooth_curvature,
            True,
            natural,
            cotan,
            device,
        )
    return _result(topology, points, patch, vertices.shape[0], was_valid)


def _rim_half_edge(topology: Topology, u: int, v: int) -> int:
    e = topology.find_edge(u, v)
    if e >= 0 and topology.lft[e] < 0:
        return e
    e = topology.find_edge(v, u)
    return e if e >= 0 and topology.lft[e] < 0 else -1


def bridge(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    step: float,
    *,
    track: bool,
) -> Result:
    """Bridge the rim edges ``edge_a`` and ``edge_b`` (vertex pairs) with a smooth strip."""
    topology, points, was_valid = _start(vertices, faces, track)
    a = _rim_half_edge(topology, *edge_a)
    b = _rim_half_edge(topology, *edge_b)
    patch = set(smooth_bridge(topology, points, a, b, step)) if a >= 0 and b >= 0 else set()
    return _result(topology, points, patch, vertices.shape[0], was_valid)


def rim_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_mask: np.ndarray,
    iterations: int,
    device: wp.DeviceLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Straighten the rim of the face region: band flips, then level-set slides."""
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    topology, points, _ = _start(vertices, faces, False)
    smooth_region_rim(topology, points, [bool(x) for x in region_mask], iterations, device)
    tris = topology.triangles()
    return points.array()[: vertices.shape[0]].copy(), tris
