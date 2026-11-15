"""
Kernels for OpenVDB's dual marching cubes (``openvdb::tools::VolumeToMesh`` at adaptivity 0).

A *dual* extraction: where marching cubes puts a vertex on every lattice edge the level crosses and
triangulates inside each cell, this puts one vertex inside each cell the level passes through (more
than one where the cell's sign pattern splits its crossing edges into separate groups) and joins
the four cells around every crossing edge with a quad. The quad is split into two triangles.

Cell ``(i, j, k)`` is the cube whose lowest corner is node ``(i, j, k)``; its corners ``0``-``7``
sit at offsets ``(0,0,0) (1,0,0) (1,0,1) (0,0,1) (0,1,0) (1,1,0) (1,1,1) (0,1,1)``, and bit ``c`` of
its sign mask is set when corner ``c``'s value is below the level (inside). ``EDGE_GROUPS[mask]``
holds the cell's vertex count and, for each of its twelve edges, the 1-based vertex the edge's
crossing belongs to (``0``: not crossed); a vertex is the mean of its edges' crossings. A cell with
an *ambiguous* face (one whose diagonal corners agree and adjacent ones differ) takes the complement
of its mask when the cell across that face has the same face ambiguous, so the two group their
crossings alike (``dmc_corrected_mask``).

Vertices are numbered by cell, cells in row-major order of their local index ``(i, j, k)``, then
by group; quads follow in the same cell order, the cell's ``x``, ``y`` and ``z`` edges in turn.
OpenVDB numbers them as its tree visits the cells (by 8**3 leaf), which gives the same mesh up to
that numbering. Cells are placed by absolute index coordinates (the lattice's ``origin`` plus the
local index).

The tables are OpenVDB's ``sEdgeGroupTable`` and ``sAmbiguousFace`` (``tools/VolumeToMesh.h``,
v12.0.1), used under the Apache License 2.0:

    Copyright Contributors to the OpenVDB Project
    SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import warp as wp

from ordito.kernels.array import OverloadTable

wp.set_module_options({"enable_backward": False})

# ``EDGE_GROUPS[mask]``: 13 digits a mask, the vertex count and then edge columns 1-12.
_EDGE_GROUPS = (
    "0000000000000110010000100011100000001001010100001100101100000001011111000010101101000000110"
    "1001100001110100110000000111010000010011111100000101101100000110110101000000111110000001011"
    "1100100000111100000000111110000100110001100110010000111001001110010101100101002011020022010"
    "1111110010010110101001111010011100101101001110011001110101001000122112100121011011010010101"
    "1010110011011111001001001121001200212221000010010111100001100010011001110011001110011000000"
    "1010111001000101101100011022211110012101101011000010100111100101020022110001021101011001101"
    "1111111000001101101100100110101110001112110022002122110011100001110000110010111000001011100"
    "1100101010100111000101100010101010100001011001011110211220202012011010010110101001101010010"
    "1001101011101110100101010121221020210021011001010001202020101221122200010102112200201012011"
    "1000001010011100000110001021001022010201110001100110101010110111010110011000001111101101000"
    "1101001100100100110110110010011011000111101001101011212210110012120110022022121010101100001"
    "1110001101001110010110010110000011011011000011111010110011111001021100221112102020211220120"
    "1011011111000222112211000022020112221002001122110200200111122101222020221100214322344113421"
    "3022011330123202022211200121100112200023100122331203200001122010210000101001101100110101110"
    "1110010100010101011010101010110101001002112220202100110101010000010011101010001001110100111"
    "2202010101221221122020001220220101010211010110100101222001010120111001101000011000010101001"
    "1000000111110110010011011011100001110101010100110010101100011110022211001102001101000111000"
    "1001100110000200220011222121010002201123211200332013201100022001220101002211022110000220102"
    "2100100221002100000011000110000001100011100100111001211000022010210101001111011011000110011"
    "2112200111012110100011011120011002222211001100110000110100011100011111001101001011000111100"
    "1010100110010111000011101011001001101101000000111110100001010100111001101000011110010101101"
    "1010110100101101101010101122211202000212101020201221200221010011210011101010001101010100000"
    "2122120201200101101010010010101101010101110010100010220021010211010000101001101000011110101"
    "2100121122101111001111000120202122110022011012210121411332442214322020211200123001123322031"
    "1001111110100220201221120022112211200002011021122000202022112021031100322331202100112210020"
    "2000021122010100000110110111001011001011110001101001101010110000120220011022123112203300132"
    "2101002201021200110220002110011011011001101001100100221120110200010110011000002010102201120"
    "2110002200120210010220102010000011000101000001010011110010101101111100010101112020201011122"
    "1011001010001222110202200111010010101012002201011102100110101001011010010110102211202020210"
    "2022001011120101010101000011100010110001100101010100100000101110010000110010111100111000011"
    "2220011002122201012200021110110110010012112211000002210102200120120022110001021001111001010"
    "1101011000010312213300132020110220002101010111001000111001100000022002110021001000011000100"
    "1000010010111220021001122211100100100112010120022011101101001010131133200221031101010010001"
    "2002210011002100111001011021010200221102122110010020201102002201010101100101002110020022100"
    "1100110010000100001001100010000000011111100100000111111000000101110101000000111011000001101"
    "2112200000102110100000100110011000000011001100001110110100000011021221000010201011000000010"
    "1010100001100111000000010011001000010000000000000000"
)

# ``AMBIGUOUS_FACES[mask]``: the cell face (1-6) the mask leaves ambiguous, ``0`` if none.
_AMBIGUOUS_FACES = (
    "0000050000500000001005104000400001002000015020000000200040000000"
    "0022050033000000660060000000000001000000300000000000000000000000"
    "0404303000500000001030000000000060600000600000000000000000000000"
    "0420000000000000000000000000000000000000000000000000000000000000"
)

# Edge column ``col`` (1-12) joins corners ``_EDGE_CORNERS[2 * (col - 1)]`` and the next entry, in
# the order OpenVDB interpolates them (``computePoint``).
_EDGE_CORNERS = (0, 1, 1, 2, 3, 2, 0, 3, 4, 5, 5, 6, 7, 6, 4, 7, 0, 4, 1, 5, 2, 6, 3, 7)

# Offsets into ``DMC_TABLE``.
_AMBIGUOUS = 256 * 13
_CORNERS = _AMBIGUOUS + 256


def _table() -> np.ndarray:
    """``DMC_TABLE``'s entries: the edge groups, the ambiguous faces, then the edge corners."""
    groups = np.frombuffer("".join(_EDGE_GROUPS).encode(), dtype=np.uint8) - ord("0")
    faces = np.frombuffer("".join(_AMBIGUOUS_FACES).encode(), dtype=np.uint8) - ord("0")
    assert groups.size == _AMBIGUOUS
    assert faces.size == 256
    return np.concatenate([groups, faces, np.asarray(_EDGE_CORNERS)]).astype(np.int32)


# The lookup tables as one ``int32`` buffer, uploaded once per device by the wrapper.
DMC_TABLE = _table()

# The cell flags ``dmc_cell_flags`` stores, OpenVDB's layout: the corrected sign mask in the low
# byte, corner 0 inside, and which of the cell's three lowest edges cross (judged on the mask
# before correction). ``0`` marks a cell the level does not pass through.
_INSIDE = 0x100
_X_EDGE = 0x200
_Y_EDGE = 0x400
_Z_EDGE = 0x800


@wp.func
def dmc_corner(c: wp.int32) -> wp.vec3i:
    # Corner ``c``'s offset from the cell's lowest node.
    return wp.vec3i((c ^ (c >> 1)) & 1, (c >> 2) & 1, (c >> 1) & 1)


@wp.func
def dmc_in_cells(cell: wp.vec3i, cells: wp.vec3i) -> wp.bool:
    return (
        cell[0] >= 0
        and cell[1] >= 0
        and cell[2] >= 0
        and cell[0] < cells[0]
        and cell[1] < cells[1]
        and cell[2] < cells[2]
    )


@wp.func
def dmc_raw_mask(field: wp.array3d[wp.Float], iso: wp.Float, cell: wp.vec3i) -> wp.int32:
    # Bit ``c`` set where corner ``c`` is strictly below ``iso`` (a value equal to it is outside).
    mask = wp.int32(0)
    for c in range(8):
        node = cell + dmc_corner(c)
        if field[node[0], node[1], node[2]] < iso:
            mask = mask | (1 << c)
    return mask


@wp.func
def dmc_corrected_mask(
    table: wp.array[wp.int32],
    field: wp.array3d[wp.Float],
    iso: wp.Float,
    cells: wp.vec3i,
    cell: wp.vec3i,
    mask: wp.int32,
) -> wp.int32:
    # OpenVDB's ``correctCellSigns``: complement the mask when the cell across its ambiguous face
    # has that same face ambiguous, judged on the neighbour's uncorrected mask. A neighbour off the
    # lattice corrects nothing.
    face = table[_AMBIGUOUS + mask]
    if face == 0:
        return mask
    step = wp.vec3i(0, 0, -1)
    partner = wp.int32(3)
    if face == 2:
        step = wp.vec3i(1, 0, 0)
        partner = 4
    elif face == 3:
        step = wp.vec3i(0, 0, 1)
        partner = 1
    elif face == 4:
        step = wp.vec3i(-1, 0, 0)
        partner = 2
    elif face == 5:
        step = wp.vec3i(0, -1, 0)
        partner = 6
    elif face == 6:
        step = wp.vec3i(0, 1, 0)
        partner = 5
    other = cell + step
    if not dmc_in_cells(other, cells):
        return mask
    if table[_AMBIGUOUS + dmc_raw_mask(field, iso, other)] == partner:
        return (~mask) & 255
    return mask


@wp.kernel
def dmc_cell_flags(
    field: wp.array3d[wp.Float],
    iso: wp.Float,
    table: wp.array[wp.int32],
    out_flags: wp.array3d[wp.uint16],
) -> None:
    """Each cell's flags (``_INSIDE``, ``_X_EDGE`` ...): ``0`` where the level does not cross it."""
    i, j, k = wp.tid()
    cell = wp.vec3i(i, j, k)
    cells = wp.vec3i(out_flags.shape[0], out_flags.shape[1], out_flags.shape[2])
    mask = dmc_raw_mask(field, iso, cell)
    flags = wp.int32(0)
    if mask != 0 and mask != 255:
        flags = dmc_corrected_mask(table, field, iso, cells, cell, mask)
        if mask & 1 != 0:
            flags = flags | _INSIDE
        if (mask & 1) != ((mask >> 1) & 1):
            flags = flags | _X_EDGE
        if (mask & 1) != ((mask >> 4) & 1):
            flags = flags | _Y_EDGE
        if (mask & 1) != ((mask >> 3) & 1):
            flags = flags | _Z_EDGE
    out_flags[i, j, k] = wp.uint16(flags)


@wp.func
def dmc_slot(flags: wp.array3d[wp.uint16], cell: wp.vec3i) -> wp.int32:
    # The cell's position in row-major cell order, its counts' slot.
    return (cell[0] * flags.shape[1] + cell[1]) * flags.shape[2] + cell[2]


@wp.func
def dmc_flags_at(flags: wp.array3d[wp.uint16], cell: wp.vec3i) -> wp.int32:
    # A cell's flags, ``0`` off the lattice.
    cells = wp.vec3i(flags.shape[0], flags.shape[1], flags.shape[2])
    if not dmc_in_cells(cell, cells):
        return 0
    return wp.int32(flags[cell[0], cell[1], cell[2]])


@wp.func
def dmc_quad_neighbour(cell: wp.vec3i, axis: wp.int32, slot: wp.int32) -> wp.vec3i:
    # The cells after ``cell`` around its lowest edge along ``axis``, in quad order (slots 1-3).
    a = wp.vec3i(0, -1, 0)
    b = wp.vec3i(0, 0, -1)
    if axis == 1:
        a = wp.vec3i(0, 0, -1)
        b = wp.vec3i(-1, 0, 0)
    elif axis == 2:
        a = wp.vec3i(0, -1, 0)
        b = wp.vec3i(-1, 0, 0)
    if slot == 1:
        return cell + a
    if slot == 2:
        return cell + a + b
    return cell + b


@wp.func
def dmc_quad_count(flags: wp.array3d[wp.uint16], cell: wp.vec3i, own: wp.int32) -> wp.int32:
    # How many of the cell's three lowest edges cross with all four cells around them crossed.
    quads = wp.int32(0)
    for axis in range(3):
        if own & (_X_EDGE << axis) != 0:
            closed = wp.bool(True)
            for slot in range(1, 4):
                if dmc_flags_at(flags, dmc_quad_neighbour(cell, axis, slot)) == 0:
                    closed = False
            if closed:
                quads += 1
    return quads


@wp.kernel
def dmc_counts(
    flags: wp.array3d[wp.uint16], table: wp.array[wp.int32], out_counts: wp.array[wp.vec2i]
) -> None:
    """Each cell's vertex and quad counts in row-major cell order, zero where nothing crosses."""
    i, j, k = wp.tid()
    cell = wp.vec3i(i, j, k)
    own = wp.int32(flags[i, j, k])
    counts = wp.vec2i(0, 0)
    if own != 0:
        counts = wp.vec2i(table[(own & 255) * 13], dmc_quad_count(flags, cell, own))
    out_counts[dmc_slot(flags, cell)] = counts


@wp.func
def dmc_vertex(
    flags: wp.array3d[wp.uint16],
    table: wp.array[wp.int32],
    counts: wp.array[wp.vec2i],
    cell: wp.vec3i,
    column: wp.int32,
) -> wp.int32:
    # The vertex of ``cell`` that its edge ``column``'s crossing belongs to.
    mask = dmc_flags_at(flags, cell) & 255
    row = mask * 13
    first = counts[dmc_slot(flags, cell)][0] - table[row]
    if table[row] > 1:
        first += table[row + column] - 1
    return first


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def dmc_emit(
    field: wp.array3d[wp.Float],
    iso: wp.Float,
    flags: wp.array3d[wp.uint16],
    table: wp.array[wp.int32],
    origin: wp.vec3i,
    counts: wp.array[wp.vec2i],
    lower: wp.vec3,
    scale: wp.vec3,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    """
    Write each crossed cell's vertices and the quads of its lowest edges, two triangles each.

    ``counts`` is ``dmc_counts``' output scanned inclusively. A vertex is OpenVDB's: each crossing
    ``(iso - v0) / (v1 - v0)`` in ``float64``, summed per coordinate in column order, scaled by the
    reciprocal of their count, added to the cell's absolute index and rounded to ``float32``; it is
    placed at ``lower + p * scale``. Built unfused, so that scaling and the add round apart.
    """
    i, j, k = wp.tid()
    cell = wp.vec3i(i, j, k)
    own = wp.int32(flags[i, j, k])
    if own == 0:
        return
    mask = own & 255
    row = mask * 13
    groups = table[row]
    scanned = counts[dmc_slot(flags, cell)]
    first = scanned[0] - groups
    absolute = wp.vec3d(
        wp.float64(origin[0] + i), wp.float64(origin[1] + j), wp.float64(origin[2] + k)
    )
    for group in range(1, groups + 1):
        total = wp.vec3d(0.0, 0.0, 0.0)
        samples = wp.int32(0)
        for column in range(1, 13):
            if table[row + column] == group:
                c0 = table[_CORNERS + 2 * (column - 1)]
                c1 = table[_CORNERS + 2 * (column - 1) + 1]
                p0 = dmc_corner(c0)
                p1 = dmc_corner(c1)
                n0 = cell + p0
                n1 = cell + p1
                v0 = wp.float64(field[n0[0], n0[1], n0[2]])
                v1 = wp.float64(field[n1[0], n1[1], n1[2]])
                t = (wp.float64(iso) - v0) / (v1 - v0)
                for axis in range(3):
                    if p0[axis] != p1[axis]:
                        total[axis] = total[axis] + t
                    elif p0[axis] != 0:
                        total[axis] = total[axis] + wp.float64(1.0)
                samples += 1
        if samples > 1:
            w = wp.float64(1.0) / wp.float64(samples)
            total = wp.vec3d(total[0] * w, total[1] * w, total[2] * w)
        point = wp.vec3(total + absolute)
        out_vertices[first + group - 1] = lower + wp.cw_mul(point, scale)

    quads = dmc_quad_count(flags, cell, own)
    if quads == 0:
        return
    corner = 6 * (scanned[1] - quads)
    # Each axis's first vertex column and the columns of the three cells after this one around
    # the edge: x ``1 / 5 7 3``, y ``9 / 12 11 10``, z ``4 / 8 6 2``.
    for axis in range(3):
        if own & (_X_EDGE << axis) != 0:
            closed = wp.bool(True)
            for slot in range(1, 4):
                if dmc_flags_at(flags, dmc_quad_neighbour(cell, axis, slot)) == 0:
                    closed = False
            if closed:
                columns = wp.vec4i(1, 5, 7, 3)
                reversed_quad = own & _INSIDE != 0
                if axis == 1:
                    columns = wp.vec4i(9, 12, 11, 10)
                elif axis == 2:
                    columns = wp.vec4i(4, 8, 6, 2)
                    reversed_quad = own & _INSIDE == 0
                q = wp.vec4i(
                    dmc_vertex(flags, table, counts, cell, columns[0]),
                    dmc_vertex(flags, table, counts, dmc_quad_neighbour(cell, axis, 1), columns[1]),
                    dmc_vertex(flags, table, counts, dmc_quad_neighbour(cell, axis, 2), columns[2]),
                    dmc_vertex(flags, table, counts, dmc_quad_neighbour(cell, axis, 3), columns[3]),
                )
                if reversed_quad:
                    q = wp.vec4i(q[3], q[2], q[1], q[0])
                # Two triangles ``(q2, q1, q0)`` and ``(q0, q3, q2)``, outward for a field
                # positive outside.
                out_faces[corner + 0] = q[2]
                out_faces[corner + 1] = q[1]
                out_faces[corner + 2] = q[0]
                out_faces[corner + 3] = q[0]
                out_faces[corner + 4] = q[3]
                out_faces[corner + 5] = q[2]
                corner += 6


# The kernels at each field precision, keyed by the field's dtype.
DMC_CELL_FLAGS: OverloadTable
DMC_EMIT: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global DMC_CELL_FLAGS, DMC_EMIT
    i32 = wp.array[wp.int32]
    flags = wp.array3d[wp.uint16]
    DMC_CELL_FLAGS = OverloadTable(
        dmc_cell_flags, {d: [wp.array3d[d], d, i32, flags] for d in (wp.float32, wp.float64)}
    )
    DMC_EMIT = OverloadTable(
        dmc_emit,
        {
            d: [
                wp.array3d[d],
                d,
                flags,
                i32,
                wp.vec3i,
                wp.array[wp.vec2i],
                wp.vec3,
                wp.vec3,
                wp.array[wp.vec3],
                i32,
            ]
            for d in (wp.float32, wp.float64)
        },
    )


_register_overloads()
