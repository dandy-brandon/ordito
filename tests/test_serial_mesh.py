"""
Tests for ``ordito._serial_mesh``: the host half-edge editing behind ``method="meshlib"``.

The public options it serves (``holes.fill_smooth``, ``stitch_smooth``, ``refill_region``,
``bridge_edges_smooth``, ``smoothing.smooth_region_boundary``) are compared with MeshLib in their
own modules; this file pins the one step whose input sizes those comparisons do not reach, the
builder's numbering of large triangle lists.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
from meshlib import mrmeshpy as mm

from ordito import _serial_mesh
from tests.conversions import meshlib_topology_to_numpy, numpy_to_meshlib_triangulation


def _icosphere_faces(subdivisions: int) -> np.ndarray:
    return np.asarray(tm.creation.icosphere(subdivisions).faces, dtype=np.int64)


def _with_fin(faces_np: np.ndarray) -> np.ndarray:
    """Add a third triangle on the first face's first edge, to a new vertex: non-manifold."""
    fin = [faces_np[0, 0], faces_np[0, 1], faces_np.max() + 1]
    return np.vstack([faces_np, fin])


def _open(faces_np: np.ndarray) -> np.ndarray:
    """Delete 20 scattered faces, and make one face degenerate: rims, and a face to skip."""
    rng = np.random.default_rng(3)
    opened = np.delete(faces_np, rng.choice(faces_np.shape[0], 20, replace=False), axis=0)
    opened[5, 2] = opened[5, 0]
    return opened


def _shuffled(faces_np: np.ndarray) -> np.ndarray:
    return faces_np[np.random.default_rng(7).permutation(faces_np.shape[0])]


# The builder's three forms, by size and manifoldness: under 8 192 triangles, face at a time; from
# there a manifold list numbered by corners (closed, and open with a skipped degenerate face); a
# non-manifold one face at a time again, in one part below 32 768 triangles and by vertex-range
# parts joined above it (in mesh order and shuffled, so the parts' border triangles interleave).
@pytest.mark.parametrize(
    "faces_np",
    [
        pytest.param(_icosphere_faces(4), id="face-by-face"),
        pytest.param(_icosphere_faces(5), id="corners-closed"),
        pytest.param(_open(_icosphere_faces(5)), id="corners-open"),
        pytest.param(_with_fin(_icosphere_faces(5)), id="non-manifold"),
        pytest.param(_with_fin(_icosphere_faces(6)), id="partitioned"),
        pytest.param(_shuffled(_with_fin(_icosphere_faces(6))), id="partitioned-shuffled"),
    ],
)
def test_build_numbers_edges_as_meshlib(faces_np: np.ndarray) -> None:
    """
    Class A against ``topologyFromTriangles``: every half-edge record, equal by id.

    Every tie the ``method="meshlib"`` options break (a split queue's, a band search's) reads
    edge ids, so the builder must number edges, link rings and pick each vertex's and face's edge
    exactly as MeshLib does, for each of its builders. The non-manifold fin is left out by both.
    """
    topology_ml = mm.topologyFromTriangles(numpy_to_meshlib_triangulation(faces_np))
    records_ml = meshlib_topology_to_numpy(topology_ml)
    topology = _serial_mesh.build(faces_np, int(faces_np.max()) + 1)
    # Non-vacuity: MeshLib linked every face but the fin or the degenerate one.
    assert np.count_nonzero(records_ml["edge_per_face"] >= 0) >= faces_np.shape[0] - 1
    for name, record_ml in records_ml.items():
        assert np.array_equal(topology.array(name), record_ml), name
