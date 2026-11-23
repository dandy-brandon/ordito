"""
Tests for ``ordito._common``: graph recording under any current stream.

Not a library comparison: no reference library records Warp graphs, so each test is ordito
against ordito, the same call under Warp's own stream and under the legacy default stream,
byte for byte.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt

_Call = Callable[[wp.array[wp.vec3], wp.array[wp.int32]], odt.ArrayNd]

# Calls that record graphs: ordito's own captures (the fixed-point smoothing passes, the heat
# solves; ``arap_parametrization`` is the call first reported failing on torch's default stream)
# and Warp's ``warp.optim.linear`` solvers, which record on the current stream themselves (the
# smoothing's BiCGSTAB fallback at ``lamb = 50``).
_CALLS: dict[str, _Call] = {
    "fixed_point": lambda vertices, faces: od.smoothing.filter_laplacian(
        vertices, faces, lamb=0.5, iterations=3, implicit_time_integration=True
    ),
    "bicgstab": lambda vertices, faces: od.smoothing.filter_laplacian(
        vertices, faces, lamb=50.0, iterations=3, implicit_time_integration=True
    ),
    "heat_geodesic": lambda vertices, faces: od.heat.heat_geodesic(
        vertices, faces, wp.array([0], dtype=wp.int32, device=vertices.device)
    ),
    "arap_parametrization": lambda vertices, faces: od.parametrization.arap_parametrization(
        vertices, faces, iterations=10
    ),
}


@pytest.mark.parametrize("call", sorted(_CALLS))
def test_recordings_run_on_the_legacy_stream(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], call: str
) -> None:
    """
    Ordito against ordito: a call that records graphs answers alike on the legacy stream.

    The legacy default stream is the one ``wp.ScopedStream(wp.stream_from_torch())`` hands Warp
    while torch runs on its default stream, and CUDA cannot record on it: every one of these calls
    raised ``Warp error: unknown stream`` there until ``_common.recordable_stream`` moved the
    recording to a private stream. The answer must be the one the call gives on Warp's own stream.
    """
    _, mesh_wp = saddle_graded
    device = wp.get_device(mesh_wp.device)
    if not device.is_cuda:
        pytest.skip("only a CUDA device has streams")
    run = _CALLS[call]
    expected_np = run(mesh_wp.points, mesh_wp.indices).numpy()
    with wp.ScopedStream(wp.Stream(device, cuda_stream=0)):
        assert device.stream.cuda_stream == 0
        actual_np = run(mesh_wp.points, mesh_wp.indices).numpy()
    assert np.array_equal(actual_np, expected_np)


def test_recordings_beside_legacy_stream_work_in_another_thread(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: a recording answers alike beside another thread's legacy stream.

    That is PyTorch running on its default stream in a second thread. While a *blocking* stream
    records, CUDA refuses the other thread's legacy-stream work ("operation would make the legacy
    stream depend on a capturing blocking stream") and invalidates the recording, so both failed
    until ``_common.scoped_capture`` recorded on a non-blocking stream. Only ordito's own
    recordings are covered: a ``warp.optim.linear`` solver still records on a blocking stream
    (``_common.recordable_stream``), so the BiCGSTAB call is not run here.
    """
    _, mesh_wp = saddle_graded
    device = wp.get_device(mesh_wp.device)
    if not device.is_cuda:
        pytest.skip("only a CUDA device has streams")
    run = _CALLS["fixed_point"]
    expected_np = run(mesh_wp.points, mesh_wp.indices).numpy()

    source = wp.zeros(1 << 16, dtype=wp.float32, device=device)
    target = wp.zeros_like(source)
    started, stop = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def use_legacy_stream() -> None:
        try:
            with wp.ScopedStream(wp.Stream(device, cuda_stream=0)):
                while not stop.is_set():
                    wp.copy(target, source)
                    started.set()
                wp.synchronize_device(device)
        except BaseException as error:  # noqa: BLE001 -- reported by the main thread
            errors.append(error)
            started.set()

    worker = threading.Thread(target=use_legacy_stream)
    worker.start()
    try:
        started.wait()
        results_np = [run(mesh_wp.points, mesh_wp.indices).numpy() for _ in range(10)]
    finally:
        stop.set()
        worker.join()
    assert not errors
    for actual_np in results_np:
        assert np.array_equal(actual_np, expected_np)
