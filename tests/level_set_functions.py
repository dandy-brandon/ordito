"""Signed distance functions as ``@wp.func`` and as rounding-identical Python twins."""

# Warp's vector types carry no index stub; these kernel-scope bodies index them.
# pyright: reportIndexIssue=false

from __future__ import annotations

import math

import warp as wp


@wp.func
def torus_sdf(p: wp.vec3d) -> wp.float64:
    q = wp.sqrt(p[0] * p[0] + p[1] * p[1]) - wp.float64(1.0)
    return wp.float64(0.35) - wp.sqrt(q * q + p[2] * p[2])


def torus_sdf_py(x: float, y: float, z: float) -> float:
    q = math.sqrt(x * x + y * y) - 1.0
    return 0.35 - math.sqrt(q * q + z * z)


@wp.func
def cube_sdf(p: wp.vec3d) -> wp.float64:
    return wp.float64(0.7) - wp.max(wp.max(wp.abs(p[0]), wp.abs(p[1])), wp.abs(p[2]))


def cube_sdf_py(x: float, y: float, z: float) -> float:
    return 0.7 - max(max(abs(x), abs(y)), abs(z))


@wp.func
def blobs_sdf(p: wp.vec3d) -> wp.float64:
    a = (p[0] - wp.float64(0.4)) * (p[0] - wp.float64(0.4)) + p[1] * p[1] + p[2] * p[2]
    b = (p[0] + wp.float64(0.4)) * (p[0] + wp.float64(0.4)) + p[1] * p[1] + p[2] * p[2]
    return wp.float64(0.3) / a + wp.float64(0.3) / b - wp.float64(1.0)


def blobs_sdf_py(x: float, y: float, z: float) -> float:
    a = (x - 0.4) * (x - 0.4) + y * y + z * z
    b = (x + 0.4) * (x + 0.4) + y * y + z * z
    return 0.3 / a + 0.3 / b - 1.0
