"""
Arc cosine, sine and cosine that round the same on every device.

The pseudo-normals' angle weights and the smoothing tangents' rotations decide ties wherever two
directions cancel, so they must round identically on the CPU and on CUDA, whose ``acos``, ``sin``
and ``cos`` differ from the C library's, and from each other, in the last bit. These are the
Manifold library's own (``include/manifold/math.h``, after fdlibm) (Apache-2.0,
https://github.com/elalish/manifold, v3.5.5; Copyright 2021 The Manifold Authors), in plain
``float64`` arithmetic; their callers build unfused (``fuse_fp``) so no contraction moves a bit.
"""

import warp as wp

from ordito.kernels.reduce import group_sum_field

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


# -- Manifold's arc cosine (the pseudo-normals' angle weights) ------------------------------------
#
# The angle weights of a vertex's pseudo-normal decide ties wherever two normals cancel (two
# boxes' matching corners), so they must round as Manifold's do, on every device. Manifold does
# not call the C library: ``include/manifold/math.h`` carries its own ``acos`` (fdlibm's, in plain
# ``float64`` arithmetic), which this ports. CUDA's ``acos`` and glibc's differ from it and from
# each other in the last bit on a tenth of inputs (2026-11-21).


@wp.func
def acos_ratio(z: wp.float64) -> wp.float64:
    p = z * (
        wp.float64(1.66666666666666657415e-01)
        + z
        * (
            wp.float64(-3.25565818622400915405e-01)
            + z
            * (
                wp.float64(2.01212532134862925881e-01)
                + z
                * (
                    wp.float64(-4.00555345006794114027e-02)
                    + z
                    * (
                        wp.float64(7.91534994289814532176e-04)
                        + z * wp.float64(3.47933107596021167570e-05)
                    )
                )
            )
        )
    )
    q = wp.float64(1.0) + z * (
        wp.float64(-2.40339491173441421878e00)
        + z
        * (
            wp.float64(2.02094576023350569471e00)
            + z
            * (wp.float64(-6.88283971605453293030e-01) + z * wp.float64(7.70381505559019352791e-02))
        )
    )
    return p / q


@wp.func
def portable_acos(x: wp.float64) -> wp.float64:
    # ``math::acos``: three ranges, the outer two through ``2 asin(sqrt((1 -+ x) / 2))`` with the
    # square root's low word split off.
    pio2_hi = wp.float64(1.57079632679489655800e00)
    pio2_lo = wp.float64(6.12323399573676603587e-17)
    bits = wp.cast(x, wp.uint64)
    hx = wp.uint32(bits >> wp.uint64(32))
    ix = hx & wp.uint32(0x7FFFFFFF)
    if ix >= wp.uint32(0x3FF00000):
        if x == wp.float64(1.0):
            return wp.float64(0.0)
        if x == wp.float64(-1.0):
            return wp.float64(2.0) * pio2_hi
        return wp.float64(0.0) / (x - x)
    if ix < wp.uint32(0x3FE00000):
        if ix <= wp.uint32(0x3C600000):
            return pio2_hi
        return pio2_hi - (x - (pio2_lo - x * acos_ratio(x * x)))
    if (hx >> wp.uint32(31)) != wp.uint32(0):
        z = (wp.float64(1.0) + x) * wp.float64(0.5)
        s = wp.sqrt(z)
        w = acos_ratio(z) * s - pio2_lo
        return wp.float64(2.0) * (pio2_hi - (s + w))
    z = (wp.float64(1.0) - x) * wp.float64(0.5)
    s = wp.sqrt(z)
    df = wp.cast(wp.cast(s, wp.uint64) & wp.uint64(0xFFFFFFFF00000000), wp.float64)
    c = (z - df * df) / (s + df)
    w = acos_ratio(z) * s + c
    return wp.float64(2.0) * (df + w)


@wp.func
def sin_polynomial(x: wp.float64, y: wp.float64, iy: wp.int32) -> wp.float64:
    z = x * x
    w = z * z
    r = (
        wp.float64(8.33333333332248946124e-03)
        + z * (wp.float64(-1.98412698298579493134e-04) + z * wp.float64(2.75573137070700676789e-06))
        + z
        * w
        * (wp.float64(-2.50507602534068634195e-08) + z * wp.float64(1.58969099521155010221e-10))
    )
    v = z * x
    if iy == 0:
        return x + v * (wp.float64(-1.66666666666666324348e-01) + z * r)
    return x - (
        (z * (wp.float64(0.5) * y - v * r) - y) - v * wp.float64(-1.66666666666666324348e-01)
    )


@wp.func
def cos_polynomial(x: wp.float64, y: wp.float64) -> wp.float64:
    z = x * x
    w = z * z
    r = z * (
        wp.float64(4.16666666666666019037e-02)
        + z * (wp.float64(-1.38888888888741095749e-03) + z * wp.float64(2.48015872894767294178e-05))
    ) + w * w * (
        wp.float64(-2.75573143513906633035e-07)
        + z * (wp.float64(2.08757232129817482790e-09) + z * wp.float64(-1.13596475577881948265e-11))
    )
    hz = wp.float64(0.5) * z
    w1 = wp.float64(1.0) - hz
    return w1 + (((wp.float64(1.0) - w1) - hz) + (z * r - x * y))


@wp.func
def high_word(x: wp.float64) -> wp.uint32:
    return wp.uint32((wp.cast(x, wp.uint64) >> wp.uint64(32)) & wp.uint64(0x7FFFFFFF))


@wp.func
def rem_pio2(x: wp.float64) -> wp.vec3d:
    # ``RemPio2`` below ``2**20 * pi / 2``: ``(y0, y1, n)``, ``x = n pi / 2 + y0 + y1``.
    pio4 = wp.float64(0.78539816339744827900)
    invpio2 = wp.float64(6.36619772367581382433e-01)
    pio2_1 = wp.float64(1.57079632673412561417e00)
    pio2_1t = wp.float64(6.07710050650619224932e-11)
    pio2_2 = wp.float64(6.07710050630396597660e-11)
    pio2_2t = wp.float64(2.02226624879595063154e-21)
    pio2_3 = wp.float64(2.02226624871116645580e-21)
    pio2_3t = wp.float64(8.47842766036889956997e-32)
    toint = wp.float64(6755399441055744.0)
    ix = high_word(x)
    negative = (wp.cast(x, wp.uint64) >> wp.uint64(63)) != wp.uint64(0)
    medium = False
    k = wp.float64(0.0)
    if ix <= wp.uint32(0x400F6A7A):
        if (ix & wp.uint32(0xFFFFF)) == wp.uint32(0x921FB):
            medium = True
        elif ix <= wp.uint32(0x4002D97C):
            k = wp.float64(1.0)
        else:
            k = wp.float64(2.0)
    elif ix <= wp.uint32(0x401C463B):
        if ix <= wp.uint32(0x4015FDBC):
            if ix == wp.uint32(0x4012D97C):
                medium = True
            else:
                k = wp.float64(3.0)
        elif ix == wp.uint32(0x401921FB):
            medium = True
        else:
            k = wp.float64(4.0)
    else:
        medium = True
    if not medium:
        if not negative:
            z = x - k * pio2_1
            y0 = z - k * pio2_1t
            return wp.vec3d(y0, (z - y0) - k * pio2_1t, k)
        z = x + k * pio2_1
        y0 = z + k * pio2_1t
        return wp.vec3d(y0, (z - y0) + k * pio2_1t, wp.float64(-1.0) * k)
    fn = (x * invpio2 + toint) - toint
    n = wp.float64(wp.int32(fn))
    r = x - fn * pio2_1
    w = fn * pio2_1t
    if r - w < wp.float64(-1.0) * pio4:
        n = n - wp.float64(1.0)
        fn = fn - wp.float64(1.0)
        r = x - fn * pio2_1
        w = fn * pio2_1t
    elif r - w > pio4:
        n = n + wp.float64(1.0)
        fn = fn + wp.float64(1.0)
        r = x - fn * pio2_1
        w = fn * pio2_1t
    y0 = r - w
    ex = wp.int32(ix >> wp.uint32(20))
    if ex - group_sum_field(y0) > 16:
        t = r
        w = fn * pio2_2
        r = t - w
        w = fn * pio2_2t - ((t - r) - w)
        y0 = r - w
        if ex - group_sum_field(y0) > 49:
            t2 = r
            w = fn * pio2_3
            r = t2 - w
            w = fn * pio2_3t - ((t2 - r) - w)
            y0 = r - w
    return wp.vec3d(y0, (r - y0) - w, n)


@wp.func
def portable_sin(x: wp.float64) -> wp.float64:
    ix = high_word(x)
    if ix <= wp.uint32(0x3FE921FB):
        if ix < wp.uint32(0x3E500000):
            return x
        return sin_polynomial(x, wp.float64(0.0), 0)
    if ix >= wp.uint32(0x7FF00000):
        return x - x
    y = rem_pio2(x)
    q = wp.int32(y[2]) & 3
    if q == 0:
        return sin_polynomial(y[0], y[1], 1)
    if q == 1:
        return cos_polynomial(y[0], y[1])
    if q == 2:
        return wp.float64(-1.0) * sin_polynomial(y[0], y[1], 1)
    return wp.float64(-1.0) * cos_polynomial(y[0], y[1])


@wp.func
def portable_cos(x: wp.float64) -> wp.float64:
    ix = high_word(x)
    if ix <= wp.uint32(0x3FE921FB):
        if ix < wp.uint32(0x3E46A09E):
            return wp.float64(1.0)
        return cos_polynomial(x, wp.float64(0.0))
    if ix >= wp.uint32(0x7FF00000):
        return x - x
    y = rem_pio2(x)
    q = wp.int32(y[2]) & 3
    if q == 0:
        return cos_polynomial(y[0], y[1])
    if q == 1:
        return wp.float64(-1.0) * sin_polynomial(y[0], y[1], 1)
    if q == 2:
        return wp.float64(-1.0) * cos_polynomial(y[0], y[1])
    return sin_polynomial(y[0], y[1], 1)
