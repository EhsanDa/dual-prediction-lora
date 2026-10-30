"""
quantize.py  --  per-client power-of-two scaling, blueprint 4.2.

The device works in "device units" u = round(kWh / 2**k), with k chosen once
per meter at commissioning. Power-of-two is not a convenience: it makes the
scaling a shift, which is exact, reversible and reproducible bit-for-bit on the
NanoPi. A general float scale would make the two sides agree only to within
whatever rounding mode each compiler happened to pick, and the whole scheme
rests on them agreeing exactly.

k may be negative. A meter averaging 2 kWh/h gets k = -3 (q = 0.125 kWh), a
meter averaging 4000 gets k = -1 or 0. The 321 real clients span four orders of
magnitude, which is why this is per client and not global.
"""

import numpy as np

from edge.fixedpoint import clamp

U_CLAMP_DEFAULT = 32000       # int16 with headroom, matches CodecParams.u_clamp


def choose_scale(train_series, target_max: int = 12000) -> int:
    """Return k such that q = 2**k maps the training maximum near target_max.

    k = ceil(log2(nanmax(train_series) / target_max)).

    target_max = 12000 against a 32000 clamp leaves a factor of 2.7 of headroom
    for test-set values above anything seen in training. Quantisation error is
    2**k / 2 ~ max/24000, under 0.005% of full scale.

    An all-zero or all-NaN training slice (a meter commissioned after the end
    of the training split, which the real store does contain) has no scale to
    infer from; it gets k = 0, i.e. 1 kWh resolution, and the caller is
    expected to have flagged the client already.
    """
    v = np.asarray(train_series, dtype=np.float64)
    v = v[np.isfinite(v)]
    mx = float(np.max(np.abs(v))) if v.size else 0.0
    if mx <= 0.0:
        return 0
    return int(np.ceil(np.log2(mx / float(target_max))))


def quantize(x_kwh: float, k: int, u_clamp: int = U_CLAMP_DEFAULT):
    """round(x / 2**k), clamped to +-u_clamp. Returns (u, clamped).

    Round-half-away-from-zero, matching C's round(), NOT numpy's round-half-even.
    The device does this in float on the way in from the ADC/meter register --
    it is the one place a float is allowed (blueprint 4.1) -- so the rounding
    rule has to be the C library's, or a value exactly on a half-unit boundary
    would quantise differently on the two sides.

    `clamped` is the CLAMP packet flag. It should effectively never fire; a
    client with more than a handful is a client whose k was chosen wrong, and
    sim/run_codec.py surfaces that as a warning.
    """
    if not np.isfinite(x_kwh):
        x_kwh = 0.0
    scaled = float(x_kwh) / (2.0 ** k)
    u = int(np.floor(scaled + 0.5)) if scaled >= 0 else -int(np.floor(-scaled + 0.5))
    cl = clamp(u, -u_clamp, u_clamp)
    return cl, cl != u


def dequantize(u: int, k: int) -> float:
    """u * 2**k. Exact in binary floating point for any k in range."""
    return float(u) * (2.0 ** k)


# ---------------------------------------------------------------- vectorised
def quantize_array(x_kwh, k: int, u_clamp: int = U_CLAMP_DEFAULT):
    """Vectorised `quantize` over a whole series. Returns (u int64, clamped bool).

    Must agree with the scalar version on every element -- asserted in
    tests/test_quantize.py, because sim/run_codec.py uses this one and the C
    port mirrors the scalar one.
    """
    x = np.asarray(x_kwh, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    scaled = x / (2.0 ** k)
    u = np.where(scaled >= 0,
                 np.floor(scaled + 0.5),
                 -np.floor(-scaled + 0.5)).astype(np.int64)
    cl = np.clip(u, -u_clamp, u_clamp)
    return cl, cl != u


def dequantize_array(u, k: int):
    return np.asarray(u, dtype=np.float64) * (2.0 ** k)


def quantisation_step(k: int) -> float:
    """The reconstruction error floor: max |dequantize(quantize(x)) - x|."""
    return 0.5 * (2.0 ** k)
