"""test_quantize.py and friends -- blueprint 9.1/9.3 supporting checks."""

import numpy as np
import pytest

from edge.codec import CodecParams
from edge.quantize import (choose_scale, dequantize, dequantize_array,
                           quantisation_step, quantize, quantize_array)

P = CodecParams()


def test_choose_scale_targets_the_headroom():
    for mx, target in ((12000, 12000), (5000, 12000), (1e6, 12000), (2.0, 12000)):
        y = np.array([0.0, mx])
        k = choose_scale(y, target)
        assert mx / 2 ** k <= target + 1e-9, (mx, k)
        # ceil means one step smaller would overshoot the target
        assert mx / 2 ** (k - 1) > target - 1e-9, (mx, k)


def test_choose_scale_leaves_room_above_the_training_max():
    """target_max 12000 against a 32000 clamp: a test value 2.6x the training
    maximum must still quantise without clamping."""
    k = choose_scale(np.array([0.0, 10000.0]))
    u, cl = quantize(10000.0 * 2.6, k, P.u_clamp)
    assert not cl


def test_choose_scale_on_a_dead_or_empty_client():
    assert choose_scale(np.zeros(100)) == 0
    assert choose_scale(np.array([np.nan, np.nan])) == 0
    assert choose_scale(np.array([])) == 0


@pytest.mark.parametrize("k", [-4, -2, 0, 1, 3])
def test_round_trip_within_half_a_step(k):
    """Half a step, on every value the clamp did not touch.

    The exclusion is not a let-off: at k = -4 a 8000 kWh reading needs 128000
    device units and the int16 clamp genuinely eats it. That is what the CLAMP
    flag is for, and `test_choose_scale_never_clamps_its_own_data` below is the
    test that says the shipped k never puts a client there.
    """
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 8000, 5000)
    u, cl = quantize_array(y, k, P.u_clamp)
    back = dequantize_array(u, k)
    err = np.abs(back - y)[~cl]
    assert err.size, "the whole series clamped; the test proves nothing"
    assert err.max() <= quantisation_step(k) + 1e-9


def test_choose_scale_never_clamps_its_own_data():
    """The commissioning rule has to be self-consistent: k chosen from a
    client's history must quantise that history without a single clamp."""
    rng = np.random.default_rng(2)
    for scale in (1.0, 40.0, 800.0, 5000.0, 120000.0):
        y = rng.uniform(0, scale, 3000)
        k = choose_scale(y)
        _, cl = quantize_array(y, k, P.u_clamp)
        assert not cl.any(), (scale, k, cl.sum())


def test_scalar_and_vector_quantisers_agree_exactly():
    """sim/ uses the array version, the C port mirrors the scalar one. If they
    ever disagree the simulation stops describing the firmware."""
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.uniform(-10, 9000, 4000),
                        np.array([0.0, 0.5, 1.5, 2.5, -0.5, -1.5])])
    for k in (-3, 0, 2):
        ua, ca = quantize_array(y, k, P.u_clamp)
        for i in range(len(y)):
            us, cs = quantize(float(y[i]), k, P.u_clamp)
            assert us == ua[i], (k, y[i], us, ua[i])
            assert cs == ca[i]


def test_half_way_rounds_away_from_zero_like_C():
    """numpy rounds halves to even; C's round() does not. The device does this
    in C, so the reference must too, or a value on a half-unit boundary
    quantises differently on the two sides."""
    assert quantize(0.5, 0, P.u_clamp)[0] == 1
    assert quantize(1.5, 0, P.u_clamp)[0] == 2
    assert quantize(2.5, 0, P.u_clamp)[0] == 3       # np.round would give 2
    assert quantize(-2.5, 0, P.u_clamp)[0] == -3


def test_clamp_flag_fires_and_is_rare():
    u, cl = quantize(1e9, 0, P.u_clamp)
    assert u == P.u_clamp and cl
    u, cl = quantize(100.0, 0, P.u_clamp)
    assert not cl


def test_nan_is_treated_as_zero_not_propagated():
    """The codec is integer-only; a NaN reaching it would become an assertion
    failure somewhere far from the cause."""
    u, cl = quantize(float("nan"), 0, P.u_clamp)
    assert u == 0 and not cl
