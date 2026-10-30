"""test_fixedpoint.py  --  blueprint 9.2."""

import numpy as np
import pytest

from edge.codec import CodecParams
from edge.codec_batch import msb_index_arr, round_shift_arr
from edge.fixedpoint import (INT16_MAX, INT16_MIN, clamp, i16, i32, i64,
                             msb_index, round_shift, sar)


# Hand-computed. Negatives and zero are the whole point: this is where C and a
# naive Python implementation part company.
SAR_TABLE = [
    (0, 0, 0), (0, 5, 0), (1, 0, 1), (8, 3, 1), (7, 3, 0),
    (-1, 1, -1),        # floor(-0.5) = -1, NOT 0
    (-8, 3, -1), (-9, 3, -2), (-7, 3, -1), (-1, 10, -1),
    (32767, 15, 0), (-32768, 15, -1),
]

ROUND_SHIFT_TABLE = [
    (0, 1, 0), (1, 1, 1), (2, 1, 1), (3, 1, 2), (-1, 1, 0), (-2, 1, -1),
    (-3, 1, -1),        # (-3 + 1) >> 1 = -1, round-half-up toward +inf
    (4096, 12, 1), (4095, 12, 1), (2047, 12, 0), (2048, 12, 1),
    (-4096, 12, -1), (5, 0, 5),
]

MSB_TABLE = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (255, 7), (256, 8),
             (1023, 9), (1024, 10), ((1 << 40) - 1, 39), (1 << 40, 40)]


@pytest.mark.parametrize("x,n,want", SAR_TABLE)
def test_sar_table(x, n, want):
    assert sar(x, n) == want


@pytest.mark.parametrize("x,n,want", ROUND_SHIFT_TABLE)
def test_round_shift_table(x, n, want):
    assert round_shift(x, n) == want


@pytest.mark.parametrize("x,want", MSB_TABLE)
def test_msb_index_table(x, want):
    assert msb_index(x) == want


def test_negative_shift_counts_rejected():
    with pytest.raises(AssertionError):
        sar(1, -1)
    with pytest.raises(AssertionError):
        round_shift(1, -1)


def test_msb_index_rejects_negative():
    # The only caller passes a sum of squares. Accepting a negative would hide
    # that invariant breaking rather than surface it.
    with pytest.raises(AssertionError):
        msb_index(-1)


def test_width_helpers_assert_rather_than_wrap():
    assert i16(INT16_MAX) == INT16_MAX
    assert i16(INT16_MIN) == INT16_MIN
    for bad in (INT16_MAX + 1, INT16_MIN - 1):
        with pytest.raises(AssertionError):
            i16(bad)
    with pytest.raises(AssertionError):
        i32(2 ** 31)
    with pytest.raises(AssertionError):
        i64(2 ** 63)
    # and they must NOT silently wrap, which is the whole reason they exist
    assert i32(2 ** 31 - 1) == 2 ** 31 - 1


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(50, 0, 10) == 10
    with pytest.raises(AssertionError):
        clamp(1, 10, 0)


def test_vectorised_helpers_match_scalar():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << 45, 5000, dtype=np.int64)
    assert np.array_equal(msb_index_arr(x),
                          np.array([msb_index(int(v)) for v in x]))
    y = rng.integers(-(1 << 40), 1 << 40, 5000, dtype=np.int64)
    for n in (0, 1, 7, 12, 31):
        assert np.array_equal(round_shift_arr(y, n),
                              np.array([round_shift(int(v), n) for v in y]))
        assert np.array_equal(y >> np.int64(n),
                              np.array([sar(int(v), n) for v in y]))


def test_accumulator_widths():
    """Blueprint 4.5. The int64 accumulator must never be at risk, and the
    int32 analysis must hold under ITS stated preconditions -- which are
    stricter than the shipped clamps, which is exactly why int64 was chosen."""
    p = CodecParams()
    rng = np.random.default_rng(1)
    worst = p.n_w * p.w_clamp * max(p.u_clamp, p.bias_input)
    assert worst < 2 ** 63 - 1, "int64 accumulator is not provably safe"

    # The int32 claim in the docstring: 6 * 16384 * 20000 < 2**31.
    assert 6 * 16384 * 20000 < 2 ** 31 - 1
    # ... and it FAILS at the shipped u_clamp, which is the point of the note.
    assert 6 * p.w_clamp * p.u_clamp > 2 ** 31 - 1

    for _ in range(2000):
        w = rng.integers(-p.w_clamp, p.w_clamp + 1, p.n_w)
        phi = np.append(rng.integers(p.u_min, p.u_clamp + 1, len(p.taps)),
                        p.bias_input)
        acc = int((w.astype(np.int64) * phi.astype(np.int64)).sum())
        assert i64(acc) == acc
