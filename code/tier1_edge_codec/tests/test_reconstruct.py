"""
test_reconstruct.py  --  blueprint 6.5.

The assembly rule is the difference between measuring this system and
measuring a strawman. Using Y_WITHIN everywhere overstates the degradation by
roughly an order of magnitude, so the slot counts are asserted directly.
"""

import numpy as np

from edge.codec import CodecParams
from sim.reconstruct import (approximate_fraction_by_issue_hour,
                             assemble_window, build_log1p_arrays, within_mask,
                             window_slot_counts)
from sim.run_codec import run_policy

P = CodecParams()
L = 336


def test_slot_counts_at_every_issue_hour():
    """h = 0 -> exactly one within-slot. h = 23 -> exactly 24. And h+1 in
    between, for every hour, not just the two endpoints."""
    origins = np.arange(1000, 1000 + 24 * 5)
    counts = window_slot_counts(origins, L)
    for t, n in zip(origins, counts):
        assert n == (t % 24) + 1, (t, n)
    assert counts[origins % 24 == 0].min() == 1
    assert counts[origins % 24 == 23].max() == 24


def test_within_slots_are_contiguous_and_at_the_end():
    """The approximate slots are the most RECENT ones. If they were anywhere
    else the per-horizon degradation story would be wrong."""
    t = np.array([1000 + h for h in range(24)])
    past, wm = within_mask(t, L)
    for i in range(len(t)):
        idx = np.nonzero(wm[i])[0]
        assert idx[-1] == L - 1, "the newest slot must always be within-day"
        assert np.array_equal(idx, np.arange(idx[0], L)), "not contiguous"


def test_assembly_picks_the_right_array():
    """Built from two arrays with disjoint marker values, so a slot taken from
    the wrong one is unmistakable."""
    T, M = 2000, 2
    y_exact = np.zeros((T, M), np.float32)
    y_within = np.ones((T, M), np.float32)
    t = np.array([1000, 1000 + 23])
    c = np.array([0, 1])
    w = assemble_window(t, c, y_exact, y_within, L)
    assert w[0].sum() == (1000 % 24) + 1
    assert w[1].sum() == ((1000 + 23) % 24) + 1
    # the within block sits at the end
    assert w[0, -1] == 1.0 and w[0, 0] == 0.0


def test_alpha_zero_window_is_the_store_up_to_quantisation(store):
    """The oracle's look-back must be the store's own series to within the
    quantisation floor -- if it is not, something other than quantisation is
    perturbing the input and every delta in the results table is polluted."""
    res = run_policy(store, 0.0, P, backend="batch")
    ex, wi = build_log1p_arrays(res, store)
    ref = np.load(f"{store.dir}/series_log1p.npy")
    t = np.arange(1100, 1100 + 48)
    c = np.zeros(len(t), np.int64)
    got = assemble_window(t, c, ex, wi, L)
    want = np.stack([ref[ti - L + 1:ti + 1, 0] for ti in t])
    # quantisation is at most 2**k/2 kWh, which in log1p space is tiny
    assert np.abs(got - want).max() < 5e-3, float(np.abs(got - want).max())


def test_alpha_inf_perturbs_only_the_current_day(store):
    """Daily batching must leave the completed days untouched. This is the
    assertion that stops the pessimistic shortcut from creeping back in."""
    res = run_policy(store, float("inf"), P, backend="batch")
    ex, wi = build_log1p_arrays(res, store)
    t = np.array([1104 + h for h in range(24)])       # 1104 % 24 == 0
    c = np.zeros(len(t), np.int64)
    got = assemble_window(t, c, ex, wi, L)
    exact_only = np.stack([ex[ti - L + 1:ti + 1, 0] for ti in t])
    for i, ti in enumerate(t):
        h = ti % 24
        d = np.abs(got[i] - exact_only[i])
        assert (d[:L - h - 1] == 0).all(), (
            f"a completed day was perturbed at issue hour {h}")
        assert d[L - h - 1:].max() >= 0


def test_exposure_closed_form_matches_the_counts():
    af = approximate_fraction_by_issue_hour(L)
    counts = window_slot_counts(np.arange(24 * 3), L)
    for h in range(24):
        assert abs(af[h] - (h + 1) / L) < 1e-12
        assert counts[h] == (h + 1)
