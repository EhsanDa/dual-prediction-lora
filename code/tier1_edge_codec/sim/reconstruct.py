"""
reconstruct.py  --  the two arrays, and the rule that combines them.
Blueprint 6.5.

    Y_EXACT_LOG1P    log1p(clip(dequantize(u_exact),  0, None))
    Y_WITHIN_LOG1P   log1p(clip(dequantize(u_within), 0, None))

Y_EXACT is the quantised truth: what the NanoPi holds for any COMPLETED day,
because every day boundary overwrites that day's 24 slots with the true values
carried in the RESYNC. Y_WITHIN is what it holds DURING the current day: u_hat
on a silent hour, u_hat + e on an exception hour.

THE ASSEMBLY RULE, AND WHY THE SHORTCUT IS WRONG
================================================
For a forecast origin at hour t with h = t % 24:

    slot t' uses Y_WITHIN_LOG1P   if t' >= t - h      (the day in progress)
    slot t' uses Y_EXACT_LOG1P    otherwise           (completed, resynced)

Using Y_WITHIN everywhere is the tempting simplification and it is wrong in the
PESSIMISTIC direction: it treats all 336 slots as approximate when at most 24
ever are, which overstates the degradation by roughly an order of magnitude. A
result that is wrong in the conservative direction is still wrong, and here it
would understate the scheme by so much that the whole rate-utility curve would
be meaningless. tests/test_reconstruct.py asserts the slot counts directly: one
within-slot at h = 0, twenty-four at h = 23.

alpha = inf (daily batching) uses the SAME rule. Nothing about the assembly
changes; what changes is that u_within is a free-running prediction with no
intra-day corrections applied to it at all.
"""

import numpy as np


def build_log1p_arrays(res, store=None):
    """(T, M) float32 arrays in log1p space, from a PolicyResult.

    Hours before a client's onboarding are filled from the store's own
    series_log1p rather than from the codec: the meter did not exist, the codec
    never ran, and inventing a reconstruction there would put fabricated values
    into the look-back of the earliest scorable windows. In practice both are
    log1p(0) = 0, but relying on that coincidence is exactly the kind of thing
    that stops being true when the store changes.
    """
    k = np.asarray(res.k, np.int64)
    q = (2.0 ** k.astype(np.float64))[None, :]
    exact = np.log1p(np.clip(res.u_exact.astype(np.float64) * q, 0, None))
    within = np.log1p(np.clip(res.u_within.astype(np.float64) * q, 0, None))
    if store is not None:
        import os
        ref = np.load(os.path.join(store.dir, "series_log1p.npy")).astype(np.float64)
        exact = np.where(res.active, exact, ref)
        within = np.where(res.active, within, ref)
    return exact.astype(np.float32), within.astype(np.float32)


def within_mask(t, L: int):
    """(B, L) boolean: True where the slot belongs to the day in progress.

    `t` is the forecast origin (the last observed hour), matching
    WindowBuilder's convention, which tier2/adapter.py asserts empirically at
    startup rather than assuming.
    """
    t = np.asarray(t, np.int64)
    past = t[:, None] + np.arange(-L + 1, 1, dtype=np.int64)[None, :]
    day_start = t - (t % 24)
    return past, past >= day_start[:, None]


def assemble_window(t, c, y_exact, y_within, L: int):
    """The look-back DLinear actually sees under this policy. (B, L) float32."""
    t = np.asarray(t, np.int64)
    c = np.asarray(c, np.int64)
    past, wmask = within_mask(t, L)
    cc = c[:, None]
    return np.where(wmask, y_within[past, cc], y_exact[past, cc]).astype(np.float32)


def day_start_horizon(t):
    """The last hour a codec policy's gateway holds EXACTLY at origin t: the
    end of the previous day, repaired by the RESYNC. With it,
    `assemble_window_horizon` reproduces `assemble_window` bit for bit."""
    t = np.asarray(t, np.int64)
    return t - (t % 24) - 1


def refresh_horizon(period: int):
    """The same for a fixed periodic refresh: the refresh at hour r delivers
    every value in (r - period, r], so at origin t the gateway holds every hour
    up to the last refresh, t - t % period, exactly."""
    def f(t):
        t = np.asarray(t, np.int64)
        return t - (t % period)
    return f


def assemble_window_horizon(t, c, y_exact, y_within, L: int, horizon_fn):
    """The look-back a gateway actually holds at origin t, for ANY policy whose
    exact knowledge ends at a known hour.

    WHY THIS EXISTS (conference revision, request F, 2026-09-29). The day rule
    in `assemble_window` is right for the codec, whose corrections arrive only
    at the hour itself (an exception) or at the day boundary (the RESYNC). It is
    wrong for a periodic refresh, which corrects the previous `period` hours
    retroactively: an hour that a later refresh on the same day already
    delivered would still be read from `y_within`, i.e. at its old zero-order-
    hold value, understating the periodic baseline.
    docs/findings/2026-09-29-periodic-baseline-assembly.md.

    Slots at or before horizon_fn(t) come from `y_exact`, later ones from
    `y_within`. (B, L) float32.
    """
    t = np.asarray(t, np.int64)
    c = np.asarray(c, np.int64)
    past = t[:, None] + np.arange(-L + 1, 1, dtype=np.int64)[None, :]
    known = np.asarray(horizon_fn(t), np.int64)
    wmask = past > known[:, None]
    cc = c[:, None]
    return np.where(wmask, y_within[past, cc], y_exact[past, cc]).astype(np.float32)


def approximate_fraction_by_issue_hour(L: int = 336):
    """Blueprint 8.3: the share of the 336-slot window that is approximate, as
    a function of issue hour. It is (h + 1) / L and nothing else -- a closed
    form, reported so that the measured degradation curve can be read against
    the exposure that produced it."""
    h = np.arange(24)
    return (h + 1) / float(L)


def window_slot_counts(t, L: int = 336):
    """Number of within-day slots per origin. Used by the tests and by the
    freshness figure."""
    _, wmask = within_mask(np.asarray(t), L)
    return wmask.sum(axis=1)
