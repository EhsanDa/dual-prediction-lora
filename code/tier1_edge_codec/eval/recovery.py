"""
recovery.py  --  recovery time after an epsilon violation.

Deliberately free of any `edge.*` / `sim.*` import so the arithmetic can be
unit-tested without the simulator, the store, or a GPU. loss_sweep.py imports
it; tests/test_loss_sweep_recovery.py tests it directly.

TWO DIFFERENT QUANTITIES, AND WHY BOTH ARE REPORTED
===================================================
RUN LENGTH (`recovery_run_*`, formerly `recovery_*`)
    Length of a maximal run of consecutive violated hours. This is TOTAL
    exposure under sustained loss and it is informative, but it is NOT a bound
    on anything: at high loss rates a new lost frame lands before the previous
    run's last violated hour has cleared, so independent failures chain into
    one contiguous run and the maximum grows with the loss rate even though
    every individual repair worked. The 34 h / 40 h / 72 h figures that tripped
    the bounded-recovery assertion were this, plus censoring (below).

PER EVENT (`recovery_event_*`)
    For each hour at which a frame was DISCARDED, the number of hours until the
    reconstruction is next inside the bound. This does not merge independent
    failures, so it is the quantity the paper's "repaired at the next
    resynchronisation" claim is actually about, and for exception loss it is
    bounded by the resync interval by construction. THIS is what the assertion
    in loss_sweep.py must test.

CENSORING
=========
A violation run still open when the series ends never recovers inside the
window, so its exposure is not observed -- only bounded below by the length of
the tail. It is reported separately and is NO LONGER folded into
`recovery_run_max_h` / `recovery_event_max_h`: doing so reported a lower bound
on an unobserved quantity as if it were a measured maximum, and that is what
made a working repair mechanism look broken.
"""

import numpy as np


def recovery_times(viol):
    """Per-violated-hour run-length recovery times for one client.

    Parameters
    ----------
    viol : 1-D boolean array, one entry per hour, True = outside the bound.

    Returns
    -------
    times    : int array of recovery times for the violated hours that DO
               recover within the window (one entry per violated hour, not per
               run).
    censored : length of a trailing violation run that never recovers, else 0.
    """
    v = np.asarray(viol, dtype=bool).ravel()
    if v.size == 0 or not v.any():
        return np.zeros(0, dtype=np.int64), 0

    d = np.diff(v.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if v[0]:
        starts = np.concatenate(([0], starts))

    censored = 0
    if v[-1]:
        censored = int(v.size - starts[-1])
        starts = starts[:-1]

    lengths = (ends - starts).astype(np.int64)
    if lengths.size == 0:
        return np.zeros(0, dtype=np.int64), censored
    times = np.concatenate([np.arange(L, 0, -1, dtype=np.int64)
                            for L in lengths])
    return times, censored


def _next_clear(v):
    """next_clear[i] = smallest j >= i with v[j] False, else len(v)."""
    n = v.size
    clear = np.flatnonzero(~v)
    if clear.size == 0:
        return np.full(n, n, dtype=np.int64)
    pos = np.searchsorted(clear, np.arange(n))
    out = np.full(n, n, dtype=np.int64)
    inside = pos < clear.size
    out[inside] = clear[pos[inside]]
    return out


def recovery_by_event(viol, event_idx):
    """Exposure attributable to each individually discarded frame.

    Parameters
    ----------
    viol      : 1-D boolean array, True = outside the bound.
    event_idx : indices (into `viol`) of hours at which a frame was DISCARDED.

    Returns
    -------
    times    : exposure in hours for each event that leaves a violation AND
               recovers inside the window. An event whose hour is not violated
               contributes 0 and is included, since "this loss cost nothing" is
               a real outcome and dropping it would bias the mean upward.
    censored : number of events whose recovery is not observed in the window.
    """
    v = np.asarray(viol, dtype=bool).ravel()
    e = np.asarray(event_idx, dtype=np.int64).ravel()
    e = e[(e >= 0) & (e < v.size)]
    if v.size == 0 or e.size == 0:
        return np.zeros(0, dtype=np.int64), 0

    nxt = _next_clear(v)
    exposure = nxt[e] - e                      # 0 where the hour is not violated
    censored = int(np.count_nonzero(nxt[e] == v.size))
    return exposure[nxt[e] != v.size], censored


def recovery_summary(viol_by_client, events_by_client=None):
    """Pool recovery statistics over clients into the numbers the paper reports.

    `viol_by_client`   : sequence of boolean masks, one per client.
    `events_by_client` : optional sequence of discarded-frame index arrays,
                         aligned to the same index base as the masks. When
                         given, the `recovery_event_*` fields are populated;
                         these are the fields the bounded-recovery assertion
                         must use.

    Means are over VIOLATED HOURS (run-length) and over EVENTS (per-event), not
    over clients, so a client with many failures weighs more -- which is the
    right weighting for "if this happens, how long until it is over".
    """
    times, censored_lengths = [], []
    for viol in viol_by_client:
        t, c = recovery_times(viol)
        if t.size:
            times.append(t)
        if c:
            censored_lengths.append(c)

    all_t = np.concatenate(times) if times else np.zeros(0, dtype=np.int64)
    resolved_max = int(all_t.max()) if all_t.size else 0
    cens_max = int(max(censored_lengths)) if censored_lengths else 0

    out = {
        # Run length: total exposure, merges independent episodes. Not a bound.
        "recovery_run_mean_h": float(all_t.mean()) if all_t.size else 0.0,
        "recovery_run_p90_h": (float(np.percentile(all_t, 90))
                               if all_t.size else 0.0),
        "recovery_run_max_h": float(resolved_max),   # RESOLVED runs only
        "recovery_violated_hours": int(all_t.size + sum(censored_lengths)),
        "recovery_censored": bool(censored_lengths),
        "recovery_censored_episodes": int(len(censored_lengths)),
        "recovery_censored_max_h": float(cens_max),
        # Back-compat aliases so older readers of the artefact do not break.
        "recovery_mean_h": float(all_t.mean()) if all_t.size else 0.0,
        "recovery_p90_h": (float(np.percentile(all_t, 90))
                           if all_t.size else 0.0),
        "recovery_max_h": float(resolved_max),
    }

    if events_by_client is None:
        return out

    ev, ev_cens = [], 0
    for viol, idx in zip(viol_by_client, events_by_client):
        t, c = recovery_by_event(viol, idx)
        if t.size:
            ev.append(t)
        ev_cens += c
    all_e = np.concatenate(ev) if ev else np.zeros(0, dtype=np.int64)
    out.update({
        "recovery_event_mean_h": float(all_e.mean()) if all_e.size else 0.0,
        "recovery_event_p90_h": (float(np.percentile(all_e, 90))
                                 if all_e.size else 0.0),
        "recovery_event_max_h": float(all_e.max()) if all_e.size else 0.0,
        "n_events": int(all_e.size + ev_cens),
        "n_events_censored": int(ev_cens),
    })
    return out
