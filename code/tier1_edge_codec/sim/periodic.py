"""
periodic.py  --  the fixed periodic refresh baseline. Reviewer item 6.

"Why event-triggered rather than just transmitting more often?" is the most
obvious objection to the whole scheme, and until now nothing in the evaluation
answered it. This closes it.

THE BASELINE
============
No codec, no predictor, no threshold. Every `period` hours the node transmits
the EXACT quantised values for the hours since the last refresh. Between
refreshes the hub holds the last delivered value (zero-order hold), which is
what a periodic-reporting deployment actually does.

  period = 2  ->  12 packets/day, 2 values each
  period = 3  ->   8 packets/day, 3 values each
  period = 4  ->   6 packets/day, 4 values each
  period = 6  ->   4 packets/day, 6 values each

No daily resync is needed: the periodic packets already deliver every value
exactly, so `u_exact` is the quantised truth everywhere. That makes this baseline
STRICTLY FAIRER than the codec on fidelity of completed days -- it has no
approximate history at all -- and the comparison is therefore only about
freshness of the current partial window, which is exactly the axis in dispute.

MATCHED PACKET RATE IS THE ONLY HONEST COMPARISON
=================================================
A periodic schedule at 12 packets/day should be compared against the codec at 12
packets/day, not against the codec at its selected alpha. eval/run_eval.py
interpolates the codec's curve onto each periodic row's packet rate and reports
the head-to-head there.

WHY THIS REUSES PolicyResult
============================
It emits exactly the arrays sim/run_codec.py emits, so it flows through
build_log1p_arrays, patch_y_past, Accum and communication_cost with no new
machinery. A baseline that needed its own scoring path would be a baseline
nobody could trust.
"""

import numpy as np

from edge.codec import CodecParams
from edge.quantize import quantize_array
from sim.run_codec import PolicyResult, _varint_len, choose_scales


def periodic_nbytes(n_values: int, values=None) -> int:
    """Encoded size of one periodic packet, on the same wire format as the
    RESYNC: a header, the first value in full, then zigzag varint deltas, then a
    CRC16. No transmit bitmap and no weight vector -- there is no predictor to
    repair, which is a genuine byte advantage for this baseline and it is
    counted rather than waved away."""
    payload = 2
    if values is not None:
        v = [int(x) for x in values]
        for a, b in zip(v, v[1:]):
            d = b - a
            payload += _varint_len((d << 1) ^ (d >> 31))
    else:
        payload += 2 * max(n_values - 1, 0)
    return 4 + payload + 2          # type|ver + node_id + payload + CRC16


def run_periodic(store, period: int, params: CodecParams = CodecParams()):
    """One periodic-refresh policy over the whole store.

    Returns a PolicyResult so it is indistinguishable downstream from a codec
    policy. `alpha` is set to the string-free sentinel -period so the row sorts
    away from the alpha sweep and can never be mistaken for one.
    """
    assert period >= 1
    T, M = store.T, store.M
    k, _ = choose_scales(store)

    u = np.zeros((T, M), np.int64)
    clamped = np.zeros((T, M), bool)
    for c in range(M):
        uc, cl = quantize_array(store.series_filled[:, c], int(k[c]),
                                params.u_clamp)
        u[:, c], clamped[:, c] = uc, cl

    t_start = store.onboarding.astype(np.int64)
    active = np.arange(T)[:, None] >= t_start[None, :]

    # Zero-order hold from the last refresh. The refresh at hour r delivers the
    # values for hours (r-period, r], so the hub holds u[r] from hour r onward,
    # exactly as an exception at hour t delivers u[t] in the codec.
    idx = np.arange(T)
    last = (idx - (idx % period))[:, None].repeat(M, axis=1)
    last = np.maximum(last, t_start[None, :])
    within = np.take_along_axis(u, last, axis=0)
    within = np.where(active, within, 0)

    # Every value is eventually delivered exactly, so completed history is exact.
    exact = np.where(active, u, 0)

    tx = active & ((idx % period == 0)[:, None])
    e = np.where(active, u - within, 0).astype(np.int32)

    # Packet sizes, measured from the actual values rather than assumed.
    n_days = (T + 23) // 24
    pkt_bytes = np.zeros((n_days, M), np.int32)
    for d in range(n_days):
        lo, hi = d * 24, min((d + 1) * 24, T)
        for c in range(M):
            r = [t for t in range(lo, hi)
                 if t % period == 0 and t >= t_start[c]]
            if not r:
                continue
            pkt_bytes[d, c] = sum(
                periodic_nbytes(period, u[max(t - period + 1, 0):t + 1, c])
                for t in r)

    # eps_eff is the L-inf bound this policy actually offers, which is
    # UNBOUNDED: a zero-order hold has no error guarantee at all. Recording it
    # as the observed maximum would be flattering it, so it is recorded as
    # infinite and linf_violation is not asserted for these rows.
    eps_eff = np.full((T, M), np.iinfo(np.int64).max, np.int64)

    return PolicyResult(
        alpha=float(-period), k=k, eps=[None] * M,
        eps_degenerate=np.zeros(M, bool), u_exact=exact, u_q=u, u_within=within,
        e=e, tx=tx, event=np.zeros((T, M), bool), resynced=active.copy(),
        eps_eff=eps_eff, active=active, clamped=clamped & active,
        resync_bytes=pkt_bytes, packets={},
        notes={"policy": "fixed periodic refresh", "period_h": period,
               "packets_per_day_nominal": 24.0 / period,
               "values_per_packet": period,
               "guarantee": ("NONE. A zero-order hold offers no L-inf bound; "
                             "the codec's epsilon guarantee has no counterpart "
                             "here, which is a qualitative difference the "
                             "packet-rate comparison does not capture.")})


def label(period: int) -> str:
    return f"periodic-{period}h"
