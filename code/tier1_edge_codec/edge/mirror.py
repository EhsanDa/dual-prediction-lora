"""
mirror.py  --  the NanoPi side. Blueprint 5.2 and 6.4.

The mirror runs the SAME CodecState the Arduino runs, driven only by what
arrives over the radio. It never sees a meter reading. Its rules:

  silent hour        fill with u_hat            guaranteed within eps
  EXCEPTION received u = u_hat + e              exact, overwrite
  RESYNC received    overwrite the previous 24  exact, then run the 24 NLMS
                     slots with the true values updates

THE PROPERTY THIS MODULE EXISTS TO PROVE
========================================
Given the same commissioning constants and a lossless channel, the mirror's
CodecState must be byte-identical to the transmitter's at EVERY hour, and its
reconstruction must equal the transmitter's own reconstruction bit for bit.
tests/test_mirror.py asserts both, and tests/test_codec.py asserts that a
transmitter which pushes u instead of u_hat on a silent hour breaks it. If
those two tests are green, the epsilon guarantee is a fact about the code
rather than a claim in a docstring.

WHY ONLY THE CURRENT PARTIAL DAY IS EVER APPROXIMATE
====================================================
Every day boundary overwrites the previous 24 slots with true values. So at a
forecast origin t with h = t % 24, the 336-hour window holds exact values
everywhere except the h+1 slots of the day in progress. sim/reconstruct.py
turns that into the two arrays the harness uses, and it is the reason the
degradation is an order of magnitude smaller than a naive "everything is
approximate" analysis would suggest.

PACKET LOSS UNDER SYNC-SAFE (revision R4)
=========================================
The weight update no longer depends on any packet, so the failure modes
collapse to one statement:

  EXCEPTION lost   The node pushed u, the mirror pushed u_hat. They diverge and
                   the epsilon bound is void until the next RESYNC, which
                   REPLAYS the day from the transmit bitmap and the true values
                   and therefore repairs the buffer exactly, however many
                   exceptions were lost.
  RESYNC lost      Costs fidelity for that day (no true values for Y_EXACT) and
                   defers any pending repair. The weight update happened
                   anyway, identically, on both sides.
  ACK lost         Nothing. The ACK carries configuration only. This is the
                   two-generals failure the review identified, removed rather
                   than documented.

So: SYNCHRONISATION IS UNCONDITIONAL given one delivered RESYNC per day, and
fidelity degrades gracefully. Running with `sync_safe=False` restores the
original behaviour and its asymmetric ACK failure, which is how the cost of the
change is measured rather than argued.
"""

import numpy as np

from edge.codec import EPS_INF, CodecParams, CodecState
from edge.protocol import (TYPE_EXCEPTION, TYPE_RESYNC, decode_exception,
                           decode_resync)


class Mirror:
    """One meter's shadow predictor on the NanoPi."""

    def __init__(self, params: CodecParams, eps, u0: int, t_start: int,
                 node_id: int = 0, strict: bool = True):
        self.state = CodecState(params, eps, u0=u0, t_start=t_start)
        self.node_id = node_id
        self.strict = strict
        self.n_exception_applied = 0
        self.n_resync_applied = 0
        self.n_desync_suspected = 0
        self.n_repaired = 0

    # ------------------------------------------------------------------ run
    def run(self, packets, t_end: int):
        """Replay a packet log and reconstruct the series.

        `packets` is an iterable of (hour, payload_bytes). The mirror decodes
        them itself -- it is given bytes, not objects, so that a change to the
        wire format cannot pass this module by.

        Returns a dict of int64 arrays indexed by absolute hour over
        [t_start, t_end):

          u_within  what the mirror holds DURING the current day: u_hat on a
                    silent hour, u_hat + e on an exception hour.
          u_exact   what it holds once the day has been resynced. Identical to
                    u_within on any hour whose day never completed (the tail of
                    the split), which is correct: no resync has happened yet.
          resynced  bool, True where u_exact came from a RESYNC rather than
                    from falling back to u_within.
        """
        st = self.state
        t0 = st.t_start
        n = t_end - t0
        assert n > 0

        exc, res = {}, {}
        for hour, payload in packets:
            kind = payload[0] >> 4
            if kind == TYPE_EXCEPTION:
                p = decode_exception(payload)
                if p.node_id != self.node_id:
                    continue
                exc[hour] = p
            elif kind == TYPE_RESYNC:
                p = decode_resync(payload)
                if p.node_id != self.node_id:
                    continue
                res[hour] = p
            else:
                raise ValueError(f"unexpected packet type {kind} on the uplink")

        u_within = np.zeros(n, np.int64)
        u_exact = np.zeros(n, np.int64)
        resynced = np.zeros(n, bool)

        for t in range(t0, t_end):
            if st.is_boundary(t):
                pkt = res.get(t)
                if st.p.sync_safe:
                    # The repair comes FIRST: replay the node's day from the
                    # bitmap and the true values, so the weight update below
                    # runs on a buffer that provably matches the node's.
                    if pkt is not None and pkt.tx_bitmap:
                        before = st.sync_snapshot()
                        st.replay_day(list(pkt.values), list(pkt.tx_bitmap))
                        if st.sync_snapshot() != before:
                            self.n_repaired += 1
                    # unconditional, no packet required
                    st.day_boundary()
                    if pkt is not None and pkt.weights:
                        st.w = [int(x) for x in pkt.weights]
                        st.u_hat = st.predict()
                        st.mark_day_start()
                elif pkt is not None:
                    st.day_boundary(list(pkt.values))
                    # The repair (CodecParams.adopt_resync_weights, OFF by
                    # default): adopt the frame's weights AFTER the updates.
                    # Valid only if the frame carries POST-update weights, i.e.
                    # an UNGATED node (firmware/node/main.ino). A gated node --
                    # the paper's -- can only send PRE-update weights, and
                    # adopting those here undoes the update above. See
                    # docs/findings/2026-09-28-frame-loss-settled.md and
                    # eval/protocol_matrix.py ("adopt_before").
                    if st.p.adopt_resync_weights and pkt.weights:
                        st.w = [int(x) for x in pkt.weights]
                        st.u_hat = st.predict()
                else:
                    # legacy: no packet means keep old weights AND old buffer.
                    # So does the paper's (gated) node, which heard no ACK --
                    # a lost RESYNC is symmetric. A lost ACK is not: this
                    # branch is skipped (the frame arrived) while the node
                    # keeps its old weights. tests/test_protocol_matrix.py.
                    st.new_day()
                if pkt is not None:
                    self.n_resync_applied += 1
                    lo = max(t - 24, t0)
                    for x in range(lo, t):
                        u_exact[x - t0] = pkt.values[x - (t - 24)]
                        resynced[x - t0] = True

            p = exc.get(t)
            if p is not None:
                r = st.apply_exception(p.e, clamped=p.clamp)
                self.n_exception_applied += 1
                if self.strict and st.eps is not EPS_INF:
                    # Every exception is a self-check: the node transmitted
                    # because |e| exceeded ITS eps, and the mirror's eps can
                    # only be smaller (it can have missed exceptions, never
                    # invented them), so the same test must fire here.
                    if not (abs(p.e) > (r.eps_eff if r.eps_eff is not EPS_INF else 0)):
                        self.n_desync_suspected += 1
            else:
                r = st.apply_silent()
            u_within[t - t0] = r.u_recon

        # Hours whose day never completed keep the within-day value: no resync
        # has occurred, so that IS what the NanoPi holds.
        fill = ~resynced
        u_exact[fill] = u_within[fill]
        return {"u_within": u_within, "u_exact": u_exact, "resynced": resynced,
                "t_start": t0, "t_end": t_end}

    def snapshot(self) -> bytes:
        return self.state.snapshot()

    def sync_snapshot(self) -> bytes:
        """Compare THIS against the transmitter's sync_snapshot(). The full
        snapshot() would differ on the novelty EWMA by design; see
        CodecState.sync_snapshot."""
        return self.state.sync_snapshot()
