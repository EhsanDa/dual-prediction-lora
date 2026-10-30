"""
codec_batch.py  --  the same codec, vectorised over clients. Blueprint 6.2.

26304 hours x 321 clients is 8.4M codec steps, and an eight-point alpha sweep
makes it 68M. The scalar reference cannot do that in Python. This module runs
M clients in parallel (vectorised) and T hours sequentially (it cannot be
otherwise -- the codec is a feedback loop).

TWO HARD REQUIREMENTS, BOTH TESTED
==================================
1. Every intermediate is np.int64. If numpy is ever allowed to promote to
   float the results become "nearly" identical, and "nearly" is exactly the
   failure mode this whole project is built to exclude. There are no float
   literals below.
2. It reproduces edge/codec.py EXACTLY -- same residuals, same flags, same
   snapshot bytes, for every client at every hour. tests/test_codec.py asserts
   it on the synthetic store. This module is an optimisation, never a second
   opinion; if the two disagree, the scalar one is right by definition.

INACTIVE CLIENTS
================
Meters commissioned mid-series (onboarding[c] > 0) are held in their
commissioned state until their start hour: their ring is uniformly u0, so
pushing u0 each hour leaves it uniform, and the counters that gate behaviour
(n_seen, weights, carry) are masked out. At t == t_start the batch state is
therefore identical to what warm_start() constructs in the scalar path, which
is what makes the equivalence test meaningful across the onboarding boundary
rather than only after it.
"""

import numpy as np

from edge.codec import EPS_INF, CodecParams

I64 = np.int64


def msb_index_arr(x):
    """Vectorised msb_index. msb_index(0) == 0, matching the scalar helper and
    the C loop. Binary descent, integer throughout."""
    x = np.asarray(x, I64)
    assert (x >= 0).all(), "msb_index of a negative value"
    r = np.zeros_like(x)
    v = x.copy()
    for sh in (32, 16, 8, 4, 2, 1):
        m = v >= (I64(1) << I64(sh))
        r = np.where(m, r + sh, r)
        v = np.where(m, v >> I64(sh), v)
    return r.astype(I64)


def round_shift_arr(x, n: int):
    x = np.asarray(x, I64)
    if n == 0:
        return x
    return (x + (I64(1) << I64(n - 1))) >> I64(n)


class BatchCodec:
    """M independent codecs advanced in lockstep.

    `eps` is per client: an int64 array, with EPS_NEVER marking alpha = inf.
    Using a sentinel instead of None keeps the comparison branchless and keeps
    the array int64, which requirement 1 above demands.
    """

    EPS_NEVER = I64(1) << I64(40)      # larger than any residual can ever be

    def __init__(self, params: CodecParams, eps, u0, t_start, T: int,
                 w_init_per_client=None):
        p = self.p = params
        self.M = M = len(u0)
        self.T = T
        self.u0 = np.asarray(u0, I64)
        self.t_start = np.asarray(t_start, I64)
        self.eps = np.asarray([self.EPS_NEVER if e is EPS_INF else e for e in eps], I64)

        self.buf = np.repeat(self.u0[:, None], p.buf_len, axis=1).astype(I64)
        self.head = 0
        # w_init_per_client is how the non-deployable per-client ridge ceiling
        # (row D of the ablation) is expressed without a second codec.
        self.w = (np.repeat(np.asarray(p.w_init, I64)[None, :], M, axis=0)
                  if w_init_per_client is None
                  else np.asarray(w_init_per_client, I64).copy())
        assert self.w.shape == (M, p.n_w), self.w.shape
        self.s = np.zeros(M, I64)
        self.n_tx_today = np.zeros(M, I64)
        self.n_seen = np.zeros(M, I64)
        self.carry = np.repeat(self.u0[:, None], max(p.n_carry, 1),
                               axis=1).astype(I64)
        self.day_true = np.repeat(self.u0[:, None], 24, axis=1).astype(I64)
        self.day_tx = np.zeros((M, 24), bool)
        self.t = 0
        self.u_hat = self._predict()

    # ------------------------------------------------------------ ring buffer
    def hist(self, j: int):
        p = self.p
        if j < p.buf_len:
            return self.buf[:, (self.head - j) % p.buf_len]
        k = j - p.buf_len
        assert k < p.n_carry, f"ring index {j} beyond ring + carry"
        return self.carry[:, k]

    def _set_hist(self, j: int, v):
        self.buf[:, (self.head - j) % self.p.buf_len] = v

    def _phi(self):
        cols = [self.hist(tap - 1) for tap in self.p.taps]
        cols.append(np.full(self.M, self.p.bias_input, I64))
        return np.stack(cols, axis=1)

    def _predict(self, phi=None):
        p = self.p
        phi = self._phi() if phi is None else phi
        acc = (self.w.astype(I64) * phi).sum(axis=1, dtype=I64)
        return np.clip(round_shift_arr(acc, p.w_q), p.u_min, p.u_clamp).astype(I64)

    # ------------------------------------------------------------- hourly step
    def eps_effective(self):
        p = self.p
        doubled = np.minimum(self.eps * 2, 0xFFFE)
        out = np.where(self.n_tx_today >= p.n_max_tx, doubled, self.eps)
        # alpha = inf must stay infinite under doubling as well
        return np.where(self.eps >= self.EPS_NEVER, self.EPS_NEVER, out).astype(I64)

    def step(self, u, active, clamped=None):
        """One hour for every client. `active` masks clients not yet
        commissioned; their state is held exactly as the scalar path would
        have it before warm_start()."""
        p = self.p
        u = np.asarray(u, I64)
        u_hat = self.u_hat
        e = np.where(active, u - u_hat, 0).astype(I64)
        ae = np.abs(e)

        s_new = np.clip(self.s + ((ae - self.s) >> I64(p.s_shift)), 0, 32767)
        self.s = np.where(active, s_new, self.s).astype(I64)

        eps_eff = self.eps_effective()
        tx = active & (ae > eps_eff)
        event = active & (self.n_seen >= p.event_warmup) & (ae > p.k_event * self.s)

        # THE LINE (blueprint 5.1 step 3): silent hour pushes the prediction.
        u_recon = np.where(tx, u, u_hat).astype(I64)
        u_push = np.where(active, u_recon, self.u0).astype(I64)
        self.head = (self.head + 1) % p.buf_len
        self.buf[:, self.head] = u_push
        col = self.t % 24
        self.day_true[:, col] = np.where(active, u, self.day_true[:, col])
        self.day_tx[:, col] = tx

        self.n_tx_today += tx.astype(I64)
        self.n_seen += active.astype(I64)
        self.t += 1
        self.u_hat = self._predict()
        return {"e": e, "tx": tx, "event": event, "u_recon": u_recon,
                "u_hat": u_hat, "eps_eff": eps_eff,
                "clamped": np.zeros(self.M, bool) if clamped is None else clamped}

    # ----------------------------------------------------------- day boundary
    def boundary_mask(self, t: int):
        if t % 24 != 0:
            return np.zeros(self.M, bool)
        return (I64(t) - self.t_start) >= 24

    def day_boundary(self, true_24, mask):
        """`mask` selects the clients whose boundary is due.

        In sync_safe mode the update runs on the RECONSTRUCTED ring and is
        unconditional -- no packet, no ACK, so `true_24` is unused and the mask
        is purely "has this client been running for a day". In legacy mode the
        ring is overwritten with `true_24` first and the mask additionally
        encodes resync delivery. See CodecState.day_boundary.
        """
        p = self.p
        m = np.asarray(mask, bool)
        if not m.any():
            self.n_tx_today = np.where(self.boundary_mask(self.t), 0,
                                       self.n_tx_today)
            self.day_tx[self.boundary_mask(self.t)] = False
            return
        if not p.sync_safe:
            for j in range(24):
                cur = self.hist(j)
                self._set_hist(j, np.where(m, true_24[:, 23 - j], cur))

        if p.adapt:
            for j in range(24):
                off_x = 23 - j
                phi = np.stack([self.hist(off_x + tap) for tap in p.taps]
                               + [np.full(self.M, p.bias_input, I64)], axis=1)
                pred = self._predict(phi)
                target = self.hist(off_x) if p.sync_safe else true_24[:, j]
                e = (target - pred).astype(I64)
                self._nlms_update(phi, e, m)

        for i in range(p.n_carry):
            self.carry[:, i] = np.where(m, self.hist(p.buf_len - 24 + i),
                                        self.carry[:, i])
        bm = self.boundary_mask(self.t)
        self.n_tx_today = np.where(bm, 0, self.n_tx_today)
        self.day_tx[bm] = False
        self.u_hat = self._predict()      # see CodecState.day_boundary

    def _nlms_update(self, phi, e, m):
        p = self.p
        P = (phi * phi).sum(axis=1, dtype=I64)
        sh = np.clip(msb_index_arr(P) - p.w_q + p.k_mu, 0, 31).astype(I64)
        prod = (e[:, None] * phi).astype(I64)
        # Always round half-up; see CodecState._nlms_update.
        one = I64(1) << np.maximum(sh - 1, 0)[:, None]
        d = np.where(sh[:, None] == 0, prod, (prod + one) >> sh[:, None])
        nw = np.clip(self.w + d, -p.w_clamp, p.w_clamp).astype(I64)
        self.w = np.where(m[:, None], nw, self.w).astype(I64)

    # ------------------------------------------------------------- snapshot
    def sync_snapshot(self, c: int) -> bytes:
        import struct

        from edge.codec import CodecState
        p = self.p
        eps = int(self.eps[c])
        eps_wire = 0xFFFF if eps >= int(self.EPS_NEVER) else min(eps, 0xFFFE)
        fmt = f"<{p.buf_len}hB{p.n_w}ihHBIh"
        return struct.pack(fmt, *[int(x) for x in self.buf[c]],
                           self.head, *[int(x) for x in self.w[c]],
                           int(self.u_hat[c]), eps_wire, int(self.n_tx_today[c]),
                           int(self.t) & 0xFFFFFFFF,
                           int(self.carry[c, 0]) if p.n_carry else 0)

    def canonical(self, c: int) -> bytes:
        """See CodecState.canonical -- ring contents in history order, so a
        rotation difference between the two implementations is not mistaken for
        a state difference."""
        import struct
        p = self.p
        eps = int(self.eps[c])
        eps_wire = 0xFFFF if eps >= int(self.EPS_NEVER) else min(eps, 0xFFFE)
        return struct.pack(
            f"<{p.buf_len}h{p.n_w}ihhHBBI",
            *[int(self.hist(j)[c]) for j in range(p.buf_len)],
            *[int(x) for x in self.w[c]], int(self.u_hat[c]), int(self.s[c]),
            eps_wire, int(self.n_tx_today[c]),
            int(min(self.n_seen[c], 255)) & 0xFF, int(self.t) & 0xFFFFFFFF)

    def snapshot(self, c: int) -> bytes:
        """Bytes identical to CodecState.snapshot() for client c, so the two
        implementations can be diffed directly rather than field by field."""
        import struct

        from edge.codec import CodecState
        p = self.p
        eps = int(self.eps[c])
        eps_wire = 0xFFFF if eps >= int(self.EPS_NEVER) else min(eps, 0xFFFE)
        ring = [int(x) for x in self.buf[c]]
        fmt = (f"<{p.buf_len}hBB{p.n_w}ihhHBBIih")
        return struct.pack(
            fmt, *ring, self.head, p.buf_len, *[int(x) for x in self.w[c]],
            int(self.u_hat[c]), int(self.s[c]), eps_wire, int(self.n_tx_today[c]),
            int(min(self.n_seen[c], 255)) & 0xFF, int(self.t) & 0xFFFFFFFF, 0,
            int(self.carry[c, 0]) if p.n_carry else 0)


def run_batch(params: CodecParams, u_series, eps, u0, t_start, T,
              resync_ok=None, progress=None, w_init_per_client=None):
    """Run every client over the whole series.

    u_series : (T, M) int64 quantised readings
    resync_ok: (n_days, M) bool or None. None means every RESYNC is delivered
               and acknowledged.

    Returns dict of (T, M) arrays: e, tx, event, u_within, u_exact, resynced.
    """
    u_series = np.asarray(u_series, I64)
    T = int(T)
    M = u_series.shape[1]
    bc = BatchCodec(params, eps, u0, t_start, T,
                    w_init_per_client=w_init_per_client)
    t_start = np.asarray(t_start, I64)

    e = np.zeros((T, M), np.int32)
    tx = np.zeros((T, M), bool)
    ev = np.zeros((T, M), bool)
    within = np.zeros((T, M), I64)
    exact = np.zeros((T, M), I64)
    resynced = np.zeros((T, M), bool)
    eps_eff = np.zeros((T, M), np.int64)

    for t in range(T):
        bmask = bc.boundary_mask(t)
        if bmask.any():
            delivered = bmask if resync_ok is None else (bmask & resync_ok[t // 24])
            # sync_safe: the weight update is unconditional, so the DEVICE-side
            # boundary uses bmask. Delivery only decides whether the hub gets
            # the true values, i.e. it costs fidelity, never synchronisation.
            bc.day_boundary(bc.day_true.copy(),
                            bmask if params.sync_safe else delivered)
            lo = max(t - 24, 0)
            exact[lo:t][:, delivered] = u_series[lo:t][:, delivered]
            resynced[lo:t, delivered] = True
        active = (I64(t) >= t_start)
        r = bc.step(u_series[t], active)
        e[t] = r["e"]
        tx[t] = r["tx"]
        ev[t] = r["event"]
        within[t] = np.where(active, r["u_recon"], 0)
        eps_eff[t] = r["eps_eff"]
        if progress and t % progress == 0:
            print(f"    hour {t:,d} / {T:,d}", flush=True)

    exact = np.where(resynced, exact, within)
    return {"e": e, "tx": tx, "event": ev, "u_within": within,
            "u_exact": exact, "resynced": resynced, "eps_eff": eps_eff}
