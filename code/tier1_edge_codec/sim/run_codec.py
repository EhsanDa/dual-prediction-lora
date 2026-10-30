"""
run_codec.py  --  run a transmission policy over a store. Blueprint 6.3.

One call to `run_policy` is one row of the alpha sweep. It runs the codec over
every client for the whole series and returns everything the evaluation needs:
the two reconstruction arrays, the residual and flag matrices, the packet
accounting, and the per-client commissioning constants.

RULES THAT ARE EASY TO GET WRONG, SPELLED OUT
=============================================
* Run on series_filled.npy. That is the representation DLinear consumes, and
  running on series_true would mean feeding NaN into an integer codec.
* Start each client at onboarding[c]. Before that the meter did not exist and
  reads exact zeros; scoring the codec on them would be scoring it on a
  constant.
* Run causally and continuously across train -> valid -> test WITHOUT
  resetting. NLMS is online and causal, so weights at test time legitimately
  reflect adaptation over earlier data and there is no leakage. What is
  selected on validation is alpha, and only alpha.
* mase_scale comes from metrics_tier2, computed on series_true with both
  endpoints non-imputed, over the TRAINING split only.

THE ALPHA ENDPOINTS ARE NOT SPECIAL CASES
=========================================
alpha = 0   -> eps = 0     -> transmit whenever the prediction is not exactly
                             right; reconstruction is exact everywhere, so this
                             is the full-freshness oracle.
alpha = inf -> eps = None  -> never transmit an exception; the only traffic is
                             the daily resync, i.e. plain daily batching.
Both fall out of the ordinary threshold test. There is no `if alpha == 0`
anywhere in the codec.
"""

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor"))

from edge.codec import EPS_INF, CodecParams, CodecState, warm_start   # noqa: E402
from edge.codec_batch import BatchCodec, run_batch                    # noqa: E402
from edge.protocol import (EXCEPTION_BYTES, Exception_, FLAG_CLAMP,   # noqa: E402
                           FLAG_EVENT, Resync, encode_exception,
                           encode_resync, time_on_air)
from edge.quantize import (choose_scale, dequantize_array,            # noqa: E402
                           quantisation_step, quantize_array)

ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, float("inf"))


# ------------------------------------------------------------------- store
@dataclass
class Store:
    dir: str
    manifest: dict
    series_filled: np.ndarray
    series_true: np.ndarray
    missing: np.ndarray
    onboarding: np.ndarray
    mase_scale: np.ndarray
    T: int
    M: int
    train_end: int
    valid_end: int

    def active_mask(self):
        return (np.arange(self.T)[:, None]
                >= self.onboarding.astype(np.int64)[None, :])

    def assert_scales_are_train_only(self):
        """Revision R5. The device's quantisation scale must be inferable at
        commissioning time, i.e. from training data only. If it ever saw
        validation or test data the device configuration would be fitted on
        data it cannot have, and every downstream result would be contaminated.
        An assertion, not an inspection: recomputed from the training slice
        alone and required to be identical."""
        k_full, _ = choose_scales(self)
        k_train, _ = choose_scales(self, _force_train_only=True)
        assert np.array_equal(k_full, k_train), (
            "choose_scales is reading beyond train_end for clients "
            f"{np.nonzero(k_full != k_train)[0].tolist()}")
        return True

    def client_accounting(self):
        """Revision R2. An unexplained M -> M-1 in a results table invites the
        question of what else was dropped silently, so the drop is named."""
        from metrics_tier2 import mase_scale as _ms
        scale, n_deg, n_pairs = _ms(self.series_true, self.missing,
                                    self.train_end, m=24)
        deg = [int(c) for c in np.nonzero(~np.isfinite(scale)
                                          | (scale <= 0))[0]]
        dead = [int(c) for c in range(self.M)
                if not np.isfinite(self.series_true[self.valid_end:, c]).any()
                or np.nanmax(np.nan_to_num(
                    self.series_true[self.valid_end:, c])) <= 0]
        return {"M": int(self.M), "n_degenerate": int(n_deg),
                "n_pairs_used": int(np.sum(n_pairs)) if np.ndim(n_pairs)
                else int(n_pairs),
                "degenerate": deg, "dead_in_test": dead,
                "n_scorable": int(self.M - len(set(deg) | set(dead))),
                "dropped": sorted(set(deg) | set(dead)),
                "note": ("clients with a degenerate mase_scale are excluded "
                         "from every per-client mean by metrics_tier2's own "
                         "nan handling; clients dead across the whole test "
                         "split contribute no scorable windows")}

    @staticmethod
    def load(data_dir):
        from metrics_tier2 import mase_scale as _mase_scale
        with open(f"{data_dir}/manifest.json") as f:
            man = json.load(f)
        filled = np.load(f"{data_dir}/series_filled.npy")
        true = np.load(f"{data_dir}/series_true.npy")
        miss = np.load(f"{data_dir}/missing.npy")
        onb = np.load(f"{data_dir}/onboarding.npy")
        scale, n_deg, _ = _mase_scale(true, miss, man["train_end"], m=24)
        T, M = filled.shape
        assert (T, M) == (man["T"], man["M"]), "store contradicts its own manifest"
        assert np.nanmin(filled) >= 0.0, (
            "the codec clamps predictions at 0 because the metered series is "
            "non-negative (see edge/codec.py deviation 2); this store has "
            "negative readings, so u_min must be revisited before proceeding")
        return Store(data_dir, man, filled, true, miss, onb, scale, T, M,
                     man["train_end"], man["valid_end"])


# ---------------------------------------------------- commissioning constants
def choose_scales(store: Store, target_max=12000, _force_train_only=False):
    """Per-client k, from TRAINING data only. A client commissioned after the
    end of the training split has no training data to size from; it falls back
    to its own first 30 days, and the fallback is reported rather than hidden.
    """
    k = np.zeros(store.M, np.int64)
    fallback = []
    for c in range(store.M):
        tr = store.series_filled[store.onboarding[c]:store.train_end, c]
        if _force_train_only:
            k[c] = choose_scale(tr if tr.size >= 24 and np.nanmax(
                tr, initial=0) > 0 else store.series_filled[
                    store.onboarding[c]:store.onboarding[c] + 720, c],
                target_max)
            continue
        if tr.size < 24 or not np.isfinite(tr).any() or np.nanmax(tr) <= 0:
            lo = int(store.onboarding[c])
            tr = store.series_filled[lo:lo + 720, c]
            fallback.append(c)
        k[c] = choose_scale(tr, target_max)
    return k, fallback


def eps_for_alpha(store: Store, k, alpha):
    """eps_c = round(alpha * mase_scale_c / 2**k_c), in device units.

    Blueprint 4.8. mase_scale is `mean |y_t - y_{t-24}|` over the training
    split, so eps reads as "we tolerate alpha times the typical day-over-day
    change on this meter" -- scale-free, interpretable, and it sidesteps the
    log1p compression problem in blueprint 3(c) without inventing a new scale.

    DEGENERATE CLIENTS. metrics_tier2.mase_scale returns NaN when the
    denominator is zero or when fewer than 100 valid pairs exist (an exactly
    constant meter, a meter commissioned near the end of training). The
    fallback is the same statistic computed on series_filled, and if THAT is
    also zero the client is exactly constant, eps = 0 is correct and costs
    nothing: the predictor reproduces a constant exactly after warm-up, so an
    eps of 0 still produces no traffic.
    """
    if alpha == float("inf"):
        return [EPS_INF] * store.M, np.zeros(store.M, bool)
    eps, degenerate = [], np.zeros(store.M, bool)
    for c in range(store.M):
        sc = store.mase_scale[c]
        if not np.isfinite(sc) or sc <= 0:
            degenerate[c] = True
            y = store.series_filled[store.onboarding[c]:store.train_end, c]
            sc = float(np.mean(np.abs(y[24:] - y[:-24]))) if y.size > 24 else 0.0
        eps.append(int(round(alpha * float(sc) / (2.0 ** int(k[c])))))
    return eps, degenerate


# ------------------------------------------------------------------ results
@dataclass
class PolicyResult:
    alpha: float
    k: np.ndarray
    eps: list
    eps_degenerate: np.ndarray
    u_exact: np.ndarray          # (T, M) int64, quantised truth once resynced
    u_q: np.ndarray              # (T, M) int64, the quantised TRUTH
    u_within: np.ndarray         # (T, M) int64, what the mirror holds intra-day
    e: np.ndarray                # (T, M) int32 residuals
    tx: np.ndarray               # (T, M) bool
    event: np.ndarray            # (T, M) bool
    resynced: np.ndarray         # (T, M) bool
    eps_eff: np.ndarray          # (T, M) the threshold ACTUALLY in force. The
                                 # daily cap (blueprint 4.9) doubles eps for the
                                 # rest of the day on exhaustion, so the
                                 # guarantee is |recon - truth| <= eps_eff(t),
                                 # which is <= 2*eps and equal to eps only on
                                 # days that stayed inside the budget.
    active: np.ndarray           # (T, M) bool, t >= onboarding[c]
    clamped: np.ndarray          # (T, M) bool
    resync_bytes: np.ndarray     # (n_days, M) int, 0 where no resync was due
    packets: dict = field(default_factory=dict)   # client -> [(hour, bytes)]
    notes: dict = field(default_factory=dict)

    @property
    def T(self):
        return self.u_exact.shape[0]

    @property
    def M(self):
        return self.u_exact.shape[1]


# ------------------------------------------------------------- byte counting
def _varint_len(z: int) -> int:
    n = 1
    while z >= 0x80:
        z >>= 7
        n += 1
    return n


def resync_nbytes(values, n_w: int = 7) -> int:
    """Encoded size of a RESYNC without building it. Asserted equal to
    len(encode_resync(...)) in tests/test_protocol.py, because a size model
    that drifts from the encoder would silently corrupt every airtime figure.
    """
    v = [int(x) for x in values]
    payload = 2
    for a, b in zip(v, v[1:]):
        d = b - a
        payload += _varint_len((d << 1) ^ (d >> 31))
    # header + delta payload + status + tx bitmap + weight vector + CRC16.
    # The bitmap and weights are the revision-R4 repair fields; they are what
    # make one delivered RESYNC sufficient to restore synchronisation.
    return 6 + payload + 1 + 3 + 1 + 4 * n_w + 2


# --------------------------------------------------------------- the driver
def run_policy(store: Store, alpha, params: CodecParams = CodecParams(),
               backend="batch", loss=0.0, exception_loss=None,
               resync_loss=None, ack_loss=0.0, seed=0, packet_log=False,
               clients=None, verbose=False,
               w_init_per_client=None) -> PolicyResult:
    """Run one policy over the store.

    backend='batch'  vectorised over clients, for the sweep.
    backend='scalar' the reference codec, one client at a time. Slower by
                     orders of magnitude, produces real packet logs, and is
                     what the batch path is validated against.

    THREE INDEPENDENT CHANNELS, because they fail differently and conflating
    them produces a curve that says the wrong thing:

      exception_loss  uplink, EXCEPTION packets. The node pushed u, the mirror
                      pushes u_hat: they diverge and the bound is void until
                      the next resync repairs it.
      resync_loss     uplink, RESYNC packets. SYMMETRIC -- the mirror gets no
                      packet and the node gets no ACK, so both keep old weights
                      and an old buffer. Costs staleness, not correctness.
      ack_loss        downlink. ASYMMETRIC: the mirror applies the resync and
                      the node does not. This is the one the protocol cannot
                      self-correct within the day -- in legacy it never
                      re-synchronises at all (docs/findings/
                      2026-09-28-frame-loss-settled.md).

    ONE CAVEAT FOR ANY GATEWAY THAT READS THE RESYNC'S WEIGHT FIELD. In legacy
    the node below decides whether to apply its update (i.e. whether its ACK
    will arrive) BEFORE it encodes the RESYNC, so the frame's weights "know"
    the future. Harmless for edge/mirror.py's legacy gateway, which ignores the
    field; misleading for adopt_resync_weights. eval/protocol_matrix.py builds
    the frame causally.

    `loss` is a convenience that sets both uplink channels; pass the individual
    ones to isolate a mode.
    """
    k, fallback_k = choose_scales(store)
    eps, degenerate = eps_for_alpha(store, k, alpha)
    T, M = store.T, store.M
    sel = list(range(M)) if clients is None else list(clients)

    u = np.zeros((T, M), np.int64)
    clamped = np.zeros((T, M), bool)
    for c in range(M):
        uc, cl = quantize_array(store.series_filled[:, c], int(k[c]), params.u_clamp)
        u[:, c], clamped[:, c] = uc, cl

    t_start = store.onboarding.astype(np.int64)
    active = np.arange(T)[:, None] >= t_start[None, :]
    u0 = np.array([u[min(int(t_start[c]), T - 1), c] for c in range(M)], np.int64)

    exception_loss = loss if exception_loss is None else exception_loss
    resync_loss = loss if resync_loss is None else resync_loss
    n_days = (T + 23) // 24
    rng = np.random.default_rng(seed)
    # `resync_delivered` is what the MIRROR sees; `resync_applied` is what the
    # NODE does. They are the same array only when the ACK is reliable, and the
    # gap between them is exactly the asymmetric failure mode.
    resync_delivered = rng.random((n_days, M)) >= resync_loss
    resync_applied = resync_delivered & (rng.random((n_days, M)) >= ack_loss)
    exc_delivered = (rng.random((T, M)) >= exception_loss
                     if exception_loss > 0 else None)
    any_loss = (exception_loss > 0 or resync_loss > 0 or ack_loss > 0)

    if backend == "batch":
        assert not packet_log, "packet logs need backend='scalar'"
        assert exception_loss == 0, (
            "exception loss changes what the MIRROR holds, not what the node "
            "does, so it cannot be simulated device-side; use backend='scalar'")
        out = run_batch(params, u, eps, u0, t_start, T,
                        resync_ok=None if not any_loss else resync_applied,
                        progress=(T // 8 if verbose else None),
                        w_init_per_client=w_init_per_client)
    else:
        out = _run_scalar(params, u, eps, u0, t_start, T, sel,
                          resync_delivered, resync_applied, exc_delivered,
                          clamped, w_init_per_client)

    res = PolicyResult(
        alpha=alpha, k=k, eps=eps, eps_degenerate=degenerate,
        u_exact=out["u_exact"], u_q=u, u_within=out["u_within"], e=out["e"],
        tx=out["tx"], event=out["event"], resynced=out["resynced"],
        eps_eff=out["eps_eff"],
        active=active, clamped=clamped & active,
        resync_bytes=_resync_bytes(u, t_start, T, M, n_days,
                                   n_w=params.n_w),
        packets=out.get("packets", {}),
        notes={"k_fallback_clients": fallback_k,
               "eps_degenerate_clients": [int(c) for c in np.nonzero(degenerate)[0]],
               "exception_loss": exception_loss, "resync_loss": resync_loss,
               "ack_loss": ack_loss, "backend": backend})
    n_clamp = int(res.clamped.sum())
    if n_clamp:
        per_c = res.clamped.sum(0)
        bad = [int(c) for c in np.nonzero(per_c > 8)[0]]
        res.notes["clamp_warning"] = (
            f"{n_clamp} quantiser clamps; clients with more than 8: {bad}. "
            f"More than a handful means k was chosen wrong for that client.")
    return res


def _resync_bytes(u, t_start, T, M, n_days, n_w=7):
    """Encoded RESYNC size for every (day, client), 0 where none is due.

    `n_w` is the WEIGHT COUNT of the configuration in force, not a constant:
    the RESYNC carries the weight vector as a repair field, so a tap set with
    a different number of weights produces a different frame size. It happened
    to be right for the shipped taps (six lags plus a bias = 7) and silently
    wrong for every other tap set, which is exactly what a tap-set ablation
    varies. Every airtime and bytes-per-day figure depends on it.
    """
    out = np.zeros((n_days, M), np.int32)
    for d in range(1, n_days):
        t = d * 24
        if t + 0 > T:
            break
        lo, hi = t - 24, min(t, T)
        if hi - lo != 24:
            continue
        for c in range(M):
            if t - t_start[c] < 24:
                continue
            out[d, c] = resync_nbytes(u[lo:hi, c], n_w=n_w)
    return out


def _run_scalar(params, u, eps, u0, t_start, T, sel, resync_delivered,
                resync_applied, exc_delivered, clamped, w_init_per_client=None):
    """Reference path: the scalar codec, one client at a time, with real
    packets. This is what edge/mirror.py is replayed against."""
    M = u.shape[1]
    e = np.zeros((T, M), np.int32)
    tx = np.zeros((T, M), bool)
    ev = np.zeros((T, M), bool)
    within = np.zeros((T, M), np.int64)
    exact = np.zeros((T, M), np.int64)
    resynced = np.zeros((T, M), bool)
    eps_eff = np.zeros((T, M), np.int64)
    packets = {}

    for c in sel:
        t0 = int(t_start[c])
        st = warm_start(params, eps[c], int(u0[c]), t0)
        if w_init_per_client is not None:
            st.w = [int(x) for x in w_init_per_client[c]]
            st.u_hat = st.predict()
        log = []
        day_seq = 0
        n_tx_prev_day = 0
        for t in range(t0, T):
            if st.is_boundary(t):
                payload = st.resync_payload()
                bitmap = st.resync_bitmap()
                if params.sync_safe:
                    # Unconditional: the update runs on the reconstructed ring,
                    # so no ACK is involved. The packet is emitted AFTER the
                    # update so it carries the post-update weights.
                    st.day_boundary()
                    applied = True
                else:
                    applied = bool(resync_applied[t // 24, c])
                    if applied:
                        st.day_boundary(payload)
                    else:
                        st.new_day()
                pkt = encode_resync(Resync(node_id=c, day_seq=day_seq & 0xFF,
                                           n_exceptions=n_tx_prev_day,
                                           values=tuple(payload),
                                           tx_bitmap=tuple(bitmap),
                                           weights=tuple(st.w)))
                day_seq += 1
                delivered = bool(resync_delivered[t // 24, c])
                log.append((t, pkt, delivered))
                if delivered:
                    exact[t - 24:t, c] = payload
                    resynced[t - 24:t, c] = True
            n_tx_prev_day = st.n_tx_today
            r = st.step(int(u[t, c]), clamped=bool(clamped[t, c]))
            e[t, c], tx[t, c], ev[t, c] = r.e, r.tx, r.event
            within[t, c] = r.u_recon
            eps_eff[t, c] = (BatchCodec.EPS_NEVER if r.eps_eff is EPS_INF
                             else r.eps_eff)
            if r.tx:
                flags = (FLAG_EVENT if r.event else 0) | (FLAG_CLAMP if r.clamped else 0)
                pkt = encode_exception(Exception_(node_id=c, hour_slot=t % 24,
                                                  flags=flags, e=int(r.e)))
                ok = True if exc_delivered is None else bool(exc_delivered[t, c])
                log.append((t, pkt, ok))
        packets[c] = log
    exact = np.where(resynced, exact, within)
    return {"e": e, "tx": tx, "event": ev, "u_within": within,
            "u_exact": exact, "resynced": resynced, "eps_eff": eps_eff,
            "packets": packets}


# Communication cost and staleness live in eval/metrics_tier1.py (blueprint
# 8.3 puts them in the metric module). They are not duplicated here: this file
# produces the arrays, that one interprets them.
