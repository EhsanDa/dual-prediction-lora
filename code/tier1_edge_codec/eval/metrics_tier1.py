"""
metrics_tier1.py  --  metrics that are new to Tier 1. Blueprint 8.3.

Everything that already exists is imported from metrics_tier2 and NOT
reimplemented: mase_scale, pooled_metrics, per_horizon, per_client,
distribution, frac_no_skill. What is here is the reconstruction, communication
and event-detection side, which Tier 2 had no reason to define.

DELIBERATELY NOT REPORTED (blueprint 8.2)
=========================================
lstnet_rse / lstnet_corr / lstnet_appendix. They exist to position against
published LTSF tables. This project measures the degradation of a FIXED model
under input perturbation, and for that RSE's between-client-variance
denominator and CORR's blindness to bias and gain are not merely uninformative,
they are misleading -- a policy that shifts every forecast by a constant scores
a perfect CORR. pinball and picp_sharpness: no probabilistic forecasts exist
here. climatology and the multi-model baseline rows: the baselines in this
project are TRANSMISSION POLICIES, not forecast models.

THE EVENT NUMBERS ARE AGAINST A PROXY
=====================================
The dataset has no anomaly labels, no event log and no transmission ground
truth. `proxy_events` defines a transparent stand-in on series_true and every
function that consumes it carries "proxy" in its name or its output keys. Any
table built from these must say so; there is no defensible way to present them
as detection performance against real events.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "vendor") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "vendor"))

from edge.protocol import (EXCEPTION_BYTES, aloha_collision_prob,   # noqa: E402
                           time_on_air)

EPS = 1e-12

# The Things Network fair-use allowance: 30 s of uplink airtime per node per
# day. Tighter than the ETSI 1% duty cycle at every spreading factor that
# matters here, and it is what the transmission budget is actually derived from
# (revision B3).
TTN_FAIR_USE_S = 30.0
ETSI_DUTY_CYCLE = 0.01


# ------------------------------------------------------- reconstruction side
def linf_violation(u_within, u_q, eps_eff, active):
    """Hours where |recon - quantised truth| exceeds the threshold that was
    ACTUALLY in force. Blueprint 9.3 wants zero, with zero tolerance.

    The threshold is eps_eff, not eps. Blueprint 4.8 says the guarantee is
    |recon - truth| <= eps, and blueprint 4.9 says the daily cap doubles eps on
    exhaustion; both cannot hold, and the code is the tie-breaker: the codec
    doubles, so the true guarantee is |recon - truth| <= eps_eff(t) <= 2*eps.
    Measured on the synthetic store at alpha = 0.5, the cap binds on 96% of
    days, so this is not a corner case -- it is the normal operating point, and
    reporting the bound as eps would be reporting a bound the system does not
    have. `frac_hours_capped` below is what makes that visible.
    """
    d = np.abs(np.asarray(u_within, np.int64) - np.asarray(u_q, np.int64))
    bad = (d > np.asarray(eps_eff, np.int64)) & np.asarray(active, bool)
    return int(bad.sum())


def frac_hours_capped(eps_eff, eps_nominal, active):
    """Share of active hours running under the DOUBLED threshold."""
    eps_eff = np.asarray(eps_eff, np.int64)
    nom = np.asarray([0 if e is None else e for e in eps_nominal], np.int64)[None, :]
    a = np.asarray(active, bool)
    if not a.any():
        return float("nan")
    return float(((eps_eff > nom) & a).mean() / max(a.mean(), EPS))


def recon_fidelity(u_recon, u_q, k, active, mase_scale=None):
    """Reconstruction error in kWh, and normalised by each client's own
    mase_scale so the four-orders-of-magnitude spread between meters does not
    turn the mean into a report about the largest one."""
    q = (2.0 ** np.asarray(k, np.float64))[None, :]
    d = (np.asarray(u_recon, np.float64) - np.asarray(u_q, np.float64)) * q
    a = np.asarray(active, bool)
    n = np.maximum(a.sum(axis=0), 1)
    mae_c = np.abs(np.where(a, d, 0.0)).sum(axis=0) / n
    rmse_c = np.sqrt(np.where(a, d, 0.0).__pow__(2).sum(axis=0) / n)
    out = {"recon_mae_kwh": float(np.abs(d[a]).mean()) if a.any() else np.nan,
           "recon_rmse_kwh": float(np.sqrt((d[a] ** 2).mean())) if a.any() else np.nan,
           "recon_mae_per_client": {int(c): float(mae_c[c]) for c in range(d.shape[1])},
           "recon_rmse_per_client": {int(c): float(rmse_c[c]) for c in range(d.shape[1])}}
    if mase_scale is not None:
        sc = np.asarray(mase_scale, np.float64)
        norm = {int(c): (float(mae_c[c] / sc[c])
                         if np.isfinite(sc[c]) and sc[c] > EPS else np.nan)
                for c in range(d.shape[1])}
        out["recon_mae_over_mase_scale"] = norm
    return out


def quantisation_mae(k, series_kwh, active=None):
    """The 2**k/2 floor, measured rather than asserted, and reported separately
    so the codec's own contribution to recon error is separable from it."""
    k = np.asarray(k, np.int64)
    q = 2.0 ** k.astype(np.float64)
    y = np.asarray(series_kwh, np.float64)
    uq = np.where(y >= 0, np.floor(y / q + 0.5), -np.floor(-y / q + 0.5)) * q
    d = np.abs(uq - y)
    if active is not None:
        a = np.asarray(active, bool)
        d = d[a] if a.any() else d
    return {"quantisation_mae_kwh": float(d.mean()),
            "quantisation_max_kwh": float(d.max()),
            "theoretical_floor_kwh": float((q / 2).mean())}


def approx_fraction_by_issue_hour(L=336):
    """Share of the look-back that is approximate, by issue hour: (h+1)/L."""
    return {int(h): float((h + 1) / L) for h in range(24)}


# --------------------------------------------------------- communication side
def communication_cost(tx, resync_bytes, active, sf_list=(7, 9, 12),
                       n_nodes=(10, 20, 40), bytes_per_tx=EXCEPTION_BYTES):
    """Blueprint 8.3. Counted from actual encoded packet sizes and COMPUTED
    airtime -- nothing here is a table lookup, because the scalability argument
    is the main claim this project makes and it has to be auditable."""
    tx = np.asarray(tx, bool)
    rb = np.asarray(resync_bytes)
    a = np.asarray(active, bool)
    days = np.maximum(a.sum(axis=0) / 24.0, EPS)

    n_exc = tx.sum(axis=0).astype(np.float64)
    n_res = (rb > 0).sum(axis=0).astype(np.float64)
    # `bytes_per_tx` is EXCEPTION_BYTES for the codec. The fixed periodic
    # baseline sends one BULK packet per refresh instead, so its per-packet size
    # is passed in and its resync_bytes array is empty -- otherwise the same
    # bytes would be counted twice.
    exc_bytes = n_exc * bytes_per_tx
    res_bytes = rb.sum(axis=0).astype(np.float64)
    pkt_day = (n_exc + n_res) / days
    byte_day = (exc_bytes + res_bytes) / days
    mean_res_bytes = float(rb[rb > 0].mean()) if (rb > 0).any() else 34.0
    n_pkt_total = max(float((n_exc + n_res).sum()), 1.0)
    mean_payload = float((exc_bytes + res_bytes).sum() / n_pkt_total)

    out = {"packets_per_day_mean": float(pkt_day.mean()),
           "packets_per_day_p50": float(np.percentile(pkt_day, 50)),
           "packets_per_day_p90": float(np.percentile(pkt_day, 90)),
           "packets_per_day_max": float(pkt_day.max()),
           "exceptions_per_day_mean": float((n_exc / days).mean()),
           "resyncs_per_day_mean": float((n_res / days).mean()),
           "bytes_per_day_mean": float(byte_day.mean()),
           "mean_payload_bytes": mean_payload,
           "mean_resync_bytes": mean_res_bytes,
           "airtime_per_day_s": {}, "duty_cycle_max": {},
           "aggregate_channel_occupancy": {}, "aloha_collision_prob": {}}

    # Per-frame time-on-air, reported so the airtime figures can be audited
    # without re-deriving them, and so the paper can say what a frame costs.
    # The two frame types differ by more than a factor of two at SF12, which
    # is why a single "mean packet" airtime is a fiction: quoting a budget in
    # PACKETS presupposes a payload mix, and that mix moves with the deadband.
    out["time_on_air_s"] = {}
    for sf in sf_list:
        t_exc = time_on_air(bytes_per_tx, sf)
        t_res = time_on_air(mean_res_bytes, sf)
        air_day = (n_exc * t_exc + n_res * t_res) / days
        mean_air = float((n_exc * t_exc + n_res * t_res).sum() / n_pkt_total)
        out["time_on_air_s"][sf] = {"exception": float(t_exc),
                                    "resync": float(t_res),
                                    "mean_realised": mean_air}
        out["airtime_per_day_s"][sf] = float(air_day.mean())
        # Worst rolling hour a node can have: the day boundary, where the
        # resync and a full exception budget can land in the same hour.
        out["duty_cycle_max"][sf] = float((t_res + 8 * t_exc) / 3600.0)
        out["aggregate_channel_occupancy"][sf] = {
            int(n): float(n * float(pkt_day.mean()) * mean_air / 86400.0)
            for n in n_nodes}
        out["aloha_collision_prob"][sf] = {
            int(n): float(aloha_collision_prob(n, float(pkt_day.mean()), mean_air))
            for n in n_nodes}
    return out


def staleness_hours(tx, active):
    """Age of the newest CORRECTED slot at each hour.

    A resync at t % 24 == 0 refreshes the whole previous day, so the reference
    resets to t-1 there. For daily batching nothing within the day is ever
    corrected and this degrades to (t % 24) + 1.

    OFF BY ONE AGAINST THE BLUEPRINT, DELIBERATELY. Blueprint 8.3 writes the
    daily-batching case as `t % 24`. That would make the window fresh at h = 0,
    but at h = 0 the resync has just delivered hours [t-24, t-1] and hour t
    itself is still a prediction, so the newest CORRECTED slot is one hour old.
    The blueprint agrees with h+1 elsewhere: 6.5 says exactly one slot is
    "within" at h = 0 and twenty-four at h = 23, i.e. h+1 approximate slots.
    Staleness and exposure are then the same count, which they have to be, and
    the freshness figure is consistent with the per-horizon one.
    """
    tx = np.asarray(tx, bool)
    a = np.asarray(active, bool)
    T, M = tx.shape
    out = np.zeros((T, M), np.int32)
    for c in range(M):
        last = -1
        for t in range(T):
            if not a[t, c]:
                continue
            if t % 24 == 0:
                last = t - 1
            if tx[t, c]:
                last = t
            out[t, c] = 0 if last < 0 else (t - last)
    return out


# --------------------------------------------------------- event side (PROXY)
def proxy_events(series_true, missing, train_end, q=99.0):
    """PROXY events: hour t is one if |y[t] - y[t-24]| exceeds that client's
    q-th percentile of the same quantity over the training split.

    There are NO ground-truth events in this dataset. This is a declared
    stand-in, chosen because it is transparent and reproducible, not because it
    is correct. Every consumer must say so.
    """
    y = np.asarray(series_true, np.float64)
    ms = np.asarray(missing, bool)
    T, M = y.shape
    d = np.full((T, M), np.nan)
    d[24:] = np.abs(y[24:] - y[:-24])
    ok = np.zeros((T, M), bool)
    ok[24:] = (~ms[24:]) & (~ms[:-24]) & np.isfinite(d[24:])
    thr = np.full(M, np.inf)
    for c in range(M):
        sel = ok[:train_end, c]
        if sel.sum() >= 100:
            thr[c] = float(np.percentile(d[:train_end, c][sel], q))
    return (ok & (d > thr[None, :])), thr


def event_detection(event_flag, proxy, active, max_latency=24):
    """Precision, recall and detection latency of the EVENT flag against the
    PROXY. Returns keys prefixed `proxy_` so no table can quietly drop the
    qualifier."""
    f = np.asarray(event_flag, bool) & np.asarray(active, bool)
    p = np.asarray(proxy, bool) & np.asarray(active, bool)
    tp = int((f & p).sum())
    fp = int((f & ~p).sum())
    fn = int((~f & p).sum())
    lat = []
    T, M = f.shape
    for c in range(M):
        fi = np.nonzero(f[:, c])[0]
        for t in np.nonzero(p[:, c])[0]:
            nxt = fi[(fi >= t) & (fi <= t + max_latency)]
            if nxt.size:
                lat.append(int(nxt[0] - t))
    return {"proxy_precision": tp / max(tp + fp, 1),
            "proxy_recall": tp / max(tp + fn, 1),
            "proxy_tp": tp, "proxy_fp": fp, "proxy_fn": fn,
            "proxy_latency_mean_h": float(np.mean(lat)) if lat else np.nan,
            "proxy_latency_p90_h": float(np.percentile(lat, 90)) if lat else np.nan,
            "proxy_detected_within_%dh" % max_latency:
                len(lat) / max(int(p.sum()), 1),
            "NOTE": "against a declared proxy; the dataset has no event labels"}


# ------------------------------------------------------------ coupling to T2
def delta_per_client(policy_pc, oracle_pc):
    """policy minus oracle, per client, over the clients both scored."""
    return {c: float(policy_pc[c] - oracle_pc[c])
            for c in policy_pc if c in oracle_pc
            and np.isfinite(policy_pc[c]) and np.isfinite(oracle_pc[c])}


if __name__ == "__main__":
    ok = 0

    def chk(name, cond, detail=""):
        global ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        assert cond, name
        ok += 1

    rng = np.random.default_rng(0)
    T, M = 24 * 40, 6
    active = np.ones((T, M), bool)
    u_q = rng.integers(100, 8000, (T, M)).astype(np.int64)
    eps = np.full(M, 50, np.int64)
    eps_eff = np.repeat(eps[None, :], T, 0)

    # --- linf_violation is an assertion, so it must catch a single bad hour.
    u_ok = u_q + rng.integers(-50, 51, (T, M))
    chk("linf_violation zero when inside eps",
        linf_violation(u_ok, u_q, eps_eff, active) == 0)
    u_bad = u_ok.copy()
    u_bad[7, 2] = u_q[7, 2] + 51
    chk("linf_violation catches one hour over by one unit",
        linf_violation(u_bad, u_q, eps_eff, active) == 1)
    # and it must respect the DOUBLED threshold, not the nominal one
    chk("linf_violation uses eps_eff, not eps",
        linf_violation(u_bad, u_q, eps_eff * 2, active) == 0)
    chk("frac_hours_capped sees the doubling",
        abs(frac_hours_capped(eps_eff * 2, list(eps), active) - 1.0) < 1e-9)
    chk("frac_hours_capped is 0 when nothing doubled",
        abs(frac_hours_capped(eps_eff, list(eps), active)) < 1e-9)

    # --- recon fidelity: a known constant offset must come back exactly.
    k = np.zeros(M, np.int64)          # q = 1 kWh
    off = np.full((T, M), 3, np.int64)
    f = recon_fidelity(u_q + off, u_q, k, active)
    chk("recon_mae_kwh exact on a constant offset",
        abs(f["recon_mae_kwh"] - 3.0) < 1e-12, f"{f['recon_mae_kwh']}")
    chk("recon_rmse equals mae for a constant offset",
        abs(f["recon_rmse_kwh"] - 3.0) < 1e-12)
    k2 = np.full(M, 2, np.int64)       # q = 4 kWh -> the same offset is 12 kWh
    f2 = recon_fidelity(u_q + off, u_q, k2, active)
    chk("recon error scales with the quantisation step",
        abs(f2["recon_mae_kwh"] - 12.0) < 1e-12)
    # normalisation must use the client's own scale
    sc = np.array([1., 2., 4., 8., np.nan, 0.0])
    f3 = recon_fidelity(u_q + off, u_q, k, active, mase_scale=sc)
    n = f3["recon_mae_over_mase_scale"]
    chk("per-client normalisation divides by that client's scale",
        abs(n[1] - 1.5) < 1e-12 and abs(n[3] - 0.375) < 1e-12)
    chk("degenerate scale gives nan, not a huge number",
        np.isnan(n[4]) and np.isnan(n[5]))

    # --- quantisation floor
    y = rng.uniform(0, 5000, (T, M))
    qm = quantisation_mae(np.full(M, 1, np.int64), y)
    chk("quantisation mae sits at half the step",
        abs(qm["quantisation_mae_kwh"] - 0.5) < 0.05, f"{qm['quantisation_mae_kwh']:.4f}")
    chk("quantisation max never exceeds half the step",
        qm["quantisation_max_kwh"] <= 1.0 + 1e-9)

    # --- window exposure closed form
    af = approx_fraction_by_issue_hour(336)
    chk("approx fraction is (h+1)/L", abs(af[0] - 1 / 336) < 1e-12
        and abs(af[23] - 24 / 336) < 1e-12)

    # --- communication cost
    tx = np.zeros((T, M), bool)
    tx[::24] = True                                  # exactly one per day
    rb = np.zeros((T // 24, M), np.int32)
    rb[1:] = 34
    cc = communication_cost(tx, rb, active)
    chk("packets/day counts exceptions and resyncs",
        abs(cc["packets_per_day_mean"] - (1 + 39 / 40)) < 1e-6,
        f"{cc['packets_per_day_mean']:.4f}")
    chk("airtime rises with spreading factor",
        cc["airtime_per_day_s"][12] > cc["airtime_per_day_s"][9]
        > cc["airtime_per_day_s"][7])
    chk("duty cycle at SF7 is inside the 1% ETSI limit",
        cc["duty_cycle_max"][7] < 0.01, f"{cc['duty_cycle_max'][7]:.5f}")
    chk("collision probability grows with node count",
        cc["aloha_collision_prob"][9][40] > cc["aloha_collision_prob"][9][10])

    # --- staleness. With no exceptions it must be exactly t % 24.
    st = staleness_hours(np.zeros((T, M), bool), active)
    chk("staleness degrades to (t%24)+1 with no exceptions",
        np.array_equal(st[24:, 0], np.arange(24, T) % 24 + 1))
    st2 = staleness_hours(np.ones((T, M), bool), active)
    chk("staleness is 0 when every hour transmits", int(st2[24:].max()) == 0)

    # --- proxy events: a planted spike must be found, a flat client must not
    #     produce events.
    # The fixture needs day-to-day variation, or the 99th percentile of
    # |y[t] - y[t-24]| is exactly 0 on a perfectly periodic series and every
    # non-zero hour becomes an "event". That degenerate case is worth knowing
    # about -- it is what the exactly-constant client 5 below exercises.
    base = np.tile(1000 + 200 * np.sin(2 * np.pi * np.arange(T) / 24)[:, None], (1, M))
    base = base + rng.normal(0, 20, (T, M))
    base[:, 5] = 500.0
    miss = np.zeros((T, M), bool)
    base[24 * 30 + 5, 0] += 5000
    pe, thr = proxy_events(base, miss, 24 * 25)
    chk("planted spike is a proxy event", pe[24 * 30 + 5, 0])
    chk("exactly constant client yields no proxy events", not pe[:, 5].any())
    ev = np.zeros((T, M), bool)
    ev[:, 1:] = pe[:, 1:]                     # perfect on every other client
    ev[24 * 30 + 5, 0] = True
    d = event_detection(pe.copy(), pe, active)
    chk("perfect flag on the planted event gives precision 1",
        abs(d["proxy_precision"] - 1.0) < 1e-9)
    ev2 = pe.copy()
    ev2[24 * 31 + 8, 0] = True                       # a flag where no proxy is
    d2 = event_detection(ev2, pe, active)
    chk("a spurious flag lowers precision", d2["proxy_precision"] < 1.0)
    chk("latency is 0 when the flag lands on the event",
        d["proxy_latency_mean_h"] == 0.0)

    # --- delta helper drops non-finite pairs rather than propagating them
    dd = delta_per_client({0: 1.0, 1: 2.0, 2: np.nan}, {0: 0.5, 1: 2.5, 2: 1.0})
    chk("delta_per_client subtracts and drops nan",
        set(dd) == {0, 1} and abs(dd[0] - 0.5) < 1e-12 and abs(dd[1] + 0.5) < 1e-12)

    print(f"\nall {ok} checks passed")
