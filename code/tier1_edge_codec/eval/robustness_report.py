"""robustness_report.py -- turn results/robustness/* into the reported intervals.

Answers the reviewers' "single seed" objection with measurements. WHAT is
stochastic and what is not is audited in docs/reference/stochasticity.md; this
module only reports how much the stochastic parts actually move.

Three independent questions, three sections:

  1. CLIENT BOOTSTRAP. Every headline figure is a mean or percentile over ~320
     meters, reported as a point estimate. Resampling clients with replacement
     gives a CI without re-running any simulation, and answers the question the
     "one seed" objection is really asking: would this hold on other meters?

  2. TABLE I SEED SPREAD. The frame-loss table is one Monte Carlo draw. Across
     several seeds we report mean, sd and range per cell.

  3. ALPHA STABILITY. alpha is selected on a 150k-window subsample of
     validation. If the selected alpha moves with the subsample, the operating
     point is an artefact of one draw.

Reads only; writes nothing. Print the report and paste the numbers, or use
--json to emit a machine-readable summary.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                                      # noqa: E402

ROB = os.path.join(paths.RESULTS_DIR, "robustness")
ROWS = ["control", "batch-daily", "C@0.25", "C@0.5", "C@0.7", "C@1", "C@2"]


# --------------------------------------------------------------- bootstrap --
def bootstrap_ci(values, stat, n_resamples, ci, rng):
    """Percentile bootstrap over UNITS (clients), not over observations.

    Clients are the quasi-independent replicates here: each meter contributes
    one MASE. Resampling them with replacement estimates how much the reported
    aggregate depends on which meters happened to be in the panel.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    draws = stat(v[idx], axis=1)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {"point": float(stat(v, axis=0) if v.ndim > 1 else stat(v[None, :], axis=1)[0]),
            "lo": float(np.percentile(draws, lo)),
            "hi": float(np.percentile(draws, hi)),
            "n": int(v.size)}


def section_bootstrap(n_resamples, ci, seed=0):
    path = os.path.join(ROB, "per_client.json")
    if not os.path.exists(path):
        return None, f"  not run -- {path} absent (repro/robustness.sh --stage bootstrap)"
    doc = json.load(open(path))
    rng = np.random.default_rng(seed)
    out, lines = {}, []
    lines.append(f"  clients scored: {doc['n_clients_scored']}  "
                 f"resamples: {n_resamples:,d}  CI: {ci}%")
    lines.append(f"  {'row':<12} {'MASE [CI]':>34} {'dMASE mean [CI]':>34}"
                 f" {'dMASE p90 [CI]':>34}")
    lines.append("  " + "-" * 116)
    for lab in ROWS:
        mp = doc["metrics_per_client"].get(lab, {}).get("MASE")
        dp = doc["delta_MASE_per_client"].get(lab)
        if mp is None:
            continue
        mase = bootstrap_ci(list(mp.values()), np.mean, n_resamples, ci, rng)
        dmean = (bootstrap_ci(list(dp.values()), np.mean, n_resamples, ci, rng)
                 if dp else None)
        dp90 = (bootstrap_ci(list(dp.values()),
                             lambda a, axis: np.percentile(a, 90, axis=axis),
                             n_resamples, ci, rng) if dp else None)
        out[lab] = {"MASE": mase, "delta_MASE_mean": dmean,
                    "delta_MASE_p90": dp90}

        def f(d):
            if d is None:
                return f"{'--':>34}"
            return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]".rjust(34)
        lines.append(f"  {lab:<12} {f(mase)} {f(dmean)} {f(dp90)}")
    return out, "\n".join(lines)


# ------------------------------------------------------------ Table I seeds --
MODES = [("resync", "Resynchronisation"), ("exception", "Exception"),
         ("ack", "Acknowledgement")]


def section_paired(n_resamples, ci, seed=0):
    """The headline claim, bootstrapped as a PAIRED comparison.

    Every meter contributes a dMASE to both rows, so the two samples are not
    independent and must be resampled together by client. The paired difference
    is also the far tighter statistic: MASE itself varies enormously across
    meters (its CI spans ~0.03), which is exactly why the paper reports dMASE
    against a per-client oracle rather than MASE levels.
    """
    path = os.path.join(ROB, "per_client.json")
    if not os.path.exists(path):
        return None, f"  not run -- {path} absent"
    doc = json.load(open(path))
    dm = doc["delta_MASE_per_client"]
    if "batch-daily" not in dm or "C@0.7" not in dm:
        return None, "  rows batch-daily / C@0.7 not both present"
    rng = np.random.default_rng(seed)
    common = sorted(set(dm["batch-daily"]) & set(dm["C@0.7"]))
    b = np.array([dm["batch-daily"][c] for c in common])
    c = np.array([dm["C@0.7"][c] for c in common])
    idx = rng.integers(0, len(common), size=(n_resamples, len(common)))
    db, dc = b[idx].mean(1), c[idx].mean(1)
    diff, ratio = db - dc, db / dc
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2

    def q(x):
        return float(np.percentile(x, lo)), float(np.percentile(x, hi))
    out = {"n_clients": len(common),
           "batch_daily_mean": float(b.mean()), "batch_daily_ci": q(db),
           "proposed_mean": float(c.mean()), "proposed_ci": q(dc),
           "paired_difference": float((b - c).mean()), "difference_ci": q(diff),
           "ratio": float(b.mean() / c.mean()), "ratio_ci": q(ratio),
           "draws_favouring_proposed": float((diff > 0).mean()),
           "clients_favouring_proposed": int((c < b).sum())}
    L = [f"  paired over {len(common)} clients, {n_resamples:,d} resamples, {ci}% CI",
         "",
         f"  batch-daily mean dMASE : {b.mean():.4f}  [{q(db)[0]:.4f}, {q(db)[1]:.4f}]",
         f"  C@0.7       mean dMASE : {c.mean():.4f}  [{q(dc)[0]:.4f}, {q(dc)[1]:.4f}]",
         f"  paired difference      : {(b-c).mean():.4f}  [{q(diff)[0]:.4f}, {q(diff)[1]:.4f}]",
         f"  ratio batch / proposed : {b.mean()/c.mean():.2f}x [{q(ratio)[0]:.2f}, {q(ratio)[1]:.2f}]",
         "",
         f"  bootstrap draws favouring the proposed scheme: "
         f"{(diff>0).mean()*100:.2f}%",
         f"  individual meters favouring it: {(c<b).sum()}/{len(common)} "
         f"({(c<b).mean()*100:.1f}%)"]
    return out, "\n".join(L)


def section_loss_seeds():
    files = sorted(glob.glob(os.path.join(ROB, "loss_sweep_seed*.json")))
    if not files:
        return None, "  not run -- no loss_sweep_seed*.json (repro/robustness.sh --stage loss)"
    runs = {}
    for p in files:
        m = re.search(r"seed(\d+)\.json$", p)
        runs[int(m.group(1))] = json.load(open(p))
    seeds = sorted(runs)
    any_doc = runs[seeds[0]]
    losses = [l for l in any_doc["losses"] if l > 0]
    out, lines = {}, []
    lines.append(f"  seeds: {seeds}   clients: {any_doc['n_clients']}   "
                 f"alpha: {any_doc['alpha']}")
    lines.append(f"  cells are violation_rate_mean x 100, over {len(seeds)} draws")
    lines.append(f"  {'row':<20} {'loss':>6} {'mean':>9} {'sd':>9} "
                 f"{'min':>9} {'max':>9} {'paper(seed 7)':>14}")
    lines.append("  " + "-" * 82)
    for key, label in MODES:
        out[key] = {}
        for loss in losses:
            vals = []
            for s in seeds:
                cur = runs[s]["sync_safe_0"]["curves"][key]
                e = next(x for x in cur if abs(x["loss"] - loss) < 1e-12)
                vals.append(e["violation_rate_mean"] * 100)
            a = np.asarray(vals)
            base = vals[seeds.index(7)] if 7 in seeds else float("nan")
            out[key][str(loss)] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                                   "min": float(a.min()), "max": float(a.max()),
                                   "seed7": float(base), "n_seeds": int(a.size)}
            lines.append(f"  {label:<20} {loss:>6.2f} {a.mean():>9.4f} "
                         f"{(a.std(ddof=1) if a.size > 1 else 0.0):>9.4f} "
                         f"{a.min():>9.4f} {a.max():>9.4f} {base:>14.4f}")
    return out, "\n".join(lines)


# ------------------------------------------------------------ alpha stability --
def section_client_count():
    """Is the 64 -> 321 client shift real, or is it seed noise?

    The published Table I used a 64-client stratified subset. The robustness
    runs use all 321. Those two differ in TWO ways at once -- panel size and,
    potentially, the packet-loss draw -- so a raw difference cannot be
    attributed without an estimate of the seed noise.

    The comparison that IS valid: hold the seed fixed at 7 (the published draw)
    and compare 64 vs 321, then ask whether that shift is larger than the
    seed-to-seed standard deviation measured at 321 clients. If it is, the shift
    is a property of the panel, not of the draw.
    """
    frozen = os.path.join(paths.FROZEN_DIR, "tier1_loss_sensitivity.json")
    seed7 = os.path.join(ROB, "loss_sweep_seed7.json")
    if not (os.path.exists(frozen) and os.path.exists(seed7)):
        return None, "  not run -- needs results_frozen + loss_sweep_seed7.json"
    fz, s7 = json.load(open(frozen)), json.load(open(seed7))

    # seed sd at 321 clients, if more than one seed has landed
    files = sorted(glob.glob(os.path.join(ROB, "loss_sweep_seed*.json")))
    runs = {int(re.search(r"seed(\d+)\.json$", p).group(1)): json.load(open(p))
            for p in files}
    seeds = sorted(runs)

    def cell(doc, mode, loss):
        cur = doc["sync_safe_0"]["curves"][mode]
        e = next(x for x in cur if abs(x["loss"] - loss) < 1e-12)
        return e["violation_rate_mean"] * 100

    losses = [l for l in s7["losses"] if l > 0]
    out, L = {}, []
    L.append(f"  panel: {fz['n_clients']} clients (published) -> "
             f"{s7['n_clients']} clients, seed 7 held fixed")
    L.append(f"  seed sd measured at {s7['n_clients']} clients over "
             f"{len(seeds)} seed(s): {seeds}")
    if len(seeds) < 2:
        L.append("  ** only one seed so far -- sd not yet estimable, "
                 "attribution deferred **")
    else:
        L.append(f"  NOTE: sd is estimated from {len(seeds)} draws, so the "
                 "2-sd rule below is a")
        L.append("        DIRECTIONAL screen, not a formal test. It separates "
                 "shifts that are")
        L.append("        orders of magnitude above the noise from those that "
                 "are not; it does")
        L.append("        not support a p-value.")
    L.append("")
    L.append(f"  {'row':<20}{'loss':>6}{'64cl':>9}{'321cl':>9}{'shift':>9}"
             f"{'seed sd':>9}  verdict")
    L.append("  " + "-" * 74)
    for mode, label in MODES:
        out[mode] = {}
        for loss in losses:
            a, b = cell(fz, mode, loss), cell(s7, mode, loss)
            vals = [cell(runs[s], mode, loss) for s in seeds]
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            shift = b - a
            if len(vals) < 2:
                verdict = "pending"
            elif sd == 0 and shift == 0:
                verdict = "identical"
            elif abs(shift) > 2 * sd:
                verdict = "PANEL (> 2 sd)"
            else:
                verdict = "within seed noise"
            out[mode][str(loss)] = {"c64": a, "c321": b, "shift": shift,
                                    "seed_sd": sd, "verdict": verdict}
            L.append(f"  {label:<20}{loss:>6.2f}{a:>9.4f}{b:>9.4f}"
                     f"{shift:>+9.4f}{sd:>9.4f}  {verdict}")
    return out, "\n".join(L)


def section_alpha():
    files = sorted(glob.glob(os.path.join(ROB, "run_eval_selectseed*.json")))
    if not files:
        return None, "  not run -- no run_eval_selectseed*.json (repro/robustness.sh --stage alpha)"
    out, lines = {}, []
    lines.append(f"  {'select_seed':>12} {'alpha':>7} {'pkt/day':>9} "
                 f"{'airtime s':>11} {'dMASE p90':>11}")
    lines.append("  " + "-" * 54)
    for p in files:
        s = int(re.search(r"selectseed(\d+)\.json$", p).group(1))
        sel = json.load(open(p))["selection"]
        out[str(s)] = {k: sel[k] for k in
                       ("selected_alpha", "packets_per_day", "airtime_sf12_s",
                        "dMASE_p90")}
        lines.append(f"  {s:>12} {sel['selected_alpha']:>7} "
                     f"{sel['packets_per_day']:>9.4f} "
                     f"{sel['airtime_sf12_s']:>11.4f} {sel['dMASE_p90']:>11.6f}")
    alphas = {v["selected_alpha"] for v in out.values()}
    lines.append("")
    if len(alphas) == 1:
        lines.append(f"  STABLE: every subsample selects alpha = {alphas.pop()}. "
                     "The operating point is not an artefact of one draw.")
    else:
        lines.append(f"  ** UNSTABLE: selected alpha varies across {sorted(alphas)}. "
                     "This is a finding -- the operating point depends on the "
                     "validation subsample. **")
    return out, "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=95)
    ap.add_argument("--json", default=None, help="also write a summary JSON here")
    a = ap.parse_args(argv)

    print("=" * 82)
    print("ROBUSTNESS REPORT -- see docs/reference/stochasticity.md for what is")
    print("stochastic at all. The deterministic core has no seed to vary.")
    print("=" * 82)

    summary = {}
    for title, fn in (("1. CLIENT BOOTSTRAP (no new simulation)",
                       lambda: section_bootstrap(a.resamples, a.ci)),
                      ("2. THE HEADLINE CLAIM, PAIRED OVER CLIENTS",
                       lambda: section_paired(a.resamples, a.ci)),
                      ("3. TABLE I ACROSS PACKET-LOSS SEEDS", section_loss_seeds),
                      ("4. PANEL SIZE: 64 vs 321 CLIENTS", section_client_count),
                      ("5. ALPHA SELECTION STABILITY", section_alpha)):
        print(f"\n{title}\n")
        data, text = fn()
        print(text)
        summary[title.split(".")[0]] = data

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
