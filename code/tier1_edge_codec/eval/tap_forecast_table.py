"""
tap_forecast_table.py  --  the weight-count ablation, forecast level, rebuilt
from tier1_results.json without re-scoring anything.

WHY THIS EXISTS
===============
run_eval.py scored the T7 and T12 rows correctly and then printed an empty
table: the `cost` dict is assembled field by field, and `tap_id` was never
copied into it, so the family-T lookup matched nothing. The rows themselves are
in the artefact. Re-running the streaming pass to recover a table that can be
derived from a file already on disk would cost forty minutes for nothing.

run_eval.py is fixed for future runs; this reproduces the table for the run you
already have. It is also the safer of the two, since it never touches the
scoring path.

WHAT IT REPORTS
===============
Forecast degradation (dMASE against the no-codec control) for each tap set, at
MATCHED PACKET RATE. Comparing at matched alpha inverts every ordering: at a
fixed deadband a worse predictor triggers more often, so it transmits more and
therefore looks better. Rate is the axis.

Node SRAM is the exact codec_t footprint: the ring at 24 + max_tap int16
samples, the staged day at 24 int16, the weights at 4 bytes each, and 22 bytes
of counters and state. Memory tracks the LARGEST LAG; accuracy tracks the
WEIGHT COUNT. They are independent axes and that is the point of the table.
"""

import argparse
import json
import re

# Artifact locations live in one place; see paths.py for the contract.
import os as _os, sys as _sys                                  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
import paths                                                   # noqa: E402

# Must stay identical to LADDER in eval/tap_ablation.py and TAP_LADDER in
# eval/run_eval.py, or the three tables disagree about what T7 means.
TAP_LADDER = {
    "T7":  (1, 2, 3, 24, 25),
    "T8":  (1, 2, 3, 24, 25, 168),
    "T9":  (1, 2, 3, 24, 25, 48, 168),
    "T10": (1, 2, 3, 24, 25, 168, 169, 336),
    "T11": tuple(range(1, 25)),
    "T12": tuple(range(1, 25)) + (168,),
}
SHIPPED = "T8"


def sram_bytes(taps):
    n_w = len(taps) + 1
    return 2 * (24 + max(taps)) + 2 * 24 + 4 * n_w + 22


def interp_at_rate(pts, rate):
    """Log-linear interpolation of dMASE onto a packet rate. None outside the
    swept range: extrapolating a curve that bends would invent a comparison."""
    pts = sorted((p, d) for p, d in pts if p is not None and d is not None)
    if len(pts) < 2:
        return None
    xs = [p for p, _ in pts]
    ys = [d for _, d in pts]
    if rate < xs[0] or rate > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if rate <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            t = (rate - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=paths.result("tier1_results.json"))
    ap.add_argument("--rates", default="15,10,6")
    ap.add_argument("--out", default=paths.result("tier1_tap_forecast.json"))
    a = ap.parse_args()

    res = json.load(open(a.results))
    cost, met = res["cost"], res["metrics"]
    rates = [float(x) for x in a.rates.split(",")]

    # Group the scored rows by tap set. The shipped configuration is family C;
    # the alternatives are labelled "T7@...", "T12@..." and so on.
    groups = {}
    for lab, c in cost.items():
        if c.get("family") == "C":
            tid = SHIPPED
        else:
            m = re.match(r"^(T\d+)@", lab)
            if not m or c.get("family") != "T":
                continue
            tid = m.group(1)
        d = met.get(lab, {}).get("delta_MASE_vs_control", {}).get("mean")
        if d is None:
            continue
        groups.setdefault(tid, []).append((c["packets_per_day_mean"], d))

    known = [t for t in TAP_LADDER if t in groups]
    if SHIPPED not in known:
        raise SystemExit("no family-C rows found; is this the right results file?")

    print("=== WEIGHT-COUNT ABLATION, forecast level, at MATCHED packets/day ===")
    print("  dMASE against the no-codec control; lower is better. Compared at")
    print("  matched packet rate, never at matched alpha.")
    print(f"\n{'id':>5s} {'w':>3s} {'maxlag':>7s} {'SRAM B':>7s}"
          + "".join(f"{str(r) + ' pkt/day':>15s}" for r in rates))

    out = {}
    for tid in known:
        taps = TAP_LADDER[tid]
        vals = [interp_at_rate(groups[tid], r) for r in rates]
        out[tid] = {"taps": list(taps), "n_weights": len(taps) + 1,
                    "max_lag": max(taps), "sram_bytes": sram_bytes(taps),
                    "shipped": tid == SHIPPED,
                    "dMASE_vs_control_at": {str(r): v
                                            for r, v in zip(rates, vals)},
                    "curve": sorted(groups[tid])}
        star = "*" if tid == SHIPPED else " "
        print(f"{tid + star:>5s} {len(taps) + 1:3d} {max(taps):7d} "
              f"{sram_bytes(taps):7d}"
              + "".join("            n/a" if v is None else f"{v:15.4f}"
                        for v in vals))
    print("  * shipped configuration.")

    ref = rates[len(rates) // 2]
    base = out[SHIPPED]["dMASE_vs_control_at"][str(ref)]
    if base:
        print(f"\nrelative to the shipped configuration at {ref:g} packets/day")
        for tid in known:
            if tid == SHIPPED:
                continue
            v = out[tid]["dMASE_vs_control_at"][str(ref)]
            if v is None:
                print(f"  {tid:>4s}: outside the swept range at {ref:g}")
                continue
            db = out[tid]["sram_bytes"] - out[SHIPPED]["sram_bytes"]
            print(f"  {tid:>4s}: dMASE {base:.4f} -> {v:.4f} "
                  f"({(v - base) / base:+.1%}), SRAM {db:+d} B")

    json.dump({"rates": rates, "shipped": SHIPPED, "rows": out},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
