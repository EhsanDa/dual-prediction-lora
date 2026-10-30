"""
gen_golden.py  --  Python -> golden_vectors.h. Blueprint 10.2.

Records a trace from the reference codec and emits it as a C header. The trace
is chosen to contain, deliberately, every branch that matters:

  * at least one day boundary WITH the resync applied
  * at least one day boundary SKIPPED (packet loss / no ACK), which is the
    branch where both sides must keep old weights and an old buffer
  * a run of silent hours (the push-u_hat path)
  * exceptions, including consecutive ones that exhaust the daily budget and
    trigger the eps doubling
  * a quantiser clamp
  * the EVENT flag firing after the warm-up

test_host.c replays it against edge_codec.c and compares u_hat, e, the flags
and the snapshot bytes. ANY mismatch is a failed build, not a warning: a C port
that is only nearly right desynchronises from the NanoPi, and the resulting
reconstruction is wrong in a way that no packet reveals.
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from edge.codec import CodecParams, warm_start          # noqa: E402
from edge.quantize import choose_scale, quantize_array  # noqa: E402

SNAP_STRIDE = 24        # a full snapshot every 24 h keeps the header small
                        # enough to sit in AVR Flash alongside the application


def build_series(n_hours, seed=11):
    """A deterministic series shaped to hit every branch."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    prof = np.array([.45, .40, .38, .37, .38, .45, .62, .85, 1.0, 1.05, 1.05,
                     1.02, .98, 1.0, 1.02, 1.0, .95, .92, .95, 1.0, .95, .80,
                     .62, .52])
    y = 1500 * prof[t % 24] * np.where((t // 24) % 7 >= 5, 0.6, 1.0)
    y = y * np.exp(rng.normal(0, 0.06, n_hours))
    # a quiet stretch, so a long silent run is guaranteed
    y[120:140] = y[119]
    # a step change, so exceptions cluster and the daily budget is exhausted
    y[200:] *= 1.9
    # and one reading far above full scale, to force the quantiser clamp
    y[260] = 1e7
    return y


def record(n_hours=384, alpha=1.0, seed=11, params=None, pretrained=None):
    params = params or CodecParams()
    if pretrained:
        import json
        params = params.with_w_init(json.load(open(pretrained))["w_global_q"])
    y = build_series(n_hours, seed)
    k = choose_scale(y[:168])
    u, clamped = quantize_array(y, k, params.u_clamp)
    mase = float(np.mean(np.abs(u[24:168] - u[:144])))
    eps = int(round(alpha * mase))

    st = warm_start(params, eps, int(u[0]), 0)
    rows, snaps, true24, boundary = [], [], [], []
    # Skip the resync on day 5 -- the packet-loss branch.
    skip_days = {5}
    for t in range(n_hours):
        if st.is_boundary(t):
            payload = st.resync_payload()
            # Under sync_safe the boundary is unconditional -- there is no ACK
            # to lose -- so the "skipped" branch only exists in legacy mode. The
            # trace still records the distinction so the C port's #if is
            # exercised by whichever build is generated.
            if params.sync_safe:
                boundary.append(1)
                true24.append(list(payload))
                st.day_boundary()
            elif (t // 24) in skip_days:
                boundary.append(2)
                st.new_day()
            else:
                boundary.append(1)
                true24.append(list(payload))
                st.day_boundary(payload)
        else:
            boundary.append(0)
        r = st.step(int(u[t]), clamped=bool(clamped[t]))
        rows.append((r.u_hat, r.e, (1 if r.tx else 0) | (2 if r.event else 0)
                     | (4 if r.clamped else 0)))
        if t % SNAP_STRIDE == 0 or t == n_hours - 1:
            snaps.append((t, st.snapshot()))

    stats = {
        "n": n_hours, "eps": eps, "k": int(k), "u0": int(u[0]),
        "n_tx": sum(1 for _, _, f in rows if f & 1),
        "n_event": sum(1 for _, _, f in rows if f & 2),
        "n_clamp": int(clamped.sum()),
        "n_boundary_applied": boundary.count(1),
        "n_boundary_skipped": boundary.count(2),
        "longest_silent_run": _longest_run([f & 1 for _, _, f in rows]),
    }
    return u, clamped, boundary, true24, rows, snaps, stats, params


def _longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = 0 if f else cur + 1
        best = max(best, cur)
    return best


def _carr(name, vals, ctype="int16_t", per_line=12):
    out = [f"static const {ctype} {name}[] = {{"]
    for i in range(0, len(vals), per_line):
        out.append("    " + ", ".join(str(int(v)) for v in vals[i:i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def render(u, clamped, boundary, true24, rows, snaps, stats, params):
    # `params` carries the w_init the trace was recorded with; the C build must
    # be generated from the same source or the very first prediction differs.
    L = []
    L.append("/* golden_vectors.h -- GENERATED by export/gen_golden.py. "
             "Do not edit. */")
    L.append("#ifndef GOLDEN_VECTORS_H\n#define GOLDEN_VECTORS_H\n")
    L.append("#include <stdint.h>\n")
    L.append(f"/* Recorded from the Python reference codec (edge/codec.py).")
    for k, v in stats.items():
        L.append(f" *   {k}: {v}")
    L.append(" */")
    L.append(f"#define GOLDEN_N              {stats['n']}")
    L.append(f"#define GOLDEN_EPS            {stats['eps']}u")
    L.append(f"#define GOLDEN_U0             {stats['u0']}")
    L.append(f"#define GOLDEN_T_START        0u")
    L.append(f"#define GOLDEN_SNAP_STRIDE    {SNAP_STRIDE}")
    L.append(f"#define GOLDEN_N_SNAP         {len(snaps)}")
    L.append(f"#define GOLDEN_N_BOUNDARY     {len(true24)}")
    L.append(f"#define GOLDEN_SNAPSHOT_BYTES {len(snaps[0][1])}\n")

    L.append(_carr("GOLDEN_U", u))
    L.append(_carr("GOLDEN_CLAMPED", clamped.astype(int), "uint8_t", 24))
    L.append("/* 0 = no boundary, 1 = resync applied, 2 = resync lost "
             "(keep old weights AND old buffer) */")
    L.append(_carr("GOLDEN_BOUNDARY", boundary, "uint8_t", 24))
    L.append(f"static const int16_t GOLDEN_TRUE24[GOLDEN_N_BOUNDARY][24] = {{")
    for v in true24:
        L.append("    { " + ", ".join(str(int(x)) for x in v) + " },")
    L.append("};")
    L.append(f"#define GOLDEN_N_W            {params.n_w}")
    L.append(_carr("GOLDEN_W_INIT", params.w_init, "int32_t", 8))
    L.append(_carr("GOLDEN_UHAT", [r[0] for r in rows]))
    L.append(_carr("GOLDEN_E", [r[1] for r in rows]))
    L.append("/* bit0 TX, bit1 EVENT, bit2 CLAMP */")
    L.append(_carr("GOLDEN_FLAGS", [r[2] for r in rows], "uint8_t", 24))
    L.append(_carr("GOLDEN_SNAP_AT", [t for t, _ in snaps], "uint16_t", 16))
    L.append("static const uint8_t GOLDEN_SNAP[GOLDEN_N_SNAP]"
             "[GOLDEN_SNAPSHOT_BYTES] = {")
    for _, b in snaps:
        L.append("    { " + ", ".join(str(x) for x in b) + " },")
    L.append("};")
    L.append("\n#endif /* GOLDEN_VECTORS_H */")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=384)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--pretrained", default=None,
                    help="edge_pretrained.json; without it the trace uses the "
                         "persistence initialisation")
    ap.add_argument("--out", default="firmware/golden_vectors.h")
    a = ap.parse_args()
    rec = record(a.hours, a.alpha, pretrained=a.pretrained)
    txt = render(*rec)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(txt)
    stats = rec[6]
    print(f"wrote {a.out}  ({len(txt) / 1024:.1f} KB of source)")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    checks = [("n_tx", "no exception in the trace"),
              ("n_event", "the EVENT branch is never exercised"),
              ("n_clamp", "the quantiser clamp is never exercised")]
    if not rec[7].sync_safe:
        checks.append(("n_boundary_skipped",
                       "the lost-ACK branch is never exercised"))
    for need, why in checks:
        assert stats[need] > 0, (
            f"{why}: the golden vectors would not test it. Adjust "
            f"build_series() rather than lowering the bar.")
    assert stats["longest_silent_run"] >= 5, (
        "no meaningful silent run: the push-u_hat path is the whole system and "
        "must be covered")


if __name__ == "__main__":
    main()
