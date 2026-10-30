"""
gen_meter_trace.py  --  golden_vectors.h -> node/meter_trace.h

THE NODE HAS NO METER AND NEVER WILL. This project's input is dataset data, so
`meter_read_kwh()` on the bench replays a recorded trace rather than reading a
register. This generator emits that trace as a C header.

WHY IT IS THE *GOLDEN* TRACE AND NOT AN ARBITRARY ONE. The golden vectors
already record, for the same 384 hours, what the reference codec produced from
these exact inputs. Because SCALE_K is negative, kWh = u * 2^SCALE_K is an
exact binary fraction and `quantise()` in main.ino returns the original integer
unchanged. So replaying these values through the WHOLE node application --
scheduling, quantiser, boundary order, packet encoders -- must reproduce
GOLDEN_UHAT, GOLDEN_E and GOLDEN_FLAGS exactly.

That turns the application into something checkable against an artifact that
already exists, rather than something that merely runs. An arbitrary trace
would exercise the same code and prove nothing about its output.

The values live in PROGMEM: 384 int16 is 768 B of Flash and zero SRAM, which
the part has in abundance. Nothing here is read at run time from SRAM.
"""

import argparse
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from edge.codec import CodecParams, warm_start          # noqa: E402
from edge.quantize import choose_scale, quantize_array  # noqa: E402
from export.gen_golden import build_series              # noqa: E402


def firmware_w_init():
    """The weight vector the FIRMWARE is compiled with.

    CodecParams() defaults to persistence, (1<<20, 0, ...), but the committed
    golden_vectors.h and edge_config.h both carry a fitted vector -- so a
    reference run left on the default predicts something the firmware never
    would. The first attempt at this file did exactly that and disagreed with
    the board at hour 0.

    It is read from golden_vectors.h's GOLDEN_W_INIT rather than re-fitted,
    because test_host.c already asserts that array equals edge_config.h's
    W_INIT and refuses to run if they have drifted. So this is the same number
    the compiler sees, traced to a file rather than recomputed.
    """
    path = os.path.join(REPO, "firmware", "golden_vectors.h")
    text = open(path, encoding="utf-8").read()
    m = re.search(r"GOLDEN_W_INIT\[\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise SystemExit("GOLDEN_W_INIT not found in " + path)
    return [int(v) for v in re.findall(r"-?\d+", m.group(1))]


def record_node(n_hours=384, alpha=1.0, seed=11):
    """The same inputs gen_golden records, run WITHOUT its injected loss.

    WHY THIS EXISTS AND WHY IT IS NOT golden_vectors.h. The golden fixture is
    branch-coverage material: gen_golden deliberately drops the resync on day 5
    (`skip_days = {5}`) so that test_host.c exercises codec_new_day as well as
    codec_day_boundary. A node running normally applies its own day boundary
    unconditionally -- main.ino says so, and the design says so, because the
    resync carries the post-update weights and needs no ACK. So a real node
    CANNOT reproduce hour 120 onward of that fixture, and the first attempt to
    run the application against it diverged there for exactly that reason.

    The application therefore needs a trajectory a node could actually have:
    identical inputs, identical eps, every boundary applied. It comes from
    edge/codec.py, the same specification, so it is generated rather than
    invented -- and `build_series` is imported from gen_golden rather than
    copied, so the two cannot drift apart.
    """
    params = CodecParams().with_w_init(firmware_w_init())
    y = build_series(n_hours, seed)
    k = choose_scale(y[:168])
    u, clamped = quantize_array(y, k, params.u_clamp)
    mase = float(np.mean(np.abs(u[24:168] - u[:144])))
    eps = int(round(alpha * mase))

    # THE TRACE STORES PRE-CLAMP VALUES, and that is the whole point of this
    # line. quantize_array returns values already clipped to u_clamp, so a node
    # replaying them would never see anything to clamp and its `clamped` flag
    # would stay false -- which is exactly what happened on the first run: 383
    # of 384 hours matched, and hour 260 lost bit 2. build_series injects a
    # spike of 40 000 000 device units there precisely so the clamp branch is
    # exercised, and the firmware's own quantise() must be the thing that
    # clamps it. So the raw quantised value is what is shipped.
    scaled = np.asarray(y, dtype=np.float64) / (2.0 ** k)
    u_raw = np.where(scaled >= 0, np.floor(scaled + 0.5),
                     -np.floor(-scaled + 0.5)).astype(np.int64)
    assert np.array_equal(np.clip(u_raw, -params.u_clamp, params.u_clamp), u), \
        "pre-clamp reconstruction disagrees with quantize_array"

    st = warm_start(params, eps, int(u[0]), 0)
    rows, n_boundary = [], 0
    for t in range(n_hours):
        if st.is_boundary(t):
            payload = st.resync_payload()
            n_boundary += 1
            if params.sync_safe:
                st.day_boundary()
            else:
                st.day_boundary(payload)
        r = st.step(int(u[t]), clamped=bool(clamped[t]))
        rows.append((r.u_hat, r.e, (1 if r.tx else 0) | (2 if r.event else 0)
                     | (4 if r.clamped else 0)))
    return u_raw, clamped, rows, eps, int(k), n_boundary


def carr(name, vals, ctype, per_line, progmem=False):
    out = ["static const %s %s[]%s = {" % (ctype, name,
                                           " PROGMEM" if progmem else "")]
    for i in range(0, len(vals), per_line):
        out.append("    " + ", ".join("%6d" % int(v) for v in vals[i:i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    fw = os.path.join(REPO, "firmware", "node")
    ap.add_argument("--out", default=os.path.join(fw, "meter_trace.h"))
    ap.add_argument("--expected", default=os.path.join(fw, "node_expected.h"))
    ap.add_argument("--hours", type=int, default=384)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args(argv)

    u_raw, clamped, rows, eps, k, n_boundary = record_node(a.hours, a.alpha)
    n = len(u_raw)
    n_clamp = int(np.asarray(clamped).sum())
    n_tx = sum(1 for _, _, f in rows if f & 1)

    # ---- the PROGMEM input trace the node replays -------------------------
    dev = ["/* meter_trace.h -- GENERATED by export/gen_meter_trace.py. "
           "Do not edit. */",
           "#ifndef METER_TRACE_H", "#define METER_TRACE_H", "",
           "#include <stdint.h>", "#include <avr/pgmspace.h>", "",
           "/* The dataset trace the node replays, in DEVICE UNITS, BEFORE the",
           " * quantiser clamp. There is no meter on the bench and there is not",
           " * meant to be: this project's input is dataset data.",
           " *",
           " * int32 and pre-clamp, deliberately. One hour carries a spike of",
           " * 40 000 000 device units that build_series injects so the clamp",
           " * branch is exercised; shipping the already-clipped value would",
           " * hide it and the node's `clamped` flag would never be set.",
           " * main.ino's quantise() is what must do the clamping.",
           " *",
           " * These are the same inputs golden_vectors.h records (clipped, the",
           " * two agree). The EXPECTED OUTPUTS are in node_expected.h and are",
           " * NOT the golden ones -- see that file for why.",
           " *",
           " * PROGMEM: read with pgm_read_dword, never dereferenced directly.",
           " */",
           "#define METER_TRACE_N       %d" % n,
           "#define METER_TRACE_EPS     %du" % eps,
           "#define METER_TRACE_T_START 0u",
           "#define METER_TRACE_N_CLAMP %d" % n_clamp,
           "",
           carr("METER_TRACE", u_raw, "int32_t", 8, progmem=True),
           "", "#endif /* METER_TRACE_H */", ""]

    # ---- the host-side expectation ---------------------------------------
    host = ["/* node_expected.h -- GENERATED by export/gen_meter_trace.py. "
            "Do not edit. */",
            "#ifndef NODE_EXPECTED_H", "#define NODE_EXPECTED_H", "",
            "#include <stdint.h>", "",
            "/* What edge/codec.py produces from METER_TRACE, with EVERY day",
            " * boundary applied -- i.e. a trajectory a real node could have.",
            " *",
            " * THIS IS DELIBERATELY NOT golden_vectors.h. That fixture drops the",
            " * resync on day 5 on purpose, so test_host.c covers codec_new_day",
            " * as well as codec_day_boundary. A node applies its boundary",
            " * unconditionally -- there is no ACK to lose -- so it cannot",
            " * reproduce the golden trace from hour 120 on, and should not be",
            " * expected to. Comparing against the golden fixture instead of",
            " * this one is a mistake that looks exactly like a firmware bug.",
            " */",
            "#define NODE_EXP_N        %d" % n,
            "#define NODE_EXP_EPS      %du" % eps,
            "#define NODE_EXP_N_TX     %d" % n_tx,
            "#define NODE_EXP_N_BOUND  %d" % n_boundary,
            "#define NODE_EXP_N_CLAMP  %d" % n_clamp,
            "",
            carr("NODE_EXP_UHAT", [r[0] for r in rows], "int16_t", 12),
            "",
            carr("NODE_EXP_E", [r[1] for r in rows], "int16_t", 12),
            "",
            carr("NODE_EXP_FLAGS", [r[2] for r in rows], "uint8_t", 24),
            "", "#endif /* NODE_EXPECTED_H */", ""]

    for path, body in ((a.out, dev), (a.expected, host)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(body))

    print("%s: %d values, eps=%d, k=%d" % (a.out, n, eps, k))
    print("%s: %d hours, %d exceptions, %d boundaries (all applied)"
          % (a.expected, n, n_tx, n_boundary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
