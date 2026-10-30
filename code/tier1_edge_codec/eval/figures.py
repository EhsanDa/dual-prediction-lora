"""
figures.py  --  five figures, and only five. Section 5.

Input is tier1_results.json (and optionally tier1_sensitivity.json). Nothing is
recomputed here and nothing is hand-edited: if a number in a figure is wrong, it
is wrong in the JSON, which is the only place to fix it.

  1  lead-1 MAE by issue hour     THE HEADLINE. batch-daily against the
                                  selected alpha, exactly matched: both predict
                                  the same absolute hour, one at lead 1 and one
                                  at lead h+1 of its morning run.
  2  rate-utility                 dMASE vs packets/day against the CONTROL.
  3  model ablation               the four rows of section 4.1 on a common
                                  packet-rate axis.
  4  rate-distortion              reconstruction MAE / mase_scale vs
                                  packets/day, with the quantisation floor.
  5  airtime at SF12              per-node fair-use fraction vs alpha.

WHAT GOES IN THE AXES, AND WHAT DOES NOT  (revision C-2)
=======================================================
An IEEE figure carries DATA. Every number that is not a data point belongs in
the caption or in the running text, where it is set in the body font, is
searchable, and does not have to survive a 3.45 in column. The in-axes summary
boxes ("delivered: 185.5 -> 121.6 kWh ...", the sensitivity band, the alpha = 0
note) are therefore GONE; they are re-emitted by write_captions() into
captions.txt, ready to paste into the .tex, and the running text already states
the same figures.

Two further clean-ups follow from that:

  * ALPHA IS A SECOND AXIS, NOT EIGHT CALLOUTS. Packet rate is a monotone
    function of alpha, so alpha is drawn as a top axis whose ticks sit at the
    measured packet rates. Eight in-panel "alpha=..." labels chasing a steep
    curve are what made Fig. 2 unreadable.
  * THE RATE AXIS IS LOGARITHMIC. The operating points are geometrically
    spaced (1.4 ... 25 pkt/day), so on a linear axis five of the eight pile up
    in the left quarter and the curve collapses into an L. On a log axis they
    are evenly spread and the rate-utility trade-off is legible.

Legends are INSIDE the axes wherever the decay of the curves leaves a corner
empty, which is most of them; only the six-entry ablation legend still needs to
go below. A legend below the axes is not free -- it is bought with plot height.

No entry is spent on the zero line: the y-axis label already says the ordinate
is measured against the no-codec control.

PUBLICATION SETTINGS
====================
Everything is sized for a SINGLE IEEE COLUMN (3.45 in) at 600 dpi, serif, 8 pt,
set once in the rcParams block below. Per-figure figsize and per-savefig dpi
arguments are NOT used: either one silently overrides the block, which is how
figures drift apart. Each figure is written as both PDF (vector, for the
camera-ready) and PNG (600 dpi, for previews).

NO FIGURE CARRIES ITS OWN TITLE. IEEE convention puts the title in the caption.
--width lets a figure that genuinely cannot fit be re-run at double-column
width (7.16 in) without touching anything else.
"""

import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

# Artifact locations live in one place; see paths.py for the contract.
import os as _os, sys as _sys                                  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
import paths                                                   # noqa: E402

# ---- IEEE two-column publication settings -------------------------------
COL_IN = 3.45                      # single IEEE column width, inches
plt.rcParams.update({
    "figure.figsize":   (COL_IN, COL_IN * 0.72),
    "figure.dpi":       600,
    "savefig.dpi":      600,
    "savefig.format":   "pdf",     # vector; also export png at 600 dpi
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.01,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,

    "font.family":      "serif",
    # Liberation Serif is metric-compatible with Times New Roman and is what
    # most Linux boxes actually have; without it the chain falls through to
    # DejaVu Serif, which is NOT Times-like and will not match the body text.
    "font.serif":       ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                         "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  6.5,

    "axes.linewidth":   0.6,
    "axes.grid":        True,
    "grid.linewidth":   0.35,
    "grid.color":       "0.8",
    "grid.alpha":       0.7,
    "lines.linewidth":  1.0,
    "lines.markersize": 2.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon":   True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.75",
    "legend.fancybox":  False,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
})
# -------------------------------------------------------------------------

import numpy as np                   # noqa: E402

# The fair-use allowance has ONE definition, in eval/metrics_tier1.py. It was
# duplicated here until 2026-09-14 (card P1-05); two copies of a number that
# every airtime percentage in the paper divides by is a silent-divergence risk.
# tests/test_revisions.py::test_the_fair_use_allowance_is_the_same_number_everywhere
# still pins them together and stays as a guard.
from eval.metrics_tier1 import TTN_FAIR_USE_S    # noqa: E402

SYNTH_BANNER = "SYNTHETIC store/checkpoint -- wiring check, not a result"
CONTROL, BATCH_DAILY = "control", "batch-daily"
FAM_STYLE = {"A": ("tab:red", "s--"), "B": ("tab:orange", "^-"),
             "C": ("tab:blue", "o-"), "D": ("tab:green", "d:")}
_SIZES = []                        # (stem, width_in, height_in) as SAVED
_CAPTIONS = []                     # (stem, caption text) for captions.txt


def family_rows(res, fam):
    labs = [l for l, c in res["cost"].items() if c.get("family") == fam]
    return sorted(labs, key=lambda l: float(res["cost"][l]["alpha"]))


def _banner(fig, res):
    if res.get("SYNTHETIC_store") or res.get("SYNTHETIC_checkpoint"):
        fig.text(0.5, 1.0, SYNTH_BANNER, ha="center", va="bottom",
                 color="crimson", fontsize=6, weight="bold")


def _fmt_alpha(a):
    a = float(a)
    return f"{a:g}"


def _alpha_axis(ax, rates, alphas, label=r"$\varepsilon$ multiplier $\alpha$"):
    """Draw alpha as a TOP axis instead of as per-point callouts.

    The rate axis must already be set (limits and scale), because the twin
    copies both: a twiny() created before set_xscale("log") silently keeps a
    linear scale and every tick lands in the wrong place.
    """
    top = ax.twiny()
    top.set_xscale(ax.get_xscale())
    top.set_xlim(ax.get_xlim())
    top.xaxis.set_major_locator(FixedLocator(list(rates)))
    # Match by nearest tick, not by dict lookup on a float: a rate of
    # 9.4000000001 in the JSON would miss an exact key and print the RATE in
    # the alpha row, which is a wrong label rather than a missing one.
    _r, _a = np.asarray(rates, float), list(alphas)
    top.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: _fmt_alpha(_a[int(np.argmin(abs(_r - v)))])))
    top.xaxis.set_minor_locator(FixedLocator([]))
    top.xaxis.set_minor_formatter(NullFormatter())
    top.tick_params(axis="x", labelsize=6, length=2, pad=1.5)
    top.set_xlabel(label, fontsize=7, labelpad=2)
    top.grid(False)
    for s in ("left", "right", "bottom"):
        top.spines[s].set_visible(False)
    return top


def _rate_axis(ax, rates, pad=1.12):
    """Log rate axis with plain (not 10^x) tick labels."""
    ax.set_xscale("log")
    lo, hi = min(rates) / pad, max(rates) * pad
    ax.set_xlim(lo, hi)
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50) if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("packets per node per day")


def _save(fig, out, caption=""):
    """Write BOTH vector and raster from one render.

    `out` may carry any extension or none; the stem is what matters. dpi and
    bbox come from the rcParams block -- passing them here is what used to
    override it and pin every figure at 140 dpi.
    """
    stem = os.path.splitext(out)[0]
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", format=ext)
    # bbox="tight" CROPS, so the file is narrower than figure.figsize. Report
    # the real width: including a 3.2 in file at width=\columnwidth silently
    # scales 8 pt text up to ~8.5 pt, which is exactly the drift the rcParams
    # block exists to prevent. Include at natural size, or set the width to
    # the number printed here.
    pad = plt.rcParams["savefig.pad_inches"]
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    _SIZES.append((os.path.basename(stem), bb.width + 2 * pad,
                   bb.height + 2 * pad))
    if caption:
        _CAPTIONS.append((os.path.basename(stem), " ".join(caption.split())))
    plt.close(fig)


# =========================================================================
# 1  lead-1 MAE by issue hour
# =========================================================================
def fig1_lead1(res, out):
    hh = np.arange(24)
    bd = np.asarray(res["metrics"][BATCH_DAILY]["MAE_lead1_by_issue_hour"])
    ctl = np.asarray(res["metrics"][CONTROL]["MAE_lead1_by_issue_hour"])
    sel = (res.get("selection") or {}).get("row")

    fig, ax = plt.subplots()
    ax.plot(hh, bd, "-.", color="darkorange", label="daily batch, 1 pkt/day")
    if sel and sel in res["metrics"]:
        v = np.asarray(res["metrics"][sel]["MAE_lead1_by_issue_hour"])
        a_sel = _fmt_alpha(res["cost"][sel]["alpha"])
        # The band IS the deliverable: it is the error the codec removes, and
        # it is the only reason the two curves are drawn on one axes.
        ax.fill_between(hh, v, bd, alpha=.12, color="tab:green", lw=0)
        ax.plot(hh, v, "--", color="tab:green",
                label=f"proposed codec, $\\alpha$={a_sel}")
    ax.plot(hh, ctl, "-", color="tab:blue", label="hourly refresh, no codec")

    ax.set_xlabel("issue hour of day (hours since the daily resync)")
    ax.set_ylabel("lead-1 forecast MAE, kWh")
    ax.set_xlim(0, 23)
    ax.set_xticks(np.arange(0, 24, 4))
    ax.set_ylim(top=max(bd.max(), ctl.max()) * 1.20)
    # Every curve here climbs to the right, so the upper LEFT corner is the one
    # that stays empty; the legend costs no plot height there.
    ax.legend(loc="upper left", ncol=1)
    _banner(fig, res)

    pk = res["cost"][sel]["packets_per_day_mean"] if sel else float("nan")
    dv = (np.asarray(res["metrics"][sel]["MAE_lead1_by_issue_hour"]).mean()
          if sel else float("nan"))
    # The peak quoted in the caption is the peak of the band the figure DRAWS,
    # i.e. the delivered gain, not the unconstrained-link gain.
    gain = bd - (np.asarray(res["metrics"][sel]["MAE_lead1_by_issue_hour"])
                 if sel else ctl)
    cap = (f"Next-hour error by issue hour. At issue hour $h$ the daily-batch "
           f"policy serves lead $h{{+}}1$ of its morning invocation, so both "
           f"curves predict the same absolute hour. Averaged over the day the "
           f"error falls from {bd.mean():.1f} kWh under daily batching to "
           f"{dv:.1f} kWh under the proposed codec at {pk:.1f} packets per "
           f"node per day ({1 - dv / bd.mean():.1%}); the unconstrained-link "
           f"bound is {ctl.mean():.1f} kWh ({1 - ctl.mean() / bd.mean():.1%}). "
           f"The shaded band is the error the codec removes. It is zero at "
           f"$h=0$ by construction, since both systems then hold the same "
           f"window, and peaks near $h={int(np.argmax(gain))}$ at "
           f"approximately {gain.max():.0f} kWh.")
    _save(fig, out, cap)


# =========================================================================
# 2  rate-utility
# =========================================================================
def fig2_rate_utility(res, out, sens=None):
    labs = family_rows(res, "C")
    x = [res["cost"][l]["packets_per_day_mean"] for l in labs]
    dm = [res["metrics"][l]["delta_MASE_vs_control"]["mean"] for l in labs]
    p90 = [res["metrics"][l]["delta_MASE_vs_control"]["p90"] for l in labs]
    al = [res["cost"][l]["alpha"] for l in labs]
    o = np.argsort(x)
    x, dm, p90, al = (list(np.asarray(v)[o]) for v in (x, dm, p90, al))

    fig, ax = plt.subplots()
    ax.axhline(0, color="0.35", lw=0.8)          # the control; see ylabel
    ax.plot(x, p90, "^--", color="tab:blue", alpha=.45, label="p90 client")
    ax.plot(x, dm, "o-", color="tab:blue", label="mean client")

    sel = res.get("selection")
    if sel:
        ax.axvline(sel["packets_per_day"], color="tab:green", ls="-.",
                   lw=0.9, label="operating point")
    b = res.get("budget")
    # The budget is AIRTIME, and the selection test is in seconds. On a packet
    # axis it can only appear as an indicative conversion at the realised mean
    # time-on-air, so the legend says "equivalent" rather than implying a
    # packet budget that nothing is actually tested against.
    b_pkt = (b or {}).get("budget_pkt_per_day_indicative",
                          (b or {}).get("budget_pkt_per_day"))
    if b_pkt:
        ax.axvline(b_pkt, color="0.45", ls=":", lw=0.9,
                   label="airtime budget (equiv.)")

    _rate_axis(ax, x)
    ax.set_ylabel(r"$\Delta$MASE vs the no-codec control")
    ax.set_ylim(bottom=min(0, min(dm)) - 0.004, top=max(p90) * 1.10)
    # Both curves decay to the right, so the upper right is empty by the shape
    # of the result itself.
    ax.legend(loc="upper right", ncol=1)
    _alpha_axis(ax, x, al)
    _banner(fig, res)

    band = ""
    if sens and "L_restricted_by_issue_hour" in sens:
        lr = np.asarray(sens["L_restricted_by_issue_hour"])
        band = (f" The first-order sensitivity bound of Section VI, "
                f"$|d\\hat{{y}}|_\\infty \\lesssim L_{{restr}}(h)\\,"
                f"\\varepsilon$ over the perturbed columns only, has "
                f"$L_{{restr}}$ growing from {lr[0]:.2f} at the resynchronisation "
                f"hour to {lr[-1]:.2f} at the end of the day; the measured "
                f"degradation lies within it throughout.")
    # No operating point is a legitimate outcome, not an error: when no alpha
    # satisfies both the airtime budget and the accuracy tolerance the curve IS
    # the result, and run_eval writes selection = null. Every sentence that
    # names the operating point is therefore built conditionally.
    if sel:
        op = (f" and the dash-dot line the selected operating point, "
              f"$\\alpha$={_fmt_alpha(sel['selected_alpha'])} at "
              f"{sel['packets_per_day']:.1f} packets per node per day.")
    else:
        op = (". No deadband satisfied both the airtime budget and the "
              "accuracy tolerance on this run, so no operating point is "
              "marked; the curve is the result.")
    cap = (f"Rate-utility of the transmission policy. Accuracy cost is measured "
           f"against the no-codec control, so the abscissa is the packet rate "
           f"and zero is the control. The upper axis gives the corresponding "
           f"deadband multiplier $\\alpha$; the rate axis is logarithmic "
           f"because the operating points are geometrically spaced. The dotted "
           f"dotted line is the airtime budget of Section V, converted to this "
           f"axis at the realised mean time-on-air"
           + op + band)
    _save(fig, out, cap)


# =========================================================================
# 3  model ablation
# =========================================================================
def fig3_ablation(res, out):
    fig, ax = plt.subplots()
    names = {"A": "persistence + deadband", "B": "global pre-trained",
             "C": "global + NLMS (proposed)", "D": "per-client ridge (ceiling)"}
    allx = []
    ax.axhline(0, color="0.35", lw=0.8)
    for f in "ABCD":
        labs = family_rows(res, f)
        if not labs:
            continue
        x = np.asarray([res["cost"][l]["packets_per_day_mean"] for l in labs])
        y = np.asarray([res["metrics"][l]["delta_MASE_vs_control"]["mean"]
                        for l in labs])
        col, st = FAM_STYLE[f]
        o = np.argsort(x)
        allx += list(x)
        ax.plot(x[o], y[o], st, color=col, label=f"{f}. {names[f]}")

    # ONE reference line, not one per evaluation rate: the grey comb of
    # dMASE_vs_control_at gridlines was drawn behind four curves that already
    # sit on a grid, and it read as chart junk.
    sel = res.get("selection") or {}
    if sel:
        ax.axvline(sel["packets_per_day"], color="0.45", ls="-.", lw=0.9,
                   zorder=0, label="operating point")

    _rate_axis(ax, allx)
    ax.set_ylabel(r"$\Delta$MASE vs the no-codec control")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.06)
    # All four families decay to the right, so five entries still fit in the
    # upper right; below the axes they would cost a fifth of the plot height.
    ax.legend(loc="upper right", ncol=1)
    _banner(fig, res)

    op3 = ((f" The dash-dot line marks the operating point selected in "
            f"Section V, {sel['packets_per_day']:.1f} packets per node per day.")
           if sel.get("packets_per_day") is not None else
           (" No operating point was selected on this run, so none is marked."))
    cap = (f"Model ablation on a common packet-rate axis. Each family is swept "
           f"over the same deadband schedule; the ordinate is the accuracy cost "
           f"against the no-codec control, so lower is better and zero is the "
           f"control." + op3)
    _save(fig, out, cap)


# =========================================================================
# 4  rate-distortion
# =========================================================================
def fig4_rate_distortion(res, out):
    labs = family_rows(res, "C")
    # alpha = 0 reconstructs EXACTLY, so its distortion is 0 and it cannot be
    # drawn on a log axis. It is stated in the caption, which is the honest way
    # to report an exact zero.
    zero = [l for l in labs
            if res["cost"][l]["recon_mae_over_mase_scale"]["mean"] <= 0]
    labs = [l for l in labs if l not in zero]
    x = [res["cost"][l]["packets_per_day_mean"] for l in labs]
    y = [res["cost"][l]["recon_mae_over_mase_scale"]["mean"] for l in labs]
    p90 = [res["cost"][l]["recon_mae_over_mase_scale"]["p90"] for l in labs]
    al = [res["cost"][l]["alpha"] for l in labs]
    o = np.argsort(x)
    x, y, p90, al = (list(np.asarray(v)[o]) for v in (x, y, p90, al))

    fig, ax = plt.subplots()
    ax.plot(x, p90, "s--", color="grey", label="p90 client")
    ax.plot(x, y, "o-", color="tab:blue", label="mean over clients")
    q = res.get("quantisation", {}).get("per_client_over_mase_scale", {})
    if q:
        ax.axhline(q["mean"], color="crimson", ls=":", lw=0.9,
                   label="quantisation floor")
    _rate_axis(ax, x)
    ax.set_ylabel("reconstruction MAE / mase scale")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper right", ncol=1)
    _alpha_axis(ax, x, al)
    _banner(fig, res)

    note = ""
    if zero:
        pk = ", ".join(f"{res['cost'][l]['packets_per_day_mean']:.1f}"
                       for l in zero)
        note = (f" The $\\alpha=0$ row is omitted: reconstruction is then exact, "
                f"the distortion is identically zero and cannot be placed on a "
                f"logarithmic axis ({pk} packets per node per day).")
    floor = ""
    if q:
        m, e = f"{q['mean']:.1e}".split("e")
        floor = (f" The dotted line is the quantisation floor, "
                 f"${m}\\times10^{{{int(e)}}}$.")
    cap = ("Rate-distortion of the codec. Reconstruction error is normalised by "
           "the MASE scale so that it is comparable across clients; the upper "
           "axis gives the deadband multiplier." + floor + note)
    _save(fig, out, cap)


# =========================================================================
# 5  airtime at SF12
# =========================================================================
def fig5_airtime(res, out, sf=12):
    labs = family_rows(res, "C")
    al = [float(res["cost"][l]["alpha"]) for l in labs]
    air = [res["cost"][l]["airtime_per_day_s"].get(
        str(sf), res["cost"][l]["airtime_per_day_s"].get(sf)) for l in labs]
    frac = np.asarray(air) / TTN_FAIR_USE_S
    o = np.argsort(al)
    al, frac = list(np.asarray(al)[o]), frac[o]

    fig, ax = plt.subplots()
    ax.plot(al, frac, "o-", color="tab:blue")
    ax.axhline(1.0, color="crimson", ls="--", lw=0.9, label="fair-use allowance")
    b = res.get("budget")
    if b:
        ax.axhline(1 - b["retry_margin"], color="0.45", ls=":", lw=0.9,
                   label="budget after retry margin")
    sel = res.get("selection")
    if sel:
        f_sel = sel["airtime_sf12_s"] / TTN_FAIR_USE_S
        ax.plot([float(sel["selected_alpha"])], [f_sel], "*", ms=7,
                color="tab:green", zorder=5, label="operating point")
    # The per-point percentages duplicated the ordinate at every marker. The
    # two numbers a reader actually needs -- the allowance and the selected
    # point -- are the two reference lines and the star.
    ax.set_xlabel(r"$\varepsilon$ multiplier $\alpha$")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_ylabel(f"airtime at SF{sf} / fair-use allowance")
    ax.set_ylim(0, max(frac.max(), 1.0) * 1.35)
    ax.legend(loc="upper right", ncol=1)
    _banner(fig, res)

    cap = (f"Per-node airtime at SF{sf} as a fraction of the "
           f"{TTN_FAIR_USE_S:g} s per day fair-use allowance, against the "
           f"deadband multiplier. The dashed line is the allowance itself.")
    if b:
        _bs = b.get("budget_airtime_s")
        cap += (f" The dotted line is the working budget that remains after a "
                f"{b['retry_margin']:.0%} retry margin"
                + (f", {_bs:.0f} s per day." if _bs else "."))
    if sel:
        cap += (f" The star is the selected operating point, $\\alpha$="
                f"{_fmt_alpha(sel['selected_alpha'])}, at "
                f"{sel['airtime_sf12_s'] / TTN_FAIR_USE_S:.0%} of the "
                f"allowance.")
    _save(fig, out, cap)


def _tex(cap):
    """A bare % is a LaTeX COMMENT: it would silently swallow the rest of the
    caption line, and every one of these captions quotes a percentage."""
    return re.sub(r"(?<!\\)%", r"\\%", " ".join(cap.split()))


def write_captions(out):
    """Everything that used to be printed inside an axes, as caption text.

    The figures no longer carry these numbers, so this file is not a
    convenience: it is where they now live, and it is regenerated from the same
    JSON as the figures, which is what keeps caption and plot from drifting.
    """
    with open(out, "w") as f:
        for stem, cap in _CAPTIONS:
            f.write(f"% {stem}\n\\caption{{{_tex(cap)}}}\n\n")


def table_system(res, out):
    rows = []
    for l in [CONTROL, BATCH_DAILY] + family_rows(res, "C"):
        m, c = res["metrics"][l], res["cost"].get(l, {})
        air = c.get("airtime_per_day_s", {})
        rows.append({
            "row": l, "alpha": c.get("alpha", ""),
            "packets_per_day": round(c.get("packets_per_day_mean", float("nan")), 3),
            "bytes_per_day": round(c.get("bytes_per_day_mean", float("nan")), 1),
            "airtime_s_SF12": round(air.get("12", air.get(12, float("nan"))), 4)
            if air else "",
            "pct_TTN": round(air.get("12", air.get(12, float("nan")))
                             / TTN_FAIR_USE_S, 4) if air else "",
            "staleness_mean_h": round(c.get("staleness_mean_h", float("nan")), 3),
            "staleness_p90_h": round(c.get("staleness_p90_h", float("nan")), 3),
            "recon_mae_kwh": round(c.get("recon_mae_kwh", float("nan")), 4),
            "linf_violation": c.get("linf_violation", ""),
            "MAE_kwh": round(m["MAE"], 3), "RMSE_kwh": round(m["RMSE"], 3),
            "sMAPE_pct": round(m["sMAPE%"], 3), "nRMSE": round(m["nRMSE"], 5),
            "MASE": round(m["MASE"], 5), "MASE_p90": round(m["MASE_p90"], 5),
            "dMASE_vs_oracle": round(m["delta_MASE"]["mean"], 6),
            "dMASE_p90_vs_oracle": round(m["delta_MASE"]["p90"], 6),
            "dMASE_vs_control": round(m["delta_MASE_vs_control"]["mean"], 6),
            "refresh_gain_lead1_kwh": round(m["refresh_gain_lead1_mean"], 4),
            "SS": round(m["SS"], 5), "SS_lt_0": round(m["SS<0"], 4),
        })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table_ablation(res, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        rates = sorted(res["ablation"]["A"]["dMASE_vs_control_at"],
                       key=float, reverse=True)
        w.writerow(["row", "description"] + [f"dMASE @ {r} pkt/day"
                                             for r in rates])
        for fam in "ABCD":
            d = res["ablation"][fam]
            w.writerow([fam, d["name"]]
                       + [("" if d["dMASE_vs_control_at"][r] is None
                           else round(d["dMASE_vs_control_at"][r], 5))
                          for r in rates])


def table_freshness(res, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        sel = (res.get("selection") or {}).get("row")
        w.writerow(["issue_hour", "batch_daily_lead1_MAE",
                    "control_lead1_MAE", "selected_alpha_lead1_MAE",
                    "gain_kWh", "approx_slots_in_window"])
        bd = res["metrics"][BATCH_DAILY]["MAE_lead1_by_issue_hour"]
        ct = res["metrics"][CONTROL]["MAE_lead1_by_issue_hour"]
        sv = res["metrics"][sel]["MAE_lead1_by_issue_hour"] if sel else [None] * 24
        for h in range(24):
            w.writerow([h, round(bd[h], 4), round(ct[h], 4),
                        "" if sv[h] is None else round(sv[h], 4),
                        round(bd[h] - ct[h], 4), h + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=paths.result("tier1_results.json"))
    ap.add_argument("--sensitivity",
                    default=paths.result("tier1_sensitivity.json"))
    ap.add_argument("--outdir", default=paths.result("figures"))
    ap.add_argument("--width", type=float, default=COL_IN,
                    help=f"figure width in inches. Default {COL_IN} (one IEEE "
                         f"column); pass 7.16 for a double-column figure.")
    ap.add_argument("--aspect", type=float, default=0.72,
                    help="height / width")
    a = ap.parse_args()
    if (a.width, a.aspect) != (COL_IN, 0.72):
        plt.rcParams["figure.figsize"] = (a.width, a.width * a.aspect)
    os.makedirs(a.outdir, exist_ok=True)
    with open(a.results) as f:
        res = json.load(f)
    sens = None
    if os.path.exists(a.sensitivity):
        with open(a.sensitivity) as f:
            sens = json.load(f)

    # No extension: _save writes .pdf and .png from the same render.
    fig1_lead1(res, f"{a.outdir}/fig1_lead1_by_issue_hour")
    fig2_rate_utility(res, f"{a.outdir}/fig2_rate_utility", sens)
    fig3_ablation(res, f"{a.outdir}/fig3_model_ablation")
    fig4_rate_distortion(res, f"{a.outdir}/fig4_rate_distortion")
    fig5_airtime(res, f"{a.outdir}/fig5_airtime_sf12")
    write_captions(f"{a.outdir}/captions.tex")
    table_system(res, f"{a.outdir}/table_system.csv")
    table_ablation(res, f"{a.outdir}/table_ablation.csv")
    table_freshness(res, f"{a.outdir}/table_freshness.csv")
    print(f"wrote 5 figures (pdf + png at "
          f"{plt.rcParams['savefig.dpi']:g} dpi), captions.tex and 3 tables "
          f"to {a.outdir}/")
    print("saved size after the tight-bbox crop -- include at THIS width, or "
          "at natural size:")
    for stem, w, h in _SIZES:
        print(f"  {stem:<28s} {w:.2f} x {h:.2f} in")
    resolved = font_manager.FontProperties(
        family=plt.rcParams["font.family"]).get_name()
    if resolved not in ("Times New Roman", "Nimbus Roman", "Liberation Serif"):
        print(f"WARNING: serif resolved to {resolved!r}, not a Times-metric "
              f"face. Install fonts-liberation (or msttcorefonts) before the "
              f"camera-ready or the figure text will not match the body.")


if __name__ == "__main__":
    main()
