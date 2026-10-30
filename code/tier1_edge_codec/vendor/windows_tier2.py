"""
windows_tier2.py  --  data layer. Imports nothing from training/runtime/federation.

Builds (y_past, x_past, x_fut, s, y_target) on demand. Nothing is materialized:
at L=336, C=21 the full sample set is ~8.1M windows x 7,413 floats ~ 240 GB.
The same information lives in a 34 MB array, so windows are gathered per batch.

COVARIATE LAYOUT  (C = 21, order is part of the feature spec hash)
  0-11  calendar          deterministic, on-device computable
  12-14 seasonal anchors  y[j-24], y[j-168], y[j-336], normalized
  15-20 rolling stats     from the look-back, constant across the horizon

REVIEW FIXES APPLIED HERE
=========================
C4  Target validity is now "the FIRST target step is not imputed", and the
    per-step mask is live. Previously `_build_index` rejected any window with
    ANY imputed target step, which made `valid` identically True -- the mask
    was dead code and the loss normalisation by `v.sum()` was a plain mean.
    Worse, it was a FILTER ON THE TEST SET, non-random and clustered on
    outages and DST transitions, i.e. the hardest periods. Measured on this
    data the any-step rule rejects 57,734 window-client pairs against 35,493
    for the first-step rule (1.63x, not the 24x the review estimated -- most
    imputation here is long contiguous runs, where the two rules differ only
    at the run edges. The 24x factor does apply to ISOLATED imputed hours, and
    the three DST spring rows are exactly that: 3 x 24 x 321 = 23,112 pairs).

C5  Metrics and baselines read `series_true` (never interpolated, NaN where
    the value did not exist). Only the model input reads `series_filled`.

M4/M6  Normalisation statistics come from `normstats_tier2`, and
    sigma_floor / eps / norm_window come from the manifest. No constructor
    default anywhere can desynchronise them.

V5  `max_missing_frac` is evaluated over the 720-hour INDEX window, not the
    model's look-back, so a window can be rejected because hours 337-720 back
    are >20% imputed even though an L=336 model never sees them. That is
    deliberate: it is what makes the origin set identical across every L, which
    is the whole point of M7. It is conservative and it filters on data the
    model does not use; the count is reported per split in the index report so
    it is never invisible.

M7  The split index is built at a CANONICAL look-back (`index_lookback`,
    default 720) independent of the model's input length, so every model --
    neural or Gram -- sees identical rows. Previously the Gram indexed at 720
    and the neural models at 336, giving different origin pools and violating
    the "every model sees identical rows" premise the comparison rests on.

M11 The training-only outlier filter reports the fraction it removes.
    Measured: 0.09% at H=24 versus 0.01% at H=1, a 7.4x amplification. Real,
    but two orders of magnitude smaller than "a substantial fraction".

M12 The MASE denominator is computed from `series_true` with both endpoints
    required non-imputed.

S1  Persistence baseline (y-hat[t+h] = y[t] for all h). At H=24 this is NOT
    Seasonal-Naive-24 and is the standard short-horizon reference.

S2  Climatology upgraded to hour-of-day x day-of-week (168 bins) with a
    holiday override, the standard reference. The previous 24x2 version was a
    strawman.
"""

import json

import numpy as np
import torch

from normstats_tier2 import norm_stats_np, z_of

INDEX_LOOKBACK = 720          # canonical, shared by every model (M7)


class WindowBuilder:
    def __init__(self, data_dir="tier2_data", lookback=336, horizon_len=24,
                 max_missing_frac=0.2, train_outlier_sigma=5.0, purge=None,
                 horizon_lag=1, index_lookback=INDEX_LOOKBACK):
        """H=24 DIRECT MULTI-STEP: targets are y[t+1 .. t+24].

        `lookback` is how much history the MODEL consumes. `index_lookback` is
        what defines the valid-origin set and is deliberately decoupled from
        it (M7) so that changing L never changes which rows are scored.
        """
        self.dir = data_dir
        with open(f"{data_dir}/manifest.json") as f:
            self.manifest = json.load(f)
        m = self.manifest
        self.sigma_floor = m["revin_sigma_floor"]      # M6: from the manifest
        self.eps = m["revin_eps"]
        self.norm_window = m["norm_window"]

        self.series = np.load(f"{data_dir}/series_log1p.npy")   # model input
        self.true = np.load(f"{data_dir}/series_true.npy")      # metrics only
        self.filled = np.load(f"{data_dir}/series_filled.npy")
        self.calendar = np.load(f"{data_dir}/calendar.npy")
        self.static = np.load(f"{data_dir}/static.npy")
        self.missing = np.load(f"{data_dir}/missing.npy")

        self.T, self.M = self.series.shape
        self.L, self.lag, self.H = lookback, horizon_lag, horizon_len
        self.index_L = max(index_lookback, lookback)
        self.n_cal = self.calendar.shape[1]
        self.C = self.n_cal + 3          # M17: rolling stats are window-level
        self.W = 6
        self.S = self.static.shape[1]
        self.i_weekend = m["calendar_features"].index("is_weekend")   # m11
        self.i_holiday = m["calendar_features"].index("is_holiday")

        # m12: the hour-of-day convention is asserted, not assumed
        assert abs(self.calendar[0, 0]) < 1e-6 and abs(self.calendar[0, 1] - 1) < 1e-6, \
            "series index 0 must be hour 0; calendar hour_sin/cos disagree"

        self.t_min = self.index_L - 1 + 336
        self.t_max = self.T - self.lag - self.H
        self.purge = (self.index_L + self.H) if purge is None else purge

        tr_end = m["train_end"]
        self.splits, self.index_report = self._build_index(
            max_missing_frac, train_outlier_sigma)

        from metrics_tier2 import mase_scale
        # M12: from series_true, both endpoints required present
        self.mase_scale, self.n_mase_degenerate, self.mase_n = mase_scale(
            self.true, self.missing, tr_end, m=24)
        self.mase_scale_168, _, _ = mase_scale(self.true, self.missing, tr_end, m=168)
        self.climatology = self._climatology(tr_end)                 # S2
        self.client_mean = np.nanmean(self.true[:tr_end], axis=0)

    # ------------------------------------------------------------- baselines
    def _climatology(self, tr_end):
        """S2: hour-of-day x day-of-week (168 bins) + holiday override.
        Returns (168, M) and (M,) holiday means."""
        tr = self.true[:tr_end]
        how = np.arange(tr_end) % 168
        hol = self.calendar[:tr_end, self.i_holiday] > 0.5
        out = np.zeros((168, self.M))
        with np.errstate(invalid="ignore"):
            for w in range(168):
                sel = (how == w) & (~hol)
                out[w] = np.nanmean(tr[sel], axis=0) if sel.any() else np.nanmean(tr, 0)
            hol_mean = (np.nanmean(tr[hol], axis=0) if hol.any()
                        else np.nanmean(tr, axis=0))
        return np.nan_to_num(out), np.nan_to_num(hol_mean)

    # ------------------------------------------------------------- indexing
    def _build_index(self, max_missing_frac, train_outlier_sigma):
        tr_end, va_end = self.manifest["train_end"], self.manifest["valid_end"]
        t_all = np.arange(self.t_min, self.t_max + 1)

        cs = np.concatenate([np.zeros((1, self.M)),
                             np.cumsum(self.missing, axis=0)], axis=0)   # m9: once
        win_missing = (cs[t_all + 1] - cs[t_all + 1 - self.index_L]) / self.index_L

        first = t_all + self.lag
        # C4: FIRST target step only. The remaining steps are handled by the
        # live per-step mask returned from build().
        tgt_missing = self.missing[first[:, None], np.arange(self.M)[None, :]]
        tgt = first[:, None]

        mu = np.nanmean(self.true[:tr_end], 0)
        sd = np.maximum(np.nanstd(self.true[:tr_end], 0), 1e-6)
        peak = np.nanmax(np.stack([self.filled[first + h] for h in range(self.H)]), 0)
        train_outlier = np.abs(peak - mu) > train_outlier_sigma * sd

        base_ok = (~tgt_missing) & (win_missing <= max_missing_frac)

        splits, rep = {}, {}
        for name, lo, hi in (("train", 0, tr_end), ("valid", tr_end, va_end),
                             ("test", va_end, self.T)):
            hi_eff = hi - self.purge if name != "test" else hi
            in_span = np.broadcast_to((tgt >= lo) & (tgt < hi_eff),
                                      tgt_missing.shape)
            cand = int(in_span.sum())
            drop_tgt = int((in_span & tgt_missing).sum())
            drop_win = int((in_span & ~tgt_missing &
                            (win_missing > max_missing_frac)).sum())
            ok = base_ok
            drop_out = 0
            if name == "train":                      # M11: train only, measured
                drop_out = int((in_span & base_ok & train_outlier).sum())
                ok = base_ok & (~train_outlier)
            sel = ok & in_span
            ti, ci = np.nonzero(sel)
            splits[name] = np.stack([t_all[ti], ci], axis=1).astype(np.int64)
            rep[name] = {"candidates": cand, "dropped_target_imputed": drop_tgt,
                         "dropped_window_missing": drop_win,
                         "dropped_train_outlier": drop_out, "kept": int(sel.sum())}
        return splits, rep

    def counts(self):
        return {k: len(v) for k, v in self.splits.items()}

    def print_index_report(self):
        print(f"{'split':6s} {'candidates':>11s} {'tgt imputed':>12s} "
              f"{'win missing':>12s} {'outlier(tr)':>12s} {'kept':>11s}")
        for k, r in self.index_report.items():
            print(f"{k:6s} {r['candidates']:11,d} {r['dropped_target_imputed']:12,d} "
                  f"{r['dropped_window_missing']:12,d} "
                  f"{r['dropped_train_outlier']:12,d} {r['kept']:11,d}")

    # ------------------------------------------------------------- batching
    def build(self, pairs, device="cpu"):
        """pairs: (B, 2) of (t, client). Targets are y[t+lag .. t+lag+H-1]."""
        t, c = pairs[:, 0], pairs[:, 1]
        B = len(t)
        past = t[:, None] + np.arange(-self.L + 1, 1)[None, :]
        fut = t[:, None] + self.lag + np.arange(self.H)[None, :]
        cc = c[:, None]

        y_past = self.series[past, cc]
        mu, sigma = norm_stats_np(y_past, self.sigma_floor)          # M4

        def anchors(idx):
            return np.stack([z_of(self.series[idx - 24, cc], mu, sigma, self.eps),
                             z_of(self.series[idx - 168, cc], mu, sigma, self.eps),
                             z_of(self.series[idx - 336, cc], mu, sigma, self.eps)],
                            axis=-1)

        y_n = z_of(y_past, mu, sigma, self.eps)
        roll = np.stack([
            y_n[:, -24:].mean(1), y_n[:, -24:].std(1), y_n[:, -168:].mean(1),
            y_n[:, -24:].min(1), y_n[:, -24:].max(1),
            y_n[:, -24:].mean(1) - y_n[:, -168:].mean(1),
        ], axis=1).astype(np.float32)

        # M17. The 6 rolling statistics are constant along the time axis, so
        # broadcasting them across L past steps fed 336 identical copies
        # through the feature projection -- the layer measured as the compute
        # bottleneck (it runs on B x (L+H) rows). 6 of 21 channels, i.e. 29% of
        # its input, was redundant. They now travel as a window-level vector
        # concatenated once in the encoder, alongside the static profile.
        # C drops 21 -> 15. This changes the feature spec, hence tier2-v3.
        x_past = np.concatenate([self.calendar[past], anchors(past)], axis=-1)
        x_fut = np.concatenate([self.calendar[fut], anchors(fut)], axis=-1)

        # C4: the mask is LIVE now -- steps 2..24 may legitimately be imputed
        y_true = self.true[fut, cc]
        valid = ~self.missing[fut, cc]

        how = fut % 168
        hol = self.calendar[fut, self.i_holiday] > 0.5
        clim = np.where(hol, self.climatology[1][c][:, None],
                        self.climatology[0][how, cc])

        tt = lambda a, d=torch.float32: torch.as_tensor(
            np.ascontiguousarray(a), dtype=d, device=device)
        return {
            "y_past": tt(y_past), "x_past": tt(x_past), "x_fut": tt(x_fut),
            "s": tt(self.static[c]), "w": tt(roll), "y_true_kwh": tt(np.nan_to_num(y_true)),
            "valid": tt(valid, torch.bool),
            # baselines, all from series_true; NaN where the reference itself
            # is imputed, so the caller can exclude those entries pairwise
            "sn24_kwh": tt(self.true[fut - 24, cc]),
            "sn168_kwh": tt(self.true[fut - 168, cc]),
            "persist_kwh": tt(np.repeat(self.true[t, c][:, None], self.H, 1)),  # S1
            "clim_kwh": tt(clim),
            "mean_kwh": tt(np.broadcast_to(self.client_mean[c][:, None], (B, self.H))),
        }

    def iter_split(self, split, batch_size=256, shuffle=False, seed=0,
                   max_samples=None, device="cpu"):
        pairs = self.splits[split]
        idx = np.arange(len(pairs))
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        if max_samples is not None and len(idx) > max_samples:
            idx = idx[:max_samples]
        for i in range(0, len(idx), batch_size):
            yield self.build(pairs[idx[i:i + batch_size]], device=device)


if __name__ == "__main__":
    wb = WindowBuilder()
    print("samples per split:", wb.counts())
    print(f"H={wb.H} (targets y[t+{wb.lag} .. t+{wb.lag+wb.H-1}])  "
          f"model L={wb.L}  index L={wb.index_L}  C={wb.C} (+{wb.W} window-level)")
    print(f"purge gap: {wb.purge} h from the tail of train and valid\n")
    wb.print_index_report()
    print(f"\nMASE-degenerate clients: {wb.n_mase_degenerate}; "
          f"min valid differences per client: {int(wb.mase_n.min()):,d}")

    for a, b in (("train", "valid"), ("valid", "test")):
        li = wb.splits[a][:, 0].max()
        fi = wb.splits[b][:, 0].min() - wb.L + 1
        print(f"  last {a} input hour {li} < first {b} input hour {fi}: "
              f"{'OK' if li < fi else 'OVERLAP'}")

    b = wb.build(wb.splits["test"][:512])
    print("\nbatch keys:", {k: tuple(v.shape) for k, v in b.items()})
    big = wb.build(wb.splits["test"][::400])
    frac = float(big["valid"].float().mean())
    print(f"live mask over {big['valid'].shape[0]:,d} test windows: "
          f"{frac:.4%} of target steps valid, "
          f"{int((~big['valid']).sum()):,d} steps masked out "
          f"(C4 unfixed would give exactly 100%)")
    print(f"NaN baseline references excluded pairwise: "
          f"sn24 {float(torch.isnan(big['sn24_kwh']).float().mean()):.4%}, "
          f"sn168 {float(torch.isnan(big['sn168_kwh']).float().mean()):.4%}")
    yp = b["y_past"].numpy()
    mu, sg = norm_stats_np(yp, wb.sigma_floor)
    for h in (0, 12, 23):
        rec = np.expm1(b["x_fut"][:, h, 12].numpy() * (sg[:, 0] + wb.eps) + mu[:, 0])
        tru = wb.filled[wb.splits["test"][:512, 0] + wb.lag + h - 24,
                        wb.splits["test"][:512, 1]]
        print(f"  anchor lag24 @h={h+1:2d}: max rel err "
              f"{np.abs(rec - tru).max() / max(tru.max(), 1.0):.2e}")
    print("\nclients with >=1 window: " + ", ".join(
        f"{k}={len(np.unique(v[:,1]))}" for k, v in wb.splits.items()))
