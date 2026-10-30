"""
train_dlinear.py  --  FALLBACK ONLY. Not part of the reported pipeline.

THE PAPER'S NUMBERS MUST COME FROM THE OWNER'S FROZEN TIER-2 CHECKPOINT. That
model being fixed is the premise of the entire project: Tier 1 measures the
degradation of a GIVEN forecaster under input perturbation, so a Tier-1 result
computed against a different forecaster is a different experiment.

This file exists because the Tier-1 work was done without access to that
checkpoint, and a Tier-1 result against random weights measures nothing. Every
checkpoint it produces is stamped `"trainer": "tier1_fallback"`, and
eval/run_eval.py prints a banner on every run that loads one. Delete this file
from the reported pipeline, or cite it as a fallback the paper does not use.

train_dlinear.py  --  produce a stand-in for the FROZEN Tier-2 checkpoint.

Tier 1 does not train anything and does not own this model; the blueprint says
the checkpoint arrives from Tier 2. But a Tier-1 result computed against a
random-weight checkpoint measures nothing, so this trains one to the format
blueprint 2.2 specifies, using the owner's own model, WindowBuilder, RevIN and
inversion. Nothing here is Tier-1 code and nothing here is novel.

LOSS. L1 in LOG1P SPACE, i.e. on `denormalize(y_hat) - log1p(y_true)`.

Not kWh: that weights the fit by meter size and produces a model excellent on
the four largest clients and useless on the rest -- the failure mode MASE exists
to expose.

And NOT the normalised z space either, which is the mistake this file made
first. Windows dominated by exact zeros -- the scheduled-shutdown meters 105,
106, 107, 299, which the preprocessor documents -- have log1p values near 0 and
a RevIN sigma pinned at the 0.05 floor. A target of log1p(500) = 6.2 in such a
window is z = 124. Trained on that, the model learns to emit very large z, and
when the same magnitude is denormalised in a window whose sigma is 3 it produces
expm1(370). Measured: forecasts up to 1.8e14 kWh and a pooled MASE of 2.7e7,
concentrated entirely on those four clients.

Log1p space fixes it at the root. The gradient with respect to z is scaled by
sigma, so a sigma-floored window can no longer drag the model toward extreme z;
it simply accepts the error there. The loss is also the quantity the inversion
actually cares about, and log1p already compresses the scale spread.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor"))
sys.path.insert(0, REPO)

from models_tier2 import TiDEConfig, build          # noqa: E402
from windows_tier2 import WindowBuilder             # noqa: E402


def log_space_error(model, batch, y_hat_n, stats):
    """denormalize(y_hat) - log1p(y_true), the quantity `to_kwh` exponentiates.

    Bounded by construction, and it uses the SAME mu/sigma the inversion will
    use, so the training objective and the reported metric agree about what the
    model is predicting.
    """
    target = torch.log1p(batch["y_true_kwh"].clamp_min(0.0))
    return model.revin.denormalize(y_hat_n, stats) - target


def evaluate(model, wb, sel, batch_size, device, max_batches=None):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(sel), batch_size):
            if max_batches and i // batch_size >= max_batches:
                break
            b = wb.build(sel[i:i + batch_size], device=device)
            yh, stats = model(b["y_past"], b["x_past"], b["x_fut"], b["s"],
                              b["w"])
            r = log_space_error(model, b, yh, stats)
            m = b["valid"]
            tot += float((torch.abs(r) * m).sum())
            n += float(m.sum())
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="tier2_data")
    ap.add_argument("--out", default="dlinear.pt")
    ap.add_argument("--lookback", type=int, default=336)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, unchanged) or cuda. Training "
                         "is the one genuinely GPU-shaped workload "
                         "here; see tools/remote/README.md.")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    wb = WindowBuilder(a.data_dir, lookback=a.lookback, horizon_len=a.horizon)
    cfg = TiDEConfig(lookback_len=a.lookback, horizon_len=a.horizon,
                     n_dynamic=wb.C, n_window=wb.W, n_static=wb.S,
                     revin_sigma_floor=wb.manifest["revin_sigma_floor"],
                     revin_eps=wb.manifest["revin_eps"])
    model = build("dlinear", cfg)
    model.to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.epochs * a.steps_per_epoch)

    tr, va = wb.splits["train"], wb.splits["valid"]
    print(f"train {len(tr):,d} windows, valid {len(va):,d}, "
          f"{sum(p.numel() for p in model.parameters()):,d} params", flush=True)

    best, best_state, history = float("inf"), None, []
    for ep in range(a.epochs):
        t0 = time.time()
        run, seen = 0.0, 0
        for step in range(a.steps_per_epoch):
            idx = rng.choice(len(tr), a.batch_size, replace=False)
            b = wb.build(tr[np.sort(idx)], device=a.device)
            yh, stats = model(b["y_past"], b["x_past"], b["x_fut"], b["s"],
                              b["w"])
            r = log_space_error(model, b, yh, stats)
            m = b["valid"].float()
            loss = (torch.abs(r) * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss)
            seen += 1
        vl = evaluate(model, wb, va, a.batch_size, a.device, a.val_batches)
        history.append({"epoch": ep, "train": run / seen, "valid": vl})
        print(f"epoch {ep}: train {run / seen:.4f}  valid {vl:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    # A blown-up forecast is invisible in a log-space loss but fatal in the kWh
    # metrics, so it is checked explicitly before the checkpoint is written.
    model.eval()
    worst = 0.0
    with torch.no_grad():
        for i in range(0, min(len(va), 200 * a.batch_size), a.batch_size):
            b = wb.build(va[i:i + a.batch_size], device=a.device)
            yh, st = model(b["y_past"], b["x_past"], b["x_fut"], b["s"], b["w"])
            worst = max(worst, float(model.to_kwh(yh, st).max()))
    print(f"worst validation forecast {worst:,.0f} kWh "
          f"(store max {float(np.nanmax(wb.true)):,.0f})")
    assert worst < 50 * float(np.nanmax(wb.true)), (
        f"forecasts reach {worst:.3g} kWh against a store maximum of "
        f"{float(np.nanmax(wb.true)):.3g}. Something is exploding through "
        f"expm1; do not ship this checkpoint.")
    # Convergence test in the blueprint-2.2 shape: the last two epochs must not
    # have improved validation by more than 1%, or the comparison would be
    # measuring the training budget rather than the model.
    conv = (len(history) >= 3
            and abs(history[-1]["valid"] - history[-2]["valid"])
            / max(history[-2]["valid"], 1e-9) < 0.01)
    ck = {"model": "dlinear", "cfg": cfg.__dict__.copy(),
          "state_dict": {k: v.detach().cpu()
                         for k, v in model.state_dict().items()},
          "feature_spec_hash": wb.manifest["feature_spec_hash"],
          "converged": bool(conv),
          "convergence_test": {"kind": "relative validation change over the "
                                       "last two epochs < 1%",
                               "history": history},
          "epochs_run": a.epochs, "val_full": None, "SYNTHETIC": False,
          "trainer": "tier1_fallback",
          "provenance": ("Trained by tier1/tier2/train_dlinear.py because no "
                         "Tier-2 checkpoint was available. NOT the frozen "
                         "model the paper reports against.")}
    torch.save(ck, a.out)
    print(f"wrote {a.out}  best valid {best:.4f}  converged={conv}")


if __name__ == "__main__":
    main()
