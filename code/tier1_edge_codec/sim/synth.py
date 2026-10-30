"""
synth.py  --  the synthetic store. Blueprint 7.

This is what makes the project buildable with no data. It produces a store that
is SCHEMA-IDENTICAL to the real one (blueprint 2.1) and statistically plausible
enough that codec behaviour on it means something, plus a checkpoint in the
blueprint 2.2 format holding a randomly-initialised DLinear.

The forecasts from that checkpoint are meaningless. That is fine and it is the
point: its job is to prove the harness runs end to end and that patch_y_past
actually reaches the model, not to produce numbers anyone would report.

SCHEMA FIDELITY IS BORROWED, NOT REIMPLEMENTED
==============================================
`calendar.npy` and `static.npy` are built by importing the real functions out
of the owner's 10_tier2_preprocess.py rather than by writing a second version
here. WindowBuilder asserts that calendar column 0/1 are hour_sin/hour_cos of
hour 0, reads `is_weekend` and `is_holiday` by name, and the manifest's
feature_spec_hash has to match what the checkpoint carries -- three separate
ways for a hand-rolled copy to drift. Importing the module does not run it (its
work is behind a __main__ guard), so blueprint 2.3's "reference only, do not
run" is respected.

PATHOLOGIES
===========
Every one that the real preprocessor documents is injected here, because the
codec has to survive them and because a store without them would let a broken
implementation pass:

  late onboarding      exact zeros before onboarding[c]
  scheduled shutdowns  recurring zero runs at 24 / 168 / 336 h periods; these
                       are NOT missing data and must not be marked as such
  outage               one long zero run, marked in missing.npy
  dead meter           zero from some hour to the end (the real data has one)
  DST rows             3 spring rows missing for all clients, 3 fall rows
                       doubled then halved, matching the real repair
  level shifts         tests NLMS drift tracking
  near-constant client tests the sigma floor and the degenerate MASE branch
"""

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_preprocess_module():
    """Import the owner's preprocessor by path. Its filename starts with a
    digit, so it is not importable by name; this is the whole reason for the
    importlib dance."""
    path = os.path.join(REPO, "vendor", "10_tier2_preprocess.py")
    assert os.path.exists(path), (
        f"{path} is missing. The synthetic store borrows the real calendar and "
        f"static-profile builders so the schema cannot drift; drop the owner's "
        f"10_tier2_preprocess.py into vendor/ and re-run.")
    spec = importlib.util.spec_from_file_location("_tier2_preprocess", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tier2_preprocess"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ profiles
_DAILY = np.array([.45, .40, .38, .37, .38, .45, .62, .85, 1.00, 1.05, 1.05,
                   1.02, .98, 1.00, 1.02, 1.00, .95, .92, .95, 1.00, .95, .80,
                   .62, .52])


def _client_series(T, rng, stamps):
    """One meter, before pathologies. Blueprint 7's generator, spelled out."""
    t = np.arange(T)
    hod = np.array([s.hour for s in stamps])
    dow = np.array([s.weekday() for s in stamps])

    base = float(np.exp(rng.uniform(np.log(1.0), np.log(5000.0))))
    amp = rng.uniform(0.2, 0.8)
    phase = rng.integers(0, 24)
    daily = 1.0 + amp * (_DAILY[(hod + phase) % 24] - _DAILY.mean())
    weekly = np.where(dow >= 5, rng.uniform(0.4, 1.0), 1.0)
    annual = 1.0 + 0.15 * np.sin(2 * np.pi * t / 8760.0 + rng.uniform(0, 2 * np.pi))
    noise = np.exp(rng.normal(0.0, rng.uniform(0.05, 0.25), T))
    return base * daily * weekly * annual * noise


def make_synthetic_store(out_dir, T=8760, M=24, seed=0, verbose=True):
    """Write a complete, schema-valid store to `out_dir`."""
    pp = _load_preprocess_module()
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    calendar, stamps, dst_slots = pp.build_calendar(T, pp.START)
    raw = np.stack([_client_series(T, rng, stamps) for _ in range(M)], axis=1)

    missing = np.zeros((T, M), bool)
    onboarding = np.zeros(M, np.int64)
    notes = {}

    # --- late onboarding: exact zeros, NOT marked missing (they are real
    #     zeros; the real preprocessor detects them separately).
    for c in range(M):
        if rng.random() < 0.25 and c > 0:
            start = int(rng.integers(24 * 7, max(24 * 8, T // 3)))
            raw[:start, c] = 0.0
            onboarding[c] = start
            notes.setdefault("onboarding", []).append(c)

    # --- scheduled shutdowns: recurring zero runs at a fixed period. These are
    #     NOT missing data. A codec that treats them as anomalies will blow its
    #     exception budget every cycle, which is exactly what we want to see.
    for c in range(M):
        if rng.random() < 0.30:
            period = int(rng.choice([24, 168, 336]))
            run = int(rng.integers(4, 12))
            off = int(rng.integers(0, period))
            idx = np.arange(onboarding[c] + off, T)
            sel = idx[(idx - off) % period < run]
            raw[sel, c] = 0.0
            notes.setdefault("shutdown", []).append((c, period, run))

    # --- one long outage per a few clients, MARKED missing
    for c in range(M):
        if rng.random() < 0.20:
            lo = int(rng.integers(onboarding[c] + 24, max(onboarding[c] + 25, T - 400)))
            hi = min(T, lo + int(rng.integers(48, 336)))
            raw[lo:hi, c] = 0.0
            missing[lo:hi, c] = True
            notes.setdefault("outage", []).append((c, lo, hi))

    # --- a dead meter: zero from some hour to the end (real data has one)
    dead = M - 1
    dead_from = int(T * 0.8)
    raw[dead_from:, dead] = 0.0
    notes["dead_meter"] = (dead, dead_from)

    # --- level shift / regime change, to exercise NLMS drift tracking
    for c in range(M):
        if rng.random() < 0.25:
            at = int(rng.integers(max(onboarding[c] + 48, T // 4), T - 48))
            raw[at:, c] *= float(rng.uniform(1.4, 2.2))
            notes.setdefault("level_shift", []).append((c, at))

    # --- a near-constant client: sigma floor + degenerate MASE denominator
    flat = 1 if M > 1 else 0
    # EXACTLY constant, so mean |y_t - y_{t-24}| is 0 and metrics_tier2's
    # mase_scale returns NaN for it. That is the degenerate branch the eps
    # calibration in sim/run_codec.py has to have a rule for, and a
    # near-constant client would quietly avoid it.
    raw[:, flat] = 40.0
    onboarding[flat] = 0          # no commissioning step, or it would not be
    missing[:, flat] = False      # constant and the scale would not be zero
    notes["near_constant"] = flat

    # --- DST, exactly as the real repair leaves it
    raw, dst_missing = pp.repair_dst(raw, stamps, dst_slots)
    for i in dst_missing:
        missing[i, :] = True
    notes["dst_spring_rows"] = list(map(int, dst_missing))

    raw = np.clip(raw, 0.0, None)
    train_end = int(T * 0.7) // 24 * 24
    valid_end = int(T * 0.8) // 24 * 24

    series_true = raw.copy()
    series_true[missing] = np.nan
    raw_filled = pp.interpolate(raw, missing)
    series_log1p = np.log1p(np.clip(raw_filled, 0, None)).astype(np.float32)
    static = pp.static_profile(raw_filled[:train_end], stamps[:train_end])

    np.save(f"{out_dir}/series_log1p.npy", series_log1p)
    np.save(f"{out_dir}/series_true.npy", series_true.astype(np.float64))
    np.save(f"{out_dir}/series_filled.npy", raw_filled.astype(np.float32))
    np.save(f"{out_dir}/missing.npy", missing)
    np.save(f"{out_dir}/calendar.npy", calendar)
    np.save(f"{out_dir}/static.npy", static)
    np.save(f"{out_dir}/onboarding.npy", onboarding)

    spec = {
        "feature_spec_version": pp.FEATURE_SPEC_VERSION,
        "calendar_features": pp.CALENDAR_FEATURES,
        "dynamic_features": pp.DYNAMIC_FEATURES,
        "window_features": pp.WINDOW_FEATURES,
        "static_features": pp.STATIC_FEATURES,
        "C": len(pp.DYNAMIC_FEATURES), "W": len(pp.WINDOW_FEATURES),
        "S": len(pp.STATIC_FEATURES),
        "target_transform": ("log1p -> RevIN(lookback) ; invert RevIN -> expm1 "
                             "-> clip0"),
        "revin_sigma_floor": 0.05, "revin_eps": 1e-5, "norm_window": 336,
        "start": pp.START.isoformat(), "freq": "1h",
        "tz": "Europe/Lisbon (wall clock)",
        "T": int(T), "M": int(M),
        "train_end": train_end, "valid_end": valid_end,
        "dst_spring_rows": [int(i) for i in dst_missing],
        "SYNTHETIC": True,
        "synthetic_seed": int(seed),
        "synthetic_notes": {k: str(v) for k, v in notes.items()},
        "note_kwh": "SYNTHETIC STORE -- no relationship to electricity.txt.",
    }
    # Identical hash recipe to the real preprocessor, over identical keys, so a
    # checkpoint trained on the real store is correctly REJECTED against this
    # one only if the feature spec really differs.
    spec["feature_spec_hash"] = hashlib.sha256(
        json.dumps({k: spec[k] for k in
                    ("feature_spec_version", "dynamic_features",
                     "window_features", "static_features", "target_transform",
                     "revin_sigma_floor", "revin_eps", "norm_window")},
                   sort_keys=True).encode()).hexdigest()
    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(spec, f, indent=2)

    if verbose:
        print(f"synthetic store -> {out_dir}  T={T} M={M} "
              f"train[0,{train_end}) valid[{train_end},{valid_end}) "
              f"test[{valid_end},{T})")
        print(f"  feature spec hash {spec['feature_spec_hash'][:16]}")
        print(f"  pathologies: onboarding={len(notes.get('onboarding', []))} "
              f"shutdown={len(notes.get('shutdown', []))} "
              f"outage={len(notes.get('outage', []))} "
              f"level_shift={len(notes.get('level_shift', []))} "
              f"dead={notes['dead_meter'][0]} near_constant={flat} "
              f"dst_rows={len(dst_missing)}")
        print(f"  imputed cells {missing.mean():.3%}   "
              f"client load range {np.nanmin(np.nanmean(series_true, 0)):.1f} "
              f"- {np.nanmax(np.nanmean(series_true, 0)):.1f} kWh/h")
    return spec


def make_synthetic_checkpoint(out_path, manifest, seed=0, lookback=336,
                              horizon=24, verbose=True):
    """A blueprint-2.2 checkpoint holding a randomly-initialised DLinear."""
    sys.path.insert(0, os.path.join(REPO, "vendor"))
    import torch
    from models_tier2 import TiDEConfig, build

    torch.manual_seed(seed)
    cfg = TiDEConfig(lookback_len=lookback, horizon_len=horizon,
                     n_dynamic=manifest["C"], n_window=manifest["W"],
                     n_static=manifest["S"],
                     revin_sigma_floor=manifest["revin_sigma_floor"],
                     revin_eps=manifest["revin_eps"])
    model = build("dlinear", cfg)
    ck = {
        "model": "dlinear",
        "cfg": cfg.__dict__.copy(),
        "state_dict": model.state_dict(),
        "feature_spec_hash": manifest["feature_spec_hash"],
        "converged": True,
        "convergence_test": {"kind": "SYNTHETIC -- untrained random weights"},
        "epochs_run": 0,
        "val_full": None,
        "SYNTHETIC": True,
        # Not supplied by Tier 2, so it trips the same provenance banner the
        # fallback trainer does. A random-weight checkpoint must never be
        # mistaken for the frozen forecaster the experiment is premised on.
        "trainer": "tier1_fallback",
        "provenance": ("Randomly initialised by sim/synth.py for wiring tests. "
                       "NOT the frozen model the paper reports against."),
    }
    torch.save(ck, out_path)
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"synthetic checkpoint -> {out_path}  dlinear L={lookback} "
              f"H={horizon}  {n:,d} params (RANDOM -- forecasts are meaningless)")
    return ck


def make_tiny_fixture(out_dir, T=240, M=3, seed=7):
    """A hand-checkable deterministic fixture for the unit tests: three
    clients, ten days, one late onboarding, one flat meter, no DST inside the
    span. Deliberately small enough that a failure can be read by eye.

    It is NOT a store (240 hours cannot satisfy WindowBuilder's 720-hour index
    window); it is a raw kWh matrix for exercising the codec alone.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = np.zeros((T, M))
    y[:, 0] = 500 * (1.0 + 0.5 * _DAILY[t % 24]) * np.exp(rng.normal(0, .05, T))
    y[:, 1] = 40.0                                   # exactly constant
    y[:, 2] = 3000 * (1.0 + 0.6 * _DAILY[(t + 7) % 24]) * np.exp(rng.normal(0, .12, T))
    onboarding = np.array([0, 0, 48], np.int64)
    y[:48, 2] = 0.0
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/y.npy", y)
    np.save(f"{out_dir}/onboarding.npy", onboarding)
    return y, onboarding


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_data")
    ap.add_argument("--T", type=int, default=8760)
    ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="synth_dlinear.pt")
    a = ap.parse_args()
    spec = make_synthetic_store(a.out, T=a.T, M=a.M, seed=a.seed)
    make_synthetic_checkpoint(a.checkpoint, spec, seed=a.seed)
    make_tiny_fixture(os.path.join(a.out, "tiny"))
    print("tiny fixture -> " + os.path.join(a.out, "tiny"))
