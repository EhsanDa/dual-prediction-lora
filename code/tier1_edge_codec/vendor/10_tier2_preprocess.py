"""
10_tier2_preprocess.py  --  Tier 2 (NanoPi) data pipeline.

Produces a COMPACT store. It deliberately does not materialize windows.

  Why: with L=336, C=21 and channel-independent sampling there are
  ~25,000 windows x 321 clients = 8.1M samples, each carrying
  336 + 337*21 = 7,413 floats. Materialized that is ~240 GB. Windows are
  therefore built on the fly by WindowBuilder (20_train_tier2.py) from a
  34 MB series array. This is not an optimization, it is the only option.

Outputs (all in `tier2_data/`):
  series_log1p.npy  (T, M) float32   log1p(kWh), DST-repaired, gap-interpolated
  series_raw.npy    (T, M) float32   untouched kWh, for metric reporting only
  missing.npy       (T, M) bool      True where the value was imputed
  calendar.npy      (T, 12) float32  deterministic covariates, computable on-device
  static.npy        (M, 8) float32   per-client profile, fitted on TRAIN ONLY
  manifest.json                      feature spec + schema hash + split bounds

TIMESTAMPS
----------
electricity.txt has no index. It is the Lai et al. (2017) aggregation:
321 clients, hourly, 2012-01-01 .. 2014-12-31 inclusive = 1096 days = 26304
rows, which this script asserts. The index is Europe/Lisbon WALL CLOCK, not
UTC -- verified empirically, see the DST section below. Wall clock is also the
right frame for the covariates, because hour-of-day drives human behaviour in
local time.

DST -- measured on this file, not assumed
-----------------------------------------
Blueprint 2.1.3 predicted the artifacts and they are all present:

  spring-forward slot (local 01:00, last Sunday of March) -- the hour does not
  exist, the file stores near-zero:
      2012-03-25 row  2017: total 17,382 vs ~577,450 the hour before
      2013-03-31 row 10921: total 17,561 vs ~541,015
      2014-03-30 row 19657: total  1,877 vs ~514,236
  fall-back slot (local 01:00, last Sunday of October) -- two real hours summed
  into one slot:
      2012-10-28 row  7225: total 955,037 vs ~586,730 / ~463,128 either side
      2013-10-27 row 15961: total 898,131
      2014-10-26 row 24697: total 937,577

Treatment: spring rows are marked missing and interpolated (there is no data
to recover). Fall rows are halved, because the slot genuinely contains two
hours of energy and we forecast one hour of energy. Both get dst_transition=1
so the model can learn they are odd. Converting to UTC is impossible here --
that needs the 15-minute source, which this aggregate has discarded.

PORTUGUESE HOLIDAYS -- read this before trusting the flag
---------------------------------------------------------
These are Portuguese commercial meters and holidays are the largest avoidable
error source in STLF. Computed here from first principles (no `holidays`
dependency, so the identical code runs on the NanoPi).

IMPORTANT AND EASY TO GET WRONG: Portugal SUSPENDED four public holidays from
2013 to 2015 under the austerity programme -- Corpus Christi, 5 October,
1 November and 1 December. That suspension covers two of our three years.
Treating them as holidays in 2013-2014 mislabels 8 days per year in the
direction that matters most. This is encoded below. It is worth verifying
against a Portuguese source before publishing, since the exact repeal dates
are the kind of thing secondary sources get wrong.
"""

import argparse
import hashlib
import json
import os
from datetime import date, datetime, timedelta

import numpy as np

START = datetime(2012, 1, 1, 0, 0)   # Europe/Lisbon wall clock
N_HOURS_EXPECTED = 1096 * 24         # 2012-01-01 .. 2014-12-31

CALENDAR_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday", "day_before_holiday", "day_after_holiday",
    "is_august", "dst_transition",
]
# M17: the rolling statistics are constant across the horizon, so they are
# window-level, not per-timestep. Keeping them in DYNAMIC_FEATURES sent 336
# identical copies through the feature projection.
DYNAMIC_FEATURES = CALENDAR_FEATURES + [
    "anchor_lag24", "anchor_lag168", "anchor_lag336",
]
WINDOW_FEATURES = [
    "roll_mean24", "roll_std24", "roll_mean168", "roll_min24", "roll_max24",
    # m18: documented as a ratio in the spec, implemented as a difference.
    # In normalised space the difference is the better-conditioned choice
    # (a ratio divides by a quantity that crosses zero); the docs now agree.
    "roll_trend_diff",
]
STATIC_FEATURES = [
    "log_mean_load", "coef_variation", "weekend_weekday_ratio",
    "night_day_ratio", "peak_hour_sin", "peak_hour_cos",
    "peak_to_mean", "temp_corr",
]
FEATURE_SPEC_VERSION = "tier2-v3"


# --------------------------------------------------------------- holidays
def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def portuguese_holidays(year: int) -> set:
    easter = easter_sunday(year)
    days = {
        date(year, 1, 1),                 # Ano Novo
        easter - timedelta(days=2),       # Sexta-feira Santa
        easter,                           # Pascoa
        date(year, 4, 25),                # Dia da Liberdade
        date(year, 5, 1),                 # Dia do Trabalhador
        date(year, 6, 10),                # Dia de Portugal
        date(year, 8, 15),                # Assuncao
        date(year, 12, 8),                # Imaculada Conceicao
        date(year, 12, 25),               # Natal
    }
    # Suspended 2013-2015 by the austerity programme; active in 2012.
    if not (2013 <= year <= 2015):
        days |= {
            easter + timedelta(days=60),  # Corpo de Deus
            date(year, 10, 5),            # Implantacao da Republica
            date(year, 11, 1),            # Todos os Santos
            date(year, 12, 1),            # Restauracao da Independencia
        }
    return days


def build_calendar(n_hours: int, start: datetime) -> np.ndarray:
    stamps = [start + timedelta(hours=i) for i in range(n_hours)]
    years = sorted({s.year for s in stamps})
    hol = set()
    for y in years:
        hol |= portuguese_holidays(y)
    hol |= portuguese_holidays(years[0] - 1)
    hol |= portuguese_holidays(years[-1] + 1)

    # spring-forward / fall-back slots, last Sunday of March / October at 01:00
    dst_slots = set()
    for y in years:
        for month in (3, 10):
            d = date(y, month, 31)
            while d.weekday() != 6:
                d -= timedelta(days=1)
            dst_slots.add(datetime(d.year, d.month, d.day, 1))

    out = np.zeros((n_hours, len(CALENDAR_FEATURES)), dtype=np.float32)
    for i, s in enumerate(stamps):
        d = s.date()
        h, dow, doy = s.hour, s.weekday(), s.timetuple().tm_yday
        is_hol = d in hol
        out[i] = (
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
            float(dow >= 5), float(is_hol),
            float((d + timedelta(days=1)) in hol),
            float((d - timedelta(days=1)) in hol),
            float(s.month == 8), float(s in dst_slots),
        )
    return out, stamps, dst_slots


# ------------------------------------------------------------ data hygiene
def repair_dst(raw, stamps, dst_slots):
    """Returns (raw_repaired, missing_mask_rows). See the module docstring."""
    missing_rows = []
    for i, s in enumerate(stamps):
        if s not in dst_slots:
            continue
        if s.month == 3:
            missing_rows.append(i)          # hour never existed
        else:
            raw[i] *= 0.5                   # two hours summed into one slot
    return raw, missing_rows


def find_onboarding(raw, win=168):
    """Per client, the first index whose trailing `win`-hour sum is non-zero.
    Meters commissioned mid-series read exactly zero before they came online;
    those zeros are not demand and must not train the model."""
    T, M = raw.shape
    csum = np.concatenate([np.zeros((1, M)), np.cumsum(raw, axis=0)], axis=0)
    roll = csum[win:] - csum[:-win]                     # (T-win+1, M)
    first = np.full(M, T, dtype=np.int64)
    for c in range(M):
        nz = np.nonzero(roll[:, c] > 0)[0]
        if not len(nz):
            continue
        # m13: roll[i] covers [i, i+win). Advance to the first non-zero HOUR
        # inside that window instead of stopping at the window start, which
        # under-flagged the onboarding boundary by up to win-1 hours.
        w0 = int(nz[0])
        inner = np.nonzero(raw[w0:w0 + win, c] > 0)[0]
        first[c] = w0 + int(inner[0]) if len(inner) else w0
    return first


def zero_periods(z_tr, candidates=(24, 168, 336, 504), min_ac=0.25):
    """Dominant period(s) of a client's zero indicator, by autocorrelation.

    Review finding M13. The previous rule tested P(zero | hour-of-week) on a
    168-bin grid, which is structurally blind to a 336-hour cycle: alternate
    weeks alias to "zero about half the time in every bin", landing exactly on
    a 0.5 threshold. Client 107's runs recur every 336 h, so it was the one
    client the rule could not see.

    Empirically the old rule did in fact leave client 107 at 3 imputed hours
    (the DST rows only), so the aliasing did not bite -- but it sat on the
    threshold by luck, and luck is not a mechanism. This tests the periods
    directly instead.
    """
    z = z_tr.astype(np.float64)
    z = z - z.mean()
    denom = (z * z).sum()
    if denom < 1e-9:
        return []
    return [p for p in candidates
            if len(z) > p and (z[p:] * z[:-p]).sum() / denom >= min_ac]


def detect_gaps(raw, onboarding, train_end, min_run=48,
                expected_zero_rate=0.5, always_outage=2160, tol=6):
    """Distinguish OUTAGES from SCHEDULED SHUTDOWNS. Zero != missing.

    The naive rule (any zero run >= 3 h is an outage) is wrong on this dataset
    and was materially damaging. These are commercial and industrial meters: a
    shop closed overnight, or a plant on a fortnightly production cycle,
    legitimately reads zero for many consecutive hours. Marking those as
    missing interpolates over real signal AND removes the windows from both
    training and evaluation. Measured: the naive rule put clients 105, 106,
    107 and 299 at 43-54% "missing" and dropped them from the test set.

    Three tests, in order:
      1. A run of >= always_outage hours (default 2160 = 90 days) is a dead
         meter, full stop. Necessary, not belt-and-braces: client 182 is zero
         for 19,914 consecutive hours from hour 6390 to the end of the series,
         so its death DOMINATES its own zero profile (65% of its training
         hours) and every recurrence test below would classify the failure
         itself as "scheduled".
      2. RECURRENCE (M13). If the same run reappears one detected period
         earlier or later, it is a schedule. Periods are found by
         autocorrelation of the zero indicator, so a 336-hour fortnightly
         cycle is visible where a 168-bin grid could not see it.
      3. Hour-of-week rate, as before, as a fallback for irregular but
         habitual downtime.
    """
    T, M = raw.shape
    miss = np.zeros((T, M), dtype=bool)
    how_all = np.arange(T) % 168
    how_tr = how_all[:train_end]
    report = []

    for c in range(M):
        z_tr = raw[:train_end, c] == 0
        rate = np.array([z_tr[how_tr == w].mean() if (how_tr == w).any() else 0.0
                         for w in range(168)])
        scheduled = rate > expected_zero_rate
        periods = zero_periods(z_tr)

        z = raw[:, c] == 0
        z[:onboarding[c]] = False
        n_sched = n_out = 0
        if z.any():
            d = np.diff(np.concatenate([[0], z.view(np.int8), [0]]))
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if b - a < min_run:
                    continue
                if b - a >= always_outage:
                    miss[a:b, c] = True
                    n_out += b - a
                    continue
                # (2) does an equivalent zero run sit one period away?
                recurs = False
                for p in periods:
                    for shift in (-p, p):
                        lo, hi = a + shift, b + shift
                        if lo < 0 or hi > T:
                            continue
                        if z[max(lo - tol, 0):min(hi + tol, T)].mean() > 0.5:
                            recurs = True
                            break
                    if recurs:
                        break
                if recurs or scheduled[how_all[a:b]].mean() > 0.5:
                    n_sched += b - a
                    continue
                miss[a:b, c] = True
                n_out += b - a
        report.append((c, periods, n_sched, n_out))
    return miss, report


def interpolate(series, miss):
    """Linear interpolation over flagged points, per client. Applied to the
    INPUT representation only; targets that are flagged are dropped from the
    sample index instead (20_train_tier2.py)."""
    out = series.copy()
    T = series.shape[0]
    t = np.arange(T)
    for c in range(series.shape[1]):
        m = miss[:, c]
        if not m.any() or m.all():
            continue
        out[m, c] = np.interp(t[m], t[~m], series[~m, c])
    return out


# ------------------------------------------------------------ static profile
def static_profile(raw_train, stamps_train):
    """8 features per client from its own training history. Computed, never
    learned -- a learned client-ID embedding is undefined for a meter the
    global model has never seen, which breaks federated aggregation."""
    T, M = raw_train.shape
    hours = np.array([s.hour for s in stamps_train])
    dows = np.array([s.weekday() for s in stamps_train])

    mean = raw_train.mean(0)
    std = raw_train.std(0)
    safe = np.maximum(mean, 1e-6)

    wk = raw_train[dows >= 5].mean(0)
    wd = raw_train[dows < 5].mean(0)
    night = raw_train[(hours >= 0) & (hours < 6)].mean(0)
    day = raw_train[(hours >= 8) & (hours < 20)].mean(0)

    prof = np.stack([raw_train[hours == h].mean(0) for h in range(24)])  # (24, M)
    peak_h = prof.argmax(0).astype(np.float64)

    out = np.stack([
        np.log1p(mean),
        std / safe,
        wk / np.maximum(wd, 1e-6),
        night / np.maximum(day, 1e-6),
        np.sin(2 * np.pi * peak_h / 24),
        np.cos(2 * np.pi * peak_h / 24),
        raw_train.max(0) / safe,
        np.zeros(M),                        # temp_corr: 0, no weather joined
    ], axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="electricity.txt")
    ap.add_argument("--out", default="tier2_data")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    raw = np.loadtxt(args.data, delimiter=",").astype(np.float64)
    T, M = raw.shape
    print(f"loaded {args.data}: {T} hours x {M} clients")
    assert T == N_HOURS_EXPECTED, (
        f"expected {N_HOURS_EXPECTED} rows for 2012-01-01..2014-12-31, got {T}. "
        f"The timestamp assignment below would be wrong -- stop and check the file.")

    calendar, stamps, dst_slots = build_calendar(T, START)
    print(f"index: {stamps[0]} .. {stamps[-1]} (Europe/Lisbon wall clock)")
    print(f"holidays flagged: {int(calendar[:, 7].sum() / 24)} days "
          f"(note the 2013-2015 austerity suspension, see docstring)")

    raw, dst_missing = repair_dst(raw, stamps, dst_slots)
    onboarding = find_onboarding(raw)
    print(f"DST: {len(dst_missing)} spring rows -> missing, "
          f"{len(dst_slots) - len(dst_missing)} fall rows halved")
    late = int((onboarding > 0).sum())
    print(f"onboarding: {late}/{M} clients start after 2012-01-01 "
          f"(latest: hour {int(onboarding.max())})")

    train_end = int(args.train_ratio * T)
    valid_end = int((args.train_ratio + args.valid_ratio) * T)
    missing, gap_report = detect_gaps(raw, onboarding, train_end)
    for c in range(M):                                  # pre-onboarding
        missing[:onboarding[c], c] = True
    for i in dst_missing:
        missing[i, :] = True
    print(f"missing/outage cells: {missing.mean():.3%}")
    # M13: a per-client imputation table, so this class of problem cannot hide
    worst = sorted(range(M), key=lambda c: -missing[:, c].sum())[:8]
    print("  most-imputed clients (client, detected zero periods, "
          "scheduled h kept, outage h flagged):")
    rep = {r[0]: r for r in gap_report}
    for c in worst:
        _, per, ns, no = rep[c]
        ps = str(per) if per else "-"
        print(f"    client {c:3d}  periods={ps:<18s} "
              f"kept={ns:6d}  flagged={no:6d}  imputed={missing[:, c].mean():6.2%}")
    for c in (105, 106, 107, 299):
        _, per, ns, no = rep[c]
        print(f"    [M13 check] client {c}: periods={per} kept={ns} h "
              f"flagged={no} h -> imputed {missing[:, c].mean():.3%}")

    # ---- C5: TWO arrays, named so they cannot be confused.
    # series_true  : DST-repaired but NEVER interpolated. NaN where the value
    #                did not exist. Every metric and every baseline is scored
    #                against this.
    # series_filled: interpolated. MODEL INPUT ONLY.
    # The previous single `series_raw.npy` was documented as "untouched kWh,
    # for metric reporting only" and was in fact both DST-halved and
    # gap-interpolated -- 0.434% of cells differed from electricity.txt, by up
    # to 118,600 kWh. Seasonal-naive baselines read y[t-24] and y[t-168],
    # which are not covered by the target-validity filter, so wherever those
    # fell on an imputed hour the baseline was a smooth interpolant: easier to
    # beat, inflating every skill score and deflating every MASE.
    series_true = raw.copy()
    series_true[missing] = np.nan
    raw_filled = interpolate(raw, missing)
    series_log1p = np.log1p(np.clip(raw_filled, 0, None)).astype(np.float32)

    static = static_profile(raw_filled[:train_end], stamps[:train_end])

    np.save(f"{args.out}/series_log1p.npy", series_log1p)
    np.save(f"{args.out}/series_true.npy", series_true.astype(np.float64))
    np.save(f"{args.out}/series_filled.npy", raw_filled.astype(np.float32))
    np.save(f"{args.out}/missing.npy", missing)
    np.save(f"{args.out}/calendar.npy", calendar)
    np.save(f"{args.out}/static.npy", static)
    np.save(f"{args.out}/onboarding.npy", onboarding)

    spec = {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "calendar_features": CALENDAR_FEATURES,
        "dynamic_features": DYNAMIC_FEATURES,
        "window_features": WINDOW_FEATURES,
        "static_features": STATIC_FEATURES,
        "C": len(DYNAMIC_FEATURES), "W": len(WINDOW_FEATURES),
        "S": len(STATIC_FEATURES),
        "target_transform": "log1p -> RevIN(lookback) ; invert RevIN -> expm1 -> clip0",
        # M6: single versioned source of truth. WindowBuilder, the Gram path
        # and the evaluator all read these instead of carrying their own
        # constructor defaults.
        "revin_sigma_floor": 0.05,
        "revin_eps": 1e-5,
        "norm_window": 336,
        "start": START.isoformat(), "freq": "1h", "tz": "Europe/Lisbon (wall clock)",
        "T": T, "M": M,
        "train_end": train_end, "valid_end": valid_end,
        "dst_spring_rows": dst_missing,
        "note_kwh": ("electricity.txt is pre-aggregated by Lai et al. 2017; this "
                     "script does not re-aggregate. State their kW->kWh convention "
                     "in the paper, it sets the absolute scale of every MAE."),
    }
    spec["feature_spec_hash"] = hashlib.sha256(
        json.dumps({k: spec[k] for k in
                    ("feature_spec_version", "dynamic_features",
                     "window_features", "static_features", "target_transform",
                     "revin_sigma_floor", "revin_eps", "norm_window")},
                   sort_keys=True).encode()).hexdigest()
    with open(f"{args.out}/manifest.json", "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nsplit 7:1:2 -> train [0,{train_end}) valid [{train_end},{valid_end}) "
          f"test [{valid_end},{T})")
    print(f"feature spec {FEATURE_SPEC_VERSION} hash {spec['feature_spec_hash'][:16]}")
    print(f"saved to {args.out}/ ({sum(os.path.getsize(f'{args.out}/{f}') for f in os.listdir(args.out))/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
