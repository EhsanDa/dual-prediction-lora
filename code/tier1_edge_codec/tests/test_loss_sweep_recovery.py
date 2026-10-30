"""
test_loss_sweep_recovery.py

Two layers:

  1. recovery_times / recovery_summary arithmetic, on hand-checked arrays.
     No simulator, no store, no data -- these run anywhere.
  2. THE CLAIM: max recovery time <= 24 h (one resynchronisation interval) for
     `resync` and `exception` loss, asserted against the sweep artefact
     tier1_loss_sensitivity.json in BOTH sync_safe configurations.

Layer 2 reads the FROZEN artefact in results_frozen/, so it runs on any
checkout that has it and does not depend on anyone having produced a fresh
sweep first. It still skips if the file is absent -- so if you ever repoint
ARTEFACT, point it somewhere that always exists, or four real assertions turn
into silent skips.
"""

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from eval.recovery import recovery_summary, recovery_times  # noqa: E402

RESYNC_INTERVAL_H = 24
# The frozen artifact, not a loose copy in the package. These layer-2 tests
# validate the numbers that back the paper, and _artefact() SKIPS when the file
# is absent -- so pointing them at a path that can vanish turns four real tests
# into silent skips. results_frozen/ is the one location guaranteed to be there.
ARTEFACT = os.path.join(os.path.dirname(os.path.dirname(REPO)),
                        "results_frozen", "tier1_loss_sensitivity.json")


# ----------------------------------------------------------------- layer 1

def test_no_violations_gives_nothing():
    t, c = recovery_times(np.zeros(48, dtype=bool))
    assert t.size == 0 and c == 0


def test_single_isolated_violation_recovers_in_one_hour():
    v = np.zeros(10, dtype=bool)
    v[4] = True
    t, c = recovery_times(v)
    assert list(t) == [1] and c == 0


def test_run_counts_down_to_one():
    # Hours 2,3,4 violated, hour 5 back inside: 3, 2, 1.
    v = np.zeros(10, dtype=bool)
    v[2:5] = True
    t, c = recovery_times(v)
    assert list(t) == [3, 2, 1] and c == 0


def test_leading_run_is_not_dropped():
    v = np.zeros(6, dtype=bool)
    v[0:2] = True
    t, c = recovery_times(v)
    assert list(t) == [2, 1] and c == 0


def test_two_runs_are_independent():
    v = np.zeros(12, dtype=bool)
    v[1:3] = True
    v[7:10] = True
    t, c = recovery_times(v)
    assert list(t) == [2, 1, 3, 2, 1] and c == 0


def test_trailing_run_is_censored_not_counted_as_recovered():
    v = np.zeros(8, dtype=bool)
    v[5:] = True                      # never returns inside the window
    t, c = recovery_times(v)
    assert t.size == 0
    assert c == 3


def test_all_violated_is_fully_censored():
    t, c = recovery_times(np.ones(24, dtype=bool))
    assert t.size == 0 and c == 24


def test_max_recovery_equals_longest_run():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.random(120) < 0.3
        v[-1] = False                 # avoid censoring for this property
        t, c = recovery_times(v)
        assert c == 0
        # longest run of True, computed the slow obvious way
        longest, run = 0, 0
        for x in v:
            run = run + 1 if x else 0
            longest = max(longest, run)
        assert (int(t.max()) if t.size else 0) == longest


def test_summary_pools_over_clients_by_hour():
    a = np.zeros(10, dtype=bool)
    a[1:3] = True                     # times 2, 1
    b = np.zeros(10, dtype=bool)
    b[5] = True                       # time 1
    s = recovery_summary([a, b])
    assert s["recovery_violated_hours"] == 3
    assert s["recovery_max_h"] == 2.0
    assert s["recovery_mean_h"] == pytest.approx((2 + 1 + 1) / 3)
    assert s["recovery_censored"] is False


def test_censored_tail_is_flagged_and_reported_separately():
    """A never-recovering tail must NOT be able to sneak past a <= 24 check.

    It is reported in its own field rather than folded into the max. This test
    previously asserted `recovery_max_h == 31.0`, i.e. the older behaviour
    where a censored tail WAS folded in. recovery.py's module docstring
    records why that changed: folding reports a lower bound on an unobserved
    quantity as if it were a measured maximum. The guarantee the test exists
    to defend -- that an open tail cannot be silently absorbed into a
    within-bound-looking maximum -- is unchanged and is asserted below against
    the fields that now carry it.
    """
    v = np.zeros(60, dtype=bool)
    v[29:] = True                     # 31 h open at the end
    s = recovery_summary([v])
    assert s["recovery_censored"] is True
    assert s["recovery_censored_episodes"] == 1
    # the tail is visible, in full, and still exceeds the resync interval
    assert s["recovery_censored_max_h"] == 31.0
    assert s["recovery_censored_max_h"] > RESYNC_INTERVAL_H
    # and the reported max covers RESOLVED runs only -- here there are none
    assert s["recovery_max_h"] == 0.0
    assert s["recovery_violated_hours"] == 31


# ----------------------------------------------------------------- layer 2

def _artefact():
    if not os.path.exists(ARTEFACT):
        pytest.skip(f"{ARTEFACT} not present -- run eval/loss_sweep.py first")
    with open(ARTEFACT) as f:
        return json.load(f)


def _config_blocks(doc):
    blocks = {k: doc[k] for k in ("sync_safe_0", "sync_safe_1") if k in doc}
    assert blocks, "no sync_safe_* configuration block in the artefact"
    return blocks


def test_artefact_records_the_flag_not_a_hardcoded_value():
    doc = _artefact()
    assert "sync_safe" in doc
    for name, block in _config_blocks(doc).items():
        expected = name.endswith("_1")
        assert bool(block["sync_safe"]) is expected, (
            f"{name} block reports sync_safe={block['sync_safe']}")


@pytest.mark.parametrize("mode", ["resync", "exception"])
def test_recovery_bounded_by_one_resync_interval(mode):
    """The paper's claim, as an assertion."""
    doc = _artefact()
    for name, block in _config_blocks(doc).items():
        for point in block["curves"][mode]:
            m = point["recovery_max_h"]
            # recovery_max_h already folds censored tails in as lower bounds,
            # so an unbounded open run cannot pass this.
            assert m <= RESYNC_INTERVAL_H, (
                f"{name}/{mode} at loss={point['loss']}: max recovery "
                f"{m} h exceeds the {RESYNC_INTERVAL_H} h resync interval")
            # A run still open when the window ends is a window-edge artefact
            # while it is shorter than one interval; longer than that and it is
            # indistinguishable from a violation that never recovers.
            assert point["recovery_censored_max_h"] <= RESYNC_INTERVAL_H, (
                f"{name}/{mode} at loss={point['loss']}: a violation run of "
                f"{point['recovery_censored_max_h']} h was still open at the "
                f"end of the window -- recovery is unproven, not bounded")


def test_recovery_mean_never_exceeds_max():
    doc = _artefact()
    for block in _config_blocks(doc).values():
        for mode, points in block["curves"].items():
            for p in points:
                assert p["recovery_mean_h"] <= p["recovery_max_h"] + 1e-9, mode


def test_zero_loss_is_the_clean_baseline():
    doc = _artefact()
    for name, block in _config_blocks(doc).items():
        for mode, points in block["curves"].items():
            for p in points:
                if p["loss"] == 0.0:
                    assert p["violation_rate_max"] == 0.0, (
                        f"{name}/{mode}: violations at zero packet loss")


def test_ack_loss_is_inert_under_sync_safe():
    """Under sync_safe the ACK carries configuration only, so ACK loss cannot
    move the bound. This is the property the whole R4 change bought; if it ever
    stops holding, the sweep is measuring something else."""
    doc = _artefact()
    if "sync_safe_1" not in doc:
        pytest.skip("sync_safe=1 configuration not in the artefact")
    for p in doc["sync_safe_1"]["curves"]["ack"]:
        assert p["violation_rate_max"] == 0.0, (
            f"ACK loss {p['loss']} produced a violation under sync_safe")
