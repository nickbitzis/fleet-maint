"""Accounting of each policy's replay of the held-out year, on the small build (120 vans, 2 bays, 1 technician)."""

import numpy as np
import pytest

from fleetmaint.config import COMPONENTS
from fleetmaint.engine import COST_KEYS, LABOR, PARTS, initial_state, simulate
from fleetmaint.features import Signals
from fleetmaint.pipeline import POLICY_ORDER
from fleetmaint.policies import RunToFailure

NAMES = list(POLICY_ORDER)


@pytest.mark.parametrize("name", NAMES)
def test_cost_categories_add_up(bundle, name):
    run = bundle.runs[name]
    s = next(p for p in bundle.results["policies"] if p["policy"] == name)
    assert sum(s["costs"].values()) == pytest.approx(s["total_usd"], abs=0.05)
    assert set(run.costs) == set(COST_KEYS)


@pytest.mark.parametrize("name", NAMES)
def test_parts_labor_towing_missed_by_hand(bundle, name):
    run, costs, f = bundle.runs[name], bundle.cfg.costs, bundle.fleet
    assert run.costs["parts"] == pytest.approx(sum(PARTS[r["comp"]] for r in run.repairs))
    labor = sum(LABOR[r["comp"]] * costs.labor_rate * (1.5 if r["kind"] == "emergency" else 1.0) for r in run.repairs)
    assert run.costs["labor"] == pytest.approx(labor)
    road = [b for b in run.breakdowns if b["roadside"]]
    assert run.costs["towing"] == pytest.approx(costs.tow_usd * len(road))
    assert run.costs["missed_deliveries"] == pytest.approx(sum(f.missed_usd[b["van"]] for b in road))
    assert run.costs["collateral"] == pytest.approx(sum(COMPONENTS[b["comp"]].collateral_usd for b in road))


@pytest.mark.parametrize("name", NAMES)
def test_shop_capacity_respected_every_day(bundle, name):
    shop = bundle.cfg.shop
    run = bundle.runs[name]
    for day in run.daily:
        assert day["bays"] <= shop.bays and day["hours"] <= shop.labor_hours + 1e-9
    by_day: dict = {}
    for r in run.repairs:
        by_day.setdefault(r["day"], set()).add(r["van"])
    assert max(len(v) for v in by_day.values()) <= shop.bays
    assert all(bundle.fleet.weekday[d] != 6 and not bundle.fleet.holiday[d] for d in by_day)


@pytest.mark.parametrize("name", NAMES)
def test_inventory_balances(bundle, name):
    """Stock never goes negative, and start + received - used = end."""
    run = bundle.runs[name]
    inv = np.array(run.inventory)
    assert (inv >= 0).all()
    cut = bundle.cfg.gen.cutoff
    start = bundle.history.inventory[cut - 1]
    used = np.bincount([r["comp"] for r in run.repairs], minlength=5)
    got = np.zeros(5, int)
    for o in run.reorders:
        if o["arrives"] < run.d1:
            got[o["comp"]] += o["qty"]
    # orders placed before the cutoff that arrive inside the replay
    for o in bundle.history.reorders:
        if o["day"] < cut <= o["arrives"]:
            got[o["comp"]] += o["qty"]
    assert np.array_equal(np.array(start) + got - used, inv[-1])


def test_run_to_failure_wastes_nothing(bundle):
    s = bundle.results["policies"][0]
    assert s["policy"] == "run_to_failure" and s["planned_replacements"] == 0 and s["wasted_life_share"] == 0
    assert s["breakdowns_avoided"] == 0


def test_early_replacements_record_true_life_left(bundle):
    planned = [r for r in bundle.runs["fixed_interval"].repairs if r["kind"] != "emergency"]
    assert planned and all(0 < r["wasted_days"] <= r["life_days"] for r in planned)
    assert all(r["life_days"] - r["age_days"] == pytest.approx(r["wasted_days"]) for r in planned)


def test_emergency_repairs_happen_after_the_failure(bundle):
    run = bundle.runs["model_based"]
    fails = {}
    for b in run.breakdowns:
        fails.setdefault((b["van"], b["comp"]), []).append(b["day"])
    for r in run.repairs:
        if r["kind"] == "emergency" and (r["van"], r["comp"]) in fails:  # else it failed before the replay began
            assert any(d < r["day"] for d in fails[(r["van"], r["comp"])])


def test_replays_share_the_same_world(bundle):
    """Same starting fleet: every policy sees the same failures on the first day, before any decision matters."""
    cut = bundle.cfg.gen.cutoff
    first = [sorted((b["van"], b["comp"]) for b in r.breakdowns if b["day"] == cut) for r in bundle.runs.values()]
    assert all(x == first[0] for x in first)


def test_simulation_is_deterministic(tiny):
    from fleetmaint.config import Costs, Shop

    def once():
        st = initial_state(tiny, Shop())
        sig = Signals.empty(tiny.V, 5, tiny.D)
        return simulate(tiny, RunToFailure(), st, sig, 0, 200, 0, Shop(), Costs(), np.full(5, 0.05))

    a, b = once(), once()
    assert a.total == b.total and len(a.breakdowns) == len(b.breakdowns) > 0


def test_policies_avoid_breakdowns(bundle):
    by = {p["policy"]: p for p in bundle.results["policies"]}
    for n in ("fixed_interval", "condition_based", "model_based"):
        assert by[n]["roadside_breakdowns"] < by["run_to_failure"]["roadside_breakdowns"]
    assert by["model_based"]["roadside_breakdowns"] < by["fixed_interval"]["roadside_breakdowns"]
    assert by["model_based"]["wasted_life_share"] < by["fixed_interval"]["wasted_life_share"]


def test_peak_days_are_mostly_protected(bundle):
    run, f = bundle.runs["model_based"], bundle.fleet
    planned = [r for r in run.repairs if r["kind"] == "planned"]
    on_peak = sum(f.peak[r["van"], r["day"]] for r in planned)
    peak_share = f.peak.sum() / f.operating.sum()
    assert on_peak / len(planned) < peak_share / 1.5


def test_reorders_fire_when_stock_runs_low(bundle):
    run = bundle.runs["fixed_interval"]
    assert len(run.reorders) > 10
    assert all(o["arrives"] == o["day"] + COMPONENTS[o["comp"]].lead_days for o in run.reorders)
