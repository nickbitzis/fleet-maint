import numpy as np
import pytest

from fleetmaint.config import COMPONENTS, Costs
from fleetmaint.engine import Context
from fleetmaint.features import Signals
from fleetmaint.policies import (
    ConditionBased,
    FixedInterval,
    RunToFailure,
    age_replacement_cost_rate,
    age_replacement_interval,
    breakdown_cost,
    early_replacement_cost,
    planned_cost,
    replace_now,
)
from fleetmaint.survival import WeibullAFT

IDX = {c.name: i for i, c in enumerate(COMPONENTS)}


def ctx(fleet, d=200, age=100, install=None):
    V = fleet.V
    sig = Signals.empty(V, 5, fleet.D)
    sig.volt[:] = 12.6
    sig.cool[:] = 195.0
    a = np.full((V, 5), age)
    inst = d - a if install is None else install
    return Context(fleet, sig, d, a, inst, np.zeros((V, 5), bool), Costs())


def test_breakdown_cost_by_hand():
    B = breakdown_cost(np.array([420.0]), Costs())
    # tow 325 + missed 420 + 1.5 days x 220 = 1,075, plus the emergency labor premium (half of hours x $125)
    assert B[0, IDX["brakes"]] == pytest.approx(1075 + 2.0 * 125 * 0.5)
    assert B[0, IDX["coolant_pump"]] == pytest.approx(1075 + 3.0 * 125 * 0.5 + 1400)


def test_cost_rule_worked_example():
    """Brakes: planned job $340 + 2 h x $125 = $590. Half its mean life left, so replacing early throws away $295,
    plus a $90 visit: $385. With a $1,200 breakdown, p = 0.30 gives $360 (wait), p = 0.35 gives $420 (replace)."""
    planned = planned_cost(Costs())[IDX["brakes"]]
    assert planned == 590
    E = early_replacement_cost(planned, np.array([200.0]), np.array([400.0]), 90.0)
    assert E[0] == pytest.approx(385.0)
    assert not replace_now(np.array([0.30]), 1200.0, E)[0]
    assert replace_now(np.array([0.35]), 1200.0, E)[0]


def test_early_cost_caps_at_a_full_part():
    assert early_replacement_cost(590.0, np.array([900.0]), np.array([400.0]), 0.0)[0] == 590.0


def test_age_replacement_optimum_exists_for_wear_out():
    m, x = WeibullAFT(mu=np.log(400), beta=np.zeros(1), sigma=1 / 3.0), np.zeros((1, 1))
    t = age_replacement_interval(m, x, cp=680, cf=1800)
    assert t is not None and 100 < t < 400
    g = age_replacement_cost_rate(m, x, np.array([t, 1600.0]), 680, 1800)
    assert g[0] < g[1]
    assert g[1] == pytest.approx(1800 / m.mean_life(x)[0], rel=0.01)  # long interval = run to failure


def test_no_interval_for_random_failures():
    m = WeibullAFT(mu=np.log(400), beta=np.zeros(1), sigma=1.0)  # exponential: age does not matter
    assert age_replacement_interval(m, np.zeros((1, 1)), cp=680, cf=1800) is None


def test_run_to_failure_proposes_nothing(tiny):
    p = RunToFailure().propose(ctx(tiny))
    assert not np.isfinite(p.score).any() and not p.opportunistic.any()


def test_fixed_interval_due_and_ride_along(tiny):
    pol = FixedInterval({"battery": ("days", 100.0), "alternator": ("days", 115.0), "coolant_pump": ("days", 300.0)})
    p = pol.propose(ctx(tiny, age=100))
    b, a, c = IDX["battery"], IDX["alternator"], IDX["coolant_pump"]
    assert np.isfinite(p.score[:, b]).all() and not np.isfinite(p.score[:, a]).any()
    assert p.opportunistic[:, a].all()  # 87% of its interval: rides along when the van is in
    assert not p.opportunistic[:, c].any() and not p.urgent[:, b].any()


def test_fixed_interval_by_miles(tiny):
    d = 200
    install = np.full((tiny.V, 5), d - 100)
    miles = tiny.cum_miles[:, d] - tiny.cum_miles[:, d - 100]
    pol = FixedInterval({"brakes": ("miles", float(np.median(miles)))})
    r = pol.ratio(ctx(tiny, d=d, install=install))
    assert np.allclose(r[:, IDX["brakes"]], miles / np.median(miles))


def test_condition_needs_two_codes_since_install(tiny):
    c = ctx(tiny, d=200, age=100)
    b = IDX["brakes"]
    c.sig.dtc[0, b, 198] = True
    assert not np.isfinite(ConditionBased().propose(c).score[0, b])  # one code is noise
    c.sig.dtc[0, b, 199] = True
    assert np.isfinite(ConditionBased().propose(c).score[0, b])
    c.install[0, b] = 199  # new part went in after the first code
    c.age[0, b] = 1
    assert not np.isfinite(ConditionBased().propose(c).score[0, b])


def test_condition_voltage_and_coolant(tiny):
    c = ctx(tiny, d=200, age=100)
    c.sig.volt[1, 194:201] = 12.1
    c.sig.cool[2, 194:201] = 215.0
    p = ConditionBased().propose(c)
    assert np.isfinite(p.score[1, IDX["battery"]]) and np.isfinite(p.score[2, IDX["coolant_pump"]])
    assert not np.isfinite(p.score[3]).any()
    assert "voltage" in p.reason[(1, IDX["battery"])]
