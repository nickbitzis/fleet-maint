"""Maintenance policies. Each one looks at a Context (telematics and part ages, never true remaining life) and
returns a Proposal for the scheduler.

The model-based policy uses the classic cost tradeoff. Replacing a part today wastes the life it has left; waiting
risks a breakdown. Replace when

    P(fail in the next 14 days) x breakdown cost  >  cost of replacing early

where breakdown cost is what a roadside failure adds over a planned job (tow, missed stops, downtime, the
emergency labor premium, collateral damage) and the cost of replacing early is the planned job's parts and labor
times the share of an average life the part still has (Weibull mean residual life / mean life), plus the visit
downtime. A visit on the van's peak day costs more, so only stronger cases pull a van then. Once a van is in the
shop the visit cost is sunk, so other parts ride along when p x breakdown cost beats the wasted life alone.
"""

from __future__ import annotations

import numpy as np

from . import features
from .config import COMPONENTS, Costs, PolicyConfig
from .engine import LABOR, PARTS, Context
from .scheduler import Proposal

C = len(COMPONENTS)
IDX = {c.name: i for i, c in enumerate(COMPONENTS)}


def breakdown_cost(missed_usd: np.ndarray, costs: Costs) -> np.ndarray:
    """(V, C): expected cost a roadside failure adds over doing the same job planned."""
    premium = LABOR * costs.labor_rate * (costs.emergency_labor_mult - 1.0)
    collateral = np.array([c.collateral_usd for c in COMPONENTS])
    per_van = costs.tow_usd + missed_usd + costs.expected_down_days * costs.downtime_per_day
    return per_van[:, None] + (premium + collateral)[None, :]


def planned_cost(costs: Costs) -> np.ndarray:
    return PARTS + LABOR * costs.labor_rate


def early_replacement_cost(planned: np.ndarray, mrl: np.ndarray, mean_life: np.ndarray, visit):
    """Value of the life thrown away (pro rata on parts and labor) plus the cost of bringing the van in."""
    return planned * np.minimum(1.0, mrl / mean_life) + visit


def replace_now(p: np.ndarray, breakdown: np.ndarray, early: np.ndarray) -> np.ndarray:
    return p * breakdown > early


class RunToFailure:
    name = "run_to_failure"
    label = "Run to failure"

    def propose(self, ctx: Context) -> Proposal:
        V = ctx.age.shape[0]
        return Proposal.empty(V, C)


class FixedInterval:
    """Replace each part at a fixed mileage or age. Parts at 85% of their interval or more ride along when the van is
    in the shop anyway; parts 25% overdue may be pulled on a peak day."""

    name = "fixed_interval"
    label = "Fixed intervals"

    def __init__(self, intervals: dict, name: str | None = None, label: str | None = None):
        self.intervals = intervals  # component -> ("miles", 18000) or ("days", 790); the value may be per van (V,)
        if name:
            self.name = name
        if label:
            self.label = label

    def ratio(self, ctx: Context) -> np.ndarray:
        V = ctx.age.shape[0]
        r = np.zeros((V, C))
        inst = np.clip(ctx.install, 0, None)
        miles_since = ctx.fleet.cum_miles[:, ctx.d][:, None] - np.take_along_axis(ctx.fleet.cum_miles, inst, axis=1)
        for name, (unit, value) in self.intervals.items():
            c = IDX[name]
            r[:, c] = (miles_since[:, c] if unit == "miles" else ctx.age[:, c]) / value
        return r

    def propose(self, ctx: Context) -> Proposal:
        r = self.ratio(ctx)
        due = r >= 1.0
        prop = Proposal(np.where(due, r, -np.inf), r >= 1.25, r >= 0.85)
        for v, c in np.argwhere(due):
            prop.reason[(int(v), int(c))] = f"interval {r[v, c] * 100:.0f}% used"
        for v, c in np.argwhere(prop.opportunistic & ~due):
            prop.reason[(int(v), int(c))] = f"interval {r[v, c] * 100:.0f}% used, van in shop"
        return prop


class ConditionBased:
    """Threshold rules on telematics: a component's DTC at least twice in the last 7 days (one hit is often noise),
    7-day mean voltage below 12.30 V (battery), or 7-day mean coolant temperature 12 F above the van's own baseline
    (coolant pump)."""

    name = "condition_based"
    label = "Condition thresholds"

    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()

    def propose(self, ctx: Context) -> Proposal:
        d, sig = ctx.d, ctx.sig
        dtc7 = features.dtc_since_install(sig, d, ctx.install, 7)
        settled = ctx.age >= 7  # a new part's first week still carries the old part's readings
        volt7 = sig.volt[:, max(0, d - 6) : d + 1].mean(axis=1)
        rise = sig.cool[:, max(0, d - 6) : d + 1].mean(axis=1) - sig.cool[:, max(0, d - 60) : max(1, d - 7)].mean(1)
        sev = np.where(dtc7 >= 2, dtc7 / 2.0, 0.0)
        low_v = (volt7 < self.cfg.condition_volt) & settled[:, IDX["battery"]]
        hot = (rise > self.cfg.condition_cool_rise) & settled[:, IDX["coolant_pump"]]
        sev[:, IDX["battery"]] += np.where(low_v, 1.0 + (self.cfg.condition_volt - volt7) * 10, 0)
        sev[:, IDX["coolant_pump"]] += np.where(hot, 1.0 + rise / 20.0, 0)
        flag = sev >= 1.0
        prop = Proposal(np.where(flag, sev, -np.inf), sev >= 1.5, flag)
        for v, c in np.argwhere(flag):
            v, c = int(v), int(c)
            bits = []
            if dtc7[v, c] >= 2:
                bits.append(f"{COMPONENTS[c].dtc} x{int(dtc7[v, c])} in 7 days")
            if c == IDX["battery"] and low_v[v]:
                bits.append(f"voltage {volt7[v]:.2f} V")
            if c == IDX["coolant_pump"] and hot[v]:
                bits.append(f"coolant +{rise[v]:.0f} F")
            prop.reason[(v, c)] = ", ".join(bits)
        return prop


class ModelBased:
    name = "model_based"
    label = "Model-based (cost rule)"

    def __init__(self, models, costs: Costs):
        self.models = models
        self.planned = planned_cost(costs)
        self.costs = costs
        self.last_p: np.ndarray | None = None

    def probabilities(self, ctx: Context) -> np.ndarray:
        return self.models.predict_p14(ctx.fleet, ctx.sig, ctx.d, ctx.age, ctx.install)

    def propose(self, ctx: Context) -> Proposal:
        p = self.probabilities(ctx)
        self.last_p = p
        m = self.models
        B = breakdown_cost(ctx.fleet.missed_usd, self.costs)
        mrl = np.column_stack([m.aft[c].mean_residual_life(np.maximum(ctx.age[:, c], 0), m.zobs) for c in range(C)])
        waste = early_replacement_cost(self.planned[None, :], mrl, m.mean_life, 0.0)
        gain = p * B
        operating = ctx.fleet.operating[:, ctx.d][:, None]
        normal = waste + np.where(operating, self.costs.visit_downtime, 0.0)
        peak = waste + np.where(operating, self.costs.peak_visit_downtime, 0.0)
        want = replace_now(p, B, normal) & ~ctx.failed
        prop = Proposal(np.where(want, gain - normal, -np.inf), replace_now(p, B, peak), replace_now(p, B, waste))
        for v, c in np.argwhere(want | prop.opportunistic):
            v, c = int(v), int(c)
            prop.reason[(v, c)] = f"p14 {p[v, c] * 100:.0f}% x ${B[v, c]:,.0f} > ${normal[v, c]:,.0f}"
        return prop


def age_replacement_cost_rate(aft, x: np.ndarray, T: np.ndarray, cp: float, cf: float) -> np.ndarray:
    """Long-run cost per day of replacing at age T or at failure, whichever comes first (Barlow and Proschan):
    (cp S(T) + cf F(T)) / integral of S from 0 to T."""
    grid = np.linspace(0, T.max(), 4001)
    S = aft.survival(grid, x)
    area = np.concatenate([[0.0], np.cumsum((S[1:] + S[:-1]) / 2 * np.diff(grid))])
    ST = np.interp(T, grid, S)
    return (cp * ST + cf * (1 - ST)) / np.interp(T, grid, area)


def age_replacement_interval(aft, x: np.ndarray, cp: float, cf: float) -> float | None:
    """The age that minimizes the cost rate, or None when running to failure is cheaper (shape near 1, or a
    breakdown that costs little more than a planned job)."""
    lam = float(aft.scale(x)[0])
    T = np.linspace(lam * 0.03, lam * 4.0, 800)
    g = age_replacement_cost_rate(aft, x, T, cp, cf)
    i = int(np.argmin(g))
    rtf = cf / float(aft.mean_life(x)[0])
    return float(T[i]) if g[i] < 0.98 * rtf and i < len(T) - 1 else None


def _interval(spec, days: float, miles_per_day: float) -> tuple[str, float]:
    if spec.interval_unit == "miles":
        return ("miles", max(1000.0, round(days * miles_per_day / 1000.0) * 1000.0))
    return ("days", max(30.44, round(days / 30.44) * 30.44))


def fixed_from_models(models, quantile: float, miles_per_day: float) -> dict[str, tuple[str, float]]:
    """Fixed intervals at the age where `quantile` of parts have failed for the average van."""
    zbar = models.zobs.mean(axis=0, keepdims=True)
    return {
        spec.name: _interval(spec, float(models.aft[c].quantile(quantile, zbar)[0]), miles_per_day)
        for c, spec in enumerate(COMPONENTS)
    }


def fixed_optimal(models, fleet, costs: Costs, miles_per_day: np.ndarray) -> dict[str, tuple[str, np.ndarray]]:
    """The strongest fixed-interval program: per component and duty cycle (urban, suburban, highway), the
    age-replacement optimum for that cycle's average van, priced with the same planned and breakdown costs the model
    policy uses. A component where no interval beats running to failure gets an infinite interval."""
    B = breakdown_cost(fleet.missed_usd, costs)
    planned = planned_cost(costs)
    out = {}
    for c, spec in enumerate(COMPONENTS):
        vals = np.full(fleet.V, np.inf)
        for t in np.unique(fleet.rtype):
            m = fleet.rtype == t
            zbar = models.zobs[m].mean(axis=0, keepdims=True)
            cp = planned[c] + costs.visit_downtime
            days = age_replacement_interval(models.aft[c], zbar, cp, planned[c] + float(B[m, c].mean()))
            if days is not None:
                vals[m] = _interval(spec, days, float(miles_per_day[m].mean()))[1]
        out[spec.name] = ("miles" if spec.interval_unit == "miles" else "days", vals)
    return out


HISTORICAL_INTERVALS = {"brakes": ("miles", 40000.0), "tires": ("miles", 50000.0)}


def historical_policy() -> FixedInterval:
    """The old practice that produced the history: brakes and tires by mileage, everything else to failure."""
    return FixedInterval(HISTORICAL_INTERVALS, name="historical", label="Historical practice")
