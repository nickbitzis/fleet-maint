"""Day-by-day fleet simulation: component wear, telematics, breakdowns, the shop, parts, and cost accounting.

One engine produces the three years of history (under the old maintenance practice) and replays the held-out year
under each policy, starting from the fleet as it stood on the cutoff day.

Order of events on day d:
1. Each component ages a day. Telematics for day d are generated from its true remaining life.
2. Components at the end of their life fail. On an operating day that is a roadside breakdown (tow, missed stops,
   collateral damage); on a Sunday or holiday the van is found dead at the depot.
3. Vans that were already down cost a day of downtime if it is an operating day.
4. Parts orders that are due arrive.
5. If the shop is open, the policy proposes work, the scheduler builds the plan, and the jobs are done that evening.
6. Parts are reordered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import COMPONENTS, Costs, Shop
from .features import Signals
from .generator import Fleet, day_signals
from .scheduler import DayPlan, Proposal, demand_rate, plan_day, reorder_qty

C = len(COMPONENTS)
PARTS = np.array([c.parts_usd for c in COMPONENTS])
LABOR = np.array([c.labor_h for c in COMPONENTS])
LEAD = np.array([c.lead_days for c in COMPONENTS])
COLLATERAL = np.array([c.collateral_usd for c in COMPONENTS])
COST_KEYS = ("parts", "labor", "towing", "missed_deliveries", "downtime", "collateral")


@dataclass
class State:
    install: np.ndarray  # (V, C) day the current part went in (negative for parts older than the data)
    life: np.ndarray  # (V, C) true life of the current part in days (hidden from policies)
    k: np.ndarray  # (V, C) install counter in the current random stream
    warn: np.ndarray  # (V, C) the current part will show a precursor before it fails (hidden from policies)
    failed: np.ndarray  # (V, C) failed and waiting for repair
    fail_day: np.ndarray  # (V, C)
    on_hand: np.ndarray  # (C,)
    orders: list[tuple[int, int, int]]  # (arrival day, component, quantity)
    consumed: np.ndarray  # (C, D) parts used per day

    def copy(self) -> State:
        return State(
            self.install.copy(),
            self.life.copy(),
            self.k.copy(),
            self.warn.copy(),
            self.failed.copy(),
            self.fail_day.copy(),
            self.on_hand.copy(),
            list(self.orders),
            self.consumed.copy(),
        )


@dataclass
class Context:
    """What a policy may look at on day d. It has no access to State.life."""

    fleet: Fleet
    sig: Signals
    d: int
    age: np.ndarray  # (V, C)
    install: np.ndarray  # (V, C)
    failed: np.ndarray  # (V, C)
    costs: Costs


@dataclass
class Run:
    policy: str
    d0: int
    d1: int
    repairs: list[dict] = field(default_factory=list)
    breakdowns: list[dict] = field(default_factory=list)
    spells: list[tuple[int, int, int, int, bool]] = field(default_factory=list)  # v, c, install, end, failed
    costs: dict = field(default_factory=lambda: dict.fromkeys(COST_KEYS, 0.0))
    daily: list[dict] = field(default_factory=list)
    plans: dict = field(default_factory=dict)  # day -> list of job dicts
    reorders: list[dict] = field(default_factory=list)
    inventory: list[list[int]] = field(default_factory=list)
    install_hist: np.ndarray | None = None
    down_van_days: int = 0

    @property
    def total(self) -> float:
        return float(sum(self.costs.values()))


def base_rates(fleet: Fleet) -> np.ndarray:
    """Planning demand per component per day if every part ran to failure."""
    return fleet.V / fleet.scale.mean(axis=0)


def initial_state(fleet: Fleet, shop: Shop) -> State:
    """Parts already in service on day 0. Install ages are drawn independently of life, then each life is drawn
    conditional on having survived to that age (H(L) = H(age) + E), so a left-truncated fit is unbiased."""
    rng = np.random.default_rng([fleet.cfg.seed, 77])
    V = fleet.V
    k = np.array([c.shape for c in COMPONENTS])[None, :]
    age0 = np.floor(rng.random((V, C)) * fleet.scale).astype(int)
    e = -np.log1p(-fleet.u_life[0, :, :, 0])
    life = fleet.scale * ((age0 / fleet.scale) ** k + e) ** (1.0 / k)
    warn = np.array([[fleet.warns(v, c, 0, 0) for c in range(C)] for v in range(V)])
    rate = base_rates(fleet)
    on_hand = np.array([reorder_qty(0, 0, rate[c], int(LEAD[c]), shop) for c in range(C)])
    return State(
        install=-age0,
        life=life,
        k=np.zeros((V, C), int),
        warn=warn,
        failed=np.zeros((V, C), bool),
        fail_day=np.zeros((V, C), int),
        on_hand=on_hand,
        orders=[],
        consumed=np.zeros((C, fleet.D), int),
    )


def shop_open(fleet: Fleet, d: int, shop: Shop) -> bool:
    return fleet.weekday[d] not in shop.closed_weekdays and not fleet.holiday[d]


def simulate(
    fleet: Fleet,
    policy,
    state: State,
    sig: Signals,
    d0: int,
    d1: int,
    stream: int,
    shop: Shop,
    costs: Costs,
    base_rate: np.ndarray,
    install_hist: np.ndarray | None = None,
    record_plans: bool = False,
    close_spells: bool = True,
    on_day: Callable[[int], None] | None = None,
) -> Run:
    run = Run(policy=policy.name, d0=d0, d1=d1)
    run.install_hist = install_hist
    for d in range(d0, d1):
        age = d - state.install
        if install_hist is not None:
            install_hist[:, :, d] = state.install
        rem = np.where(state.failed, 0.0, state.life - age)
        volt, cool, dtc, noise = day_signals(fleet, d, rem, state.warn)
        sig.volt[:, d], sig.cool[:, d], sig.dtc[:, :, d], sig.noise[:, :, d] = volt, cool, dtc, noise

        was_down = state.failed.any(axis=1)
        new = (rem <= 0) & ~state.failed
        broke_today: set[int] = set()
        for v, c in np.argwhere(new):
            v, c = int(v), int(c)
            roadside = bool(fleet.operating[v, d]) and not was_down[v] and v not in broke_today
            ev = {"van": v, "comp": c, "day": d, "roadside": roadside, "tow": 0.0, "missed": 0.0, "collateral": 0.0}
            if roadside:
                broke_today.add(v)
                ev["tow"] = costs.tow_usd
                ev["missed"] = float(fleet.missed_usd[v])
                ev["collateral"] = float(COLLATERAL[c])
            run.costs["towing"] += ev["tow"]
            run.costs["missed_deliveries"] += ev["missed"]
            run.costs["collateral"] += ev["collateral"]
            run.breakdowns.append(ev)
        state.failed |= new
        state.fail_day[new] = d
        down_op = was_down & fleet.operating[:, d]
        run.costs["downtime"] += costs.downtime_per_day * float(down_op.sum())
        run.down_van_days += int(was_down.sum())

        if state.orders:
            keep = []
            for arrive, c, q in state.orders:
                if arrive <= d:
                    state.on_hand[c] += q
                else:
                    keep.append((arrive, c, q))
            state.orders = keep

        plan = DayPlan([], 0, 0.0)
        if shop_open(fleet, d, shop):
            ctx = Context(fleet, sig, d, age, state.install, state.failed, costs)
            prop: Proposal = policy.propose(ctx)
            plan = plan_day(state.failed, state.fail_day, prop, fleet.peak[:, d], state.on_hand, shop, today=d)
            visit_vans: dict[int, bool] = {}
            day_jobs = []
            for job in plan.jobs:
                v, c = job.van, job.comp
                emergency = job.kind == "emergency"
                visit_vans[v] = visit_vans.get(v, False) or emergency
                run.costs["parts"] += PARTS[c]
                run.costs["labor"] += LABOR[c] * costs.labor_rate * (costs.emergency_labor_mult if emergency else 1.0)
                life = float(state.life[v, c])
                wasted = 0.0 if emergency else max(0.0, life - float(age[v, c]))
                run.repairs.append(
                    {
                        "van": v,
                        "comp": c,
                        "day": d,
                        "kind": job.kind,
                        "parts": float(PARTS[c]),
                        "labor_h": float(LABOR[c]),
                        "age_days": int(age[v, c]),
                        "wasted_days": wasted,
                        "life_days": life,
                    }
                )
                end = int(state.fail_day[v, c]) if emergency else d
                run.spells.append((v, c, int(state.install[v, c]), end, emergency))
                state.k[v, c] += 1
                state.install[v, c] = d
                state.life[v, c] = fleet.life(v, c, int(state.k[v, c]), stream)
                state.warn[v, c] = fleet.warns(v, c, int(state.k[v, c]), stream)
                state.failed[v, c] = False
                state.on_hand[c] -= 1
                state.consumed[c, d] += 1
                if record_plans:
                    day_jobs.append(
                        {
                            "van": v,
                            "comp": c,
                            "kind": job.kind,
                            "score": None if emergency else round(job.score, 2),
                            "reason": prop.reason.get((v, c), "breakdown" if emergency else ""),
                        }
                    )
            for v, had_emergency in visit_vans.items():
                if not had_emergency and fleet.operating[v, d]:
                    run.costs["downtime"] += costs.peak_visit_downtime if fleet.peak[v, d] else costs.visit_downtime
            if record_plans:
                run.plans[d] = day_jobs

        reorders = 0
        for c in range(C):
            on_order = sum(q for _, cc, q in state.orders if cc == c)
            rate = demand_rate(state.consumed[c], d, float(base_rate[c]), shop.trailing_days)
            q = reorder_qty(int(state.on_hand[c]), on_order, rate, int(LEAD[c]), shop)
            if q > 0:
                state.orders.append((d + int(LEAD[c]), c, q))
                run.reorders.append({"day": d, "comp": c, "qty": q, "arrives": d + int(LEAD[c])})
                reorders += 1
        run.inventory.append(state.on_hand.astype(int).tolist())
        run.daily.append(
            {
                "day": d,
                "jobs": len(plan.jobs),
                "bays": plan.bays_used,
                "hours": plan.hours_used,
                "waiting": int(state.failed.any(axis=1).sum()),
                "skipped_peak": plan.skipped_peak,
                "skipped_parts": plan.skipped_parts,
                "skipped_capacity": plan.skipped_capacity,
                "reorders": reorders,
            }
        )
        if on_day:
            on_day(d)
    if not close_spells:
        return run
    for v, c in np.argwhere(np.ones_like(state.failed)):
        v, c = int(v), int(c)
        if state.failed[v, c]:
            run.spells.append((v, c, int(state.install[v, c]), int(state.fail_day[v, c]), True))
        else:
            run.spells.append((v, c, int(state.install[v, c]), d1, False))
    return run
