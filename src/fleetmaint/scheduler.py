"""Daily shop plan and parts reorders.

The plan fills bays in priority order: vans that broke down first (oldest failure first), then planned work by the
policy's score. A van takes one bay for the day however many jobs it gets, so once a van is in, the scheduler adds
its other components that the policy would replace opportunistically. Planned work skips vans on their route's peak
day unless the policy marks it urgent. Every job needs technician hours and a part on the shelf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import COMPONENTS, Shop

LABOR_H = np.array([c.labor_h for c in COMPONENTS])


@dataclass
class Proposal:
    """What a policy wants done today. score is -inf where it wants nothing."""

    score: np.ndarray  # (V, C)
    urgent: np.ndarray  # (V, C) bool: worth doing even on a peak day
    opportunistic: np.ndarray  # (V, C) bool: worth doing if the van is in the shop anyway
    reason: dict = field(default_factory=dict)  # (v, c) -> short text, for the plan view

    @classmethod
    def empty(cls, V: int, C: int) -> Proposal:
        return cls(np.full((V, C), -np.inf), np.zeros((V, C), bool), np.zeros((V, C), bool))


@dataclass
class Job:
    van: int
    comp: int
    kind: str  # emergency, planned, grouped
    score: float = 0.0


@dataclass
class DayPlan:
    jobs: list[Job]
    bays_used: int
    hours_used: float
    skipped_peak: int = 0
    skipped_parts: int = 0
    skipped_capacity: int = 0


def plan_day(
    failed: np.ndarray,
    fail_day: np.ndarray,
    proposal: Proposal,
    peak_today: np.ndarray,
    on_hand: np.ndarray,
    shop: Shop,
    today: int | None = None,
) -> DayPlan:
    """A part that failed today is towed in and diagnosed first; it can be repaired from the next shop day."""
    bays = shop.bays
    hours = shop.labor_hours
    stock = on_hand.astype(int).copy()
    jobs: list[Job] = []
    in_shop: set[int] = set()
    done: set[tuple[int, int]] = set()
    skipped = {"peak": 0, "parts": 0, "capacity": 0}

    def take(v: int, c: int, kind: str, score: float) -> bool:
        nonlocal hours
        if (v, c) in done:
            return False
        if stock[c] <= 0:
            skipped["parts"] += 1
            return False
        if LABOR_H[c] > hours + 1e-9:
            skipped["capacity"] += 1
            return False
        stock[c] -= 1
        hours -= LABOR_H[c]
        jobs.append(Job(v, c, kind, score))
        done.add((v, c))
        return True

    def group(v: int) -> None:
        for c in np.flatnonzero(proposal.opportunistic[v] & ~failed[v]):
            take(v, int(c), "grouped", float(proposal.score[v, c]))

    # 1. Broken-down vans, oldest failure first.
    ready = failed if today is None else failed & (fail_day < today)
    down = np.flatnonzero(ready.any(axis=1))
    first_fail = np.where(failed, fail_day, np.iinfo(np.int32).max).min(axis=1)
    for v in down[np.argsort(first_fail[down], kind="stable")]:
        if bays == 0:
            skipped["capacity"] += int(ready[v].sum())
            continue
        got = False
        for c in np.flatnonzero(ready[v]):
            got |= take(int(v), int(c), "emergency", np.inf)
        if got:
            bays -= 1
            in_shop.add(int(v))
            group(int(v))

    # 2. Planned work by score.
    van_down = failed.any(axis=1)
    cand = np.argwhere(np.isfinite(proposal.score) & ~failed & ~van_down[:, None])
    if len(cand):
        order = np.argsort(-proposal.score[cand[:, 0], cand[:, 1]], kind="stable")
        for v, c in cand[order]:
            v, c = int(v), int(c)
            s = float(proposal.score[v, c])
            if v in in_shop:
                take(v, c, "planned", s)
                continue
            if peak_today[v] and not proposal.urgent[v, c]:
                skipped["peak"] += 1
                continue
            if bays == 0:
                skipped["capacity"] += 1
                continue
            if take(v, c, "planned", s):
                bays -= 1
                in_shop.add(v)
                group(v)
    return DayPlan(
        jobs, shop.bays - bays, shop.labor_hours - hours, skipped["peak"], skipped["parts"], skipped["capacity"]
    )


def reorder_point(rate_per_day: float, lead_days: int, safety_days: int) -> int:
    return math.ceil(rate_per_day * (lead_days + safety_days))


def reorder_qty(on_hand: int, on_order: int, rate_per_day: float, lead_days: int, shop: Shop) -> int:
    """(s, S) policy on inventory position. s covers demand over the lead time plus safety days; S adds cover_days
    more. Returns 0 when the position is above s."""
    position = on_hand + on_order
    s = reorder_point(rate_per_day, lead_days, shop.safety_days)
    S = max(s + 1, s + math.ceil(rate_per_day * shop.cover_days))
    return S - position if position <= s else 0


def demand_rate(consumed: np.ndarray, d: int, base_rate: float, trailing_days: int) -> float:
    """Daily demand estimate: the larger of the planning rate and the trailing consumption."""
    lo = max(0, d - trailing_days + 1)
    recent = consumed[lo : d + 1].sum() / max(1, d + 1 - lo)
    return max(base_rate, float(recent))
