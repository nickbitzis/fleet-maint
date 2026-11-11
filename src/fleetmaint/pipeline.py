"""End to end: generate the fleet, run three years of history, fit both models on the first two, score them on the
third as observed, replay the third under each policy, and assemble what the API serves."""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as M
from .config import COMPONENTS, Costs, PolicyConfig, Shop
from .engine import COST_KEYS, Run, State, base_rates, initial_state, simulate
from .features import Signals, observed_covariates
from .generator import Fleet, GenConfig, generate
from .policies import (
    HISTORICAL_INTERVALS,
    ConditionBased,
    FixedInterval,
    ModelBased,
    RunToFailure,
    fixed_from_models,
    fixed_optimal,
    historical_policy,
)
from .survival import fit_weibull_aft

C = len(COMPONENTS)
POLICY_ORDER = ("run_to_failure", "fixed_interval", "condition_based", "model_based")


@dataclass
class PipelineConfig:
    gen: GenConfig = field(default_factory=GenConfig)
    shop: Shop = field(default_factory=Shop)
    costs: Costs = field(default_factory=Costs)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train_step: int = 3  # days between training snapshots
    test_step: int = 7
    sweep: bool = True  # also replay fixed intervals at other quantiles

    @classmethod
    def small(cls, vans: int = 120, seed: int = 7) -> PipelineConfig:
        return cls(gen=GenConfig(seed=seed, vans=vans), shop=Shop(bays=2, technicians=1), sweep=False, train_step=4)


@dataclass
class Bundle:
    cfg: PipelineConfig
    fleet: Fleet
    history: Run
    models: M.Models
    runs: dict[str, Run]
    model_sig: Signals
    model_state: State
    hist_sig: Signals
    results: dict
    seconds: float = 0.0


def to_json(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def merge(a: Run, b: Run) -> Run:
    out = Run(policy=a.policy, d0=a.d0, d1=b.d1)
    for k in ("repairs", "breakdowns", "spells", "daily", "reorders", "inventory"):
        setattr(out, k, getattr(a, k) + getattr(b, k))
    out.costs = {k: a.costs[k] + b.costs[k] for k in COST_KEYS}
    out.install_hist = a.install_hist
    out.down_van_days = a.down_van_days + b.down_van_days
    return out


def fixed_table(fixed: dict, fleet: Fleet) -> list[dict]:
    from .config import ROUTE_TYPES

    rows = []
    for t, rt in enumerate(ROUTE_TYPES):
        m = fleet.rtype == t
        if not m.any():
            continue
        row = {"duty_cycle": rt.name, "vans": int(m.sum())}
        for name, (unit, vals) in fixed.items():
            v = float(np.asarray(vals)[m][0]) if np.ndim(vals) else float(vals)
            row[name] = None if not np.isfinite(v) else {"unit": unit, "value": round(v, 1)}
        rows.append(row)
    return rows


def summarize(run: Run, fleet: Fleet, label: str) -> dict:
    planned = [r for r in run.repairs if r["kind"] != "emergency"]
    wasted = np.array([r["wasted_days"] for r in planned]) if planned else np.zeros(0)
    frac = np.array([r["wasted_days"] / max(r["life_days"], 1e-9) for r in planned]) if planned else np.zeros(0)
    by_comp = []
    for c, spec in enumerate(COMPONENTS):
        by_comp.append(
            {
                "component": spec.name,
                "failures": sum(1 for b in run.breakdowns if b["comp"] == c),
                "roadside": sum(1 for b in run.breakdowns if b["comp"] == c and b["roadside"]),
                "planned": sum(1 for r in planned if r["comp"] == c),
            }
        )
    daily = run.daily
    open_days = [x for x in daily if x["bays"] > 0 or x["jobs"] > 0]
    return {
        "policy": run.policy,
        "label": label,
        "total_usd": round(run.total, 2),
        "costs": {k: round(v, 2) for k, v in run.costs.items()},
        "failures": len(run.breakdowns),
        "roadside_breakdowns": sum(1 for b in run.breakdowns if b["roadside"]),
        "planned_replacements": len(planned),
        "emergency_repairs": len(run.repairs) - len(planned),
        "wasted_life_days": round(float(wasted.sum()), 1),
        "wasted_life_share": round(float(frac.mean()), 4) if len(frac) else 0.0,
        "down_van_days": run.down_van_days,
        "peak_skips": sum(x["skipped_peak"] for x in daily),
        "parts_waits": sum(x["skipped_parts"] for x in daily),
        "capacity_waits": sum(x["skipped_capacity"] for x in daily),
        "max_bays": max((x["bays"] for x in daily), default=0),
        "mean_bays": round(float(np.mean([x["bays"] for x in open_days])) if open_days else 0.0, 2),
        "reorders": len(run.reorders),
        "by_component": by_comp,
        "cost_per_van": round(run.total / fleet.V, 2),
    }


def build(cfg: PipelineConfig | None = None, log: Callable[[str], None] = print) -> Bundle:
    cfg = cfg or PipelineConfig()
    t0 = time.perf_counter()
    fleet = generate(cfg.gen)
    D, cut = fleet.D, cfg.gen.cutoff
    shop, costs = cfg.shop, cfg.costs

    # History: the old practice for three years. Snapshot the fleet on the cutoff day for the replays.
    sig = Signals.empty(fleet.V, C, D)
    state = initial_state(fleet, shop)
    inst_hist = np.zeros((fleet.V, C, D), np.int16)
    hist = historical_policy()
    rate0 = base_rates(fleet)
    h1 = simulate(fleet, hist, state, sig, 0, cut, 0, shop, costs, rate0, install_hist=inst_hist, close_spells=False)
    cut_state, cut_sig = state.copy(), sig.copy()
    h2 = simulate(fleet, hist, state, sig, cut, D, 0, shop, costs, rate0, install_hist=inst_hist)
    history = merge(h1, h2)
    log(
        f"history: {len(history.breakdowns)} failures, {len(history.repairs)} repairs ({time.perf_counter() - t0:.1f}s)"
    )

    # Models, trained on the first two years only.
    zobs = observed_covariates(fleet, 0, cut)
    aft = []
    for c in range(C):
        t, e, X, entry = M.survival_data(history.spells, zobs, cut, c)
        aft.append(fit_weibull_aft(t, e, X, entry))
    end_arr, ev_arr = M.spell_lookup(history.spells, inst_hist, D)
    train = M.snapshots(fleet, sig, inst_hist, end_arr, ev_arr, range(60, cut - M.HORIZON, cfg.train_step), zobs, aft)
    # Training labels must not see past the cutoff.
    tte_train = np.minimum(train.tte, cut - train.day[None, :])
    ev_train = train.event & (train.tte <= tte_train)
    train = M.Snapshots(train.X, tte_train, ev_train, train.day, train.van)
    gbm = []
    for c in range(C):
        known, y = train.labels(c)
        gbm.append(M.fit_gbm(train.X[c][known], y[known], seed=cfg.gen.seed))
    mean_life = np.column_stack([aft[c].mean_life(zobs) for c in range(C)])
    models = M.Models(aft, gbm, zobs, mean_life)
    test = M.snapshots(fleet, sig, inst_hist, end_arr, ev_arr, range(cut, D - M.HORIZON, cfg.test_step), zobs, aft)
    evaluation = M.evaluate(models, test)
    log(f"models fitted and scored ({time.perf_counter() - t0:.1f}s)")

    # Replays of the held-out year.
    train_rate = np.maximum(cut_state.consumed[:, :cut].sum(axis=1) / cut, 0.05)
    van_miles_per_day = fleet.miles[:, :cut].mean(axis=1)
    miles_per_day = float(van_miles_per_day.mean())
    fixed = fixed_optimal(models, fleet, costs, van_miles_per_day)
    policies = {
        "run_to_failure": RunToFailure(),
        "fixed_interval": FixedInterval(fixed, label="Fixed intervals"),
        "condition_based": ConditionBased(cfg.policy),
        "model_based": ModelBased(models, costs),
    }
    if cfg.sweep:
        for q in cfg.policy.fixed_quantiles_sweep:
            iv = fixed_from_models(models, q, miles_per_day)
            policies[f"fixed_q{int(q * 100):02d}"] = FixedInterval(
                iv, f"fixed_q{int(q * 100):02d}", f"Fixed at {q:.0%} failed"
            )
    runs: dict[str, Run] = {}
    model_sig = model_state = None
    for name, pol in policies.items():
        st = cut_state.copy()
        st.k[:] = -1  # held-out installs draw from stream 1, index 0 up, for every policy alike
        sg = cut_sig.copy()
        runs[name] = simulate(
            fleet, pol, st, sg, cut, D, 1, shop, costs, train_rate, record_plans=name == "model_based"
        )
        if name == "model_based":
            model_sig, model_state = sg, st
        log(f"replay {name}: ${runs[name].total:,.0f} ({time.perf_counter() - t0:.1f}s)")

    summaries = {n: summarize(r, fleet, policies[n].label) for n, r in runs.items()}
    rtf = summaries["run_to_failure"]
    for s in summaries.values():
        s["breakdowns_avoided"] = rtf["roadside_breakdowns"] - s["roadside_breakdowns"]
        s["savings_vs_run_to_failure"] = round(rtf["total_usd"] - s["total_usd"], 2)
    fixed_names = [n for n in summaries if n.startswith("fixed")]
    best_fixed = min(fixed_names, key=lambda n: summaries[n]["total_usd"])
    gate = {
        "passed": summaries["model_based"]["total_usd"] < summaries["fixed_interval"]["total_usd"],
        "model_usd": summaries["model_based"]["total_usd"],
        "fixed_usd": summaries["fixed_interval"]["total_usd"],
        "best_fixed": best_fixed,
        "best_fixed_usd": summaries[best_fixed]["total_usd"],
        "beats_best_fixed": summaries["model_based"]["total_usd"] < summaries[best_fixed]["total_usd"],
    }
    results = {
        "fleet": {
            "company": "Larkspur Parcel Co.",
            "vans": fleet.V,
            "routes": int(fleet.cfg.routes),
            "start": fleet.dates[0].isoformat(),
            "end": fleet.dates[-1].isoformat(),
            "cutoff": fleet.dates[cut].isoformat(),
            "days": D,
            "heldout_days": D - cut,
        },
        "shop": {"bays": shop.bays, "technicians": shop.technicians, "labor_hours": shop.labor_hours},
        "costs": costs.__dict__,
        "components": [
            {
                "name": s.name,
                "label": s.label,
                "parts_usd": s.parts_usd,
                "labor_h": s.labor_h,
                "lead_days": s.lead_days,
                "dtc": s.dtc,
                "dtc_text": s.dtc_text,
                "true_shape": s.shape,
                "true_scale_days": s.scale_days,
                "true_beta": dict(zip(("miles", "harsh", "idle", "urban"), s.beta, strict=True)),
            }
            for s in COMPONENTS
        ],
        "history": {
            "failures": len(history.breakdowns),
            "roadside": sum(1 for b in history.breakdowns if b["roadside"]),
            "repairs": len(history.repairs),
            "intervals": HISTORICAL_INTERVALS,
        },
        "survival": M.aft_summary(aft),
        "evaluation": evaluation,
        "leading_indicators": M.leading_indicators(fleet, sig, history.breakdowns, D),
        "fixed_intervals": fixed_table(fixed, fleet),
        "policies": [summaries[n] for n in POLICY_ORDER],
        "sweep": [summaries[n] for n in summaries if n.startswith("fixed")],
        "gate": gate,
        "train": {"rows": int(train.X.shape[1]), "last_label_day": fleet.dates[cut - 1].isoformat()},
    }
    results = json.loads(json.dumps(results, default=to_json))
    b = Bundle(cfg, fleet, history, models, runs, model_sig, model_state, sig, results)
    b.seconds = round(time.perf_counter() - t0, 1)
    results["build_seconds"] = b.seconds
    log(f"done in {b.seconds}s; gate {'passed' if gate['passed'] else 'FAILED'}")
    return b


def save(b: Bundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(b, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load(path: Path) -> Bundle:
    with open(path, "rb") as fh:
        return pickle.load(fh)
