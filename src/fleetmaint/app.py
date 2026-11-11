"""FastAPI service: fleet risk, van detail, the shop plan for a date, policy comparison, model evaluation, and the UI.

The service loads data/bundle.pkl from `fleetmaint train`. If it is missing it builds the bundle on startup (about
20 seconds for 400 vans) and tries to cache it. "Today" is the last day of the held-out year, with the fleet as the
model-based policy left it.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import date
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse

from . import pipeline
from .config import COMPONENTS, ROUTE_TYPES
from .logs import get_logger, log_event
from .policies import breakdown_cost
from .prom import Metrics

STATIC = Path(__file__).parent / "static"
C = len(COMPONENTS)


def data_dir() -> Path:
    return Path(os.environ.get("FLEETMAINT_DATA_DIR", "data"))


def load_or_build(path: Path | None = None) -> pipeline.Bundle:
    path = path or data_dir() / "bundle.pkl"
    if path.exists():
        return pipeline.load(path)
    b = pipeline.build(log=lambda m: log_event(get_logger(), "build", step=m))
    try:
        pipeline.save(b, path)
    except OSError:
        pass  # read-only data dir: serve from memory
    return b


def r2(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


class FleetView:
    """Precomputed answers for the API."""

    def __init__(self, b: pipeline.Bundle):
        self.b = b
        f = b.fleet
        self.fleet = f
        self.today = f.D - 1
        st = b.model_state
        self.age = self.today - st.install
        p = b.models.predict_p14(f, b.model_sig, self.today, self.age, st.install)
        self.p14 = np.where(st.failed, 1.0, p)
        self.bcost = breakdown_cost(f.missed_usd, b.cfg.costs)
        self.risk = 1.0 - np.prod(1.0 - self.p14, axis=1)
        self.exp_cost = (self.p14 * self.bcost).sum(axis=1)
        self.ids = {f.van_id(v): v for v in range(f.V)}
        self.cut = b.cfg.gen.cutoff

    def date_index(self, s: str) -> int:
        try:
            d = date.fromisoformat(s)
        except ValueError as e:
            raise HTTPException(400, "date must be YYYY-MM-DD") from e
        i = (d - self.fleet.dates[0]).days
        if not self.cut <= i < self.fleet.D:
            raise HTTPException(
                404, f"plans exist for {self.fleet.dates[self.cut]} to {self.fleet.dates[-1]}, the held-out year"
            )
        return i

    def fleet_rows(self) -> list[dict]:
        f, st = self.fleet, self.b.model_state
        rows = []
        for v in np.argsort(-self.risk, kind="stable"):
            v = int(v)
            top = int(np.argmax(self.p14[v]))
            rows.append(
                {
                    "van": f.van_id(v),
                    "route": f"R{int(f.route[v]) + 1:02d}",
                    "duty_cycle": ROUTE_TYPES[int(f.rtype[v])].name,
                    "risk_14d": round(float(self.risk[v]), 4),
                    "expected_breakdown_usd": round(float(self.exp_cost[v]), 0),
                    "top_component": COMPONENTS[top].name,
                    "top_p14": round(float(self.p14[v, top]), 4),
                    "p14": {COMPONENTS[c].name: round(float(self.p14[v, c]), 4) for c in range(C)},
                    "down": bool(st.failed[v].any()),
                    "odometer": round(float(f.cum_miles[v, self.today])),
                }
            )
        return rows

    def van(self, vid: str, days: int) -> dict:
        if vid not in self.ids:
            raise HTTPException(404, f"unknown van {vid}")
        v = self.ids[vid]
        f, b, st = self.fleet, self.b, self.b.model_state
        sig = b.model_sig
        lo = max(0, self.today - days + 1)
        tele = []
        for d in range(lo, self.today + 1):
            codes = [COMPONENTS[c].dtc for c in range(C) if sig.dtc[v, c, d]]
            tele.append(
                {
                    "date": f.dates[d].isoformat(),
                    "miles": round(float(f.miles[v, d]), 1),
                    "engine_h": round(float(f.engine_h[v, d]), 2),
                    "idle_h": round(float(f.idle_h[v, d]), 2),
                    "harsh_brakes": int(f.harsh[v, d]),
                    "battery_v": r2(sig.volt[v, d]),
                    "coolant_f": r2(sig.cool[v, d]),
                    "dtc": codes,
                    "peak_day": bool(f.peak[v, d]),
                }
            )
        ts = np.arange(0, 181, 3)
        x = b.models.zobs[v : v + 1]
        comps = []
        for c, spec in enumerate(COMPONENTS):
            a = max(0, int(self.age[v, c]))
            aft = b.models.aft[c]
            comps.append(
                {
                    "component": spec.name,
                    "label": spec.label,
                    "age_days": a,
                    "installed": f.dates[int(st.install[v, c])].isoformat() if st.install[v, c] >= 0 else None,
                    "failed": bool(st.failed[v, c]),
                    "p14_gbm": round(float(self.p14[v, c]), 4),
                    "p14_survival": round(float(aft.cond_fail_prob(np.array([a]), 14, x)[0]), 4),
                    "mean_residual_life_days": round(float(aft.mean_residual_life(np.array([a]), x)[0]), 1),
                    "breakdown_cost_usd": round(float(self.bcost[v, c]), 0),
                    "curve": [round(float(s), 4) for s in aft.cond_survival_curve(a, ts, x[0])],
                }
            )
        work = []
        for src, run in (("history", b.history), ("replay", b.runs["model_based"])):
            for r in run.repairs:
                if r["van"] == v and (src == "replay" or r["day"] < self.cut):
                    work.append(
                        {
                            "date": f.dates[r["day"]].isoformat(),
                            "component": COMPONENTS[r["comp"]].name,
                            "kind": r["kind"],
                            "parts_usd": r["parts"],
                            "labor_h": r["labor_h"],
                            "part_age_days": r["age_days"],
                        }
                    )
            for e in run.breakdowns:
                if e["van"] == v and (src == "replay" or e["day"] < self.cut):
                    work.append(
                        {
                            "date": f.dates[e["day"]].isoformat(),
                            "component": COMPONENTS[e["comp"]].name,
                            "kind": "roadside breakdown" if e["roadside"] else "failed at depot",
                            "tow_usd": e["tow"],
                            "missed_usd": e["missed"],
                        }
                    )
        work.sort(key=lambda w: w["date"], reverse=True)
        return {
            "van": vid,
            "route": f"R{int(f.route[v]) + 1:02d}",
            "duty_cycle": ROUTE_TYPES[int(f.rtype[v])].name,
            "peak_weekday": ["Mon", "Tue", "Wed", "Thu", "Fri"][int(f.route_peak_wd[f.route[v]])],
            "today": f.dates[self.today].isoformat(),
            "risk_14d": round(float(self.risk[v]), 4),
            "usage": dict(zip(("z_miles", "z_harsh", "z_idle", "urban"), np.round(x[0], 3).tolist(), strict=True)),
            "telematics": tele,
            "curve_days": ts.tolist(),
            "components": comps,
            "work_history": work[:60],
        }

    def plan(self, d: int) -> dict:
        f, run = self.fleet, self.b.runs["model_based"]
        i = d - self.cut
        day = run.daily[i]
        jobs = [
            {
                "van": f.van_id(j["van"]),
                "route": f"R{int(f.route[j['van']]) + 1:02d}",
                "component": COMPONENTS[j["comp"]].name,
                "kind": j["kind"],
                "labor_h": COMPONENTS[j["comp"]].labor_h,
                "reason": j["reason"],
            }
            for j in run.plans.get(d, [])
        ]
        orders = [
            {
                "component": COMPONENTS[o["comp"]].name,
                "qty": o["qty"],
                "arrives": f.dates[min(o["arrives"], f.D - 1)].isoformat(),
            }
            for o in run.reorders
            if o["day"] == d
        ]
        shop = self.b.cfg.shop
        return {
            "date": f.dates[d].isoformat(),
            "weekday": f.dates[d].strftime("%a"),
            "shop_open": d in run.plans,
            "bays": shop.bays,
            "labor_hours": shop.labor_hours,
            "bays_used": day["bays"],
            "hours_used": day["hours"],
            "vans_waiting": day["waiting"],
            "skipped_peak": day["skipped_peak"],
            "jobs": jobs,
            "reorders": orders,
            "inventory": dict(zip([c.name for c in COMPONENTS], run.inventory[i], strict=True)),
        }


def create_app(bundle: pipeline.Bundle | None = None) -> FastAPI:
    log = get_logger()
    t0 = time.perf_counter()
    b = bundle if bundle is not None else load_or_build()
    view = FleetView(b)
    res = b.results
    metrics = Metrics()
    log_event(
        log, "bundle_ready", seconds=round(time.perf_counter() - t0, 2), vans=b.fleet.V, gate=res["gate"]["passed"]
    )
    for p in res["policies"]:
        metrics.set_gauge(
            "fleetmaint_policy_cost_usd", "Held-out year total cost by policy.", p["total_usd"], policy=p["policy"]
        )
        metrics.set_gauge(
            "fleetmaint_policy_roadside_breakdowns",
            "Held-out year roadside breakdowns.",
            p["roadside_breakdowns"],
            policy=p["policy"],
        )
    for e in res["evaluation"]:
        for m in ("survival", "gbm"):
            metrics.set_gauge(
                "fleetmaint_model_cindex",
                "Concordance on held-out snapshots.",
                round(e[m]["cindex"], 4),
                component=e["component"],
                model=m,
            )
            if e[m]["auc"] is not None:
                metrics.set_gauge(
                    "fleetmaint_model_auc_14d",
                    "AUC for failure within 14 days.",
                    round(e[m]["auc"], 4),
                    component=e["component"],
                    model=m,
                )
    metrics.set_gauge("fleetmaint_gate_passed", "1 if model-based beat fixed intervals.", int(res["gate"]["passed"]))
    metrics.set_gauge(
        "fleetmaint_vans_high_risk", "Vans with 14-day failure risk of 30% or more.", int((view.risk >= 0.3).sum())
    )

    app = FastAPI(title="Fleet predictive maintenance", version="1.0.0")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.observe_request(route, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if route not in ("/healthz", "/metrics"):
            log_event(log, "request", request_id=rid, method=request.method, path=request.url.path,
                      status=response.status_code, ms=round(dt * 1000, 1))  # fmt: skip
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "vans": b.fleet.V, "today": res["fleet"]["end"], "gate_passed": res["gate"]["passed"]}

    @app.get("/metrics")
    def prom():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/summary")
    def summary():
        keys = ("fleet", "shop", "costs", "components", "history", "gate", "fixed_intervals", "train", "build_seconds")
        return {k: res[k] for k in keys} | {"high_risk_vans": int((view.risk >= 0.3).sum())}

    @app.get("/api/fleet")
    def fleet(limit: int = Query(400, ge=1, le=1000), duty_cycle: str | None = None):
        rows = view.fleet_rows()
        if duty_cycle is not None:
            if duty_cycle not in [r.name for r in ROUTE_TYPES]:
                raise HTTPException(400, "duty_cycle must be urban, suburban, or highway")
            rows = [r for r in rows if r["duty_cycle"] == duty_cycle]
        return {"today": res["fleet"]["end"], "count": len(rows), "vans": rows[:limit]}

    @app.get("/api/vans/{van_id}")
    def van(van_id: str, days: int = Query(120, ge=7, le=365)):
        return view.van(van_id, days)

    @app.get("/api/plan")
    def plan(date: str | None = None, days: int = Query(1, ge=1, le=14)):
        start = view.date_index(date) if date else view.today - 6
        out = [view.plan(d) for d in range(start, min(start + days, b.fleet.D))]
        return {"start": out[0]["date"], "days": out}

    @app.get("/api/policies")
    def policies():
        return {
            "heldout": {"start": res["fleet"]["cutoff"], "end": res["fleet"]["end"]},
            "policies": res["policies"],
            "sweep": res["sweep"],
            "fixed_intervals": res["fixed_intervals"],
            "gate": res["gate"],
        }

    @app.get("/api/models")
    def models():
        return {
            "survival": res["survival"],
            "evaluation": res["evaluation"],
            "leading_indicators": res["leading_indicators"],
        }

    return app
