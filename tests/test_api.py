import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from fleetmaint.app import create_app
from fleetmaint.cli import markdown
from fleetmaint.logs import JsonFormatter, get_logger


@pytest.fixture(scope="module")
def client(bundle):
    return TestClient(create_app(bundle))


def test_healthz(client):
    r = client.get("/healthz").json()
    assert r["status"] == "ok" and r["vans"] == 120 and r["today"] == "2026-11-10"


def test_metrics(client):
    client.get("/api/fleet")
    body = client.get("/metrics").text
    assert 'fleetmaint_http_requests_total{route="/api/fleet",status="200"}' in body
    assert 'fleetmaint_policy_cost_usd{policy="model_based"}' in body and "fleetmaint_model_cindex" in body


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "Shop schedule" in r.text and "prefers-color-scheme: dark" in r.text


def test_summary(client):
    s = client.get("/api/summary").json()
    assert s["fleet"]["company"] == "Larkspur Parcel Co." and len(s["components"]) == 5
    assert s["fleet"]["cutoff"] == "2025-11-11"


def test_fleet_sorted_by_risk(client):
    r = client.get("/api/fleet").json()
    risk = [v["risk_14d"] for v in r["vans"]]
    assert r["count"] == 120 and risk == sorted(risk, reverse=True)
    v = r["vans"][0]
    assert v["top_p14"] == max(v["p14"].values())


def test_fleet_filter(client):
    r = client.get("/api/fleet?duty_cycle=highway&limit=5").json()
    assert 0 < len(r["vans"]) <= 5 and all(v["duty_cycle"] == "highway" for v in r["vans"])
    assert client.get("/api/fleet?duty_cycle=boats").status_code == 400


def test_van_detail(client):
    vid = client.get("/api/fleet?limit=1").json()["vans"][0]["van"]
    v = client.get(f"/api/vans/{vid}").json()
    assert len(v["telematics"]) == 120 and v["telematics"][-1]["date"] == "2026-11-10"
    assert len(v["components"]) == 5
    for c in v["components"]:
        curve = c["curve"]
        assert curve[0] == 1.0 and all(0 <= a <= 1 for a in curve)
        assert all(a >= b for a, b in zip(curve, curve[1:], strict=False))  # survival only goes down
        assert len(curve) == len(v["curve_days"])
    assert v["work_history"] and all(w["date"] <= "2026-11-10" for w in v["work_history"])


def test_unknown_van(client):
    assert client.get("/api/vans/VAN-9999").status_code == 404


def test_plan_for_a_date(client, bundle):
    r = client.get("/api/plan?date=2026-03-03").json()
    d = r["days"][0]
    assert d["date"] == "2026-03-03" and d["weekday"] == "Tue" and d["shop_open"]
    assert d["bays_used"] <= d["bays"] and sum(j["labor_h"] for j in d["jobs"]) <= d["labor_hours"]
    assert len({j["van"] for j in d["jobs"]}) == d["bays_used"]
    sunday = client.get("/api/plan?date=2026-03-01").json()["days"][0]
    assert not sunday["shop_open"] and sunday["jobs"] == []


def test_plan_week_and_errors(client):
    r = client.get("/api/plan?days=7").json()
    assert len(r["days"]) == 7 and r["days"][-1]["date"] == "2026-11-10"
    assert client.get("/api/plan?date=2024-01-01").status_code == 404
    assert client.get("/api/plan?date=yesterday").status_code == 400


def test_policies(client):
    r = client.get("/api/policies").json()
    assert [p["policy"] for p in r["policies"]] == [
        "run_to_failure",
        "fixed_interval",
        "condition_based",
        "model_based",
    ]
    assert "passed" in r["gate"] and r["fixed_intervals"]


def test_models_endpoint(client):
    r = client.get("/api/models").json()
    assert len(r["evaluation"]) == 5 and "battery" in r["leading_indicators"]


def test_request_id_and_json_logs(client):
    r = client.get("/api/summary", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    log = get_logger()
    log.addHandler(h)
    try:
        client.get("/api/summary", headers={"x-request-id": "xyz789"})
    finally:
        log.removeHandler(h)
    line = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert line["request_id"] == "xyz789" and line["status"] == 200 and line["path"] == "/api/summary"


def test_markdown_table(bundle):
    md = markdown(bundle.results)
    assert "| Model-based (cost rule) |" in md and "Gate:" in md
