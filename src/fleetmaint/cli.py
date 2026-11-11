"""Command line: fleetmaint train | evaluate | serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pipeline


def usd(v: float) -> str:
    return f"${v:,.0f}"


def markdown(r: dict) -> str:
    lines = [
        f"Held-out year {r['fleet']['cutoff']} to {r['fleet']['end']}, {r['fleet']['vans']} vans.",
        "",
        "| Policy | Total cost | Roadside breakdowns | Planned replacements | Wasted life | Parts | Labor | Towing "
        "| Missed deliveries | Downtime | Collateral |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in r["policies"]:
        c = p["costs"]
        lines.append(
            f"| {p['label']} | {usd(p['total_usd'])} | {p['roadside_breakdowns']} | {p['planned_replacements']} | "
            f"{p['wasted_life_share'] * 100:.0f}% | {usd(c['parts'])} | {usd(c['labor'])} | {usd(c['towing'])} | "
            f"{usd(c['missed_deliveries'])} | {usd(c['downtime'])} | {usd(c['collateral'])} |"
        )
    g = r["gate"]
    lines += [
        "",
        f"Gate: {'PASSED' if g['passed'] else 'FAILED'}. Model-based {usd(g['model_usd'])} vs fixed intervals "
        f"{usd(g['fixed_usd'])}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fleetmaint")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="generate, fit, replay every policy; writes bundle.pkl and results.json")
    e = sub.add_parser("evaluate", help="run everything and print the policy table")
    e.add_argument("--gate", action="store_true", help="exit 1 unless model-based beats fixed intervals")
    e.add_argument("--markdown", type=Path, help="append the table to this file (for $GITHUB_STEP_SUMMARY)")
    e.add_argument("--json", type=Path)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8884)
    a = ap.parse_args(argv)

    def log(m):
        print(m, file=sys.stderr)

    if a.cmd == "train":
        b = pipeline.build(log=log)
        pipeline.save(b, a.data_dir / "bundle.pkl")
        (a.data_dir / "results.json").write_text(json.dumps(b.results, indent=1, default=float))
        log(f"wrote {a.data_dir / 'bundle.pkl'} and results.json")
    elif a.cmd == "evaluate":
        b = pipeline.build(log=log)
        md = markdown(b.results)
        print(md)
        if a.markdown:
            with open(a.markdown, "a") as fh:
                fh.write(md)
        if a.json:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps(b.results, indent=1, default=float))
        if a.gate and not b.results["gate"]["passed"]:
            log("gate failed: the model-based policy did not beat fixed intervals")
            sys.exit(1)
    elif a.cmd == "serve":
        import os

        import uvicorn

        os.environ.setdefault("FLEETMAINT_DATA_DIR", str(a.data_dir))
        from .app import create_app

        uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
