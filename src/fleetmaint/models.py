"""Training data from history, the two failure models per component, and their evaluation on a time split.

Survival: a Weibull AFT per component on part lifetimes (spells) that started in the training window, censored at
the cutoff day or at a preventive replacement. Classifier: gradient boosting per component on daily snapshots,
label = the part fails within the next 14 days. Training snapshots stop 15 days before the cutoff, so no label
looks past it. Both models are scored on snapshots from the held-out year as it was observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import features
from .config import COMPONENTS, COVARIATES
from .features import Signals
from .generator import Fleet
from .survival import WeibullAFT, concordance_index

C = len(COMPONENTS)
HORIZON = 14


def survival_data(spells, zobs: np.ndarray, cutoff: int, c: int):
    """Ages at failure or censoring, events, covariates, and entry ages for component c, from every part in service
    before the cutoff. Parts already installed on day 0 enter at their age that day (left truncation); install dates
    come from the work orders, so their age is known."""
    rows = [(v, inst, end, failed) for v, cc, inst, end, failed in spells if cc == c and inst < cutoff]
    v = np.array([r[0] for r in rows], int)
    inst = np.array([r[1] for r in rows])
    end = np.array([r[2] for r in rows])
    failed = np.array([r[3] for r in rows], bool)
    event = failed & (end < cutoff)
    dur = np.minimum(end, cutoff) - inst
    entry = np.maximum(0, -inst)
    keep = dur > entry
    return dur[keep], event[keep], zobs[v[keep]], entry[keep]


def spell_lookup(spells, install_hist: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    """For every (van, component, day), the end day and failure flag of the part in service that day."""
    V, Cn, _ = install_hist.shape
    by_key: dict[tuple[int, int], list] = {}
    for v, c, inst, end, failed in spells:
        by_key.setdefault((v, c), []).append((inst, end, failed))
    end_arr = np.full((V, Cn, D), D, np.int32)
    ev_arr = np.zeros((V, Cn, D), bool)
    for (v, c), lst in by_key.items():
        lst.sort()
        ins = np.array([x[0] for x in lst])
        ends = np.array([x[1] for x in lst])
        fl = np.array([x[2] for x in lst])
        idx = np.searchsorted(ins, install_hist[v, c].astype(int))
        idx = np.clip(idx, 0, len(ins) - 1)
        ok = ins[idx] == install_hist[v, c]
        end_arr[v, c] = np.where(ok, ends[idx], D)
        ev_arr[v, c] = ok & fl[idx]
    return end_arr, ev_arr


@dataclass
class Snapshots:
    X: np.ndarray  # (C, N, F)
    tte: np.ndarray  # (C, N) days from the snapshot to the end of the spell
    event: np.ndarray  # (C, N) the spell ended in a failure
    day: np.ndarray  # (N,)
    van: np.ndarray  # (N,)

    def labels(self, c: int, horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
        """(mask, y): rows with a known 14-day outcome, and whether the part failed in that window."""
        tte, ev = self.tte[c], self.event[c]
        known = (tte > 0) & (ev | (tte > horizon))
        return known, ev & (tte <= horizon)


def snapshots(
    fleet: Fleet,
    sig: Signals,
    install_hist: np.ndarray,
    end_arr: np.ndarray,
    ev_arr: np.ndarray,
    days,
    zobs: np.ndarray,
    aft: list[WeibullAFT],
) -> Snapshots:
    Xs, tte, ev, dd, vv = [], [], [], [], []
    V = fleet.V
    for d in days:
        inst = install_hist[:, :, d].astype(int)
        age = d - inst
        sp = features.survival_p14(aft, age, zobs)
        Xs.append(features.compute(fleet, sig, d, age, inst, zobs, sp))
        tte.append((end_arr[:, :, d] - d).T)
        ev.append(ev_arr[:, :, d].T)
        dd.append(np.full(V, d))
        vv.append(np.arange(V))
    return Snapshots(
        np.concatenate(Xs, axis=1),
        np.concatenate(tte, axis=1),
        np.concatenate(ev, axis=1),
        np.concatenate(dd),
        np.concatenate(vv),
    )


def fit_gbm(X: np.ndarray, y: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    m = HistGradientBoostingClassifier(
        max_iter=220, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    )  # fmt: skip
    m.fit(X, y)
    return m


@dataclass
class Models:
    aft: list[WeibullAFT]
    gbm: list[HistGradientBoostingClassifier]
    zobs: np.ndarray  # (V, 4)
    mean_life: np.ndarray  # (V, C)

    def predict_p14(self, fleet: Fleet, sig: Signals, d: int, age: np.ndarray, install: np.ndarray) -> np.ndarray:
        a = np.maximum(age, 0)
        sp = features.survival_p14(self.aft, a, self.zobs)
        X = features.compute(fleet, sig, d, a, install, self.zobs, sp)
        return np.column_stack([self.gbm[c].predict_proba(X[c])[:, 1] for c in range(C)])


def reliability(p: np.ndarray, y: np.ndarray, edges=(0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0001)) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 20:
            out.append(
                {"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "pred": float(p[m].mean()), "obs": float(y[m].mean())}
            )
    return out


def ece(p: np.ndarray, y: np.ndarray) -> float:
    """Expected calibration error over 10 equal-count bins."""
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    return float(sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)) / len(p))


def evaluate(models: Models, test: Snapshots, max_cindex_rows: int = 5000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for c, spec in enumerate(COMPONENTS):
        known, y = test.labels(c)
        X = test.X[c]
        p_surv = X[:, features.FEATURES.index("survival_p14")]
        p_gbm = models.gbm[c].predict_proba(X)[:, 1]
        alive = test.tte[c] > 0
        idx = np.flatnonzero(alive)
        if len(idx) > max_cindex_rows:
            idx = rng.choice(idx, max_cindex_rows, replace=False)
        row = {
            "component": spec.name,
            "label": spec.label,
            "test_rows": int(known.sum()),
            "positives": int(y[known].sum()),
        }
        for name, p in (("survival", p_surv), ("gbm", p_gbm)):
            pk, yk = p[known], y[known]
            row[name] = {
                "auc": float(roc_auc_score(yk, pk)) if 0 < yk.sum() < len(yk) else None,
                "brier": float(brier_score_loss(yk, pk)),
                "ece": ece(pk, yk),
                "mean_pred": float(pk.mean()),
                "cindex": concordance_index(test.tte[c][idx], test.event[c][idx], p[idx]),
                "reliability": reliability(pk, yk),
            }
        row["base_rate"] = float(y[known].mean())
        out.append(row)
    return out


def aft_summary(aft: list[WeibullAFT]) -> list[dict]:
    return [
        {"component": spec.name, "label": spec.label} | aft[c].to_dict(COVARIATES) for c, spec in enumerate(COMPONENTS)
    ]


def leading_indicators(fleet: Fleet, sig: Signals, breakdowns: list[dict], d_max: int, lookback: int = 60) -> dict:
    """Mean signal by days before failure, relative to the same van 45 to 60 days before (battery and alternator:
    voltage; coolant pump: coolant temperature; brakes and tires: share of days with the component's DTC)."""
    out = {}
    ks = np.arange(-lookback, 1)
    for c, spec in enumerate(COMPONENTS):
        evs = [b for b in breakdowns if b["comp"] == c and lookback <= b["day"] < d_max]
        if spec.name in ("battery", "alternator"):
            series = [sig.volt[b["van"], b["day"] + ks] for b in evs]
            unit = "V"
        elif spec.name == "coolant_pump":
            series = [sig.cool[b["van"], b["day"] + ks] for b in evs]
            unit = "F"
        else:
            series = [sig.dtc[b["van"], c, b["day"] + ks].astype(float) for b in evs]
            unit = "share of days with " + spec.dtc
        if not series:
            continue
        arr = np.array(series, dtype=float)
        if unit in ("V", "F"):
            arr = arr - arr[:, :16].mean(axis=1, keepdims=True)
        out[spec.name] = {
            "days": ks.tolist(),
            "mean": np.round(arr.mean(axis=0), 4).tolist(),
            "unit": unit,
            "n": len(evs),
        }
    return out
