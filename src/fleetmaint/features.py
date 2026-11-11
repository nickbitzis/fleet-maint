"""Features for the 14-day failure classifier, computed from telematics up to and including day d.

Nothing here reads a value after day d. The same function serves training snapshots from history and the daily
decisions inside the policy replays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import COMPONENTS
from .generator import Fleet, zscores

FEATURES = (
    "age_days",
    "kmiles_since_install",
    "z_miles",
    "z_harsh",
    "z_idle",
    "urban",
    "survival_p14",
    "volt_mean7",
    "volt_drop",
    "volt_min7",
    "cool_rise7",
    "cool_max_rise7",
    "dtc7",
    "dtc28",
    "ambient7",
    "harsh_per100_7",
)


@dataclass
class Signals:
    volt: np.ndarray  # (V, D) float32
    cool: np.ndarray  # (V, D) float32
    dtc: np.ndarray  # (V, C, D) bool, the component's own code
    noise: np.ndarray  # (V, 3, D) bool, unrelated codes

    @classmethod
    def empty(cls, V: int, C: int, D: int) -> Signals:
        return cls(
            np.full((V, D), np.nan, np.float32),
            np.full((V, D), np.nan, np.float32),
            np.zeros((V, C, D), bool),
            np.zeros((V, 3, D), bool),
        )

    def copy(self) -> Signals:
        return Signals(self.volt.copy(), self.cool.copy(), self.dtc.copy(), self.noise.copy())


def observed_covariates(fleet: Fleet, d0: int, d1: int) -> np.ndarray:
    """Van usage covariates from telematics in days [d0, d1): what the fleet can measure, not the true rates."""
    op = fleet.operating[:, d0:d1]
    miles = fleet.miles[:, d0:d1]
    mrate = miles.sum(1) / np.maximum(op.sum(1), 1)
    harsh = fleet.harsh[:, d0:d1].sum(1) / np.maximum(miles.sum(1), 1) * 100
    idle = fleet.idle_h[:, d0:d1].sum(1) / np.maximum(fleet.engine_h[:, d0:d1].sum(1), 1e-6)
    return zscores(mrate, harsh, idle, fleet.rtype == 0)


def _mean(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo = max(0, lo)
    hi = max(lo + 1, hi)
    return a[:, lo:hi].mean(axis=1)


def compute(
    fleet: Fleet,
    sig: Signals,
    d: int,
    age: np.ndarray,
    install_day: np.ndarray,
    zobs: np.ndarray,
    surv_p14: np.ndarray,
) -> np.ndarray:
    """Feature tensor (C, V, F) for every van and component on day d.

    age and install_day are (V, C); surv_p14 is the Weibull AFT probability of failing in the next 14 days.
    """
    V, C = age.shape
    volt7 = _mean(sig.volt, d - 6, d + 1)
    volt_prev = _mean(sig.volt, d - 27, d - 6)
    volt_min7 = sig.volt[:, max(0, d - 6) : d + 1].min(axis=1)
    cool7 = _mean(sig.cool, d - 6, d + 1)
    cool_base = _mean(sig.cool, d - 60, d - 7)
    cool_max7 = sig.cool[:, max(0, d - 6) : d + 1].max(axis=1)
    dtc7, dtc28 = dtc_since_install(sig, d, install_day, 7), dtc_since_install(sig, d, install_day, 28)
    amb7 = float(fleet.ambient_f[max(0, d - 6) : d + 1].mean())
    m7 = fleet.miles[:, max(0, d - 6) : d + 1].sum(axis=1)
    h7 = fleet.harsh[:, max(0, d - 6) : d + 1].sum(axis=1) / np.maximum(m7, 1) * 100
    inst = np.clip(install_day, 0, None)
    kmiles = (fleet.cum_miles[:, d][:, None] - np.take_along_axis(fleet.cum_miles, inst, axis=1)) / 1000.0
    out = np.empty((C, V, len(FEATURES)), np.float32)
    for c in range(C):
        out[c] = np.column_stack(
            [
                age[:, c],
                kmiles[:, c],
                zobs,
                surv_p14[:, c],
                volt7,
                volt_prev - volt7,
                volt_min7,
                cool7 - cool_base,
                cool_max7 - cool_base,
                dtc7[:, c],
                dtc28[:, c],
                np.full(V, amb7),
                h7,
            ]
        )
    return out


def dtc_since_install(sig: Signals, d: int, install_day: np.ndarray, window: int) -> np.ndarray:
    """(V, C) count of each component's own DTC in the last `window` days, counting only days after the current part
    went in. Codes the old part raised do not follow the new one."""
    lo = max(0, d - window + 1)
    days = np.arange(lo, d + 1)
    return (sig.dtc[:, :, lo : d + 1] & (days[None, None, :] > install_day[:, :, None])).sum(axis=2)


def survival_p14(models, age: np.ndarray, zobs: np.ndarray, horizon: int = 14) -> np.ndarray:
    return np.column_stack([models[c].cond_fail_prob(age[:, c], horizon, zobs) for c in range(len(COMPONENTS))])
