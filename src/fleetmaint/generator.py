"""Synthetic fleet: vans, routes, calendar, daily usage, component lifetimes, and the telematics signal model.

Everything is seeded. Usage (miles, harsh braking, idle) is exogenous: it does not depend on maintenance, so every
policy replays the same days. Component lifetimes come from pre-drawn uniforms indexed by (van, component, install
number), so two policies that install the same part on the same van draw the same life (common random numbers).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .config import COMPONENTS, NOISE_DTCS, ROUTE_TYPES, Z_REF, Costs

K_DRAWS = 48  # installs per (van, component) and stream; far more than three years needs


@dataclass(frozen=True)
class GenConfig:
    seed: int = 7
    vans: int = 400
    days: int = 1096
    start: date = date(2023, 11, 11)
    heldout_days: int = 365
    routes: int = 40

    @property
    def cutoff(self) -> int:
        """First day index of the held-out year."""
        return self.days - self.heldout_days


def holidays(year: int) -> set[date]:
    nov1 = date(year, 11, 1)
    thanksgiving = nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21)
    return {date(year, 1, 1), date(year, 7, 4), thanksgiving, date(year, 12, 25)}


@dataclass
class Fleet:
    cfg: GenConfig
    dates: list[date]
    weekday: np.ndarray  # (D,)
    holiday: np.ndarray  # (D,) bool
    ambient_f: np.ndarray  # (D,)
    route: np.ndarray  # (V,) route id
    rtype: np.ndarray  # (V,) index into ROUTE_TYPES
    route_peak_wd: np.ndarray  # (routes,) weekday of each route's peak day
    operating: np.ndarray  # (V, D) bool
    peak: np.ndarray  # (V, D) bool
    miles: np.ndarray  # (V, D)
    harsh: np.ndarray  # (V, D) harsh braking events
    idle_h: np.ndarray  # (V, D)
    engine_h: np.ndarray  # (V, D)
    cum_miles: np.ndarray  # (V, D) odometer at the end of each day
    z_true: np.ndarray  # (V, 4) true usage covariates
    scale: np.ndarray  # (V, C) true Weibull scale in days
    volt_off: np.ndarray  # (V,)
    cool_base: np.ndarray  # (V,)
    missed_usd: np.ndarray  # (V,) cost of the stops a roadside breakdown misses
    u_life: np.ndarray  # (2, V, C, K) uniforms: stream 0 = history, stream 1 = held-out replays
    u_warn: np.ndarray  # (2, V, C, K) uniforms: does this part give a warning before it fails

    @property
    def V(self) -> int:
        return len(self.route)

    @property
    def D(self) -> int:
        return len(self.dates)

    def life(self, v: int, c: int, k: int, stream: int) -> float:
        u = self.u_life[stream, v, c, k % K_DRAWS]
        return float(self.scale[v, c] * (-np.log1p(-u)) ** (1.0 / COMPONENTS[c].shape))

    def warns(self, v: int, c: int, k: int, stream: int) -> bool:
        return bool(self.u_warn[stream, v, c, k % K_DRAWS] < COMPONENTS[c].warn_prob)

    def van_id(self, v: int) -> str:
        return f"VAN-{v + 1:04d}"


def zscores(miles_rate, harsh_rate, idle_share, urban) -> np.ndarray:
    return np.column_stack(
        [
            (np.asarray(miles_rate) - Z_REF["miles"][0]) / Z_REF["miles"][1],
            (np.asarray(harsh_rate) - Z_REF["harsh"][0]) / Z_REF["harsh"][1],
            (np.asarray(idle_share) - Z_REF["idle"][0]) / Z_REF["idle"][1],
            np.asarray(urban, dtype=float),
        ]
    )


def true_scale(z: np.ndarray) -> np.ndarray:
    beta = np.array([c.beta for c in COMPONENTS])  # (C, 4)
    base = np.array([c.scale_days for c in COMPONENTS])
    return base[None, :] * np.exp(z @ beta.T)


def generate(cfg: GenConfig | None = None) -> Fleet:
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    V, D = cfg.vans, cfg.days
    dates = [cfg.start + timedelta(days=i) for i in range(D)]
    weekday = np.array([d.weekday() for d in dates])
    hol = set().union(*(holidays(y) for y in range(cfg.start.year, dates[-1].year + 1)))
    holiday = np.array([d in hol for d in dates])
    doy = np.array([d.timetuple().tm_yday for d in dates])
    ambient = 56 + 24 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 6, D)
    peak_season = np.array([d.month == 12 and d.day <= 23 for d in dates])

    shares = np.array([r.share for r in ROUTE_TYPES])
    route_type_of = rng.choice(len(ROUTE_TYPES), size=cfg.routes, p=shares)
    route_peak_wd = rng.integers(0, 5, cfg.routes)  # Monday to Friday
    route = np.arange(V) % cfg.routes
    rng.shuffle(route)
    rtype = route_type_of[route]

    def draw(attr_mu, attr_sd, lo):
        mu = np.array([getattr(ROUTE_TYPES[t], attr_mu) for t in rtype])
        sd = np.array([getattr(ROUTE_TYPES[t], attr_sd) for t in rtype])
        return np.maximum(lo, rng.normal(mu, sd))

    base_miles = draw("miles_mu", "miles_sd", 30)
    harsh_rate = draw("harsh_mu", "harsh_sd", 0.4)
    idle_share = np.clip(draw("idle_mu", "idle_sd", 0.04), 0.04, 0.6)
    speed = np.array([ROUTE_TYPES[t].speed_mph for t in rtype])

    operating = np.broadcast_to((weekday != 6) & ~holiday, (V, D)).copy()
    peak = (weekday[None, :] == route_peak_wd[route][:, None]) | peak_season[None, :]
    peak &= operating
    miles = base_miles[:, None] * rng.lognormal(-0.007, 0.12, (V, D)) * np.where(peak, 1.12, 1.0) * operating
    harsh = rng.poisson(miles / 100 * harsh_rate[:, None]).astype(np.float32)
    drive_h = miles / speed[:, None]
    idle_h = drive_h * (idle_share / (1 - idle_share))[:, None] * rng.lognormal(-0.011, 0.15, (V, D))
    engine_h = drive_h + idle_h

    z_true = zscores(base_miles, harsh_rate, idle_share, rtype == 0)
    scale = true_scale(z_true)
    missed = np.array([ROUTE_TYPES[t].stops * ROUTE_TYPES[t].stop_value_usd for t in rtype]) * Costs().missed_share
    return Fleet(
        cfg=cfg,
        dates=dates,
        weekday=weekday,
        holiday=holiday,
        ambient_f=ambient,
        route=route,
        rtype=rtype,
        route_peak_wd=route_peak_wd,
        operating=operating,
        peak=peak,
        miles=miles.astype(np.float32),
        harsh=harsh,
        idle_h=idle_h.astype(np.float32),
        engine_h=engine_h.astype(np.float32),
        cum_miles=np.cumsum(miles, axis=1),
        z_true=z_true,
        scale=scale,
        volt_off=rng.normal(0, 0.04, V),
        cool_base=194 + rng.normal(0, 2.0, V),
        missed_usd=missed,
        u_life=rng.random((2, V, len(COMPONENTS), K_DRAWS)),
        u_warn=rng.random((2, V, len(COMPONENTS), K_DRAWS)),
    )


def day_signals(
    fleet: Fleet, d: int, rem: np.ndarray, warn: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Telematics for day d given each component's true remaining life in days, rem (V, C), and whether each part
    gives a warning before it fails, warn (V, C). A part that does not warn fails with no precursor.

    Returns battery voltage (V,), coolant temperature (V,), component DTC flags (V, C), and noise DTC flags (V, 3).
    The random draws depend only on (seed, d), so every policy sees the same noise on the same day.
    """
    rng = np.random.default_rng([fleet.cfg.seed, 1009, d])
    V = fleet.V
    t = fleet.ambient_f[d]
    w = warn.astype(float)
    ramp = lambda r, win: np.clip(1.0 - r / win, 0.0, 1.0)  # noqa: E731
    volt = (
        12.62
        + fleet.volt_off
        - 0.006 * max(0.0, 45.0 - t)
        - 0.40 * w[:, 1] * ramp(rem[:, 1], 45.0) ** 2
        - 0.25 * w[:, 3] * ramp(rem[:, 3], 21.0)
        + rng.normal(0, 0.08, V)
    )
    spike = (rng.random(V) < 0.01) * rng.uniform(6, 16, V)  # heavy load or a hot idle, not a failing pump
    cool = (
        fleet.cool_base
        + 0.12 * (t - 60.0)
        + 20.0 * w[:, 4] * ramp(rem[:, 4], 28.0) ** 2
        + spike
        + rng.normal(0, 3.0, V)
    )
    amp = np.array([c.dtc_amp for c in COMPONENTS])
    win = np.array([c.dtc_window for c in COMPONENTS])
    p = 0.002 + w * amp[None, :] * ramp(rem, win[None, :])
    on = fleet.operating[:, d][:, None]
    dtc = (rng.random(rem.shape) < p) & on
    noise = (rng.random((V, len(NOISE_DTCS))) < 0.004) & on
    return volt.astype(np.float32), cool.astype(np.float32), dtc, noise
