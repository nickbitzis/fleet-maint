"""Fleet, component, cost, and shop constants. Every number the simulation and the policies use lives here."""

from __future__ import annotations

from dataclasses import dataclass

COVARIATES = ("miles", "harsh", "idle", "urban")
# Fixed reference points for the usage covariates, so z-scores mean the same thing in every run.
Z_REF = {"miles": (110.0, 45.0), "harsh": (4.5, 2.0), "idle": (0.27, 0.08)}


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    label: str
    shape: float  # Weibull shape k
    scale_days: float  # Weibull scale for a reference van (all z = 0, not urban)
    beta: tuple[float, float, float, float]  # log-scale effect of (miles, harsh, idle, urban); negative = shorter life
    parts_usd: float
    labor_h: float
    lead_days: int  # supplier lead time for a reorder
    dtc: str  # fictional diagnostic trouble code this component raises before it fails
    dtc_text: str
    dtc_amp: float  # daily probability of the code on the last day of life
    dtc_window: float  # days before failure the code starts to appear
    collateral_usd: float  # extra damage when it fails on the road (a dead coolant pump cooks the engine)
    interval_unit: str  # what a fixed-interval policy counts: "miles" or "months"
    warn_prob: float  # share of failures that show a precursor at all (a puncture or an internal short does not)


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        "brakes", "Brakes", 3.0, 430, (-0.15, -0.30, 0.0, -0.15), 340, 2.0, 3,
        "FM-B220", "Brake pad wear sensor", 0.35, 18, 0, "miles", 0.9,
    ),
    ComponentSpec(
        "battery", "Battery", 2.6, 980, (0.0, 0.0, -0.12, -0.10), 215, 0.5, 4,
        "FM-P310", "Low cranking voltage", 0.12, 30, 0, "months", 0.7,
    ),
    ComponentSpec(
        "tires", "Tires", 2.2, 520, (-0.30, -0.08, 0.0, 0.05), 760, 1.5, 4,
        "FM-T405", "Tire pressure loss", 0.18, 8, 150, "miles", 0.45,
    ),
    ComponentSpec(
        "alternator", "Alternator", 1.9, 1150, (-0.10, 0.0, -0.18, 0.0), 485, 2.5, 7,
        "FM-E140", "Charging circuit low output", 0.25, 21, 0, "months", 0.7,
    ),
    ComponentSpec(
        "coolant_pump", "Coolant pump", 1.7, 1250, (-0.15, 0.0, -0.06, -0.05), 395, 3.0, 10,
        "FM-C510", "Coolant flow restricted", 0.15, 14, 1400, "months", 0.65,
    ),
)  # fmt: skip
COMP_NAMES = tuple(c.name for c in COMPONENTS)
NOISE_DTCS = ("FM-N901", "FM-N907", "FM-N915")  # unrelated codes that fire at random


@dataclass(frozen=True)
class RouteType:
    name: str
    share: float
    miles_mu: float
    miles_sd: float
    harsh_mu: float  # harsh braking events per 100 miles
    harsh_sd: float
    idle_mu: float  # idle share of engine hours
    idle_sd: float
    speed_mph: float
    stops: int
    stop_value_usd: float  # cost of a missed or redelivered stop


ROUTE_TYPES: tuple[RouteType, ...] = (
    RouteType("urban", 0.5, 75, 12, 6.5, 1.2, 0.34, 0.05, 17, 120, 7.0),
    RouteType("suburban", 0.3, 115, 15, 4.0, 0.8, 0.24, 0.04, 28, 80, 9.0),
    RouteType("highway", 0.2, 185, 25, 1.8, 0.5, 0.14, 0.03, 46, 25, 30.0),
)


@dataclass(frozen=True)
class Costs:
    labor_rate: float = 125.0
    emergency_labor_mult: float = 1.5  # unplanned work pulls a technician off other jobs
    tow_usd: float = 325.0
    downtime_per_day: float = 220.0  # spare van rental and dispatch for each operating day a van is down
    visit_downtime: float = 90.0  # half a day off the route for a planned visit
    peak_visit_downtime: float = 400.0  # pulling a van on its route's peak day
    missed_share: float = 0.5  # a roadside breakdown misses half the day's stops on average
    expected_down_days: float = 1.5  # used by the model policy to price a breakdown before it happens


@dataclass(frozen=True)
class Shop:
    bays: int = 7  # vans per day
    technicians: int = 5
    hours_per_tech: float = 8.0
    closed_weekdays: tuple[int, ...] = (6,)  # Sunday
    safety_days: int = 4  # reorder point covers lead time plus this many days of demand
    cover_days: int = 14  # order up to this many days of demand above the reorder point
    trailing_days: int = 28  # demand estimate looks back this far

    @property
    def labor_hours(self) -> float:
        return self.technicians * self.hours_per_tech


@dataclass(frozen=True)
class PolicyConfig:
    horizon_days: int = 14  # the "next period" for failure probability
    fixed_quantiles_sweep: tuple[float, ...] = (0.05, 0.10, 0.20)  # sensitivity runs: replace at these failure shares
    condition_volt: float = 12.30  # 7-day mean battery voltage below this flags battery and alternator
    condition_cool_rise: float = 12.0  # 7-day mean coolant temperature this far above the van's baseline
