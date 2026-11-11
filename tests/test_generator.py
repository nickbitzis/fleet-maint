from datetime import date

import numpy as np

from fleetmaint.config import COMPONENTS
from fleetmaint.generator import GenConfig, day_signals, generate, holidays, true_scale, zscores

IDX = {c.name: i for i, c in enumerate(COMPONENTS)}


def test_shapes_and_dates():
    f = generate()
    assert f.V == 400 and f.D == 1096
    assert f.dates[0] == date(2023, 11, 11) and f.dates[-1] == date(2026, 11, 10)
    assert f.dates[f.cfg.cutoff] == date(2025, 11, 11) and f.D - f.cfg.cutoff == 365
    assert f.miles.shape == (400, 1096) and f.scale.shape == (400, 5)


def test_same_seed_same_fleet(tiny):
    again = generate(tiny.cfg)
    assert np.array_equal(again.miles, tiny.miles) and np.array_equal(again.u_life, tiny.u_life)
    other = generate(GenConfig(seed=4, vans=24, days=400, heldout_days=100))
    assert not np.array_equal(other.miles, tiny.miles)


def test_no_driving_on_sundays_and_holidays(tiny):
    off = (tiny.weekday == 6) | tiny.holiday
    assert tiny.miles[:, off].sum() == 0 and tiny.harsh[:, off].sum() == 0
    assert (tiny.miles[:, ~off] > 0).all()
    assert date(2024, 11, 28) in holidays(2024)  # Thanksgiving, fourth Thursday


def test_peak_days_follow_route_weekday(tiny):
    for v in range(tiny.V):
        wd = tiny.route_peak_wd[tiny.route[v]]
        days = np.flatnonzero(tiny.peak[v])
        dec = np.array([tiny.dates[d].month == 12 for d in days])
        assert (tiny.weekday[days][~dec] == wd).all()
    assert not (tiny.peak & ~tiny.operating).any()


def test_duty_cycles_differ():
    f = generate()
    urban, highway = f.rtype == 0, f.rtype == 2
    assert f.miles[urban].mean() < f.miles[highway].mean()
    hr = f.harsh.sum(1) / f.miles.sum(1)
    assert hr[urban].mean() > 2 * hr[highway].mean()


def test_usage_shortens_life():
    """Ground truth: heavier usage gives a shorter Weibull scale, and the drawn lives follow it."""
    f = generate()
    z = f.z_true
    hi, lo = z[:, 0] > np.quantile(z[:, 0], 0.75), z[:, 0] < np.quantile(z[:, 0], 0.25)
    t = IDX["tires"]
    assert f.scale[hi, t].mean() < 0.8 * f.scale[lo, t].mean()
    lives = np.array([[f.life(v, t, k, 0) for k in range(20)] for v in range(f.V)])
    assert lives[hi].mean() < 0.8 * lives[lo].mean()
    hh, hl = z[:, 1] > np.quantile(z[:, 1], 0.75), z[:, 1] < np.quantile(z[:, 1], 0.25)
    b = IDX["brakes"]
    assert f.scale[hh, b].mean() < 0.75 * f.scale[hl, b].mean()


def test_true_scale_matches_spec():
    z = zscores([110], [4.5], [0.27], [False])
    assert np.allclose(z, 0)
    assert np.allclose(true_scale(z)[0], [c.scale_days for c in COMPONENTS])


def test_life_draws_are_deterministic_and_streams_differ(tiny):
    assert tiny.life(3, 1, 2, 0) == tiny.life(3, 1, 2, 0)
    assert tiny.life(3, 1, 2, 0) != tiny.life(3, 1, 2, 1)
    # Weibull scale check: median life over many draws is near scale * ln(2)^(1/k)
    f = generate()
    c = IDX["battery"]
    ratio = np.array([f.life(v, c, k, 0) / f.scale[v, c] for v in range(f.V) for k in range(10)])
    assert abs(np.median(ratio) - np.log(2) ** (1 / COMPONENTS[c].shape)) < 0.03


def test_battery_voltage_decays_before_failure(tiny):
    rem = np.full((tiny.V, 5), 500.0)
    warn = np.ones((tiny.V, 5), bool)
    v_ok, *_ = day_signals(tiny, 100, rem, warn)
    rem2 = rem.copy()
    rem2[:, IDX["battery"]] = 1.0
    v_bad, *_ = day_signals(tiny, 100, rem2, warn)
    assert (v_ok - v_bad).mean() > 0.3
    v_quiet, *_ = day_signals(tiny, 100, rem2, np.zeros_like(warn))
    assert np.allclose(v_quiet, v_ok)  # a part that gives no warning shows nothing


def test_dtc_rate_rises_near_failure_and_noise_is_shared(tiny):
    warn = np.ones((tiny.V, 5), bool)
    near = np.full((tiny.V, 5), 0.5)
    far = np.full((tiny.V, 5), 900.0)
    days = [d for d in range(50, 150) if tiny.operating[0, d]]
    hits_near = sum(day_signals(tiny, d, near, warn)[2].sum() for d in days)
    hits_far = sum(day_signals(tiny, d, far, warn)[2].sum() for d in days)
    assert hits_near > 20 * max(hits_far, 1)
    a, b = day_signals(tiny, 77, far, warn), day_signals(tiny, 77, far, warn)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))  # common random numbers
