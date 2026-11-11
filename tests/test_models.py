import numpy as np

from fleetmaint import models as M
from fleetmaint.config import COMPONENTS
from fleetmaint.features import FEATURES, compute, observed_covariates, survival_p14

IDX = {c.name: i for i, c in enumerate(COMPONENTS)}


def test_survival_data_respects_the_cutoff(bundle):
    cut = bundle.cfg.gen.cutoff
    z = bundle.models.zobs
    for c in range(5):
        t, e, X, entry = M.survival_data(bundle.history.spells, z, cut, c)
        assert (t > entry).all() and (entry >= 0).all() and len(t) == len(X)
        assert e.sum() > 20 and (~e).sum() > 20  # failures and censored parts
        assert (t - entry <= cut).all()  # observed time never runs past the cutoff


def test_time_split_has_no_leakage(bundle):
    """Rebuild training snapshots after scrambling everything from the cutoff on: telematics and the fate of every
    part that was still running. Features and labels must not change."""
    b, f = bundle, bundle.fleet
    cut = b.cfg.gen.cutoff
    inst = b.history.install_hist
    days = range(60, cut - M.HORIZON, 25)
    end, ev = M.spell_lookup(b.history.spells, inst, f.D)
    base = M.snapshots(f, b.hist_sig, inst, end, ev, days, b.models.zobs, b.models.aft)
    sig = b.hist_sig.copy()
    rng = np.random.default_rng(0)
    sig.volt[:, cut:] = rng.normal(11, 1, sig.volt[:, cut:].shape)
    sig.dtc[:, :, cut:] = True
    spells = [
        (v, c, i, (e + 37 if e >= cut else e), (not fl if e >= cut else fl)) for v, c, i, e, fl in b.history.spells
    ]
    end2, ev2 = M.spell_lookup(spells, inst, f.D)
    other = M.snapshots(f, sig, inst, end2, ev2, days, b.models.zobs, b.models.aft)
    assert np.array_equal(base.X, other.X)
    for c in range(5):
        k1, y1 = base.labels(c)
        k2, y2 = other.labels(c)
        assert np.array_equal(k1, k2) and np.array_equal(y1, y2)
    assert max(days) + M.HORIZON < cut


def test_spell_lookup_points_at_the_part_in_service(bundle):
    b = bundle
    inst = b.history.install_hist
    end, ev = M.spell_lookup(b.history.spells, inst, b.fleet.D)
    v, c, i, e, fl = next(s for s in b.history.spells if s[4] and s[2] > 10)
    assert end[v, c, i + 1] == e and ev[v, c, i + 1]
    assert (inst[v, c, i + 1 : e + 1] == i).all()


def test_labels_drop_censored_windows():
    snap = M.Snapshots(
        np.zeros((1, 4, 1)), np.array([[5, 5, 30, 0]]), np.array([[True, False, False, True]]), np.zeros(4), np.zeros(4)
    )
    known, y = snap.labels(0)
    assert known.tolist() == [True, False, True, False] and y.tolist() == [True, False, False, True]


def test_usage_effects_have_the_right_sign(bundle):
    s = {r["component"]: r for r in bundle.results["survival"]}
    assert s["brakes"]["beta"]["harsh"] < -0.15
    assert s["tires"]["beta"]["miles"] < -0.1
    assert 2.4 < s["brakes"]["shape"] < 3.6


def test_both_models_beat_chance_and_gbm_wins_on_brakes(bundle):
    ev = {e["component"]: e for e in bundle.results["evaluation"]}
    for e in ev.values():
        assert e["survival"]["auc"] > 0.55 and e["gbm"]["auc"] > 0.55
        assert 0.5 < e["survival"]["cindex"] < 1
        assert e["gbm"]["brier"] < 0.05
    assert ev["brakes"]["gbm"]["auc"] > ev["brakes"]["survival"]["auc"]


def test_voltage_falls_before_battery_failure(bundle):
    li = bundle.results["leading_indicators"]["battery"]
    assert li["mean"][-1] < -0.1 and abs(li["mean"][0]) < 0.03


def test_reliability_and_ece():
    p = np.repeat([0.01, 0.3], 500)
    y = np.concatenate([np.zeros(495), np.ones(5), np.zeros(350), np.ones(150)])
    rows = M.reliability(p, y)
    assert [round(r["obs"], 2) for r in rows] == [0.01, 0.3] and sum(r["n"] for r in rows) == 1000
    rng = np.random.default_rng(0)
    q = rng.random(100_000)
    assert M.ece(q, rng.random(100_000) < q) < 0.01  # calibrated by construction
    assert M.ece(q, rng.random(100_000) < q / 2) > 0.2


def test_features_shape_and_survival_column(bundle):
    b = bundle
    d = b.fleet.D - 1
    st = b.model_state
    age = d - st.install
    sp = survival_p14(b.models.aft, np.maximum(age, 0), b.models.zobs)
    X = compute(b.fleet, b.model_sig, d, np.maximum(age, 0), st.install, b.models.zobs, sp)
    assert X.shape == (5, b.fleet.V, len(FEATURES)) and np.isfinite(X).all()
    assert np.allclose(X[:, :, FEATURES.index("survival_p14")], sp.T)
    p = b.models.predict_p14(b.fleet, b.model_sig, d, age, st.install)
    assert p.shape == (b.fleet.V, 5) and (p >= 0).all() and (p <= 1).all()


def test_observed_covariates_track_truth(bundle):
    f = bundle.fleet
    z = observed_covariates(f, 0, bundle.cfg.gen.cutoff)
    for j in range(3):
        assert np.corrcoef(z[:, j], f.z_true[:, j])[0, 1] > 0.95
