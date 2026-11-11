import numpy as np

from fleetmaint.config import Shop
from fleetmaint.scheduler import Proposal, demand_rate, plan_day, reorder_point, reorder_qty

V, C = 10, 5
STOCK = np.full(C, 50)


def prop():
    return Proposal.empty(V, C)


def nofail():
    return np.zeros((V, C), bool), np.zeros((V, C), int)


def test_bays_limit():
    p = prop()
    p.score[:, 1] = np.arange(V)  # battery jobs, 0.5 h each
    failed, fd = nofail()
    plan = plan_day(failed, fd, p, np.zeros(V, bool), STOCK, Shop(bays=3, technicians=5), today=10)
    assert plan.bays_used == 3 and len({j.van for j in plan.jobs}) == 3
    assert [j.van for j in plan.jobs] == [9, 8, 7]  # highest score first


def test_hours_limit():
    p = prop()
    p.score[:, 4] = 1.0  # coolant pumps, 3 h each
    failed, fd = nofail()
    shop = Shop(bays=10, technicians=1, hours_per_tech=8.0)
    plan = plan_day(failed, fd, p, np.zeros(V, bool), STOCK, shop, today=10)
    assert len(plan.jobs) == 2 and plan.hours_used == 6.0 <= shop.labor_hours


def test_emergencies_first_oldest_first_and_not_same_day():
    failed, fd = nofail()
    failed[2, 0], fd[2, 0] = True, 8
    failed[5, 3], fd[5, 3] = True, 6
    failed[7, 1], fd[7, 1] = True, 10  # failed today: towed in, diagnosed tomorrow
    p = prop()
    p.score[0, 0] = 99.0
    plan = plan_day(failed, fd, p, np.zeros(V, bool), STOCK, Shop(bays=2), today=10)
    assert [(j.van, j.kind) for j in plan.jobs] == [(5, "emergency"), (2, "emergency")]


def test_peak_day_skip_unless_urgent():
    p = prop()
    p.score[0, 0] = p.score[1, 0] = 1.0
    p.urgent[1, 0] = True
    peak = np.zeros(V, bool)
    peak[[0, 1]] = True
    failed, fd = nofail()
    plan = plan_day(failed, fd, p, peak, STOCK, Shop(), today=10)
    assert [j.van for j in plan.jobs] == [1] and plan.skipped_peak == 1


def test_grouping_uses_one_bay():
    p = prop()
    p.score[4, 0] = 5.0
    p.opportunistic[4, [1, 2]] = True
    p.opportunistic[6, 1] = True  # van 6 is not coming in
    failed, fd = nofail()
    plan = plan_day(failed, fd, p, np.zeros(V, bool), STOCK, Shop(), today=10)
    assert sorted((j.comp, j.kind) for j in plan.jobs) == [(0, "planned"), (1, "grouped"), (2, "grouped")]
    assert plan.bays_used == 1


def test_grouping_onto_a_breakdown():
    failed, fd = nofail()
    failed[3, 2], fd[3, 2] = True, 4
    p = prop()
    p.opportunistic[3, 1] = True
    plan = plan_day(failed, fd, p, np.zeros(V, bool), STOCK, Shop(), today=10)
    assert {(j.comp, j.kind) for j in plan.jobs} == {(2, "emergency"), (1, "grouped")} and plan.bays_used == 1


def test_no_part_no_job():
    failed, fd = nofail()
    failed[0, 3], fd[0, 3] = True, 2
    p = prop()
    p.score[1, 3] = 3.0
    stock = STOCK.copy()
    stock[3] = 1
    plan = plan_day(failed, fd, p, np.zeros(V, bool), stock, Shop(), today=10)
    assert [(j.van, j.kind) for j in plan.jobs] == [(0, "emergency")]  # the breakdown gets the last alternator
    stock[3] = 0
    plan = plan_day(failed, fd, p, np.zeros(V, bool), stock, Shop(), today=10)
    assert plan.jobs == [] and plan.skipped_parts == 2 and plan.bays_used == 0


def test_reorder_point_and_quantity_by_hand():
    shop = Shop(safety_days=4, cover_days=14)
    # 1.2 parts a day, 3-day lead: s = ceil(1.2 x 7) = 9, S = 9 + ceil(1.2 x 14) = 26
    assert reorder_point(1.2, 3, 4) == 9
    assert reorder_qty(5, 3, 1.2, 3, shop) == 26 - 8
    assert reorder_qty(6, 4, 1.2, 3, shop) == 0  # position 10 is above s
    assert reorder_qty(0, 0, 0.01, 10, shop) == 2  # slow movers: s = 1, S = 2


def test_demand_rate_uses_trailing_usage():
    used = np.zeros(100, int)
    assert demand_rate(used, 99, 0.5, 28) == 0.5
    used[72:100] = 2
    assert demand_rate(used, 99, 0.5, 28) == 2.0
