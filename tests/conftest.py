"""Shared fixtures: one small end to end build (120 vans, a 2-bay shop) and a tiny fleet for unit tests."""

from __future__ import annotations

import pytest

from fleetmaint.generator import GenConfig, generate
from fleetmaint.pipeline import PipelineConfig, build


@pytest.fixture(scope="session")
def bundle():
    return build(PipelineConfig.small(), log=lambda m: None)


@pytest.fixture(scope="session")
def tiny():
    return generate(GenConfig(seed=3, vans=24, days=400, heldout_days=100))
