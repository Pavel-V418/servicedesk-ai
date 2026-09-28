import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pytest

from sla.risk import SLARiskEstimator
from sla.sla_analytics import SLAAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    return SLAAnalyzer()


@pytest.fixture(scope="module")
def estimator(analyzer):
    return SLARiskEstimator(analyzer)


def test_dashboard_is_valid_json(analyzer):
    data = json.loads(analyzer.get_dashboard_data())
    assert data["summary"]["total_requests"] > 0


def test_priority_matching(estimator):
    assert estimator._match_priority("Наивысший") == "(1) Наивысший"
    assert estimator._match_priority("(3) Средний") == "(3) Средний"
    assert estimator._match_priority("3") == "(3) Средний"
    assert estimator._match_priority("") is None


def test_highest_priority_is_riskier_than_low(estimator):
    high = estimator.assess(priority="Наивысший", line_code="L3")
    low = estimator.assess(priority="Низкий", line_code="L1")
    assert high["sla_risk_score"] > low["sla_risk_score"]
    assert high["sla_risk_level"] == "высокий"
    assert low["sla_risk_level"] == "низкий"


def test_works_without_inputs(estimator):
    r = estimator.assess()
    assert r["sla_risk_level"] in {"низкий", "средний", "высокий"}
    assert "Приоритет не указан" in r["sla_risk_reason"]
