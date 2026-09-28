import pytest
from src.data.validation import assert_no_leakage


def test_allowed_features_pass():
    assert_no_leakage(["Описание 2", "Услуга", "Приоритет"])


def test_forbidden_features_fail():
    with pytest.raises(ValueError):
        assert_no_leakage(["Описание 2", "Результат работ"])
