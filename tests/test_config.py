from pathlib import Path

from t2i_exp3.runner import validate_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_is_valid():
    config, count = validate_experiment(ROOT / "configs/smoke.json")
    assert count == 3
    assert config.attempts_per_task == 4
    assert config.feedback_mode.value == "target_observable"
