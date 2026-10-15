from types import SimpleNamespace

import pytest

from t2i_exp3 import qwen_image_target as guards


def test_target_clip_checker_implementation_removed():
    assert not hasattr(guards, 'ClipTextImageOutputChecker')
    with pytest.raises(ValueError):
        guards._build_output_checker({'mode': 'clip_text_image_checker'}, SimpleNamespace())


def test_clip_semantic_scorer_retained():
    from t2i_exp3.clip_score import ClipImageTextScorer
    assert callable(ClipImageTextScorer.score)
