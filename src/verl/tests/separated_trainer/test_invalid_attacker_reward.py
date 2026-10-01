"""An invalid attacker rewrite must be excluded from the task-reward update but still pay the format penalty: with
FORMAT_REWARD_VALUE=0.5 a fixed task reward of 0 beat a valid rewrite the defender wins (-1.0 vs -0.5), so the
attacker learned to break the format. Run from the repo root:
    python -m pytest -q src/verl/tests/separated_trainer/test_invalid_attacker_reward.py
"""
import importlib.util
from pathlib import Path

import pytest

GAME_PY = Path(__file__).resolve().parents[2] / "verl" / "utils" / "reward_score" / "game.py"


@pytest.fixture
def game(monkeypatch):
    monkeypatch.setenv("FORMAT_REWARD_VALUE", "0.5")
    spec = importlib.util.spec_from_file_location("game_invalid_attacker", GAME_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VALID = "<think>plan</think><answer>attack</answer>"
NO_CLOSE = "<think>plan plan plan"
EMPTY = "<think>plan</think><answer></answer>"


def test_invalid_gets_group_mean_of_valid(game):
    # one seed, 4 rollouts: three valid rewrites the defender beats (-1.5) and one that half-succeeds (0)
    task = [-1.5, -1.5, 0.0, 0.0]
    invalid = [False, False, False, True]
    out = game.mask_invalid_attacker_task(task, invalid, [True] * 4, ["s"] * 4)
    assert out[:3] == [-1.5, -1.5, 0.0]
    assert out[3] == pytest.approx(-1.0)


def test_invalid_never_beats_valid_after_format_penalty(game):
    # defender wins every rollout: the invalid one must end below the valid ones, not above as with a fixed 0
    task = game.mask_invalid_attacker_task([-1.5, -1.5, -1.5, 0.0], [False, False, False, True], [True] * 4, ["s"] * 4)
    fmt = [game.format_reward_func(c) for c in (VALID, VALID, VALID, NO_CLOSE)]
    total = [t + f for t, f in zip(task, fmt)]
    assert total[:3] == [-1.0, -1.0, -1.0]
    assert total[3] == pytest.approx(-2.0)


def test_groups_are_independent(game):
    task = [-1.5, 0.0, 1.5, 0.0]
    invalid = [False, True, False, True]
    out = game.mask_invalid_attacker_task(task, invalid, [True] * 4, ["a", "a", "b", "b"])
    assert out == [-1.5, -1.5, 1.5, 1.5]


def test_judge_invalid_samples_do_not_feed_the_mean(game):
    out = game.mask_invalid_attacker_task([-1.5, 7.0, 0.0], [False, False, True], [True, False, True], ["s"] * 3)
    assert out[2] == pytest.approx(-1.5)


def test_all_invalid_group_gets_zero(game):
    out = game.mask_invalid_attacker_task([0.0, 0.0], [True, True], [True, True], ["s", "s"])
    assert out == [0.0, 0.0]


def test_valid_samples_untouched(game):
    task = [0.3, -0.7]
    assert game.mask_invalid_attacker_task(task, [False, False], [True, True], ["s", "t"]) == task


@pytest.mark.parametrize("content", [NO_CLOSE, EMPTY, "no tags at all"])
def test_format_penalty_for_invalid_rewrites(game, content):
    assert game.format_reward_func(content) == -0.5
    assert game.format_reward_func(VALID) == 0.5
