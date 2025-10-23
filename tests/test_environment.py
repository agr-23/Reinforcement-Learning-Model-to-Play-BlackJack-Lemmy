import pytest

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.env.blackjack_env import (
    BlackjackEnv, Rules,
    ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT
)

# ---------- Fixtures ----------

@pytest.fixture
def env_default():
    """Default environment with 1 deck and deterministic seed."""
    return BlackjackEnv(Rules(n_decks=6, burn_first_card=False), seed=7)

# ---------- State structure ----------

def test_reset_returns_full_obs(env_default):
    obs, r, done, info = env_default.reset()
    expected_keys = {
        "player_total", "usable_ace", "dealer_up",
        "can_double", "can_split",
        "true_count", "run_count", "cards_remaining",
        "hand_index", "num_hands"
    }
    assert expected_keys.issubset(obs.keys())
    assert isinstance(obs["player_total"], int)
    assert done in (False, True)

# ---------- Deal / Hand resolution ----------

def test_player_blackjack_payout():
    env = BlackjackEnv(Rules(n_decks=6, peek=True, burn_first_card=False), seed=7)
    obs, r, done, info = env.reset()
    if done and info.get("reason") == "player_blackjack":
        assert r == pytest.approx(env.rules.blackjack_payout)

def test_dealer_blackjack_peek():
    env = BlackjackEnv(Rules(n_decks=6, peek=True, burn_first_card=False), seed=13)
    obs, r, done, info = env.reset()
    if done and info.get("reason") == "dealer_blackjack_peek":
        assert r in (0.0, -1.0)

def test_split_creates_two_hands():
    env = BlackjackEnv(Rules(n_decks=6, burn_first_card=False), seed=99)
    obs, r, done, info = env.reset()
    if obs["can_split"]:
        obs, r, done, info = env.step(ACTION_SPLIT)
        assert obs["num_hands"] == 2
        assert "split" in info

def test_double_marks_hand_done():
    env = BlackjackEnv(Rules(n_decks=6, burn_first_card=False), seed=21)
    obs, r, done, info = env.reset()
    if obs["can_double"]:
        obs, r, done, info = env.step(ACTION_DOUBLE)
        assert env.hands_doubled[0] is True
        assert done or env.hands_done[0] is True

def test_hit_until_bust_forces_done():
    env = BlackjackEnv(Rules(n_decks=6, burn_first_card=False), seed=33)
    obs, r, done, info = env.reset()
    for _ in range(10):
        if done:
            break
        obs, r, done, info = env.step(ACTION_HIT)
    assert done is True or obs["player_total"] >= 21

# ---------- Count updates ----------

def test_run_count_and_cards_remaining_update(env_default):
    obs, r, done, info = env_default.reset()
    while done:  # asegurar que no arranque terminado
        obs, r, done, info = env_default.reset()

    initial_remaining = obs["cards_remaining"]
    obs, r, done, info = env_default.step(ACTION_HIT)
    assert obs["cards_remaining"] < initial_remaining
    assert isinstance(obs["run_count"], int)



# ---------- Action constraints ----------

def test_illegal_action_forced_to_stand(env_default):
    obs, r, done, info = env_default.reset()
    while done:
        obs, r, done, info = env_default.reset()

    obs, r, done, info = env_default.step(99)
    assert "illegal_action_forced_to_stand" in info
    assert done in (False, True)


def test_available_actions_respect_rules(env_default):
    obs, r, done, info = env_default.reset()
    while done:
        obs, r, done, info = env_default.reset()

    acts = env_default.available_actions()
    assert ACTION_HIT in acts
    assert ACTION_STAND in acts
    if obs["can_double"]:
        assert ACTION_DOUBLE in acts
    if obs["can_split"]:
        assert ACTION_SPLIT in acts



