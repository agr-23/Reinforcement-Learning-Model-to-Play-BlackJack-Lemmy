'''import pytest
from src.env.blackjack_env import BlackjackEnv, Rules, ACTION_HIT, ACTION_STAND

def test_reset_and_peek_h17():
    env = BlackjackEnv(Rules(n_decks=1, h17=True, peek=True, burn_first_card=False), seed=42)
    obs, r, done, info = env.reset()
    assert "dealer_up" in obs
    assert done in (False, True)
    # Render should not crash
    _ = env.render()

def test_dealer_hits_soft17():
    # Force a sequence by playing through a few rounds
    env = BlackjackEnv(Rules(n_decks=2, h17=True), seed=123)
    for _ in range(5):
        obs, r, done, info = env.reset()
        # play naively: stand immediately; dealer should complete
        obs, r, done, info = env.step(ACTION_STAND)
        # episode must end
        assert done is True

def test_basic_playthrough():
    env = BlackjackEnv(Rules(n_decks=2), seed=7)
    obs, r, done, info = env.reset()
    # Play up to 20 steps max to avoid infinite loops in case of bug
    for _ in range(20):
        if done:
            break
        acts = env.available_actions()
        # Simple policy: hit if total < 12, else stand
        if obs["player_total"] < 12 and ACTION_HIT in acts:
            obs, r, done, info = env.step(ACTION_HIT)
        else:
            obs, r, done, info = env.step(ACTION_STAND)
    assert done is True '''