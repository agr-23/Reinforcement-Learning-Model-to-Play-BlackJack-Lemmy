
import pytest
import numpy as np
import sys, os
from agents.qlearning import QLearningAgent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_bellman_update_sanity():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    s, a, sp = "s3", "hit", "s4"
    r = 1.0

    # configuration of Q inicial table
    agent.Q = {
        s: {a: 0.0},
        sp: {"hit": 2.0, "stand": 1.0}
    }

    legal_next = ["hit", "stand"]
    agent.update(s, a, r, sp, legal_next, done=False)

    expected = 0.0 + 0.5 * (1.0 + 0.9 * 2.0 - 0.0)
    assert pytest.approx(agent.Q[s][a], rel=1e-6) == expected

