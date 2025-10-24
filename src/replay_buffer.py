# src/replay_buffer.py
import random
import pickle
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=200_000, seed=None):
        self.capacity = int(capacity)
        self.buffer = []
        self.pos = 0
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        # store raw numpy arrays (float32) and python scalars
        s = np.array(state, dtype=np.float32)
        ns = np.array(next_state, dtype=np.float32)
        item = (s, int(action), float(reward), ns, bool(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        # Return a list of transitions (s,a,r,s',done) as used by train_dqn.py
        return self.rng.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    # --- checkpoint helpers ---
    def save_state(self):
        """Return a pickle-serializable snapshot of the buffer (may be large)."""
        return {
            "capacity": self.capacity,
            "buffer": self.buffer,
            "pos": self.pos
        }

    def load(self, state):
        """Load previously saved buffer state (from checkpoint)."""
        if not state:
            return
        self.capacity = int(state.get("capacity", self.capacity))
        self.buffer = state.get("buffer", [])  # list of tuples
        self.pos = int(state.get("pos", 0))