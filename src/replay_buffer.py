"""
Experience Replay Buffer Implementation for Deep Q-Learning

This module implements a circular buffer for storing and sampling transitions in 
reinforcement learning, specifically designed for the DQN (Deep Q-Network) agent
training in the Blackjack environment.

Key Features:
- Circular buffer implementation with fixed capacity
- Efficient memory usage with numpy arrays
- Random sampling for experience replay
- Checkpoint support for saving/loading buffer state
- Type conversion handling for numerical stability

The buffer stores transitions as tuples of (state, action, reward, next_state, done):
- state: Current state as numpy float32 array
- action: Integer representing the action taken
- reward: Float value of the reward received
- next_state: Next state as numpy float32 array
- done: Boolean indicating if the episode ended

Usage Example:
    buffer = ReplayBuffer(capacity=200_000, seed=42)
    
    # Store transition
    buffer.push(state, action, reward, next_state, done)
    
    # Sample batch for training
    batch = buffer.sample(batch_size=32)
    
    # Save/load for checkpointing
    state = buffer.save_state()
    buffer.load(state)
"""

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