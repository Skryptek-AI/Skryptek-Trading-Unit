"""
Deep Q-Network (DQN) Agent
===========================
- Double DQN to reduce overestimation bias
- Prioritised Experience Replay (PER) for faster learning
- Dueling architecture (value + advantage streams)
- Epsilon-greedy exploration with decay
- Periodic target-network sync
"""

import numpy as np
import random
import json
import os
from collections import deque
from typing import List, Tuple, Optional

# ── Optional GPU acceleration (falls back to CPU seamlessly) ──────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────
#  NEURAL NETWORK  (Dueling DQN)
# ─────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Dueling architecture separates state-value V(s) from
    advantage A(s,a), making learning more stable.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        shared   = self.shared(x)
        value    = self.value_stream(shared)
        advantage= self.advantage_stream(shared)
        q        = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


# ─────────────────────────────────────────────
#  PRIORITISED REPLAY BUFFER
# ─────────────────────────────────────────────

class PrioritisedReplayBuffer:
    """
    Samples transitions proportional to TD-error magnitude
    so the agent learns more from surprising transitions.
    """

    def __init__(self, capacity: int = 50_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha    = alpha
        self.buffer   : List[tuple] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos      = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos]      = (state, action, reward, next_state, done)
        self.priorities[self.pos]  = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        n      = len(self.buffer)
        prios  = self.priorities[:n] ** self.alpha
        probs  = prios / prios.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32),
                indices,
                weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
#  NUMPY FALLBACK (no PyTorch)
# ─────────────────────────────────────────────

class SimpleQTable:
    """
    Lightweight Q-table fallback using discretised state bins.
    Used automatically if PyTorch is not installed.
    """

    def __init__(self, obs_dim: int, n_actions: int, n_bins: int = 5):
        self.n_actions = n_actions
        self.n_bins    = n_bins
        self.obs_dim   = obs_dim
        self.q: dict   = {}

    def _discretise(self, obs: np.ndarray) -> tuple:
        binned = np.digitize(np.clip(obs, -3, 3),
                             np.linspace(-3, 3, self.n_bins))
        # Use only first 6 features to keep state space manageable
        return tuple(binned[:6])

    def predict(self, obs: np.ndarray) -> np.ndarray:
        key = self._discretise(obs)
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions)
        return self.q[key]

    def update(self, obs, action, target):
        key = self._discretise(obs)
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions)
        self.q[key][action] = target


# ─────────────────────────────────────────────
#  DQN AGENT
# ─────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN agent with:
    - Prioritised Experience Replay
    - Dueling network (when PyTorch available)
    - Epsilon-greedy exploration with cosine annealing
    - Target network with soft/hard updates
    - Checkpoint save/load
    """

    def __init__(
        self,
        obs_dim:          int,
        n_actions:        int   = 3,
        lr:               float = 3e-4,
        gamma:            float = 0.99,
        epsilon_start:    float = 1.0,
        epsilon_end:      float = 0.05,
        epsilon_decay:    float = 0.995,
        batch_size:       int   = 64,
        replay_capacity:  int   = 50_000,
        target_update:    int   = 200,     # steps between hard target updates
        hidden_size:      int   = 256,
        checkpoint_dir:   str   = "checkpoints",
    ):
        self.obs_dim        = obs_dim
        self.n_actions      = n_actions
        self.gamma          = gamma
        self.epsilon        = epsilon_start
        self.epsilon_end    = epsilon_end
        self.epsilon_decay  = epsilon_decay
        self.batch_size     = batch_size
        self.target_update  = target_update
        self.checkpoint_dir = checkpoint_dir
        self.steps          = 0
        self.learn_steps    = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.online_net = DuelingDQN(obs_dim, n_actions, hidden_size).to(self.device)
            self.target_net = DuelingDQN(obs_dim, n_actions, hidden_size).to(self.device)
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_net.eval()
            self.optimizer  = optim.Adam(self.online_net.parameters(), lr=lr)
            self.replay     = PrioritisedReplayBuffer(replay_capacity)
            print(f"[Agent] DQN with Dueling network on {self.device}")
        else:
            self.q_table = SimpleQTable(obs_dim, n_actions)
            self.replay  = deque(maxlen=replay_capacity)
            print("[Agent] PyTorch not found — using Q-table fallback")

        self.loss_history    = []
        self.epsilon_history = []

    # ── action selection ───────────────────────────────────────────────

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        if TORCH_AVAILABLE:
            with torch.no_grad():
                t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q = self.online_net(t)
                return int(q.argmax().item())
        else:
            q = self.q_table.predict(obs)
            return int(np.argmax(q))

    # ── store transition ───────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        if TORCH_AVAILABLE:
            self.replay.push(state, action, reward, next_state, done)
        else:
            self.replay.append((state, action, reward, next_state, done))
        self.steps += 1

    # ── learn from replay ─────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if TORCH_AVAILABLE:
            return self._learn_torch()
        else:
            return self._learn_qtable()

    def _learn_torch(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        beta = min(1.0, 0.4 + self.learn_steps * 6e-6)
        (states, actions, rewards, next_states,
         dones, indices, weights) = self.replay.sample(self.batch_size, beta)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)
        w  = torch.FloatTensor(weights).to(self.device)

        # Double DQN: online selects action, target evaluates value
        q_current = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_actions = self.online_net(ns).argmax(1)
            q_next       = self.target_net(ns).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            q_target     = r + self.gamma * q_next * (1 - d)

        td_errors = (q_target - q_current).detach().cpu().numpy()
        loss = (w * F.smooth_l1_loss(q_current, q_target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.replay.update_priorities(indices, td_errors)

        # Hard target update
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.learn_steps += 1
        loss_val = float(loss.item())
        self.loss_history.append(loss_val)
        return loss_val

    def _learn_qtable(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        batch = random.sample(self.replay, min(self.batch_size, len(self.replay)))
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            q_vals  = self.q_table.predict(state)
            nq_vals = self.q_table.predict(next_state)
            target  = reward + (0 if done else self.gamma * np.max(nq_vals))
            error   = target - q_vals[action]
            self.q_table.update(state, action, q_vals[action] + 0.1 * error)
            total_loss += abs(error)
        return total_loss / self.batch_size

    # ── epsilon decay ─────────────────────────────────────────────────

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    # ── checkpoint ────────────────────────────────────────────────────

    def save(self, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"agent_{tag}.pt")
        if TORCH_AVAILABLE:
            torch.save({
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optim":  self.optimizer.state_dict(),
                "epsilon":self.epsilon,
                "steps":  self.steps,
            }, path)
        else:
            import pickle
            with open(path.replace(".pt", ".pkl"), "wb") as f:
                pickle.dump({"q": self.q_table.q, "epsilon": self.epsilon}, f)
        print(f"[Agent] Saved → {path}")

    def load(self, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"agent_{tag}.pt")
        if TORCH_AVAILABLE and os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.online_net.load_state_dict(ckpt["online"])
            self.target_net.load_state_dict(ckpt["target"])
            self.optimizer.load_state_dict(ckpt["optim"])
            self.epsilon = ckpt["epsilon"]
            self.steps   = ckpt["steps"]
            print(f"[Agent] Loaded ← {path}")
        else:
            pkl = path.replace(".pt", ".pkl")
            if os.path.exists(pkl):
                import pickle
                with open(pkl, "rb") as f:
                    d = pickle.load(f)
                self.q_table.q = d["q"]
                self.epsilon   = d["epsilon"]
