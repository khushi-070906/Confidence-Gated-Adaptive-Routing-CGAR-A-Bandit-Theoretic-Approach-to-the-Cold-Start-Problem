"""
routers.py
----------
Four baseline routers for comparison against CGAR:

1. RoundRobinRouter       — stateless, zero exploration cost
2. LeastConnectionsRouter — stateless, load-sensitive
3. PureDQNRouter          — DQN router, no gating (baseline)
4. EpsGreedyDQNRouter     — DQN with decaying epsilon (ablation baseline)

Compatible with fast_qnet.QNetwork which takes:
    predict(state_matrix)            -> (n_active,)
    update(state_matrix, action, reward, next_state_matrix)
"""

import numpy as np
from cgar import QNetwork, ReplayBuffer


# ── Round Robin ────────────────────────────────────────────────────────────

class RoundRobinRouter:
    """Cycles through active backends in order. Stateless."""

    def __init__(self):
        self._counter = 0

    def route(self, active_backends) -> int:
        if not active_backends:
            return 0
        idx = self._counter % len(active_backends)
        self._counter += 1
        return idx

    def update(self, active_backends, action, reward, result):
        pass

    def notify_pool_change(self, new_ids, removed_ids):
        pass


# ── Least Connections ──────────────────────────────────────────────────────

class LeastConnectionsRouter:
    """Routes to backend with lowest current queue depth. Stateless."""

    def route(self, active_backends) -> int:
        if not active_backends:
            return 0
        return min(range(len(active_backends)),
                   key=lambda i: active_backends[i].queue_depth)

    def update(self, active_backends, action, reward, result):
        pass

    def notify_pool_change(self, new_ids, removed_ids):
        pass


# ── Pure DQN ───────────────────────────────────────────────────────────────

class PureDQNRouter:
    """
    Pure DQN router with fixed epsilon-greedy exploration.
    No gating — represents existing RL-gateway literature baseline.
    Uses fast_qnet.QNetwork via import from cgar.py.
    """

    def __init__(self, epsilon: float = 0.15, max_backends: int = 20,
                 n_features: int = 5, lr: float = 0.02, seed: int = 42):
        self.epsilon      = epsilon
        self.rng          = np.random.default_rng(seed)
        self.max_backends = max_backends
        self.qnet         = QNetwork(n_features, max_backends, lr=lr, seed=seed)
        self.replay       = ReplayBuffer(3000)
        self.batch_size   = 16
        self.train_every  = 20
        self.t            = 0

        self._prev_state  = None
        self._prev_action = None

        # Per-backend statistics for state building
        self.mu_hat     = {}
        self.n_visits   = {}
        self.sum_reward = {}

    def route(self, active_backends) -> int:
        self.t += 1
        self._ensure_tracked(active_backends)
        state = self._build_state(active_backends)

        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(len(active_backends)))
        else:
            q_vals = self.qnet.predict(state)          # ← no slot_indices
            action = int(np.argmax(q_vals))

        action = min(action, len(active_backends) - 1)
        self._prev_state  = state
        self._prev_action = action
        return action

    def update(self, active_backends, action, reward, result):
        action = min(action, len(active_backends) - 1)
        b   = active_backends[action]
        bid = b.id

        self.n_visits[bid]   = self.n_visits.get(bid, 0) + 1
        self.sum_reward[bid] = self.sum_reward.get(bid, 0.0) + reward
        self.mu_hat[bid]     = self.sum_reward[bid] / self.n_visits[bid]

        if self._prev_state is not None:
            next_state = self._build_state(active_backends)
            self.replay.push((self._prev_state, action, reward, next_state))

        if self.t % self.train_every == 0 and len(self.replay) >= self.batch_size:
            for (s, a, r, ns) in self.replay.sample(self.batch_size):
                n = min(s.shape[0], ns.shape[0])
                if n > 0:
                    self.qnet.update(s[:n], min(a, n-1), r, ns[:n])  # ← no slot args

    def notify_pool_change(self, new_ids, removed_ids):
        for bid in new_ids:
            if bid not in self.n_visits:
                self.n_visits[bid]   = 0
                self.mu_hat[bid]     = 0.0
                self.sum_reward[bid] = 0.0
        for bid in removed_ids:
            for d in (self.n_visits, self.mu_hat, self.sum_reward):
                d.pop(bid, None)

    def _ensure_tracked(self, active_backends):
        for b in active_backends:
            if b.id not in self.n_visits:
                self.n_visits[b.id]   = 0
                self.mu_hat[b.id]     = 0.0
                self.sum_reward[b.id] = 0.0

    def _build_state(self, active_backends) -> np.ndarray:
        rows = []
        for b in active_backends:
            bid = b.id
            rows.append([
                float(np.clip(self.mu_hat.get(bid, 0.0), -5.0, 0.0) / -5.0),
                min(1.0, self.n_visits.get(bid, 0) / 500.0),
                min(1.0, b.queue_depth / 20.0),
                float(np.clip(b.cpu, 0.0, 1.0)),
                0.0,   # no age tracking in pure DQN
            ])
        return np.array(rows, dtype=np.float32)


# ── Eps-Greedy DQN ─────────────────────────────────────────────────────────

class EpsGreedyDQNRouter(PureDQNRouter):
    """
    DQN with decaying epsilon — critical ablation baseline.
    Epsilon decays from epsilon_start → epsilon_min over decay_steps requests.
    If CGAR beats this, the adaptive gate contributes beyond a simple
    exploration schedule.
    """

    def __init__(self, epsilon_start: float = 0.9, epsilon_min: float = 0.05,
                 decay_steps: int = 2000, max_backends: int = 20,
                 n_features: int = 5, lr: float = 0.02, seed: int = 42):
        super().__init__(epsilon=epsilon_start, max_backends=max_backends,
                         n_features=n_features, lr=lr, seed=seed)
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        self.decay_steps   = decay_steps

    def route(self, active_backends) -> int:
        # Decay epsilon linearly
        progress    = min(1.0, self.t / self.decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_min - self.epsilon_start)
        return super().route(active_backends)