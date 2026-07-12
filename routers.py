"""
routers.py
----------
Four baseline routers for comparison against CGAR:

1. RoundRobinRouter       — stateless, zero exploration cost
2. LeastConnectionsRouter — stateless, load-sensitive
3. PureDQNRouter          — same Q-network as CGAR but no gating
4. EpsGreedyDQNRouter     — DQN with decaying epsilon exploration
                            (critical ablation: proves CGAR's gate
                             contributes beyond a simple decay schedule)

--------------------------------------------------------------------------
FIXES APPLIED (see accompanying review notes on cgar.py):

PureDQNRouter shares the QNetwork class with CGARRouter and had the exact
same bug: Q-values were predicted/updated by each backend's *position* in
the active_backends list for a given call, not by a persistent identity.
Since the position mapping changes every time the pool churns or the list
reorders, W-rows got silently applied to the wrong backend after every
churn event — this is a strong candidate for why Pure DQN was the
second-worst performer on P99 latency (116.3ms), right behind CGAR
(117.3ms) pre-fix, out of all 8 methods.

Applied here:
  1. Persistent backend-id -> action-slot mapping, same approach as the
     fixed CGARRouter, so Pure DQN and Eps-Greedy DQN survive churn and
     reordering correctly.
  2. Updated all qnet.predict()/qnet.update() calls to match the new
     QNetwork signature (explicit slot_indices / slot_index args) added
     in the fixed cgar.py — this file would otherwise raise a TypeError
     against the corrected QNetwork.
  3. mu_hat is now normalized via a running scale estimate before going
     into the state vector, and b.cpu is defensively clipped to [0,1],
     mirroring the same fix applied in cgar.py's _build_state. Previously
     mu_hat was raw/unbounded while the other three features were already
     squashed to [0,1] — same magnitude-dominance risk as in CGAR.

These are the same three fix categories as cgar.py, applied here because
this file was silently relying on the pre-fix QNetwork behavior rather
than because the bug originated independently.
--------------------------------------------------------------------------
"""

import numpy as np
import random

from cgar import QNetwork, ReplayBuffer


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
        pass   # no learning

    def notify_pool_change(self, new_ids, removed_ids):
        pass


class LeastConnectionsRouter:
    """Routes to the backend with the lowest current queue depth. Stateless."""

    def route(self, active_backends) -> int:
        if not active_backends:
            return 0
        return min(range(len(active_backends)),
                   key=lambda i: active_backends[i].queue_depth)

    def update(self, active_backends, action, reward, result):
        pass

    def notify_pool_change(self, new_ids, removed_ids):
        pass


class PureDQNRouter:
    """
    Pure DQN router with epsilon-greedy exploration (fixed epsilon).
    No gating — represents the existing RL-gateway literature.

    Shares the QNetwork class with CGARRouter, so it now also carries a
    persistent backend-id -> action-slot mapping (see module docstring).
    """

    def __init__(self, epsilon: float = 0.15, max_backends: int = 20,
                 n_features: int = 5, lr: float = 0.02, seed: int = 42):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.max_backends = max_backends
        self.qnet = QNetwork(n_features, max_backends, lr=lr, seed=seed)
        self.replay = ReplayBuffer(3000)
        self.batch_size = 32
        self.train_every = 10
        self.t = 0

        self._prev_state = None
        self._prev_active_ids = None
        self._prev_action_bid = None

        # Per-backend tracking (for state building only)
        self.mu_hat = {}
        self.n_visits = {}
        self.sum_reward = {}

        # Persistent backend-id -> action-slot mapping (Fix #1, same as CGAR)
        self.slot = {}
        self._free_slots = list(range(max_backends - 1, -1, -1))

        # Running scale estimate for mu_hat normalization (Fix #3, same as CGAR)
        self._mu_scale = 1.0
        self._mu_scale_decay = 0.99

    def _assign_slot(self, bid):
        if bid in self.slot:
            return
        if not self._free_slots:
            raise RuntimeError(
                f"{type(self).__name__}: exceeded max_backends={self.max_backends} "
                "live slots. Increase max_backends."
            )
        self.slot[bid] = self._free_slots.pop()

    def _release_slot(self, bid):
        s = self.slot.pop(bid, None)
        if s is not None:
            self._free_slots.append(s)

    def route(self, active_backends) -> int:
        self.t += 1
        self._ensure_tracked(active_backends)

        active_ids = [b.id for b in active_backends]
        slot_indices = [self.slot[bid] for bid in active_ids]

        state = self._build_state(active_backends)

        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(len(active_backends)))
        else:
            q_vals = self.qnet.predict(state, slot_indices)
            action = int(np.argmax(q_vals))

        action = min(action, len(active_backends) - 1)

        self._prev_state = state
        self._prev_active_ids = active_ids
        self._prev_action_bid = active_ids[action]

        return action

    def update(self, active_backends, action, reward, result):
        action = min(action, len(active_backends) - 1)
        b = active_backends[action]
        bid = b.id

        self.n_visits[bid] = self.n_visits.get(bid, 0) + 1
        self.sum_reward[bid] = self.sum_reward.get(bid, 0.0) + reward
        self.mu_hat[bid] = self.sum_reward[bid] / self.n_visits[bid]

        self._mu_scale = max(
            self._mu_scale * self._mu_scale_decay,
            abs(self.mu_hat[bid]),
            1e-6,
        )

        if self._prev_state is not None and self._prev_active_ids is not None:
            try:
                prev_row_idx = self._prev_active_ids.index(self._prev_action_bid)
            except ValueError:
                prev_row_idx = None

            if prev_row_idx is not None:
                prev_slot = self.slot[self._prev_action_bid]
                prev_state_row = self._prev_state[prev_row_idx]

                next_state = self._build_state(active_backends)
                next_ids = [bb.id for bb in active_backends]
                next_slots = [self.slot[i] for i in next_ids]

                self.replay.push(
                    (prev_state_row, prev_slot, reward, next_state, next_slots)
                )

        if self.t % self.train_every == 0 and len(self.replay) >= self.batch_size:
            for (s_row, slot_idx, r, ns, next_slots) in self.replay.sample(self.batch_size):
                self.qnet.update(s_row, slot_idx, r, ns, next_slots)

    def notify_pool_change(self, new_ids, removed_ids):
        for bid in new_ids:
            if bid not in self.n_visits:
                self.n_visits[bid] = 0
                self.mu_hat[bid] = 0.0
                self.sum_reward[bid] = 0.0
            self._assign_slot(bid)

        for bid in removed_ids:
            for d in (self.n_visits, self.mu_hat, self.sum_reward):
                d.pop(bid, None)
            self._release_slot(bid)

    def _ensure_tracked(self, active_backends):
        for b in active_backends:
            if b.id not in self.n_visits:
                self.n_visits[b.id] = 0
                self.mu_hat[b.id] = 0.0
                self.sum_reward[b.id] = 0.0
            self._assign_slot(b.id)  # no-op if already assigned

    def _build_state(self, active_backends):
        rows = []
        for b in active_backends:
            bid = b.id
            mu = self.mu_hat.get(bid, 0.0)
            mu_norm = float(np.clip(mu / (self._mu_scale + 1e-8), -1.0, 1.0))
            cpu = float(np.clip(b.cpu, 0.0, 1.0))

            rows.append([
                mu_norm,
                min(1.0, self.n_visits.get(bid, 0) / 500.0),
                min(1.0, b.queue_depth / 20.0),
                cpu,
                0.0,   # no age tracking in pure DQN
            ])
        return np.array(rows, dtype=np.float32)


class EpsGreedyDQNRouter(PureDQNRouter):
    """
    DQN with decaying epsilon — the critical ablation baseline.
    Epsilon decays from epsilon_start to epsilon_min over decay_steps.
    If CGAR beats this, the gate contributes beyond a simple decay schedule.

    Inherits the fixed slot-mapping and state-normalization logic from
    PureDQNRouter unchanged.
    """

    def __init__(self, epsilon_start: float = 0.9, epsilon_min: float = 0.05,
                 decay_steps: int = 2000, max_backends: int = 20,
                 n_features: int = 5, lr: float = 0.02, seed: int = 42):
        super().__init__(epsilon=epsilon_start, max_backends=max_backends,
                         n_features=n_features, lr=lr, seed=seed)
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps

    def route(self, active_backends) -> int:
        # Decay epsilon
        progress = min(1.0, self.t / self.decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_min - self.epsilon_start)
        return super().route(active_backends)