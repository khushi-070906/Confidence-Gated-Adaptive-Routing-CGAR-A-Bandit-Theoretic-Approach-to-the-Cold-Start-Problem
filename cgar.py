"""
cgar.py
-------
Confidence-Gated Adaptive Routing (CGAR)

Core idea:
  For each backend i, maintain a visit count n_i and empirical mean reward mu_i.
  Compute a gating weight:
      lambda_i(t) = exp(-n_i / tau)
  This weight is 1 (trust heuristic fully) when n_i is small,
  and 0 (trust learned policy fully) when n_i is large.

  Final routing weight:
      w_i = lambda_i * H_i  +  (1 - lambda_i) * sigma(Q_i)

  Where H_i = heuristic score (inverse least-connections)
        Q_i = DQN Q-value for backend i (updated asynchronously)

--------------------------------------------------------------------------
FIXES APPLIED (see accompanying review notes):

1. Persistent action-slot mapping.
   Previously Q-values were indexed by each backend's *position* in the
   active_backends list for that call, not by backend identity. Since the
   pool reorders on every churn event, W[i] silently got applied to a
   different backend than the one it was trained on. Now every backend is
   assigned a fixed slot (self.slot[bid]) on first sight / pool-change
   notification, and that slot is used consistently for both predict()
   and update() regardless of list order.

2. Q-value normalization changed from softmax to min-max.
   Softmax forces sum(q_norm) == 1, so each individual value shrinks as
   the number of active backends K grows (~1/K on average), while H_i
   (heuristic score) does not shrink with K. This meant the heuristic
   term could dominate the blend even when lambda_i -> 0, defeating the
   intended full handoff to the learned policy. Min-max normalization
   keeps Q-values on a comparable, K-independent scale to H_i.

3. State feature normalization.
   mu_hat (raw empirical reward) was going into the state vector
   unnormalized and unbounded, while three other features were already
   squashed to [0,1]. In a linear Q-approximator this lets whichever
   feature has the largest raw magnitude dominate the dot product
   regardless of actual importance. mu_hat is now squashed via a
   running-scale-based clip; cpu is defensively clipped to [0,1]
   (confirm your simulator's actual cpu units/range and adjust the
   scaling in _build_state if it reports e.g. 0-100 instead of 0-1).
--------------------------------------------------------------------------
"""

import numpy as np
from collections import deque
import random


# ─── Tiny DQN ──────────────────────────────────────────────────────────────

class QNetwork:
    """
    Lightweight tabular Q-approximation.

    W has one row per fixed action *slot* (not per position-in-list).
    predict()/update() take explicit slot indices so callers never rely
    on list ordering to determine which weight row applies to which
    backend.
    """

    def __init__(self, n_features: int, n_actions: int, lr: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_actions, n_features))
        self.lr = lr
        self.n_actions = n_actions

    def predict(self, state_matrix: np.ndarray, slot_indices) -> np.ndarray:
        """
        state_matrix: (n, n_features) rows in the SAME order as slot_indices.
        slot_indices: sequence of length n giving the persistent action slot
                      for each row.
        """
        slot_indices = np.asarray(slot_indices, dtype=int)
        W = self.W[slot_indices]
        return np.einsum('ij,ij->i', W, state_matrix)

    def update(self, state_row, slot_index: int, reward: float, next_state_matrix,
               next_slot_indices, gamma: float = 0.9):
        """
        Single-transition update for the backend at slot_index.
        next_state_matrix / next_slot_indices describe the *next* active
        pool, used to bootstrap max_a' Q(s', a').
        """
        q_val = float(np.dot(self.W[slot_index], state_row))

        if len(next_slot_indices) > 0:
            q_next = self.predict(next_state_matrix, next_slot_indices)
            bootstrap = np.max(q_next)
        else:
            bootstrap = 0.0

        target = reward + gamma * bootstrap
        error = target - q_val
        self.W[slot_index] += self.lr * error * state_row


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buf = deque(maxlen=capacity)

    def push(self, transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self):
        return len(self.buf)


# ─── CGAR Router ───────────────────────────────────────────────────────────

class CGARRouter:
    def __init__(self,
                 tau: float = 100.0,
                 mode: str = "adaptive",
                 static_lambda: float = 0.5,
                 n_features: int = 5,
                 max_backends: int = 20,
                 lr: float = 0.02,
                 seed: int = 42):

        self.tau = tau
        self.mode = mode
        self.static_lambda = static_lambda
        self.rng = np.random.default_rng(seed)
        self.max_backends = max_backends

        self.n_visits = {}
        self.mu_hat = {}
        self.age = {}
        self.sum_reward = {}

        # Persistent backend-id -> action-slot mapping (Fix #1)
        self.slot = {}
        self._free_slots = list(range(max_backends - 1, -1, -1))  # pop() from end = slot 0 first

        self.qnet = QNetwork(n_features, max_backends, lr=lr, seed=seed)
        self.replay = ReplayBuffer(capacity=3000)

        self.batch_size = 32
        self.train_every = 10
        self.t = 0

        # Running scale estimate for mu_hat normalization (Fix #3)
        self._mu_scale = 1.0
        self._mu_scale_decay = 0.99

        self._prev_state = None
        self._prev_action_bid = None
        self._prev_active_ids = None

    # ─── Pool membership ────────────────────────────────────────────────

    def notify_pool_change(self, new_backend_ids: list, removed_ids: list):
        for bid in new_backend_ids:
            self.n_visits[bid] = 0
            self.mu_hat[bid] = 0.0
            self.age[bid] = 0
            self.sum_reward[bid] = 0.0
            self._assign_slot(bid)

        for bid in removed_ids:
            for d in (self.n_visits, self.mu_hat, self.age, self.sum_reward):
                d.pop(bid, None)
            self._release_slot(bid)

    def _assign_slot(self, bid):
        if bid in self.slot:
            return
        if not self._free_slots:
            raise RuntimeError(
                f"CGARRouter: exceeded max_backends={self.max_backends} live slots. "
                "Increase max_backends."
            )
        self.slot[bid] = self._free_slots.pop()

    def _release_slot(self, bid):
        s = self.slot.pop(bid, None)
        if s is not None:
            self._free_slots.append(s)

    def _ensure_tracked(self, active_backends):
        for b in active_backends:
            if b.id not in self.n_visits:
                self.n_visits[b.id] = 0
                self.mu_hat[b.id] = 0.0
                self.age[b.id] = 0
                self.sum_reward[b.id] = 0.0
            self._assign_slot(b.id)  # no-op if already assigned

    # ─── Routing ────────────────────────────────────────────────────────

    def route(self, active_backends) -> int:
        self.t += 1
        self._ensure_tracked(active_backends)

        active_ids = [b.id for b in active_backends]
        slot_indices = [self.slot[bid] for bid in active_ids]

        state_matrix = self._build_state(active_backends)
        q_vals = self.qnet.predict(state_matrix, slot_indices)
        q_norm = self._normalize_q(q_vals)

        weights = []

        for i, b in enumerate(active_backends):
            bid = b.id
            n_i = self.n_visits.get(bid, 0)

            if self.mode == "heuristic":
                lam = 1.0
            elif self.mode == "rl":
                lam = 0.0
            elif self.mode == "static":
                lam = self.static_lambda
            else:
                lam = np.exp(-n_i / self.tau)

            H_i = 1.0 / (b.queue_depth + 1.0)

            w_i = lam * H_i + (1.0 - lam) * q_norm[i]
            weights.append(w_i)

        weights = np.array(weights)
        weights = np.clip(weights, 0, None)

        if weights.sum() == 0:
            weights = np.ones(len(weights))

        weights /= weights.sum()

        if self.rng.random() < 0.9:
            action = int(np.argmax(weights))
        else:
            action = self.rng.choice(len(active_backends), p=weights)

        # Store by backend id, not position, so update() can't be fooled
        # by the list reordering between route() and update().
        self._prev_state = state_matrix
        self._prev_active_ids = active_ids
        self._prev_action_bid = active_ids[action]

        return action

    # ─── Learning ───────────────────────────────────────────────────────

    def update(self, active_backends, action: int, reward: float, result: dict):
        if action >= len(active_backends):
            return

        b = active_backends[action]
        bid = b.id

        self.n_visits[bid] = self.n_visits.get(bid, 0) + 1
        self.sum_reward[bid] = self.sum_reward.get(bid, 0.0) + reward
        self.mu_hat[bid] = self.sum_reward[bid] / self.n_visits[bid]
        self.age[bid] = self.age.get(bid, 0) + 1

        # Update running mu_hat scale used for state normalization (Fix #3)
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
            self._train_step()

    # ─── State / normalization helpers ─────────────────────────────────

    def _build_state(self, active_backends) -> np.ndarray:
        rows = []
        for b in active_backends:
            bid = b.id
            mu = self.mu_hat.get(bid, 0.0)
            mu_norm = float(np.clip(mu / (self._mu_scale + 1e-8), -1.0, 1.0))

            n_norm = min(1.0, self.n_visits.get(bid, 0) / 500.0)
            q_norm_depth = min(1.0, b.queue_depth / 20.0)

            # Defensive clip: confirm b.cpu is already a 0-1 fraction from
            # the simulator. If it reports 0-100 instead, divide by 100
            # here before clipping.
            cpu = float(np.clip(b.cpu, 0.0, 1.0))

            age_norm = min(1.0, self.age.get(bid, 0) / 500.0)

            rows.append([mu_norm, n_norm, q_norm_depth, cpu, age_norm])

        return np.array(rows, dtype=np.float32)

    def _normalize_q(self, x: np.ndarray) -> np.ndarray:
        """
        Min-max normalize Q-values to [0, 1], independent of how many
        backends are active (Fix #2). Falls back to a neutral 0.5 vector
        if all values are equal (avoids div-by-zero and avoids biasing
        toward slot 0).
        """
        lo = x.min()
        hi = x.max()
        span = hi - lo
        if span < 1e-8:
            return np.full_like(x, 0.5)
        return (x - lo) / span

    # ─── Training ───────────────────────────────────────────────────────

    def _train_step(self):
        batch = self.replay.sample(self.batch_size)

        for (state_row, slot_index, reward, next_state, next_slots) in batch:
            self.qnet.update(state_row, slot_index, reward, next_state, next_slots)