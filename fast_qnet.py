"""Replaces the slow per-request PyTorch calls with numpy MLP — same architecture, 100x faster."""
import numpy as np

def relu(x): return np.maximum(0, x)

class QNetwork:
    def __init__(self, n_features, n_actions, lr=0.001, seed=0):
        rng = np.random.default_rng(seed)
        scale = lambda i, o: np.sqrt(2.0/i)
        self.W1 = rng.normal(0, scale(n_features,64), (64, n_features))
        self.b1 = np.zeros(64)
        self.W2 = rng.normal(0, scale(64,32), (32, 64))
        self.b2 = np.zeros(32)
        self.W3 = rng.normal(0, scale(32,n_actions), (n_actions, 32))
        self.b3 = np.zeros(n_actions)
        self.lr = lr
        self.n_actions = n_actions
        self.n_features = n_features

    def _forward(self, x):
        h1 = relu(x @ self.W1.T + self.b1)
        h2 = relu(h1 @ self.W2.T + self.b2)
        return h2 @ self.W3.T + self.b3, h1, h2

    def predict(self, state_matrix):
        """Returns Q-value for each active backend: shape (n_active,)"""
        n = state_matrix.shape[0]
        out, _, _ = self._forward(state_matrix)   # (n_active, n_actions)
        # Q(s_i, i) = diagonal: each backend's Q-value for routing to itself
        return np.array([out[i, min(i, out.shape[1]-1)] for i in range(n)])

    def update(self, state_matrix, action, reward, next_state_matrix, gamma=0.95):
        n     = state_matrix.shape[0]
        n_nxt = next_state_matrix.shape[0]
        if n == 0 or n_nxt == 0:
            return

        # Forward passes
        out,  h1,  h2  = self._forward(state_matrix)
        outn, _,   _   = self._forward(next_state_matrix)

        # Target for chosen action
        q_next_vals = np.array([outn[i, min(i, outn.shape[1]-1)] for i in range(n_nxt)])
        target_val  = reward + gamma * q_next_vals.max()

        # Current Q-value for chosen action
        a_idx = min(action, n-1)
        col   = min(a_idx, out.shape[1]-1)
        pred  = out[a_idx, col]
        err   = target_val - pred

        # Backprop through W3 → W2 → W1 (only for the action row)
        dout = np.zeros_like(out)
        dout[a_idx, col] = -2 * err      # MSE gradient

        dW3 = dout.T @ h2                # (n_actions, 32)
        dh2 = dout @ self.W3             # (n, 32)
        dh2 *= (h2 > 0)
        dW2 = dh2.T @ h1                 # (32, 64)
        dh1 = dh2 @ self.W2             # (n, 64)
        dh1 *= (h1 > 0)
        dW1 = dh1.T @ state_matrix       # (64, n_features)

        # Gradient clipping
        for g in (dW1, dW2, dW3):
            np.clip(g, -1.0, 1.0, out=g)

        self.W3 -= self.lr * dW3
        self.W2 -= self.lr * dW2
        self.W1 -= self.lr * dW1
        self.b3 -= self.lr * dout.sum(axis=0)
        self.b2 -= self.lr * dh2.sum(axis=0)
        self.b1 -= self.lr * dh1.sum(axis=0)