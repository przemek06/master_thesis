import numpy as np
import torch


class ESNFeedback:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=0.9, sparsity=0.9, input_scaling=1.0, leaky_rate=1.0, ridge=1e-6, noise=0.0, feedback_scaling=0.56, W_in=None, W_fb=None, seed=None, device=None):
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.leaky_rate = leaky_rate
        self.ridge = ridge
        self.noise = noise
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        rng = np.random.default_rng(seed)

        if W_in is None:
            W_in = rng.uniform(-1, 1, (n_reservoir, n_inputs + 1))
        if W_fb is None:
            W_fb = rng.uniform(-1, 1, (n_reservoir, n_outputs))

        W = rng.uniform(-1, 1, (n_reservoir, n_reservoir))
        W[rng.random((n_reservoir, n_reservoir)) < sparsity] = 0.0
        W = W / np.max(np.abs(np.linalg.eigvals(W))) * spectral_radius

        self.W_in = torch.tensor(W_in * input_scaling, dtype=torch.float32, device=self.device)
        self.W_fb = torch.tensor(W_fb * feedback_scaling, dtype=torch.float32, device=self.device)
        self.W = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.W_out = None

    def _run(self, u, y=None, initial_state=None):
        u = torch.tensor(u, dtype=torch.float32, device=self.device)
        state = torch.zeros(self.n_reservoir, device=self.device) if initial_state is None else initial_state.clone()
        states = torch.empty(u.shape[0], self.n_reservoir, device=self.device)
        bias = torch.full((1,), 0.2, device=self.device)
        fb = torch.zeros(self.n_outputs, device=self.device)

        for t in range(u.shape[0]):
            pre = self.W_in @ torch.cat([u[t], bias]) + self.W @ state + self.W_fb @ fb
            if self.noise > 0.0:
                pre = pre + self.noise * torch.randn_like(pre)
            state = (1 - self.leaky_rate) * state + self.leaky_rate * torch.tanh(pre)
            states[t] = state
            if y is not None:
                fb = torch.tensor(y[t], dtype=torch.float32, device=self.device).reshape(self.n_outputs)
            else:
                fb = state @ self.W_out.T

        return states

    def fit(self, u, y, warmup=0):
        with torch.no_grad():
            states = self._run(u, y=y)
            X = states[warmup:]
            Y = torch.tensor(y[warmup:], dtype=torch.float32, device=self.device)
            A = X.T @ X + self.ridge * torch.eye(self.n_reservoir, device=self.device)
            self.W_out = torch.linalg.solve(A, X.T @ Y).T
        return self

    def fit_batch(self, U, Y, warmup=0):
        with torch.no_grad():
            U = torch.tensor(U, dtype=torch.float32, device=self.device)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
            B, T, _ = U.shape
            bias = torch.full((1,), 0.2, device=self.device).expand(B, 1)
            state = torch.zeros(B, self.n_reservoir, device=self.device)
            fb = torch.zeros(B, self.n_outputs, device=self.device)
            states = torch.empty(T, B, self.n_reservoir, device=self.device)

            for t in range(T):
                inp = torch.cat([U[:, t, :], bias], dim=1)
                pre = inp @ self.W_in.T + state @ self.W.T + fb @ self.W_fb.T
                if self.noise > 0.0:
                    pre = pre + self.noise * torch.randn_like(pre)
                state = (1 - self.leaky_rate) * state + self.leaky_rate * torch.tanh(pre)
                states[t] = state
                fb = Y_t[:, t, :]

            X = states[warmup:].reshape(-1, self.n_reservoir)
            Y_flat = Y_t[:, warmup:, :].permute(1, 0, 2).reshape(-1, self.n_outputs)
            A = X.T @ X + self.ridge * torch.eye(self.n_reservoir, device=self.device)
            self.W_out = torch.linalg.solve(A, X.T @ Y_flat).T
        return self

    def reset_state(self):
        self.last_state = torch.zeros(self.n_reservoir, device=self.device)

    def predict(self, u, initial_state=None):
        with torch.no_grad():
            states = self._run(u, initial_state=initial_state)
            self.last_state = states[-1]
            return (states @ self.W_out.T).cpu().numpy()

    def predict_autonomous(self, n_steps, initial_state=None):
        with torch.no_grad():
            state = self.last_state if initial_state is None else initial_state
            bias = torch.full((1,), 0.2, device=self.device)

            preds = []
            inp = state @ self.W_out.T
            for _ in range(n_steps):
                pre = self.W_in @ torch.cat([inp, bias]) + self.W @ state + self.W_fb @ inp
                if self.noise > 0.0:
                    pre = pre + self.noise * torch.randn_like(pre)
                state = (1 - self.leaky_rate) * state + self.leaky_rate * torch.tanh(pre)
                inp = state @ self.W_out.T
                preds.append(inp.cpu().numpy())

            return np.array(preds)
