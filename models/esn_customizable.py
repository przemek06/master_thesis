import numpy as np
import torch


class ESNCustomizable:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=0.9, sparsity=0.9, input_scaling=1.0,
                 leaky_rate=1.0, ridge=1e-6, noise=0.0, feedback_scaling=0.56,
                 W_in=None, W=None, W_fb=None, bias=None, readout_inputs=False, seed=None, device=None):
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.leaky_rate = leaky_rate
        self.ridge = ridge
        self.noise = noise
        self.readout_inputs = readout_inputs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        rng = np.random.default_rng(seed)

        n_in_extended = n_inputs + (len(bias) if bias is not None else 0)

        if W_in is None:
            W_in = rng.uniform(-1, 1, (n_reservoir, n_in_extended))
        if W_fb is None:
            W_fb = rng.uniform(-1, 1, (n_reservoir, n_outputs))
        if W is None:
            W = rng.uniform(-1, 1, (n_reservoir, n_reservoir))
            W[rng.random((n_reservoir, n_reservoir)) < sparsity] = 0.0
            W = W / np.max(np.abs(np.linalg.eigvals(W))) * spectral_radius

        self._is_complex = np.iscomplexobj(W)
        self._state_size = n_reservoir * 2 if self._is_complex else n_reservoir

        self.W_in = torch.tensor(W_in * input_scaling, dtype=torch.float32, device=self.device)
        self.W_fb = torch.tensor(W_fb * feedback_scaling, dtype=torch.float32, device=self.device)
        self.W = torch.tensor(W, dtype=torch.complex64 if self._is_complex else torch.float32, device=self.device)
        self.bias = torch.tensor(bias, dtype=torch.float32, device=self.device) if bias is not None else None
        self.W_out = None

    def _flat(self, state):
        return torch.cat([state.real, state.imag]) if self._is_complex else state

    def _run(self, u, y=None, initial_state=None):
        u = torch.tensor(u, dtype=torch.float32, device=self.device)
        dtype = torch.complex64 if self._is_complex else torch.float32
        state = torch.zeros(self.n_reservoir, dtype=dtype, device=self.device) if initial_state is None else initial_state.clone()
        states = torch.empty(u.shape[0], self._state_size, device=self.device)
        fb = torch.zeros(self.n_outputs, device=self.device)

        for t in range(u.shape[0]):
            inp = torch.cat([u[t], self.bias]) if self.bias is not None else u[t]
            pre = self.W_in @ inp + self.W @ state + self.W_fb @ fb
            if self.noise > 0.0:
                pre = pre + self.noise * torch.randn(self.n_reservoir, dtype=dtype, device=self.device)
            state = (1 - self.leaky_rate) * state + self.leaky_rate * torch.tanh(pre)
            states[t] = self._flat(state)
            if y is not None:
                fb = torch.tensor(y[t], dtype=torch.float32, device=self.device).reshape(self.n_outputs)
            else:
                fb = self._readout_step(self._flat(state), u[t])

        return states, state

    def _readout_step(self, flat_state, u_t):
        if not self.readout_inputs:
            return flat_state @ self.W_out.T
        return torch.cat([flat_state, u_t, torch.ones(1, device=self.device)]) @ self.W_out.T

    def _readout_features(self, states, u):
        if not self.readout_inputs:
            return states
        u_t = torch.tensor(u, dtype=torch.float32, device=self.device)
        ones = torch.ones(states.shape[0], 1, device=self.device)
        return torch.cat([states, u_t, ones], dim=1)

    def fit(self, u, y, warmup=0):
        with torch.no_grad():
            states, _ = self._run(u, y=y)
            X = self._readout_features(states, u)[warmup:]
            Y = torch.tensor(y[warmup:], dtype=torch.float32, device=self.device)
            A = X.T @ X + self.ridge * torch.eye(X.shape[1], device=self.device)
            self.W_out = torch.linalg.solve(A, X.T @ Y).T
        return self

    def fit_batch(self, U, Y, warmup=0):
        with torch.no_grad():
            U = torch.tensor(U, dtype=torch.float32, device=self.device)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
            B, T, _ = U.shape
            dtype = torch.complex64 if self._is_complex else torch.float32
            state = torch.zeros(B, self.n_reservoir, dtype=dtype, device=self.device)
            fb = torch.zeros(B, self.n_outputs, device=self.device)
            states = torch.empty(T, B, self._state_size, device=self.device)

            for t in range(T):
                inp = torch.cat([U[:, t, :], self.bias.expand(B, -1)], dim=1) if self.bias is not None else U[:, t, :]
                pre = inp @ self.W_in.T + state @ self.W.T + fb @ self.W_fb.T
                if self.noise > 0.0:
                    pre = pre + self.noise * torch.randn(B, self.n_reservoir, dtype=dtype, device=self.device)
                state = (1 - self.leaky_rate) * state + self.leaky_rate * torch.tanh(pre)
                flat = torch.cat([state.real, state.imag], dim=1) if self._is_complex else state
                states[t] = flat
                fb = Y_t[:, t, :]

            X = states[warmup:].reshape(-1, self._state_size)
            Y_flat = Y_t[:, warmup:, :].permute(1, 0, 2).reshape(-1, self.n_outputs)
            A = X.T @ X + self.ridge * torch.eye(self._state_size, device=self.device)
            self.W_out = torch.linalg.solve(A, X.T @ Y_flat).T
        return self

    def reset_state(self):
        dtype = torch.complex64 if self._is_complex else torch.float32
        self.last_state = torch.zeros(self.n_reservoir, dtype=dtype, device=self.device)

    def predict(self, u, initial_state=None):
        with torch.no_grad():
            states, last_state = self._run(u, initial_state=initial_state)
            self.last_state = last_state
            return (self._readout_features(states, u) @ self.W_out.T).cpu().numpy()

    def predict_autonomous(self, n_steps, initial_state=None):
        with torch.no_grad():
            state = self.last_state if initial_state is None else initial_state

            preds = []
            inp = self._flat(state) @ self.W_out.T
            for _ in range(n_steps):
                inp_with_bias = torch.cat([inp, self.bias]) if self.bias is not None else inp
                pre = self.W_in @ inp_with_bias + self.W @ state + self.W_fb @ inp
                if self.noise > 0.0:
                    dtype = torch.complex64 if self._is_complex else torch.float32
                    pre = pre + self.noise * torch.randn(self.n_reservoir, dtype=dtype, device=self.device)
                state = (1 - self.leaky_rate) * state + self.leaky_rate * torch.tanh(pre)
                inp = self._flat(state) @ self.W_out.T
                preds.append(inp.cpu().numpy())

            return np.array(preds)
