import numpy as np
import torch
from distribution import sample_eigenvalues
from generate import generate_isospectral_sparse_matrix


class CustomESN2:
    def __init__(self, n_inputs, n_reservoir, n_outputs, r, alpha, beta, mu, kappa, sparsity=0.9, input_scaling=1.0, ridge=1e-6, iterations=200, threshold=0.01, seed=None, device=None):
        self.n_reservoir = n_reservoir
        self.ridge = ridge
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        rng = np.random.default_rng(seed)

        W_in = rng.uniform(-1, 1, (n_reservoir, n_inputs)) * input_scaling

        W = generate_isospectral_sparse_matrix(
            lambda size: sample_eigenvalues(r, alpha, beta, mu, kappa, size),
            n_reservoir,
            sparsity,
            iterations=iterations,
            threshold=threshold,
            seed=int(rng.integers(0, 2**31)),
        )
        print(f"W sparsity: {np.mean(W == 0):.4f}")
        self.W_in = torch.tensor(W_in, dtype=torch.float32, device=self.device)
        self.W = torch.tensor(W, dtype=torch.complex64, device=self.device)
        self.W_out = None

    def _run(self, u, initial_state=None):
        u = torch.tensor(u, dtype=torch.float32, device=self.device)
        state = torch.zeros(self.n_reservoir, dtype=torch.complex64, device=self.device) if initial_state is None else initial_state.clone()
        states = torch.empty(u.shape[0], self.n_reservoir * 2, device=self.device)

        for t in range(u.shape[0]):
            state = torch.tanh(self.W_in @ u[t] + self.W @ state)
            states[t] = torch.cat([state.real, state.imag])

        return states

    def fit(self, u, y, warmup=0):
        with torch.no_grad():
            states = self._run(u)
            X = states[warmup:]
            Y = torch.tensor(y[warmup:], dtype=torch.float32, device=self.device)
            A = X.T @ X + self.ridge * torch.eye(self.n_reservoir * 2, device=self.device)
            self.W_out = torch.linalg.solve(A, X.T @ Y).T
        return self

    def predict(self, u, initial_state=None):
        with torch.no_grad():
            states = self._run(u, initial_state)
            return (states @ self.W_out.T).cpu().numpy()

    def predict_autonomous(self, u_warmup, n_steps):
        with torch.no_grad():
            u_w = torch.tensor(u_warmup, dtype=torch.float32, device=self.device)
            state = torch.zeros(self.n_reservoir, dtype=torch.complex64, device=self.device)

            for t in range(u_w.shape[0]):
                state = torch.tanh(self.W_in @ u_w[t] + self.W @ state)

            preds = []
            flat = torch.cat([state.real, state.imag])
            inp = flat @ self.W_out.T
            for _ in range(n_steps):
                state = torch.tanh(self.W_in @ inp + self.W @ state)
                flat = torch.cat([state.real, state.imag])
                inp = flat @ self.W_out.T
                preds.append(inp.cpu().numpy())

            return np.array(preds)
