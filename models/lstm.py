import numpy as np
import torch
import torch.nn as nn


class _LSTMNet(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, n_hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)


class LSTM:
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _LSTMNet(n_inputs, n_hidden, n_outputs, n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def fit(self, u_train, y_train, u_val, y_val, n_epochs, warmup=0):
        u_tr = self._to_tensor(u_train).unsqueeze(0)
        y_tr = self._to_tensor(y_train).unsqueeze(0)
        u_vl = self._to_tensor(u_val).unsqueeze(0)
        y_vl = self._to_tensor(y_val).unsqueeze(0)

        train_losses, val_losses = [], []

        for _ in range(n_epochs):
            self.net.train()
            self.optimizer.zero_grad()
            pred = self.net(u_tr)
            loss = self.loss_fn(pred[:, warmup:], y_tr[:, warmup:])
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())

            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(u_vl)
                val_loss = self.loss_fn(val_pred, y_vl)
            val_losses.append(val_loss.item())

        return train_losses, val_losses

    def predict(self, u):
        self.net.eval()
        with torch.no_grad():
            u_t = self._to_tensor(u).unsqueeze(0)
            pred = self.net(u_t).squeeze(0)
        return pred.cpu().numpy()
