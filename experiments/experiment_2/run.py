import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from tasks.narma10 import load
from models.lstm import LSTM

HERE = os.path.dirname(__file__)


def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)


u_train, y_train, u_val, y_val, u_test, y_test = load()

model = LSTM(n_inputs=1, n_hidden=64, n_outputs=1, n_layers=1, lr=1e-3)
train_losses, val_losses = model.fit(u_train, y_train, u_val, y_val, n_epochs=300, warmup=100)

y_pred = model.predict(u_test)

print(f"Test NMSE: {nmse(y_test, y_pred):.6f}")

epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(8, 4))
plt.plot(epochs, train_losses, label="Train loss")
plt.plot(epochs, val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("LSTM — NARMA-10")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "loss.png"), dpi=150)
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
for i, ax in enumerate(axes):
    start = i * 100
    end = start + 200
    ax.plot(y_test[start:end], label="Target", color="steelblue")
    ax.plot(y_pred[start:end], label="Prediction", color="tomato", linestyle="--")
    ax.set_title(f"Timesteps {start}–{end}")
    ax.legend(loc="upper right")

fig.suptitle("LSTM — NARMA-10 predictions vs target")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.show()
