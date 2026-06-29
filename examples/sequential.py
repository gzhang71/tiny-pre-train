"""
Sequential example: fit a noisy sine wave.
Run: python -m examples.sequential
"""
import numpy as np
from models.sequential import Sequential
from layers.linear import Linear
from layers.activations import Tanh
from losses.losses import MSELoss
from optim.adam import ADAM
from training.trainer import Trainer


def main():
    np.random.seed(0)
    x = np.linspace(-np.pi, np.pi, 400)[:, None].astype(np.float32)
    y = np.sin(x) + 0.05 * np.random.randn(*x.shape).astype(np.float32)

    model = Sequential([Linear(1, 32), Tanh(), Linear(32, 32), Tanh(), Linear(32, 1)])
    trainer = Trainer(model, MSELoss(), ADAM(model.parameters(), lr=1e-3))
    trainer.fit(x, y, epochs=200, batch_size=64, verbose=False)

    mse = float(np.mean((trainer.predict(x) - y) ** 2))
    print(f"Final MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
