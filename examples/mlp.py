"""
MLP example: classify a 2-class spiral dataset.
Run: python -m examples.mlp
"""
import numpy as np
from models.mlp import MLP
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM
from training.trainer import Trainer


def make_spiral(n: int = 200, noise: float = 0.15):
    X, y = [], []
    for c in range(2):
        t = np.linspace(0, 2 * np.pi, n)
        r = t / (2 * np.pi)
        X.append(np.stack([r * np.cos(t + c * np.pi), r * np.sin(t + c * np.pi)], axis=1)
                 + noise * np.random.randn(n, 2))
        y.append(np.full(n, c, dtype=int))
    return np.concatenate(X).astype(np.float32), np.concatenate(y)


def main():
    np.random.seed(42)
    x, y = make_spiral()

    model = MLP([2, 64, 32, 2])
    trainer = Trainer(model, SoftmaxCrossEntropy(), ADAM(model.parameters(), lr=3e-3))
    trainer.fit(x, y, epochs=100, batch_size=64, verbose=False)

    logits = trainer.predict(x)
    acc = (logits.argmax(axis=1) == y).mean()
    print(f"Accuracy: {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
