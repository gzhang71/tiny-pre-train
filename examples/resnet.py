"""
ResNet example: binary classification on a checkerboard pattern.
Run: python -m examples.resnet
"""
import numpy as np
from models.resnet import ResNet
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM
from training.trainer import Trainer


def make_checkerboard(n: int = 400):
    x = np.random.uniform(-2, 2, (n, 2)).astype(np.float32)
    y = ((np.floor(x[:, 0]) + np.floor(x[:, 1])) % 2).astype(int)
    return x, y


def main():
    np.random.seed(7)
    x, y = make_checkerboard()

    model = ResNet(in_features=2, hidden_features=32, out_features=2, num_blocks=3)
    trainer = Trainer(model, SoftmaxCrossEntropy(), ADAM(model.parameters(), lr=1e-3))
    trainer.fit(x, y, epochs=150, batch_size=64, verbose=False)

    logits = trainer.predict(x)
    acc = (logits.argmax(axis=1) == y).mean()
    print(f"Accuracy: {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
