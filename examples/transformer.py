"""
Transformer example: learn to predict the next token in a repeating sequence.
Run: python -m examples.transformer
"""
import numpy as np
from models.transformer import Transformer
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM


VOCAB = 8
SEQ_LEN = 16


def make_batch(batch_size: int = 32):
    """Sequences that repeat a short pattern; target is next token."""
    pattern = np.arange(VOCAB)
    full = np.tile(pattern, SEQ_LEN // VOCAB + 1)
    starts = np.random.randint(0, VOCAB, size=batch_size)
    x = np.stack([full[s: s + SEQ_LEN] for s in starts])
    y = np.stack([full[s + 1: s + SEQ_LEN + 1] for s in starts])
    return x, y


def main():
    np.random.seed(0)
    model = Transformer(vocab_size=VOCAB, d_model=32, n_heads=2, n_layers=2, max_seq_len=SEQ_LEN)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = ADAM(model.parameters(), lr=1e-3)

    for step in range(1, 201):
        x, y = make_batch()
        optimizer.zero_grad()
        logits = model.forward(x)                          # (B, T, V)
        loss = loss_fn.forward(logits.reshape(-1, VOCAB), y.reshape(-1))
        model.backward(loss_fn.backward().reshape(x.shape[0], SEQ_LEN, VOCAB))
        optimizer.step()
        if step % 50 == 0:
            preds = logits.reshape(-1, VOCAB).argmax(axis=1)
            acc = (preds == y.reshape(-1)).mean()
            print(f"Step {step:3d}  loss={loss:.4f}  acc={acc * 100:.1f}%")


if __name__ == "__main__":
    main()
