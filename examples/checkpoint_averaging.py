"""
Checkpoint averaging: the mean of several late-training snapshots often beats
the last (noisy) checkpoint. Same setup as examples/gpt2.py, but with a
deliberately high learning rate, small batches, and 15% label noise so late
steps keep bouncing around the optimum; evaluation is on clean data.
Run: python -m examples.checkpoint_averaging
"""
import numpy as np
from models.gpt2 import GPT2
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM
from training.checkpoint import get_state, set_state, average_states


VOCAB = 10
SEQ_LEN = 12


def make_batch(batch_size: int = 32, label_noise: float = 0.0):
    pattern = np.arange(VOCAB)
    full = np.tile(pattern, SEQ_LEN // VOCAB + 2)
    starts = np.random.randint(0, VOCAB, size=batch_size)
    x = np.stack([full[s: s + SEQ_LEN] for s in starts])
    y = np.stack([full[s + 1: s + SEQ_LEN + 1] for s in starts])
    if label_noise:  # corrupt a fraction of targets → gradients stay noisy forever
        flip = np.random.rand(*y.shape) < label_noise
        y = np.where(flip, np.random.randint(0, VOCAB, size=y.shape), y)
    return x, y


def evaluate(model, loss_fn, x, y) -> float:
    logits = model.forward(x)
    return float(loss_fn.forward(logits.reshape(-1, VOCAB), y.reshape(-1)))


def main():
    np.random.seed(1)
    model = GPT2(vocab_size=VOCAB, d_model=32, n_heads=2, n_layers=2, max_seq_len=SEQ_LEN)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = ADAM(model.parameters(), lr=2e-2)  # high on purpose → noisy late phase
    x_eval, y_eval = make_batch(64)

    snapshots = []
    for step in range(1, 301):
        x, y = make_batch(8, label_noise=0.15)  # small noisy batches
        optimizer.zero_grad()
        logits = model.forward(x)
        loss = loss_fn.forward(logits.reshape(-1, VOCAB), y.reshape(-1))
        model.backward(loss_fn.backward().reshape(x.shape[0], SEQ_LEN, VOCAB))
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step:3d}  train loss={loss:.4f}")
        if step > 200 and step % 5 == 0:  # snapshot the last 20 checkpoints
            snapshots.append(get_state(model))

    last_loss = evaluate(model, loss_fn, x_eval, y_eval)
    set_state(model, average_states(snapshots))
    avg_loss = evaluate(model, loss_fn, x_eval, y_eval)

    print(f"\nEval loss, last checkpoint:                {last_loss:.4f}")
    print(f"Eval loss, average of last {len(snapshots)} checkpoints: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
