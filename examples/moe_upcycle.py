"""
Dense→MoE upcycling: train a dense GPT-2, copy each block's FFN into 4 experts
with a zero-initialized router (output is exactly the dense model's), then
continue training so the experts specialize.
Run: python -m examples.moe_upcycle
"""
import numpy as np
from models.gpt2 import GPT2
from layers.moe import MoEFeedForward
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM


VOCAB = 10
SEQ_LEN = 12


def make_batch(batch_size: int = 32):
    pattern = np.arange(VOCAB)
    full = np.tile(pattern, SEQ_LEN // VOCAB + 2)
    starts = np.random.randint(0, VOCAB, size=batch_size)
    x = np.stack([full[s: s + SEQ_LEN] for s in starts])
    y = np.stack([full[s + 1: s + SEQ_LEN + 1] for s in starts])
    return x, y


def train(model, loss_fn, optimizer, steps: int, tag: str):
    for step in range(1, steps + 1):
        x, y = make_batch()
        optimizer.zero_grad()
        logits = model.forward(x)
        loss = loss_fn.forward(logits.reshape(-1, VOCAB), y.reshape(-1))
        model.backward(loss_fn.backward().reshape(x.shape[0], SEQ_LEN, VOCAB))
        optimizer.step()
        if step % 50 == 0:
            print(f"[{tag}] step {step:3d}  loss={loss:.4f}")


def main():
    np.random.seed(1)
    model = GPT2(vocab_size=VOCAB, d_model=32, n_heads=2, n_layers=2, max_seq_len=SEQ_LEN)
    loss_fn = SoftmaxCrossEntropy()

    train(model, loss_fn, ADAM(model.parameters(), lr=1e-3), 150, "dense")

    x, y = make_batch(64)
    dense_logits = model.forward(x)

    for block in model.blocks:
        block.ffn = MoEFeedForward.from_dense(block.ffn, n_experts=4, top_k=2)
    moe_logits = model.forward(x)
    diff = np.abs(moe_logits - dense_logits).max()
    print(f"\nmax |MoE − dense| logit diff after upcycling: {diff:.2e}")
    assert diff < 1e-9, "upcycled MoE must reproduce the dense model"

    # A zero router over identical experts is a fixed point (identical expert
    # grads, exactly-zero router grad) — jitter the router to break symmetry.
    from core.backend import randn
    for block in model.blocks:
        block.ffn.router.W.data = randn(*block.ffn.router.W.data.shape) * 1e-2

    train(model, loss_fn, ADAM(model.parameters(), lr=1e-3), 150, "moe")

    model.forward(x)  # refresh routing stats
    print("\nGate mass per expert (per block):")
    for i, block in enumerate(model.blocks):
        frac = block.ffn._gates.reshape(-1, block.ffn.n_experts).mean(axis=0)
        print(f"  block {i}: " + "  ".join(f"{f:.2f}" for f in frac))


if __name__ == "__main__":
    main()
