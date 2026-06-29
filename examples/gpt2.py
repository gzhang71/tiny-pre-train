"""
GPT-2 example: learn a repeating alphabet, then generate from a prompt.
Run: python -m examples.gpt2
"""
import numpy as np
from models.gpt2 import GPT2
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


def main():
    np.random.seed(1)
    model = GPT2(vocab_size=VOCAB, d_model=32, n_heads=2, n_layers=2, max_seq_len=SEQ_LEN)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = ADAM(model.parameters(), lr=1e-3)

    for step in range(1, 201):
        x, y = make_batch()
        optimizer.zero_grad()
        logits = model.forward(x)
        loss = loss_fn.forward(logits.reshape(-1, VOCAB), y.reshape(-1))
        model.backward(loss_fn.backward().reshape(x.shape[0], SEQ_LEN, VOCAB))
        optimizer.step()
        if step % 50 == 0:
            acc = (logits.reshape(-1, VOCAB).argmax(axis=1) == y.reshape(-1)).mean()
            print(f"Step {step:3d}  loss={loss:.4f}  acc={acc * 100:.1f}%")

    prompt = np.array([[3, 4, 5]])
    generated = model.generate(prompt, max_new_tokens=7, temperature=0.5)
    print(f"\nPrompt: {generated[:3].tolist()}  →  generated: {generated[3:].tolist()}")


if __name__ == "__main__":
    main()
