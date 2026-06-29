"""
T5 example: learn to copy a source sequence to the target (seq2seq).
  src: [1, 2, 3, 4]
  tgt input:  [0, 1, 2, 3]   (0 = BOS; decoder input is right-shifted)
  tgt labels: [1, 2, 3, 4]
Run: python -m examples.t5
"""
import numpy as np
from models.t5 import T5
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM


VOCAB = 6    # tokens 1-5, 0 = BOS
SRC_LEN = 4
TGT_LEN = 4
BOS = 0


def make_batch(batch_size: int = 32):
    src = np.random.randint(1, VOCAB, size=(batch_size, SRC_LEN))
    tgt_in = np.concatenate([np.full((batch_size, 1), BOS), src[:, :-1]], axis=1)
    tgt_out = src
    return src, tgt_in, tgt_out


def main():
    np.random.seed(2)
    model = T5(vocab_size=VOCAB, d_model=32, n_heads=2,
               n_encoder_layers=2, n_decoder_layers=2)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = ADAM(model.parameters(), lr=3e-3)

    for step in range(1, 401):
        src, tgt_in, tgt_out = make_batch()
        optimizer.zero_grad()
        logits = model.forward(src, tgt_in)                    # (B, T, V)
        loss = loss_fn.forward(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))
        model.backward(loss_fn.backward().reshape(src.shape[0], TGT_LEN, VOCAB))
        optimizer.step()
        if step % 100 == 0:
            acc = (logits.reshape(-1, VOCAB).argmax(axis=1) == tgt_out.reshape(-1)).mean()
            print(f"Step {step:3d}  loss={loss:.4f}  acc={acc * 100:.1f}%")


if __name__ == "__main__":
    main()
