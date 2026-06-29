"""
VAE example: encode/decode 2D data from two Gaussian clusters.
Run: python -m examples.vae
"""
import numpy as np
from models.vae import VAE
from losses.losses import MSELoss
from optim.adam import ADAM


def make_data(n: int = 500):
    c1 = np.random.randn(n // 2, 2) + np.array([2.0, 2.0])
    c2 = np.random.randn(n // 2, 2) + np.array([-2.0, -2.0])
    return np.concatenate([c1, c2]).astype(np.float32)


def main():
    np.random.seed(3)
    x = make_data()

    model = VAE(input_dim=2, hidden_dims=[32, 16], latent_dim=2)
    loss_fn = MSELoss()
    optimizer = ADAM(model.parameters(), lr=1e-3)

    beta = 0.1
    for epoch in range(1, 301):
        idx = np.random.permutation(len(x))
        recon_losses = []
        for start in range(0, len(x), 64):
            batch = x[idx[start: start + 64]]
            optimizer.zero_grad()
            recon = model.forward(batch)
            recon_loss = loss_fn.forward(recon, batch)
            recon_losses.append(recon_loss)
            model.backward(loss_fn.backward(), beta=beta)
            optimizer.step()
        if epoch % 75 == 0:
            kl = model.kl_loss()
            print(f"Epoch {epoch:3d}  recon={np.mean(recon_losses):.4f}  kl={kl:.4f}")

    recon = model.forward(x)
    print(f"\nFinal reconstruction MSE: {float(np.mean((recon - x) ** 2)):.4f}")

    z_sample = model.sample(4)
    print(f"Sampled points from prior:\n{z_sample.round(2)}")


if __name__ == "__main__":
    main()
