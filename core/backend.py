"""Array backend selection: numpy (default) or JAX.

Select the backend with an environment variable, set BEFORE importing any
tiny_ml module:

    TINY_ML_BACKEND=jax python -c "from examples.gpt2 import main; main()"

In jax mode every `np.` call inside the library resolves to `jax.numpy`, so
matmuls and elementwise ops run through XLA. float64 is enabled by default so
results match numpy exactly; set TINY_ML_JAX_X64=0 for float32, which is
considerably faster and the usual choice when speed is the point.

JAX arrays are immutable, so the library never mutates arrays in place:
gradient accumulation rebinds attributes (`p.grad = p.grad + g`, which is what
`p.grad += g` falls back to), and scattered writes go through `scatter_add`
below. Randomness always comes from real numpy (`randn`, `sample_categorical`)
so seeding behaves identically in both modes.
"""
import os

import numpy as _np

BACKEND = os.environ.get("TINY_ML_BACKEND", "numpy").lower()

if BACKEND == "jax":
    from jax import config as _jax_config
    _jax_config.update("jax_enable_x64", os.environ.get("TINY_ML_JAX_X64", "1") != "0")
    import jax.numpy as xp
else:
    BACKEND = "numpy"
    xp = _np


def scatter_add(arr, index, updates):
    """`arr[index] += updates` that works on both backends.

    Returns the updated array — always assign the result back
    (`a = scatter_add(a, idx, u)`); numpy mutates in place, JAX cannot.
    `index` may be anything numpy fancy indexing accepts, including slices
    and tuples of arrays.
    """
    if BACKEND == "jax":
        # asarray: tolerate plain-numpy targets (e.g. user-zeroed grads)
        return xp.asarray(arr).at[index].add(updates)
    _np.add.at(arr, index, updates)
    return arr


def randn(*shape):
    """Standard-normal sample as a backend array, drawn with numpy's RNG
    so `np.random.seed(...)` gives identical draws in both modes."""
    return xp.asarray(_np.random.randn(*shape))


def sample_categorical(probs) -> int:
    """Draw one index from a probability vector (any backend's array)."""
    p = _np.asarray(probs, dtype=_np.float64)
    return int(_np.random.choice(p.size, p=p / p.sum()))


def to_numpy(x) -> _np.ndarray:
    return _np.asarray(x)
