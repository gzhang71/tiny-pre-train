"""
Checkpoint utilities: snapshot, restore, and average model parameters.

A "state" is a plain list of arrays in `model.parameters()` order — the same
ordering contract the optimizers rely on. Averaging states (over an annealing
window, or SWA-style over late training) often beats the last noisy checkpoint.
"""
from core.backend import xp as np


def get_state(model) -> list:
    """Snapshot all parameters as copies (safe to keep while training continues)."""
    return [np.array(p.data) for p in model.parameters()]


def set_state(model, state: list) -> None:
    """Restore a snapshot taken by get_state on the same architecture."""
    for p, arr in zip(model.parameters(), state, strict=True):
        p.data = arr


def average_states(states: list[list]) -> list:
    """Elementwise mean of several states (checkpoint averaging)."""
    n = len(states)
    return [sum(arrs) / n for arrs in zip(*states, strict=True)]
