"""
Mixture-of-Experts feed-forward layer with top-k routing.

Dense educational implementation: every expert runs on every token and the
outputs are mixed by the (sparse) gates, so there is no scatter/gather and the
shapes stay static in jax mode. Real MoE saves FLOPs by dispatching each token
to only its top-k experts — here the routing math is the point, not the speed
(same spirit as FlashAttention-in-numpy).
"""
import copy

from core.backend import xp as np, randn
from core.module import Layer
from layers.linear import Linear
from layers.feedforward import FeedForward


class MoEFeedForward(Layer):
    """Drop-in FFN replacement: y = Σ_e gate_e(x) · expert_e(x).

    Router: Linear(d_model, n_experts) → softmax → keep each token's top-k
    probabilities → renormalize over the kept experts. Experts are independent
    ffn_cls instances (FeedForward or SwiGLU).

    Use `MoEFeedForward.from_dense(block.ffn, ...)` to upcycle a trained dense
    FFN: each expert starts as a copy and the router starts at zero, so the
    MoE output exactly equals the dense output at init.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, n_experts: int = 4,
                 top_k: int = 2, ffn_cls=FeedForward, activation_cls=None):
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = Linear(d_model, n_experts)
        self.experts = [ffn_cls(d_model, d_ff, activation_cls=activation_cls)
                        for _ in range(n_experts)]

    @classmethod
    def from_dense(cls, ffn, n_experts: int = 4, top_k: int = 2,
                   router_scale: float = 0.0) -> "MoEFeedForward":
        """Upcycle a trained dense FFN (Komatsuzaki et al. 2022).

        Experts are deep copies of `ffn`. With router_scale=0 the router is
        zero and gates are uniform over identical experts, so the MoE output
        equals the dense FFN exactly — but that symmetry never breaks under
        training (identical experts get identical gradients and the router
        gradient is exactly zero). Pass router_scale > 0 (e.g. 1e-2) to jitter
        the router: outputs stay near-identical while the slightly unequal
        gates let the experts specialize.
        """
        d_model = ffn.linear2.W.data.shape[1]
        moe = cls.__new__(cls)
        moe.n_experts, moe.top_k = n_experts, top_k
        moe.router = Linear(d_model, n_experts)
        moe.router.W.data = randn(d_model, n_experts) * router_scale
        moe.experts = [copy.deepcopy(ffn) for _ in range(n_experts)]
        return moe

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits = self.router.forward(x)                        # (..., E)
        z = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = z / z.sum(axis=-1, keepdims=True)
        threshold = np.sort(probs, axis=-1)[..., -self.top_k, None]
        mask = np.where(probs >= threshold, 1.0, 0.0)          # ties keep >k experts
        gates = probs * mask
        gates = gates / gates.sum(axis=-1, keepdims=True)
        self._probs, self._mask, self._gates = probs, mask, gates
        self._expert_out = [e.forward(x) for e in self.experts]
        out = self._gates[..., 0, None] * self._expert_out[0]
        for e in range(1, self.n_experts):
            out = out + self._gates[..., e, None] * self._expert_out[e]
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # d_gate_e = ⟨grad, expert_e(x)⟩ per token
        d_gates = np.stack(
            [(grad * out).sum(axis=-1) for out in self._expert_out], axis=-1
        )                                                      # (..., E)
        # gates = mask·probs / Σ(mask·probs), mask treated as constant
        denom = (self._mask * self._probs).sum(axis=-1, keepdims=True)
        d_probs = self._mask / denom * (
            d_gates - (d_gates * self._gates).sum(axis=-1, keepdims=True)
        )
        d_logits = self._probs * (
            d_probs - (d_probs * self._probs).sum(axis=-1, keepdims=True)
        )
        dx = self.router.backward(d_logits)
        for e, expert in enumerate(self.experts):
            dx = dx + expert.backward(grad * self._gates[..., e, None])
        return dx

    def parameters(self) -> list:
        params = self.router.parameters()
        for expert in self.experts:
            params.extend(expert.parameters())
        return params
