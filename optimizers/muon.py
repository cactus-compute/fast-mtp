"""
Muon optimizer - Momentum Orthogonalized Update.

Reference: https://github.com/KellerJordan/modded-nanogpt
"""

import torch
from torch.optim import Optimizer
from optimizers.adamw import WarmupCosineScheduler


def newton_schulz_(G, steps=5, eps=1e-7):
    """
    Apply Newton-Schulz iteration to orthogonalize G in-place.
    Computes approximate G @ (G^T @ G)^{-1/2}.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimal coefficients for 5 iterations
    X = G.to(torch.bfloat16)

    # If rows > cols, it's more efficient to orthogonalize the transpose
    # and then transpose back.
    if X.shape[0] > X.shape[1]:
        X = X.T

    # Normalize to ensure convergence
    X /= X.norm() + eps

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.shape[0] > G.shape[1]:
        X = X.T

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer: applies Newton-Schulz orthogonalization to gradients of 2D params.

    Args:
        params: Parameters to optimize (should be 2D weight matrices only)
        lr: Learning rate (typically 0.02 for Muon)
        momentum: Momentum coefficient (default 0.95)
        nesterov: Use Nesterov momentum (default True)
        ns_steps: Newton-Schulz iteration steps (default 5)
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Apply Newton-Schulz orthogonalization
                g = newton_schulz_(g, steps=ns_steps)

                # Scale by sqrt(ratio) to preserve update magnitude for rectangular matrices
                # This matches current Modded-NanoGPT practice
                scale = max(1, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(g, alpha=-lr * scale)


def build_muon_optimizer(model, lr=0.02, weight_decay=0.01, warmup_steps=0, max_steps=1000):
    """
    Build Muon optimizer with separate param groups:
    - 2D weight matrices (non-embedding): Muon
    - Embeddings, biases, norms, 1D params: AdamW
    """
    muon_params = []
    adamw_params = []
    seen_params = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # Avoid adding the same parameter twice if it's tied (e.g. embedding and head)
        if p in seen_params:
            continue
        seen_params.add(p)

        # Use Muon for 2D weights that aren't embeddings
        if p.ndim == 2 and "embedding" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    # Create optimizers
    optimizers = []

    if muon_params:
        muon_opt = Muon(muon_params, lr=lr)
        optimizers.append(muon_opt)

    if adamw_params:
        # AdamW for embeddings/norms with lower LR
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=lr * 0.1,  # 10x lower LR for non-Muon params
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
        optimizers.append(adamw_opt)

    # Combine into single optimizer interface
    combined_opt = CombinedOptimizer(optimizers)
    scheduler = WarmupCosineScheduler(combined_opt, warmup_steps, max_steps)

    return combined_opt, scheduler


class CombinedOptimizer(Optimizer):
    """Wraps multiple optimizers to provide a unified interface compatible with accelerate."""

    def __init__(self, optimizers):
        # Collect all params for the base Optimizer init
        all_params = []
        for opt in optimizers:
            for group in opt.param_groups:
                all_params.extend(group["params"])

        # Initialize base Optimizer with dummy defaults
        super().__init__(all_params, defaults={})

        # Store the actual optimizers
        self.optimizers = optimizers

        # Replace param_groups with the combined groups from all optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def state_dict(self):
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers]
        }

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)
