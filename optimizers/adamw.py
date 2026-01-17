import torch

from optimizers.scheduler import WarmupCosineScheduler


def build_adamw_optimizer(model, lr=3e-4, weight_decay=0.01, warmup_steps=0, max_steps=1000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, max_steps)
    return optimizer, scheduler
