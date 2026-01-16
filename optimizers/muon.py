import torch
from optimizers.adamw import WarmupCosineScheduler

def build_muon_optimizer(model, lr=1e-3, weight_decay=0.01, warmup_steps=0, max_steps=1000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, max_steps)
    return optimizer, scheduler
