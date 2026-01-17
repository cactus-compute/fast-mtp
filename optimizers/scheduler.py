import math
import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1, min_lr_ratio=0.1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio
            if self.warmup_steps > 0 and step <= self.warmup_steps:
                lr = base_lr * step / max(1, self.warmup_steps)
            else:
                if self.max_steps <= self.warmup_steps or self.max_steps <= 0:
                    lr = base_lr
                else:
                    progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                    progress = min(max(progress, 0.0), 1.0)
                    lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
