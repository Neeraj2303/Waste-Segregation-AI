import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine Learning Rate Scheduler with Warmup.
    Linearly warms up learning rate from 0 to base_lr over warmup_epochs,
    then performs cosine annealing decay down to min_lr over the remaining epochs.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
            
        epoch = self.last_epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup: scale from min_lr to base_lr
            alpha = epoch / max(1, self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing decay from base_lr to min_lr
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            # Clip progress to [0.0, 1.0]
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
