import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping implementation.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change in monitored quantity to qualify as improvement.
        load_best_model: Whether to save and restore the best model state.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum improvement threshold.
        counter: Current number of epochs without improvement.
        best_loss: Best validation loss observed so far.
        best_model_state: Saved state dict of the best model.
        load_best_model: Flag for model state restoration.
    """

    def __init__(
        self, patience: int = 20, min_delta: float = 0.0, load_best_model: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.load_best_model = load_best_model

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.load_best_model:
                self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing learning rate scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer to schedule.
        warmup: Number of warmup steps for linear increase.
        max_iters: Total number of training iterations.
        min_lr: Minimum learning rate after annealing.

    Attributes:
        warmup: Number of warmup iterations.
        max_num_iters: Maximum number of training iterations.
        min_lr: Minimum learning rate value.
        max_lr: Maximum learning rate from optimizer.
        lr: Current learning rate value.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup: int,
        max_iters: int,
        min_lr: float,
    ):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_lr = min_lr
        self.max_lr = max(group["lr"] for group in optimizer.param_groups)
        self.lr: Optional[float] = None
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        lr = self._get_lr(self.last_epoch)
        self.lr = lr
        return [lr for _ in self.optimizer.param_groups]

    def _get_lr(self, it: int) -> float:
        if it < self.warmup:
            return self.max_lr * (it + 1) / self.warmup
        if it > self.max_num_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup) / (self.max_num_iters - self.warmup)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def get_last_lr(self) -> List[float]:
        return [self.lr]


class LROnPlateauScheduler:
    """Learning rate scheduler that reduces LR when validation loss plateaus.

    Args:
        optimizer: PyTorch optimizer to schedule.
        min_lr: Minimum learning rate threshold.
        factor: Factor by which to reduce learning rate.
        patience: Number of steps to wait before reducing learning rate.
        threshold: Relative threshold for measuring improvement.

    Attributes:
        optimizer: The optimizer being scheduled.
        min_lr: Minimum allowed learning rate.
        factor: Learning rate reduction factor.
        patience: Steps to wait before reduction.
        threshold: Improvement threshold as relative change.
        counter: Current number of steps without improvement.
        best_val_loss: Best validation loss observed so far.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr: float = 1e-8,
        factor: float = 0.1,
        patience: int = 3,
        threshold: float = 0.01,
    ):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_val_loss = float("inf")

    def step(self, val_loss: Union[float, List[float]]) -> None:
        if isinstance(val_loss, list):
            val_loss = sum(val_loss)

        if val_loss < self.best_val_loss * (1 - self.threshold):
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for group in self.optimizer.param_groups:
                    group["lr"] *= self.factor
                    if group["lr"] < self.min_lr:
                        group["lr"] = self.min_lr
                self.counter = 0

    def get_last_lr(self) -> List[float]:
        return [self.optimizer.param_groups[0]["lr"]]
