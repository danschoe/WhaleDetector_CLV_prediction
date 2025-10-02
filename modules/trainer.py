from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing.dataset import TimeSeriesDataset
from modules.modling_utils import (
    CosineWarmupScheduler,
    EarlyStopping,
    LROnPlateauScheduler,
)


class Trainer:
    """
    Trainer class for LTV prediction models following the paper's methodology.

    Args:
        model: PyTorch model to train.
        config: Configuration dictionary containing training parameters.
        device: Device to train on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.config = config
        self.bfloat_16 = config.get("bfloat_16", False)

        # Setup optimizer based on weight decay
        weight_decay = self.config.get("weight_decay", 0.0)
        if weight_decay is not None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=float(config["learning_rate"]),
                betas=config["betas"],
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(config["learning_rate"]),
                betas=config["betas"],
            )

        self.criterion = nn.MSELoss()

        # Setup early stopping
        patience = self.config.get("patience", 10)
        delta = self.config.get("min_delta", 0.001)
        self.load_best_model = self.config.get("load_best_model", True)
        self.early_stopping = EarlyStopping(
            patience=patience, min_delta=delta, load_best_model=self.load_best_model
        )

        # Gradient clipping
        self.grad_clip_norm: Optional[float] = self.config.get("grad_clip_norm")

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def prepare_data(
        self,
        train_data: TimeSeriesDataset,
        val_data: TimeSeriesDataset,
        test_data: TimeSeriesDataset,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            test_data: Test dataset.
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loading.

        Returns:
            Tuple of train, validation, and test data loaders.
        """

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for customer_id, batch_X, batch_y in tqdm(train_loader):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.bfloat_16 and self.device.type == "cuda",
            ):
                outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping if specified
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()

            # LR scheduler step for cosine scheduler (per iteration)
            if isinstance(self.lr_scheduler, CosineWarmupScheduler):
                self.lr_scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(
        self, val_loader: DataLoader, return_customer_id: bool = False
    ) -> Union[float, Tuple[float, List[Any]]]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader.
            return_customer_id: Whether to return customer IDs.

        Returns:
            Validation loss, optionally with customer IDs.
        """
        self.model.eval()
        total_loss = 0.0
        customer_ids: List[Any] = []

        with torch.no_grad():
            for customer_id, batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.bfloat_16 and self.device.type == "cuda",
                ):
                    outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                if return_customer_id:
                    customer_ids.extend(customer_id.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        if return_customer_id:
            return avg_loss, customer_ids
        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Train the model with early stopping as described in the paper.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Dictionary containing training history and best validation loss.
        """
        max_epochs = self.config["num_epochs"]
        print(f"Training {self.model.__class__.__name__} on {self.device}")

        # Setup lr scheduler if specified
        self.lr_scheduler: Optional[
            Union[CosineWarmupScheduler, LROnPlateauScheduler]
        ] = None
        lr_scheduler_type = self.config.get("lr_scheduler")
        if lr_scheduler_type == "cosine":
            lr_config = self.config.get("lr_scheduler_config", {})
            warmup = lr_config.get("warmup_steps", 1000)

            max_iters = lr_config.get("max_iters")
            if max_iters is None:
                max_iters = max_epochs * len(train_loader)

            min_lr = float(lr_config.get("min_lr", 1e-5))
            self.lr_scheduler = CosineWarmupScheduler(
                self.optimizer, warmup, max_iters, min_lr
            )
        elif lr_scheduler_type == "on_plateau":
            lr_config = self.config.get("lr_scheduler_config", {})
            min_lr = float(lr_config.get("min_lr", 1e-5))
            factor = lr_config.get("factor", 0.1)
            lr_patience = lr_config.get("lr_patience", 3)
            threshold = lr_config.get("threshold", 0.01)
            self.lr_scheduler = LROnPlateauScheduler(
                self.optimizer, min_lr, factor, lr_patience, threshold
            )

        for epoch in range(max_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # LR scheduler step for plateau scheduler (per epoch)
            if isinstance(self.lr_scheduler, LROnPlateauScheduler):
                self.lr_scheduler.step(val_loss)

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                # Load best model state
                if self.load_best_model:
                    self.model.load_state_dict(self.early_stopping.best_model_state)
                break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.early_stopping.best_loss,
        }

    def evaluate(
        self, test_loader: DataLoader
    ) -> Tuple[pl.DataFrame, Dict[str, float]]:
        """
        Evaluate the model on test set.

        Args:
            test_loader: Test data loader.

        Returns:
            Tuple of predictions DataFrame and evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        predictions: List[float] = []
        targets: List[float] = []
        customer_ids: List[Any] = []

        with torch.no_grad():
            for customer_id, batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.bfloat_16 and self.device.type == "cuda",
                ):
                    outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
                customer_ids.extend(customer_id)

        predictions_array = np.array(predictions)
        targets_array = np.array(targets)

        df = pl.DataFrame(
            {
                "customer_id": customer_ids,
                "predictions": predictions_array,
            }
        )

        # Calculate metrics mentioned in the paper
        mse = np.mean((predictions_array - targets_array) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_array - targets_array))
        wmape = np.sum(np.abs(predictions_array - targets_array)) / np.sum(
            targets_array
        )

        return df, {
            "test_loss": total_loss / len(test_loader),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "wmape": wmape,
        }
