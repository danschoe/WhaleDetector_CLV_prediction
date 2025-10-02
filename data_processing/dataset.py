import warnings
from typing import Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for Customer Lifetime Value time series data.

    The dataset creates 1D time series from sparse day-price pairs where
    days_before_lst contains indices (1-360) counted from the back, and
    total_price_lst contains corresponding price values.

    Args:
        dataframe: Polars DataFrame containing columns:
                   - customer_id (str): Customer identifier
                   - days_before_lst (list[int]): Days before (1-360)
                   - total_price_lst (list[float]): Price values
                   - CLV_label (float): Customer lifetime value label
        sequence_length: Length of the output time series (default 360)
        enable_shared_memory: Whether to move tensors to shared memory (default True)

    Returns:
        Tuple of (time_series, customer_id, clv_label) where:
        - time_series: 1D tensor of shape (sequence_length,)
        - customer_id: String customer identifier
        - clv_label: Scalar tensor with CLV label
    """

    def __init__(
        self,
        dataframe: pl.DataFrame,
        sequence_length: int = 360,
        enable_shared_memory: bool = True,
    ) -> None:
        """
        Initialize the dataset with efficient tensor-based storage.

        Args:
            dataframe: Input Polars DataFrame with required columns
            sequence_length: Length of output time series
            enable_shared_memory: Whether to enable shared memory for multiprocessing
        """
        self.sequence_length = sequence_length
        self.enable_shared_memory = enable_shared_memory
        self._length = len(dataframe)

        # Validate required columns
        required_cols = {
            "customer_id",
            "days_before_lst",
            "total_price_lst",
            "CLV_label",
        }
        available_cols = set(dataframe.columns)
        if not required_cols.issubset(available_cols):
            missing = required_cols - available_cols
            raise ValueError(f"Missing required columns: {missing}")

        # Pre-process and validate all data
        self._preprocess_data(dataframe)

        # Convert to memory-efficient tensor storage
        self._create_tensor_storage()

        # Enable shared memory for multiprocessing efficiency
        if self.enable_shared_memory:
            self._enable_shared_memory()

    def _preprocess_data(self, dataframe: pl.DataFrame) -> None:
        """Preprocess and validate all data during initialization."""
        # Extract and validate data
        customer_ids = dataframe["customer_id"].to_list()
        days_before_lists = dataframe["days_before_lst"].to_list()
        price_lists = dataframe["total_price_lst"].to_list()
        clv_labels = np.array(dataframe["CLV_label"].to_list())

        # Pre-validate and determine maximum sequence size needed
        max_seq_len = 0
        for i in range(self._length):
            days_before = days_before_lists[i]
            total_prices = price_lists[i]

            if len(days_before) != len(total_prices):
                raise ValueError(
                    f"Row {i}: Mismatched lengths: days_before_lst({len(days_before)}) "
                    f"vs total_price_lst({len(total_prices)})"
                )

            if len(days_before) > 0:
                min_day, max_day = min(days_before), max(days_before)
                if min_day < 1 or max_day > self.sequence_length:
                    raise ValueError(
                        f"Row {i}: days_before_lst values must be in range [1, {self.sequence_length}]"
                    )
                max_seq_len = max(max_seq_len, len(days_before))

        # Store for tensor creation
        self._customer_ids = customer_ids
        self._days_before_lists = days_before_lists
        self._price_lists = price_lists
        self._clv_labels = clv_labels
        self._max_seq_len = max_seq_len

    def _create_tensor_storage(self) -> None:
        """Convert all data to efficient tensor storage."""
        # Store CLV labels as tensor
        self._clv_tensor = torch.from_numpy(self._clv_labels).float()

        # Using a packed representation: [start_idx, length, day1, price1, day2, price2, ...]
        total_elements = sum(len(days) * 2 + 2 for days in self._days_before_lists)

        # Create packed storage tensor
        self._packed_data = torch.zeros(total_elements, dtype=torch.float32)
        self._row_offsets = torch.zeros(self._length, dtype=torch.long)

        offset = 0
        for i in range(self._length):
            days_before = self._days_before_lists[i]
            total_prices = self._price_lists[i]

            self._row_offsets[i] = offset

            # Pack: [start_offset, length, day1, price1, day2, price2, ...]
            seq_len = len(days_before)
            self._packed_data[offset] = offset  # Start offset (for validation)
            self._packed_data[offset + 1] = seq_len  # Length

            if seq_len > 0:
                # Interleave days and prices for cache efficiency
                for j, (day, price) in enumerate(zip(days_before, total_prices)):
                    self._packed_data[offset + 2 + j * 2] = float(day)
                    self._packed_data[offset + 2 + j * 2 + 1] = price

            offset += 2 + seq_len * 2

        # Clean up temporary data to save memory
        del self._days_before_lists
        del self._price_lists
        del self._clv_labels

    def _enable_shared_memory(self) -> None:
        """Move tensors to shared memory for efficient multiprocessing."""
        try:
            self._packed_data.share_memory_()
            self._row_offsets.share_memory_()
            self._clv_tensor.share_memory_()
        except RuntimeError as e:
            warnings.warn(
                f"Could not enable shared memory: {e}. "
                f"This may cause memory bloat with num_workers > 0"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """
        Retrieve a sample from the dataset with efficient tensor operations.

        The time series is assembled by mapping days_before_lst indices
        (counted from the back) to total_price_lst values:
        - day 1 maps to index -1 (last position)
        - day 360 maps to index -360 (first position)

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple containing:
                - time_series: 1D tensor of shape (sequence_length,) with prices
                - customer_id: Customer identifier string
                - clv_label: CLV label as scalar tensor

        Raises:
            IndexError: If idx is out of bounds
        """
        if idx >= self._length or idx < 0:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {self._length}"
            )

        # Initialize time series with zeros
        time_series = torch.zeros(self.sequence_length, dtype=torch.float32)

        # Get packed data for this row
        offset = self._row_offsets[idx].item()
        seq_len = int(self._packed_data[offset + 1].item())

        if seq_len > 0:
            # Extract days and prices efficiently
            start_idx = offset + 2
            end_idx = start_idx + seq_len * 2

            # Get interleaved data and reshape
            packed_row = self._packed_data[start_idx:end_idx]
            days_and_prices = packed_row.view(seq_len, 2)

            days = days_and_prices[:, 0].long()  # Convert to long for indexing
            prices = days_and_prices[:, 1]

            # Vectorized index mapping: day 1 -> index 359, day 360 -> index 0
            array_indices = self.sequence_length - days

            # Vectorized assignment
            time_series[array_indices] = prices

        return (
            self._customer_ids[idx],
            time_series.unsqueeze(0),  # Add channel dimension: (360,) -> (1, 360)
            self._clv_tensor[idx],
        )
