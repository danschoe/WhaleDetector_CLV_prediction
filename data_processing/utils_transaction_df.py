import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl


def group_by_customer(df_pl: pl.DataFrame, group_by_channel_id: bool = False):
    """Aggregates purchase data by customer and date with optional sales channel grouping.

    Args:
        df_pl: Polars DataFrame containing purchase data with columns:
            - customer_id: Customer identifier
            - date: Purchase date
            - article_id: Product identifier
            - price: Individual item price
            - sales_channel_id: Sales channel identifier
        group_by_channel_id: If True, includes sales_channel_id in grouping for
            multivariate analysis. If False, groups only by customer_id and date.

    Returns:
        Polars DataFrame with aggregated purchase data containing:
            - customer_id: Customer identifier
            - date: Purchase date
            - sales_channel_id: Sales channel (if group_by_channel_id=True)
            - article_ids: List of article IDs purchased
            - total_price: Sum of all item prices, rounded to 2 decimals
            - prices: List of individual item prices
            - sales_channel_ids: List of sales channels (if group_by_channel_id=False)
            - num_items: Count of items purchased
    """
    # Used for multi variate time series
    if group_by_channel_id:
        grouped_df = (
            df_pl.lazy()
            .group_by(["customer_id", "date", "sales_channel_id"])
            .agg(
                [
                    pl.col("article_id").explode().alias("article_ids"),
                    pl.col("price").sum().round(2).alias("total_price"),
                    pl.col("price").explode().alias("prices"),
                ]
            )
            .with_columns(pl.col("article_ids").list.len().alias("num_items"))
        )
    else:
        grouped_df = (
            df_pl.lazy()
            .group_by(["customer_id", "date"])
            .agg(
                [
                    pl.col("article_id").explode().alias("article_ids"),
                    pl.col("price").sum().round(2).alias("total_price"),
                    pl.col("sales_channel_id").explode().alias("sales_channel_ids"),
                    pl.col("price").explode().alias("prices"),
                ]
            )
            .with_columns(pl.col("article_ids").list.len().alias("num_items"))
        )

    return grouped_df.collect()


def train_test_split(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    subset: int = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Splits data into train, validation, and test sets with optional subsampling.

    The function performs the following operations:
    1. Optional subsampling of both train and test data
    2. Optional percentage-based subsampling of training data
    3. Creates a validation set from 10% of the training data

    Args:
        train_df (pl.DataFrame): Training dataset.
        test_df (pl.DataFrame): Test dataset.
        subset (int, optional): If provided, limits both train and test sets to first n rows.
            Defaults to None.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Tuple containing:
            - train_df: Final training dataset (90% of training data after subsampling)
            - val_df: Validation dataset (10% of training data)
            - test_df: Test dataset (potentially subsampled)
    """

    if subset is not None:
        train_df = train_df[:subset]
        test_df = test_df[:subset]

    # Train-val-split
    # Calculate 10% of the length of the array
    sampled_indices = np.random.choice(
        len(train_df), size=int(0.1 * len(train_df)), replace=False
    )
    val_df = train_df[sampled_indices]
    train_df = train_df.filter(~pl.arange(0, pl.count()).is_in(sampled_indices))

    return train_df, val_df, test_df


def map_article_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Maps article IDs to new running IDs by sorting unique article IDs and assigning sequential IDs.

    Args:
        df (pl.DataFrame): DataFrame with 'article_id' column to be mapped.

    Returns:
        pl.DataFrame: DataFrame with mapped article IDs, where original article IDs are replaced with running IDs (0, 1, 2, ...).
    """
    # Get unique article IDs and sort them
    unique_article_ids = df.select("article_id").unique().sort("article_id")

    # Create mapping from original article_id to running id
    mapping_df = unique_article_ids.with_row_index("new_id").with_columns(
        pl.col("article_id").alias("old_id"), pl.col("new_id").cast(pl.Int32)
    )

    # Join and replace article_id with new running id
    df = df.join(
        mapping_df, left_on="article_id", right_on="old_id", how="inner"
    ).select(
        pl.col("new_id").alias("article_id"),
        pl.all().exclude(["article_id", "old_id", "new_id"]),
    )
    df = df.sort("article_id")

    return df
