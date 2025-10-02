from pathlib import Path

from data_processing.dataset import TimeSeriesDataset
from data_processing.transaction_df import get_tx_article_dfs


def get_datasets(data_path: Path, config: dict):
    train_article, val_article, test_article = get_tx_article_dfs(
        data_path=data_path,
        config=config,
        cols_to_aggregate=[
            "days_before",
            "total_price",
        ],
        keep_customer_id=True,
    )
    train_dataset = TimeSeriesDataset(
        dataframe=train_article, sequence_length=config["sequence_length"]
    )
    val_dataset = TimeSeriesDataset(
        dataframe=val_article, sequence_length=config["sequence_length"]
    )
    test_dataset = TimeSeriesDataset(
        dataframe=test_article, sequence_length=config["sequence_length"]
    )

    return train_dataset, val_dataset, test_dataset
