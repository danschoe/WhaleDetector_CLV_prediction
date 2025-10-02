import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from data_processing.get_data import get_datasets
from modules.trainer import Trainer
from modules.whale_detector import CNNModel


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():

    # parser = argparse.ArgumentParser(description="Train whale detector model")
    # parser.add_argument(
    #     "--config", type=str, required=True, help="Path to the config.yaml file"
    # )
    # args = parser.parse_args()

    # Load configuration
    config = load_config("configs/config.yaml")

    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    data_path = Path(config["data_path"])

    train_begin = datetime.strptime(config.get("train_begin"), "%Y-%m-%d")
    train_label_begin = datetime.strptime(config.get("train_label_begin"), "%Y-%m-%d")
    sequence_length = (train_label_begin - train_begin).days

    config["sequence_length"] = sequence_length

    train_data, val_data, test_data = get_datasets(data_path=data_path, config=config)

    # Train Single Time Series CNN Model
    print("\n" + "=" * 50)
    print("Training Single Time Series CNN Model")
    print("=" * 50)

    device = config["device"] if torch.cuda.is_available() else "cpu"
    device = device if not torch.backends.mps.is_available() else "mps"

    cnn_model = CNNModel(
        input_channels=config["input_channels"],
        sequence_length=config["sequence_length"],
    )

    if config.get("compile", False):
        cnn_model = torch.compile(cnn_model)

    cnn_trainer = Trainer(cnn_model, config=config, device=device)

    cnn_train_loader, cnn_val_loader, cnn_test_loader = cnn_trainer.prepare_data(
        train_data,
        val_data,
        test_data,
        batch_size=config.get("batch_size", 64),
        num_workers=config.get("num_workers", 0),
    )

    cnn_results = cnn_trainer.train(cnn_train_loader, cnn_val_loader)
    print(f"Best Validation Loss: {cnn_results['best_val_loss']:.4f}")

    print("\nEvaluating on Test Set...")
    predictions_df, cnn_metrics = cnn_trainer.evaluate(cnn_test_loader)

    predictions_path = data_path / "predictions.csv"

    predictions_df.write_csv(predictions_path)

    print("\nCNN Model Test Metrics:")
    for key, value in cnn_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
