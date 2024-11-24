import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class StockDataLoader:
    def __init__(self, csv_file, sequence_length=60, horizon=60, batch_size=32, train_ratio=0.7, val_ratio=0.15, asset_column=None, top_n_symbols=5, normalize_method="zscore", start_cutoff=0.0):
        self.csv_file = csv_file
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.asset_column = asset_column
        self.top_n_symbols = top_n_symbols  # Limit to top N symbols
        self.normalize_method = normalize_method
        self.start_cutoff = start_cutoff  # Remove the first `start_cutoff` percent of rows

        self._prepare_data()
        self._create_datasets()
        self._create_dataloaders()

    def _prepare_data(self):
        # Load and sort data
        data = pd.read_csv(self.csv_file, parse_dates=['date'], index_col='date')
        data.sort_index(inplace=True)

        # Apply start cutoff
        cutoff_idx = int(len(data) * self.start_cutoff)
        data = data.iloc[cutoff_idx:]  # Remove the first `start_cutoff` percent of rows

        # Select the first `top_n_symbols` columns if asset_column is not specified
        if self.asset_column is not None:
            data = data[[self.asset_column]]
        else:
            data = data.iloc[:, :self.top_n_symbols]

        # Calculate indices for splitting
        train_end = int(len(data) * self.train_ratio)
        val_end = train_end + int(len(data) * self.val_ratio)

        # Assign data splits
        self.train_data = data.iloc[:train_end]
        self.val_data = data.iloc[train_end:val_end]
        self.test_data = data.iloc[val_end:]

        # Store unscaled returns (target values) separately
        self.train_returns = self.train_data.copy()
        self.val_returns = self.val_data.copy()
        self.test_returns = self.test_data.copy()

    def _create_datasets(self):
        self.train_dataset = StockDataset(
            self.train_data,
            self.train_returns,  # Pass unscaled returns
            self.sequence_length,
            self.horizon,
            self.normalize_method
        )
        self.val_dataset = StockDataset(
            self.val_data,
            self.val_returns,  # Pass unscaled returns
            self.sequence_length,
            self.horizon,
            self.normalize_method
        )
        self.test_dataset = StockDataset(
            self.test_data,
            self.test_returns,  # Pass unscaled returns
            self.sequence_length,
            self.horizon,
            self.normalize_method
        )

    def _create_dataloaders(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader


class StockDataset(Dataset):
    def __init__(self, data, returns, sequence_length, horizon, normalize_method):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.data = data.values  # Original input data
        self.returns = returns.values  # Unscaled returns
        self.num_samples = len(self.data) - sequence_length - horizon + 1
        self.normalize_method = normalize_method

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx + self.sequence_length + self.horizon > len(self.data):
            raise IndexError("Sequence length exceeds dataset length.")

        seq_x = self.data[idx: idx + self.sequence_length]
        seq_y = self.returns[idx + self.sequence_length: idx + self.sequence_length + self.horizon]

        # Normalize the batch
        if self.normalize_method == "minmax":
            scaler = MinMaxScaler()
        elif self.normalize_method == "zscore":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalize_method}")

        seq_x = scaler.fit_transform(seq_x)

        seq_x = torch.tensor(seq_x, dtype=torch.float32)  # Scaled input
        seq_y = torch.tensor(seq_y, dtype=torch.float32)  # Unscaled target
        return seq_x, seq_y


# Example usage
if __name__ == "__main__":
    # Create data loader for a specific asset
    data_loader = StockDataLoader(
        csv_file='data/grouped_data_return_daily.csv',
        sequence_length=60,
        horizon=7,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        asset_column="AAPL"  # Specify the column for a specific asset, e.g., "AAPL"
    )

    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()

    seq_x, seq_y = next(iter(train_loader))
    print("Input sequence shape:", seq_x.shape)  
    print("Target returns shape:", seq_y.shape)
