"""Country sequences and padded batches for the temporal model."""

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class COVIDSequenceDataset(Dataset):
    """
    Group corresponding feature and raw rows into country sequences.

    Both frames must contain the same rows in the same order; their index
    labels may differ. Rows must be in time order within each country.
    The returned row positions allow inference to restore the input order.
    """

    def __init__(
        self,
        df_norm: pd.DataFrame,
        df_raw: pd.DataFrame,
        temp_cols: list[str],
        stat_cols: list[str],
        target_col: str = "target_surge",
    ):
        if len(df_norm) != len(df_raw):
            raise ValueError("Feature and raw frames must have the same number of rows.")
        if df_raw["iso_code"].isna().any():
            raise ValueError("Country codes must not be missing.")

        self.temp_seqs = []
        self.stat_seqs = []
        self.targets = []
        self.orig_indices = []

        for positions in df_raw.groupby("iso_code", sort=False).indices.values():
            features = df_norm.iloc[positions]
            group = df_raw.iloc[positions]

            # Extract temporal sequences and static context
            temp_val = features[temp_cols].to_numpy(dtype=np.float32)
            stat_val = features[stat_cols].iloc[0].to_numpy(dtype=np.float32)

            # Handle targets (default to zeros during inference if target_col is absent)
            if target_col in group.columns:
                target_val = group[target_col].values.astype(np.float32)
            else:
                target_val = np.zeros(len(positions), dtype=np.float32)

            self.temp_seqs.append(torch.tensor(temp_val))
            self.stat_seqs.append(torch.tensor(stat_val))
            self.targets.append(torch.tensor(target_val))

            self.orig_indices.append(positions)

    def __len__(self):
        return len(self.temp_seqs)

    def __getitem__(self, idx):
        return self.temp_seqs[idx], self.stat_seqs[idx], self.targets[idx], self.orig_indices[idx]


def collate_sequences(batch):
    """
    Right-pad temporal sequences and labels while retaining row positions.
    """
    temps, stats, targets, orig_indices = zip(*batch)

    temps_padded = pad_sequence(temps, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0.0)
    stats_stacked = torch.stack(stats)

    return temps_padded, stats_stacked, targets_padded, orig_indices


def make_sequence_loader(
    dataset: COVIDSequenceDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a loader with the sequence dataset's padding and collation rules.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sequences,
        num_workers=num_workers,
    )
