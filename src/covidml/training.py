"""Explicit PyTorch training and inference for country sequence models."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_tcn(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int = 20,
    learning_rate: float = 0.001,
    pos_weight: float = 2.0,
    verbose: bool = False,
) -> list[float]:
    """
    Train a model in place and return the mean loss for each epoch.

    The loader supplies temporal features, static features, targets, and
    original row positions. Padded timesteps do not contribute to the loss.
    Tensors follow the model's device; set the random seed before building
    the model when reproducible initialization and shuffling are needed.
    """
    if len(loader) == 0:
        raise ValueError("Training requires a non-empty data loader.")

    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device), reduction="none"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_steps = 0
        for temps, stats, targets, positions in loader:
            temps, stats, targets = temps.to(device), stats.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(temps, stats)
            lengths = torch.tensor([len(rows) for rows in positions], device=device)
            valid = torch.arange(targets.shape[1], device=device)[None, :] < lengths[:, None]
            step_losses = criterion(logits, targets)[valid]
            step_losses.mean().backward()
            optimizer.step()
            total_loss += step_losses.detach().sum().item()
            total_steps += step_losses.numel()

        average_loss = total_loss / total_steps
        losses.append(average_loss)
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"Epoch [{epoch + 1:02d}/{epochs}] | Loss: {average_loss:.4f}")
    return losses


def predict_tcn(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """
    Return surge probabilities in input row order, discarding padding.

    Use a loader that visits each country once, without dropping batches.
    Batch size and country shuffling do not change the returned row order.
    """
    device = next(model.parameters()).device
    model.eval()
    probabilities = []
    row_positions = []
    with torch.inference_mode():
        for temps, stats, _, positions in loader:
            logits = model(temps.to(device), stats.to(device))
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            for index, rows in enumerate(positions):
                probabilities.append(batch_probs[index, :len(rows)])
                row_positions.append(rows)

    if not probabilities:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(np.concatenate(row_positions))
    return np.concatenate(probabilities)[order]
