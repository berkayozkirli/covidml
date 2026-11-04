"""Causal temporal convolutional network architecture."""

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0
        ))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Pad only the left side (past time steps) of the temporal dimension
        x_padded = nn.functional.pad(x, (self.padding, 0))
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv(x_padded)
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, temp_channels, static_channels, hidden_channels, num_layers, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_c = temp_channels if i == 0 else hidden_channels
            layers.append(TCNResidualBlock(in_c, hidden_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_channels + static_channels, 1)

    def forward(self, x_temp, x_stat):
        x = x_temp.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)

        x_stat_expanded = x_stat.unsqueeze(1).repeat(1, x.shape[1], 1)
        combined = torch.cat([x, x_stat_expanded], dim=-1)

        return self.final_linear(combined).squeeze(-1)

