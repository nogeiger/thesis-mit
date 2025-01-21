import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class NoisePredictor(nn.Module):
    """
    A feedforward neural network to predict clean 3D trajectories from noisy inputs, 
    including forces as extra features if the flag is set.
    """
    def __init__(self, seq_length, hidden_dim, use_forces=False):
        super(NoisePredictor, self).__init__()
        self.use_forces = use_forces
        input_dim = seq_length * 3  # Clean and noisy trajectories (pos_0 and pos)
        
        if self.use_forces:
            input_dim += seq_length * 3  # Add forces (force_x, force_y, force_z)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, seq_length * 3)  # Output clean trajectory (pos_0)
        self.relu = nn.ReLU()

    def forward(self, noisy_trajectory, forces=None):
        """
        Forward pass to predict clean 3D trajectory.

        Args:
            noisy_trajectory (torch.Tensor): Input noisy trajectory of shape [batch_size, seq_length, 3].
            forces (torch.Tensor, optional): Input forces of shape [batch_size, seq_length, 3].

        Returns:
            torch.Tensor: Predicted clean trajectory of shape [batch_size, seq_length, 3].
        """
        batch_size, seq_length, _ = noisy_trajectory.shape

        # If using forces, concatenate them with noisy trajectory
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)  # Concatenate noisy trajectory and forces
        else:
            x = noisy_trajectory

        x = x.view(batch_size, -1)  # Flatten to [batch_size, seq_length * 6] if forces are included
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        predicted_noise = self.output_layer(x)
        return predicted_noise.view(batch_size, seq_length, 3)  # Reshape back to [batch_size, seq_length, 3]

class NoisePredictorLSTM(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=False):
        super(NoisePredictorLSTM, self).__init__()
        self.use_forces = use_forces
        input_dim = 3  # Each timestep has (x, y, z)

        if self.use_forces:
            input_dim += 3  # Add forces (force_x, force_y, force_z)

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)  # Two LSTM layers
        self.fc = nn.Linear(hidden_dim, 3)  # Predict (x, y, z) noise for each timestep

    def forward(self, noisy_trajectory, forces=None):
        batch_size, seq_length, _ = noisy_trajectory.shape

        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)  # Concatenate noisy trajectory and forces
        else:
            x = noisy_trajectory

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Shape: [batch_size, seq_length, hidden_dim]

        # Predict noise for each timestep
        predicted_noise = self.fc(lstm_out)  # Shape: [batch_size, seq_length, 3]

        return predicted_noise