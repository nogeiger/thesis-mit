import os
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
import torch.nn.functional as F

#FF
class NoisePredictorInitial(nn.Module):
    """
    A feedforward neural network to predict clean 3D trajectories from noisy inputs, 
    including forces as extra features if the flag is set.
    """
    def __init__(self, seq_length, hidden_dim, use_forces=False):
        super(NoisePredictorInitial, self).__init__()
        self.use_forces = use_forces
        input_dim = seq_length * 7  # noisy trajectory(3) and noisy quaternion(4)
        
        
        if self.use_forces:
            input_dim += seq_length * 6  # Add forces (force_x, force_y, force_z,m_x,m_y,m_z)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_6 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_7 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, seq_length * 7)  # Output clean trajectory (pos_0 and error quaternion)
        self.relu = nn.ReLU()

    def forward(self, noisy_pos, noisy_q, forces=None, moment=None):
        """
        Forward pass to predict clean 3D trajectory.

        Args:
            noisy_pos (torch.Tensor): Input noisy trajectory of shape [batch_size, seq_length, 3].
            noisy_q (torch.Tensor): Input noisy forces of shape [batch_size, seq_length, 4].
            forces (torch.Tensor, optional): Input forces of shape [batch_size, seq_length, 3].

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Predicted noise for pos,u,thea of shape [batch_size, seq_length, 3].

        """
        batch_size, seq_length, _ = noisy_pos.shape
        
        # If using forces, concatenate them with noisy trajectory
        if self.use_forces:
            x = torch.cat((noisy_pos, noisy_q, forces, moment), dim=-1)  # Concatenate noisy trajectory and forces
        else:
            x = torch.cat((noisy_pos, noisy_q), dim=-1)  # Concatenate noisy trajectory

        x = x.view(batch_size, -1)  # Flatten to [batch_size, seq_length * 13] if forces are included
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        x = self.hidden_layer_3(x)
        x = self.relu(x)
        x = self.hidden_layer_4(x)
        x = self.relu(x)
        predicted_noise = self.output_layer(x)
        return predicted_noise.view(batch_size, seq_length, 7)  # Reshape back to [batch_size, seq_length, 7]


#Transformer
class NoisePredictorTransformer(nn.Module):
    def __init__(self, seq_length, hidden_dim, num_heads=8, num_layers=4, use_forces=False):
        super(NoisePredictorTransformer, self).__init__()
        self.use_forces = use_forces
        input_dim = 7  # (x, y, z)

        if self.use_forces:
            input_dim += 6  # Include force dimensions (fx, fy, fz)

        self.embedding = nn.Linear(input_dim, hidden_dim)  # Embed trajectory + forces into hidden_dim
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))  # Learnable positional encodings

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, 7)  # Output clean trajectory (x, y, z)

    def forward(self, noisy_pos, noisy_q, forces=None, moment=None):
        """
        Forward pass to predict clean 3D trajectory.

        Args:
            noisy_trajectory (torch.Tensor): Input noisy trajectory of shape [batch_size, seq_length, 7].
            forces (torch.Tensor, optional): Input forces of shape [batch_size, seq_length, 7].

        Returns:
            torch.Tensor: Predicted clean trajectory of shape [batch_size, seq_length, 7].
        """
        batch_size, seq_length, _ = noisy_pos.shape

        # Concatenate forces with trajectory if used
        if self.use_forces:
            x = torch.cat((noisy_pos, noisy_q, forces, moment), dim=-1)  # Shape: [batch_size, seq_length, 13]
        else:
            x = torch.cat(noisy_pos, noisy_q)  # Shape: [batch_size, seq_length, 7]

        # Embed input and add positional encoding
        x = self.embedding(x) + self.positional_encoding  # Shape: [batch_size, seq_length, hidden_dim]

        # Pass through Transformer Encoder
        x = self.transformer(x)  # Shape: [batch_size, seq_length, hidden_dim]

        # Decode to output clean trajectory
        x = self.fc1(x)
        x = self.relu(x)
        predicted_trajectory = self.fc2(x)  # Shape: [batch_size, seq_length, 7]

        return predicted_trajectory



#LSTM
class NoisePredictorLSTM(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=False):
        super(NoisePredictorLSTM, self).__init__()
        self.use_forces = use_forces
        input_dim = 3  # (x, y, z)

        if self.use_forces:
            self.fusion_fc = nn.Sequential(
                nn.Linear(6, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(seq_length),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU()
            )
            input_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1, dropout=0.)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activation1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.activation2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim // 4, 3)

    def forward(self, noisy_trajectory, forces=None):
        if self.use_forces:
            # Concatenate forces and trajectory
            x = torch.cat((noisy_trajectory, forces), dim=-1)
            x = self.fusion_fc(x)  # Learn features from forces + trajectory
        else:
            x = noisy_trajectory  # Shape: [batch_size, seq_length, input_dim]

        lstm_out, _ = self.lstm(x)  # Shape: [batch_size, seq_length, hidden_dim]
        lstm_out = self.layer_norm(lstm_out)  # Normalize LSTM output

        # Add skip connection at this stage, after the LSTM
        residual = lstm_out

        x_refined = self.fc1(lstm_out)  # [batch_size, seq_length, hidden_dim // 2]
        x_refined = self.activation1(x_refined)

        x_refined = self.fc2(x_refined)  # [batch_size, seq_length, hidden_dim // 4]
        x_refined = self.activation2(x_refined)

        # Add skip connection to the last intermediate layer
        x_refined += residual  # Ensure the dimensions match for addition

        predicted_noise = self.fc3(x_refined)  # Final output: [batch_size, seq_length, 3]

        # Optional: If the original input has the same shape, add a final residual
        predicted_noise += noisy_trajectory

        return predicted_noise
    
    
class NoisePredictorGRU(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=False, num_layers=2, dropout=0.3):
        super(NoisePredictorGRU, self).__init__()
        self.use_forces = use_forces
        input_dim = 3 if not use_forces else 6  

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, noisy_trajectory, forces=None):
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)
        else:
            x = noisy_trajectory

        gru_out, _ = self.gru(x)
        predicted_noise = self.fc(gru_out)

        return predicted_noise



class NoisePredictorConv1D(nn.Module):
    def __init__(self, seq_length, hidden_dim, num_layers=5, use_forces=False, kernel_size=3, dropout=0.2):
        super(NoisePredictorConv1D, self).__init__()
        self.use_forces = use_forces
        input_dim = 3 if not use_forces else 6  # Input: (x, y, z) or (x, y, z, fx, fy, fz)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else 3  # Final layer outputs 3 channels (x, y, z)
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2  # Maintain same sequence length
                )
            )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, noisy_trajectory, forces=None):
        """
        Forward pass for the Conv1D model.

        Args:
            noisy_trajectory (torch.Tensor): Shape [batch_size, seq_length, 3].
            forces (torch.Tensor, optional): Shape [batch_size, seq_length, 3].

        Returns:
            torch.Tensor: Predicted clean trajectory, shape [batch_size, seq_length, 3].
        """
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)  # Concatenate trajectory and forces
        else:
            x = noisy_trajectory  # Shape: [batch_size, seq_length, 3]

        x = x.permute(0, 2, 1)  # Change shape for Conv1D: (batch, channels, seq_length)

        # Pass through Conv1D layers
        for i, layer in enumerate(self.layers[:-1]):  # All layers except the last
            x = self.relu(layer(x))
            x = self.dropout(x)

        # Final layer (no activation, no dropout)
        x = self.layers[-1](x)

        x = x.permute(0, 2, 1)  # Convert back to (batch, seq_length, channels)
        return x

class NoisePredictorTCN(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=False, num_layers=5, kernel_size=3, dropout=0.15):
        super(NoisePredictorTCN, self).__init__()
        self.use_forces = use_forces
        input_dim = 3 if not use_forces else 6  # Input dimensions: (x, y, z) or (x, y, z, fx, fy, fz)

        # Define TCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else 3  # Output layer reduces to 3 channels (x, y, z)
            self.layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) * dilation // 2, dilation=dilation)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, noisy_trajectory, forces=None):
        """
        Forward pass for TCN.
        
        Args:
            noisy_trajectory (torch.Tensor): Shape [batch_size, seq_length, 3].
            forces (torch.Tensor, optional): Shape [batch_size, seq_length, 3].
        
        Returns:
            torch.Tensor: Predicted clean trajectory, shape [batch_size, seq_length, 3].
        """
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)  # Concatenate trajectory and forces
        else:
            x = noisy_trajectory  # Shape: [batch_size, seq_length, 3]

        x = x.permute(0, 2, 1)  # Change to [batch_size, features, seq_length] for Conv1d

        # Pass through TCN layers
        for layer in self.layers[:-1]:  # Exclude the final layer
            x = F.relu(layer(x))
            x = self.dropout(x)

        # Final layer (no activation or dropout)
        x = self.layers[-1](x)

        x = x.permute(0, 2, 1)  # Back to [batch_size, seq_length, features]
        return x

