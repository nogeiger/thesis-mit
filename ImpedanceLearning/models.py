import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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

#LSTM
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
    
#Transformer
class NoisePredictorTransformer(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=False, num_layers=4, nhead=4, dropout=0.2):
        super(NoisePredictorTransformer, self).__init__()
        self.use_forces = use_forces
        input_dim = 3 if not use_forces else 6  # (x, y, z) or (x, y, z + force_x, force_y, force_z)
        
        # Initial embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, noisy_trajectory, forces=None):
        """
        Args:
            noisy_trajectory: [batch_size, seq_length, 3]
            forces: [batch_size, seq_length, 3] (optional)

        Returns:
            predicted_noise: [batch_size, seq_length, 3]
        """
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)  # Concatenate force with trajectory
        else:
            x = noisy_trajectory

        x = self.embedding(x)  # Linear projection to hidden_dim
        x = self.transformer_encoder(x)  # Pass through Transformer Encoder
        predicted_noise = self.fc(x)  # Final prediction

        return predicted_noise
    
#Attention class for other models
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, forces, noisy_trajectory):
        """
        Args:
            forces: [batch_size, seq_length, 3]
            noisy_trajectory: [batch_size, seq_length, 3]
        
        Returns:
            context_vector: [batch_size, seq_length, hidden_dim]
        """
        Q = self.query(forces)  # Query from force input
        K = self.key(noisy_trajectory)  # Key from noisy trajectory
        V = self.value(noisy_trajectory)  # Value from noisy trajectory

        attention_weights = self.softmax(torch.bmm(Q, K.transpose(1, 2)))  # Compute attention scores
        context_vector = torch.bmm(attention_weights, V)  # Apply attention
        
        return context_vector

#LSTM with attention
class NoisePredictorLSTMWithAttention(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=True, num_layers=2, dropout=0.3):
        super(NoisePredictorLSTMWithAttention, self).__init__()
        self.use_forces = use_forces
        input_dim = 3  # (x, y, z)

        if self.use_forces:
            input_dim += 3  # Include forces (force_x, force_y, force_z)

        self.attention = AttentionLayer(input_dim=3, hidden_dim=hidden_dim)

        self.lstm = nn.LSTM(
            input_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, noisy_trajectory, forces=None):
        """
        Args:
            noisy_trajectory: [batch_size, seq_length, 3]
            forces: [batch_size, seq_length, 3] (optional)

        Returns:
            predicted_noise: [batch_size, seq_length, 3]
        """
        if self.use_forces:
            context_vector = self.attention(forces, noisy_trajectory)
            x = torch.cat((noisy_trajectory, forces, context_vector), dim=-1)  # Concatenate attention output
        else:
            x = noisy_trajectory

        lstm_out, _ = self.lstm(x)  
        predicted_noise = self.fc(lstm_out)

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
    def __init__(self, seq_length, hidden_dim, use_forces=False):
        super(NoisePredictorConv1D, self).__init__()
        self.use_forces = use_forces
        input_dim = 3 if not use_forces else 6  

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=3, kernel_size=1)  # Output layer

        self.relu = nn.ReLU()

    def forward(self, noisy_trajectory, forces=None):
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)
        else:
            x = noisy_trajectory

        x = x.permute(0, 2, 1)  # Change shape for Conv1D: (batch, channels, seq_length)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        x = x.permute(0, 2, 1)  # Convert back: (batch, seq_length, channels)
        return x

class NoisePredictorConvLSTM(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=False, num_layers=2, dropout=0.3):
        super(NoisePredictorConvLSTM, self).__init__()
        self.use_forces = use_forces
        input_dim = 3 if not use_forces else 6  

        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, noisy_trajectory, forces=None):
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)
        else:
            x = noisy_trajectory

        x = x.permute(0, 2, 1)  # Conv1D expects (batch, channels, seq_length)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)  # Convert back to (batch, seq_length, channels)

        lstm_out, _ = self.lstm(x)
        predicted_noise = self.fc(lstm_out)

        return predicted_noise




class NoisePredictorTCN(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=False):
        super(NoisePredictorTCN, self).__init__()
        self.use_forces = use_forces
        input_dim = 3 if not use_forces else 6  

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, 3, kernel_size=1)  # Output layer

    def forward(self, noisy_trajectory, forces=None):
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)
        else:
            x = noisy_trajectory

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.permute(0, 2, 1)

        return x

#attention,cnn,lstm
class NoisePredictorHybrid(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=True, num_layers=2, dropout=0.3):
        super(NoisePredictorHybrid, self).__init__()
        self.use_forces = use_forces
        input_dim = 3 if not use_forces else 6  

        # Attention Layer
        self.attention = AttentionLayer(input_dim=3, hidden_dim=hidden_dim)

        # 1D CNN for Local Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim + hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # LSTM for Temporal Dependencies
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, noisy_trajectory, forces=None):
        """
        Args:
            noisy_trajectory: [batch_size, seq_length, 3]
            forces: [batch_size, seq_length, 3] (optional)

        Returns:
            predicted_noise: [batch_size, seq_length, 3]
        """
        if self.use_forces:
            context_vector = self.attention(forces, noisy_trajectory)  
            x = torch.cat((noisy_trajectory, forces, context_vector), dim=-1)  
        else:
            x = noisy_trajectory

        x = x.permute(0, 2, 1)  # Change shape for CNN (batch, channels, seq_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Convert back for LSTM (batch, seq_length, channels)

        lstm_out, _ = self.lstm(x)  
        predicted_noise = self.fc(lstm_out)  

        return predicted_noise