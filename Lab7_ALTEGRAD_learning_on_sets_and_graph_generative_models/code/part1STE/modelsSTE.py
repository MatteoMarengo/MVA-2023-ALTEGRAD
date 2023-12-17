"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        ############## Task 3
    
        x = self.embedding(x)  # Apply embedding
        x = self.tanh(self.fc1(x))  # Apply first fully connected layer and tanh activation

        x = torch.sum(x, dim=1)  # Sum aggregator

        x = self.fc2(x)  # Apply output layer
        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        ############## Task 4
    
        x = self.embedding(x)  # Apply embedding

        # LSTM layer
        # We only use the hidden state of the last time step
        _, (h_n, _) = self.lstm(x)  # h_n is the hidden state for the last time step

        # Fully connected layer
        x = self.fc(h_n)
        
        return x.squeeze()