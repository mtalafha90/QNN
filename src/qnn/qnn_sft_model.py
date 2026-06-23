"""
Hybrid quantum-classical QNN model for B(lambda,t).
"""

import torch
import torch.nn as nn

from config import N_QUBITS, N_Q_LAYERS, READOUT_WIDTH
from src.qnn.quantum_circuit import quantum_circuit


class QNNSFTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_weights = nn.Parameter(
            0.01 * torch.randn(N_Q_LAYERS, N_QUBITS, 3)
        )

        self.readout = nn.Sequential(
            nn.Linear(N_QUBITS, READOUT_WIDTH),
            nn.Tanh(),
            nn.Linear(READOUT_WIDTH, READOUT_WIDTH),
            nn.Tanh(),
            nn.Linear(READOUT_WIDTH, 1),
        )

    def forward(self, x):
        """
        x shape: (N, 2), where columns are [lambda, t].
        Returns B shape: (N,)
        """
        x = x.float()
        q_outputs = []
        for xi in x:
            q_out = quantum_circuit(xi, self.q_weights)
            q_out = torch.stack(q_out).float()
            q_outputs.append(q_out)
        q_outputs = torch.stack(q_outputs, dim=0).float()
        B = self.readout(q_outputs).squeeze(-1)
        return B
