"""
Hybrid quantum-classical QNN model for B(lambda,t).
"""

import torch
import torch.nn as nn

from config import N_QUBITS, N_Q_LAYERS, READOUT_WIDTH, Q_WEIGHT_INIT_SCALE
from src.qnn.quantum_circuit import quantum_circuit


class QNNSFTModel(nn.Module):
    def __init__(self):
        super().__init__()

        # A larger init than the original 0.01 so the variational layers are
        # not initialised as a near-identity (0.01 rad rotations contribute
        # almost nothing, leaving the quantum feature map essentially fixed).
        self.q_weights = nn.Parameter(
            Q_WEIGHT_INIT_SCALE * torch.randn(N_Q_LAYERS, N_QUBITS, 3)
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

        The quantum circuit is evaluated on the whole batch in one execution
        via PennyLane broadcasting rather than looping over samples, which is
        what makes training tractable.
        """
        x = x.float()

        q_out = quantum_circuit(x, self.q_weights)   # list of N_QUBITS tensors (N,)
        q_out = torch.stack(q_out, dim=-1).float()    # (N, N_QUBITS)

        B = self.readout(q_out).squeeze(-1)           # (N,)
        return B
