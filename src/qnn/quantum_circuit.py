"""
Quantum circuit for the QNN-SFT model.

The circuit receives x = [latitude, time] and returns expectation values.
"""

import torch
import pennylane as qml

from config import N_QUBITS, N_Q_LAYERS

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(x, weights):
    """
    Parameters
    ----------
    x : torch tensor, shape (2,)
        x[0] = latitude lambda in radians
        x[1] = normalized time
    weights : torch tensor
        Shape (N_Q_LAYERS, N_QUBITS, 3)
    """

    lam = x[0]
    t = x[1]

    # Input encoding
    qml.RY(lam, wires=0)
    qml.RZ(t, wires=0)

    if N_QUBITS > 1:
        qml.RY(2.0 * lam, wires=1)
        qml.RZ(2.0 * t, wires=1)

    if N_QUBITS > 2:
        qml.RY(lam * t, wires=2)
        qml.RZ(lam + t, wires=2)

    if N_QUBITS > 3:
        qml.RY(torch.sin(lam), wires=3)
        qml.RZ(torch.cos(lam), wires=3)

    for q in range(4, N_QUBITS):
        qml.RY((q + 1) * lam, wires=q)
        qml.RZ((q + 1) * t, wires=q)

    # Variational layers
    for layer in range(N_Q_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)

        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])

        if N_QUBITS > 2:
            qml.CNOT(wires=[N_QUBITS - 1, 0])

    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]
