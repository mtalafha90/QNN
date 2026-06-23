"""
Plotting functions for QNN-SFT.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from src.diagnostics.metrics import axial_dipole, polar_field


def _save_butterfly(filename, lat_grid, time_grid, field, title, cbar_label):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    lat_deg = np.rad2deg(lat_grid)

    vmax = np.nanmax(np.abs(field))
    if vmax <= 0:
        vmax = 1.0

    im = ax.pcolormesh(
        time_grid,
        lat_deg,
        field.T,
        shading="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / filename, dpi=200)
    plt.close(fig)


def plot_training_loss(history):
    if history is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for key, values in history.items():
        values = np.asarray(values)
        if np.any(values > 0):
            ax.semilogy(values, label=key)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("QNN-SFT training loss")
    ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "training_loss.png", dpi=200)
    plt.close(fig)


def plot_polar_field(lat_grid, time_grid, B_classical, B_qnn):
    Pn_c, Ps_c = polar_field(lat_grid, B_classical)
    Pn_q, Ps_q = polar_field(lat_grid, B_qnn)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(time_grid, Pn_c, label="Classical north")
    ax.plot(time_grid, Ps_c, label="Classical south")
    ax.plot(time_grid, Pn_q, "--", label="QNN north")
    ax.plot(time_grid, Ps_q, "--", label="QNN south")

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Polar field proxy")
    ax.set_title("Polar field evolution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "polar_field.png", dpi=200)
    plt.close(fig)


def plot_axial_dipole(lat_grid, time_grid, B_classical, B_qnn):
    D_c = axial_dipole(lat_grid, B_classical)
    D_q = axial_dipole(lat_grid, B_qnn)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(time_grid, D_c, label="Classical")
    ax.plot(time_grid, D_q, "--", label="QNN")

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Axial dipole proxy")
    ax.set_title("Axial dipole evolution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "axial_dipole.png", dpi=200)
    plt.close(fig)


def make_all_plots(
    lat_grid,
    time_grid,
    source_map,
    B_classical,
    B_qnn,
    history=None,
):
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    _save_butterfly(
        "source_butterfly.png",
        lat_grid,
        time_grid,
        source_map,
        "Synthetic source S(lambda,t)",
        "S",
    )

    _save_butterfly(
        "classical_sft_butterfly.png",
        lat_grid,
        time_grid,
        B_classical,
        "Classical finite-difference SFT",
        "B",
    )

    _save_butterfly(
        "qnn_sft_butterfly.png",
        lat_grid,
        time_grid,
        B_qnn,
        "QNN-SFT solution",
        "B",
    )

    _save_butterfly(
        "qnn_minus_classical.png",
        lat_grid,
        time_grid,
        B_qnn - B_classical,
        "Difference: QNN - Classical",
        "Delta B",
    )

    plot_polar_field(lat_grid, time_grid, B_classical, B_qnn)
    plot_axial_dipole(lat_grid, time_grid, B_classical, B_qnn)
    plot_training_loss(history)
