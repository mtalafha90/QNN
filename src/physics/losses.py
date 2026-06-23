"""
Loss functions for QNN-SFT.
"""

import torch

from config import (
    W_PDE,
    W_IC,
    W_BC,
    W_REF,
    USE_REFERENCE_LOSS,
)
from src.physics.sft_equation import sft_residual


def loss_qnn_sft(
    model,
    batches,
    lat_grid_t,
    time_grid_t,
    source_map_t,
):
    x_f = batches["x_f"]
    x_ic = batches["x_ic"]
    B_ic = batches["B_ic"]
    x_bc = batches["x_bc"]
    B_bc = batches["B_bc"]

    residual = sft_residual(
        model=model,
        x=x_f,
        lat_grid_t=lat_grid_t,
        time_grid_t=time_grid_t,
        source_map_t=source_map_t,
    )

    loss_pde = torch.mean(residual**2)

    B_ic_pred = model(x_ic)
    loss_ic = torch.mean((B_ic_pred - B_ic) ** 2)

    B_bc_pred = model(x_bc)
    loss_bc = torch.mean((B_bc_pred - B_bc) ** 2)

    if USE_REFERENCE_LOSS and "x_ref" in batches:
        B_ref_pred = model(batches["x_ref"])
        loss_ref = torch.mean((B_ref_pred - batches["B_ref"]) ** 2)
    else:
        loss_ref = torch.tensor(0.0, dtype=loss_pde.dtype, device=loss_pde.device)

    total = (
        W_PDE * loss_pde
        + W_IC * loss_ic
        + W_BC * loss_bc
        + W_REF * loss_ref
    )

    loss_dict = {
        "total": total.detach().cpu().item(),
        "pde": loss_pde.detach().cpu().item(),
        "ic": loss_ic.detach().cpu().item(),
        "bc": loss_bc.detach().cpu().item(),
        "ref": loss_ref.detach().cpu().item(),
    }

    return total, loss_dict
