"""
Training script for QNN-SFT.

Two important changes relative to the very first version:

1. Field normalisation. The SFT field is O(1e-3). Training directly on such
   tiny targets leaves every squared-error loss term at O(1e-6) and the loss
   landscape almost flat, so the optimiser settles on the trivial near-zero
   solution. We therefore train on the non-dimensional field
   ``b = B / field_scale`` (field_scale = RMS of the classical reference) and
   scale back to physical units only at the very end. The PDE is linear in B,
   so the source is normalised by the same factor and the residual is left
   unchanged in form.

2. Early stopping + LR scheduling on a held-out reference set, so a run no
   longer burns thousands of epochs after it has already converged.
"""

import copy

import numpy as np
import torch
from tqdm import trange

from config import (
    EPOCHS,
    LR,
    N_COLLOCATION,
    N_IC,
    N_BC,
    N_REF,
    N_VAL,
    EARLY_STOP_PATIENCE,
    LR_FACTOR,
    LR_PATIENCE,
    MIN_LR,
    FIELD_SCALE,
    BOUNDARY_VALUE,
    DEVICE,
    RESULTS_DIR,
)
from src.qnn.qnn_sft_model import QNNSFTModel
from src.physics.losses import loss_qnn_sft


def _make_training_batches(
    lat_grid,
    time_grid,
    B_reference,
    device=DEVICE,
):
    lat_min, lat_max = lat_grid[0], lat_grid[-1]
    t_min, t_max = time_grid[0], time_grid[-1]

    # Collocation points
    lam_f = np.random.uniform(lat_min, lat_max, N_COLLOCATION)
    t_f = np.random.uniform(t_min, t_max, N_COLLOCATION)
    x_f = np.stack([lam_f, t_f], axis=1)

    # Initial condition at t=0
    ic_idx = np.linspace(0, len(lat_grid) - 1, N_IC).astype(int)
    lam_ic = lat_grid[ic_idx]
    t_ic = np.full_like(lam_ic, t_min)
    x_ic = np.stack([lam_ic, t_ic], axis=1)
    B_ic = np.zeros_like(lam_ic)

    # Boundary condition at lat_min and lat_max
    t_bc = np.random.uniform(t_min, t_max, N_BC)
    lam_left = np.full(N_BC // 2, lat_min)
    lam_right = np.full(N_BC - N_BC // 2, lat_max)
    lam_bc = np.concatenate([lam_left, lam_right])
    t_bc_all = np.concatenate([t_bc[: N_BC // 2], t_bc[N_BC // 2 :]])
    x_bc = np.stack([lam_bc, t_bc_all], axis=1)
    B_bc = np.full(len(x_bc), BOUNDARY_VALUE)

    batches = {
        "x_f": torch.tensor(x_f, dtype=torch.float32, device=device),
        "x_ic": torch.tensor(x_ic, dtype=torch.float32, device=device),
        "B_ic": torch.tensor(B_ic, dtype=torch.float32, device=device),
        "x_bc": torch.tensor(x_bc, dtype=torch.float32, device=device),
        "B_bc": torch.tensor(B_bc, dtype=torch.float32, device=device),
    }

    # Weak reference points from the (normalised) classical solution
    if B_reference is not None:
        n_time, n_lat = B_reference.shape

        j = np.random.randint(0, n_time, size=N_REF)
        i = np.random.randint(0, n_lat, size=N_REF)

        x_ref = np.stack([lat_grid[i], time_grid[j]], axis=1)
        B_ref = B_reference[j, i]

        batches["x_ref"] = torch.tensor(x_ref, dtype=torch.float32, device=device)
        batches["B_ref"] = torch.tensor(B_ref, dtype=torch.float32, device=device)

    return batches


def _make_validation_set(lat_grid, time_grid, B_reference, device=DEVICE):
    """Fixed held-out reference points used to monitor convergence."""
    if B_reference is None:
        return None, None

    n_time, n_lat = B_reference.shape
    j = np.random.randint(0, n_time, size=N_VAL)
    i = np.random.randint(0, n_lat, size=N_VAL)

    x_val = np.stack([lat_grid[i], time_grid[j]], axis=1)
    B_val = B_reference[j, i]

    return (
        torch.tensor(x_val, dtype=torch.float32, device=device),
        torch.tensor(B_val, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def _validation_loss(model, x_val, B_val, batch_size=4096):
    if x_val is None:
        return float("nan")
    model.eval()
    preds = []
    for k in range(0, x_val.shape[0], batch_size):
        preds.append(model(x_val[k : k + batch_size]))
    pred = torch.cat(preds)
    return torch.mean((pred - B_val) ** 2).item()


def _predict_on_grid(model, lat_grid, time_grid, device=DEVICE, batch_size=2048):
    model.eval()

    LAT, T = np.meshgrid(lat_grid, time_grid)  # shape (n_time, n_lat)
    xs = np.stack([LAT.ravel(), T.ravel()], axis=1)
    xs = torch.tensor(xs, dtype=torch.float32, device=device)

    preds = []
    with torch.no_grad():
        for k in range(0, xs.shape[0], batch_size):
            preds.append(model(xs[k : k + batch_size]).cpu())

    B = torch.cat(preds).numpy()
    B = B.reshape(len(time_grid), len(lat_grid))
    return B


def train_qnn_sft(
    lat_grid,
    time_grid,
    source_map,
    B_reference=None,
):
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = DEVICE

    # --- Field normalisation -------------------------------------------------
    if FIELD_SCALE is not None:
        field_scale = float(FIELD_SCALE)
    elif B_reference is not None:
        field_scale = float(np.sqrt(np.mean(B_reference ** 2)))
    else:
        field_scale = 1.0
    if not np.isfinite(field_scale) or field_scale <= 0.0:
        field_scale = 1.0

    source_map_n = source_map / field_scale
    B_reference_n = None if B_reference is None else B_reference / field_scale
    print(f"   field_scale (RMS of reference) = {field_scale:.6e}")

    model = QNNSFTModel().to(device)

    lat_grid_t = torch.tensor(lat_grid, dtype=torch.float32, device=device)
    time_grid_t = torch.tensor(time_grid, dtype=torch.float32, device=device)
    source_map_t = torch.tensor(source_map_n, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )

    x_val, B_val = _make_validation_set(lat_grid, time_grid, B_reference_n, device)

    history = {
        "total": [],
        "pde": [],
        "ic": [],
        "bc": [],
        "ref": [],
        "val": [],
    }

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in trange(EPOCHS, desc="Training QNN-SFT"):
        model.train()
        batches = _make_training_batches(
            lat_grid=lat_grid,
            time_grid=time_grid,
            B_reference=B_reference_n,
            device=device,
        )

        optimizer.zero_grad()
        loss, loss_dict = loss_qnn_sft(
            model=model,
            batches=batches,
            lat_grid_t=lat_grid_t,
            time_grid_t=time_grid_t,
            source_map_t=source_map_t,
        )
        loss.backward()
        optimizer.step()

        val_loss = _validation_loss(model, x_val, B_val)
        scheduler.step(val_loss)

        for key in ("total", "pde", "ic", "bc", "ref"):
            history[key].append(loss_dict[key])
        history["val"].append(val_loss)

        # Track best model on the held-out reference set.
        if np.isfinite(val_loss) and val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 100 == 0:
            print(
                f"epoch={epoch:05d} "
                f"loss={loss_dict['total']:.3e} "
                f"pde={loss_dict['pde']:.3e} "
                f"ic={loss_dict['ic']:.3e} "
                f"bc={loss_dict['bc']:.3e} "
                f"ref={loss_dict['ref']:.3e} "
                f"val={val_loss:.3e}"
            )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val improvement for {EARLY_STOP_PATIENCE} epochs; "
                f"best val={best_val:.3e})"
            )
            break

    # Restore the best weights seen during training.
    model.load_state_dict(best_state)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), RESULTS_DIR / "models" / "qnn_sft_model.pt")

    # Predict in normalised units, then convert back to physical field.
    B_qnn = field_scale * _predict_on_grid(model, lat_grid, time_grid, device=device)

    return model, B_qnn, history
