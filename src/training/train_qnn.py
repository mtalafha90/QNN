"""
Training script for QNN-SFT.
"""

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

    # Weak reference points from classical solution
    if B_reference is not None:
        n_time, n_lat = B_reference.shape

        j = np.random.randint(0, n_time, size=N_REF)
        i = np.random.randint(0, n_lat, size=N_REF)

        x_ref = np.stack([lat_grid[i], time_grid[j]], axis=1)
        B_ref = B_reference[j, i]

        batches["x_ref"] = torch.tensor(x_ref, dtype=torch.float32, device=device)
        batches["B_ref"] = torch.tensor(B_ref, dtype=torch.float32, device=device)

    return batches


def _predict_on_grid(model, lat_grid, time_grid, device=DEVICE, batch_size=256):
    model.eval()

    xs = []
    for t in time_grid:
        for lat in lat_grid:
            xs.append([lat, t])

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

    model = QNNSFTModel().to(device)

    lat_grid_t = torch.tensor(lat_grid, dtype=torch.float32, device=device)
    time_grid_t = torch.tensor(time_grid, dtype=torch.float32, device=device)
    source_map_t = torch.tensor(source_map, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "total": [],
        "pde": [],
        "ic": [],
        "bc": [],
        "ref": [],
    }

    for epoch in trange(EPOCHS, desc="Training QNN-SFT"):
        batches = _make_training_batches(
            lat_grid=lat_grid,
            time_grid=time_grid,
            B_reference=B_reference,
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

        for key in history:
            history[key].append(loss_dict[key])

        if epoch % 100 == 0:
            print(
                f"epoch={epoch:05d} "
                f"loss={loss_dict['total']:.3e} "
                f"pde={loss_dict['pde']:.3e} "
                f"ic={loss_dict['ic']:.3e} "
                f"bc={loss_dict['bc']:.3e} "
                f"ref={loss_dict['ref']:.3e}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), RESULTS_DIR / "models" / "qnn_sft_model.pt")

    B_qnn = _predict_on_grid(model, lat_grid, time_grid, device=device)

    return model, B_qnn, history
