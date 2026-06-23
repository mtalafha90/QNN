"""
SFT residual for QNN training.
"""

import torch

from config import U0, ETA, TAU


def torch_meridional_flow(lam, u0=U0):
    return u0 * torch.sin(2.0 * lam)


def interpolate_source_torch(x, lat_grid_t, time_grid_t, source_map_t):
    """
    Bilinear interpolation of source_map at collocation points.

    x shape: (N, 2), columns [lat, time]
    source_map_t shape: (N_TIME, N_LAT)
    """

    lam = x[:, 0]
    t = x[:, 1]

    lat_min = lat_grid_t[0]
    lat_max = lat_grid_t[-1]
    time_min = time_grid_t[0]
    time_max = time_grid_t[-1]

    n_lat = lat_grid_t.numel()
    n_time = time_grid_t.numel()

    # Convert to fractional indices
    lat_pos = (lam - lat_min) / (lat_max - lat_min) * (n_lat - 1)
    time_pos = (t - time_min) / (time_max - time_min) * (n_time - 1)

    lat_pos = torch.clamp(lat_pos, 0, n_lat - 1 - 1e-6)
    time_pos = torch.clamp(time_pos, 0, n_time - 1 - 1e-6)

    i0 = torch.floor(lat_pos).long()
    j0 = torch.floor(time_pos).long()

    i1 = i0 + 1
    j1 = j0 + 1

    wi = lat_pos - i0.float()
    wj = time_pos - j0.float()

    S00 = source_map_t[j0, i0]
    S10 = source_map_t[j0, i1]
    S01 = source_map_t[j1, i0]
    S11 = source_map_t[j1, i1]

    S0 = S00 * (1.0 - wi) + S10 * wi
    S1 = S01 * (1.0 - wi) + S11 * wi
    S = S0 * (1.0 - wj) + S1 * wj

    return S


def sft_residual(
    model,
    x,
    lat_grid_t,
    time_grid_t,
    source_map_t,
    u0=U0,
    eta=ETA,
    tau=TAU,
):
    """
    Compute SFT residual at collocation points.

    Residual:
    R =
    B_t
    + 1/cos(lambda) d/dlambda [u B cos(lambda)]
    - eta/cos(lambda) d/dlambda [cos(lambda) B_lambda]
    - S
    + B/tau
    """

    x = x.clone().detach().requires_grad_(True)

    lam = x[:, 0]

    B = model(x)

    grad_B = torch.autograd.grad(
        B,
        x,
        grad_outputs=torch.ones_like(B),
        create_graph=True,
        retain_graph=True,
    )[0]

    B_lam = grad_B[:, 0]
    B_t = grad_B[:, 1]

    coslam = torch.clamp(torch.cos(lam), min=1.0e-4)

    u = torch_meridional_flow(lam, u0=u0)

    adv_quantity = u * B * coslam

    grad_adv = torch.autograd.grad(
        adv_quantity,
        x,
        grad_outputs=torch.ones_like(adv_quantity),
        create_graph=True,
        retain_graph=True,
    )[0]

    d_adv_dlam = grad_adv[:, 0]

    diff_quantity = coslam * B_lam

    grad_diff = torch.autograd.grad(
        diff_quantity,
        x,
        grad_outputs=torch.ones_like(diff_quantity),
        create_graph=True,
        retain_graph=True,
    )[0]

    d_diff_dlam = grad_diff[:, 0]

    S = interpolate_source_torch(x, lat_grid_t, time_grid_t, source_map_t)

    R = (
        B_t
        + d_adv_dlam / coslam
        - eta * d_diff_dlam / coslam
        - S
        + B / tau
    )

    return R
