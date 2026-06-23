"""
Diagnostic metrics for QNN-SFT.
"""

import numpy as np


def axial_dipole(lat_grid, B_map):
    """
    Simple axial dipole proxy:

    D(t) = integral B(lambda,t) sin(lambda) cos(lambda) dlambda
           / integral cos(lambda) dlambda
    """

    weights = np.cos(lat_grid)
    norm = np.trapezoid(weights, lat_grid)

    D = []
    for B in B_map:
        D.append(np.trapezoid(B * np.sin(lat_grid) * weights, lat_grid) / norm)

    return np.array(D)


def polar_field(lat_grid, B_map, cap_deg=60.0):
    cap = np.deg2rad(cap_deg)

    north = lat_grid >= cap
    south = lat_grid <= -cap

    weights = np.cos(lat_grid)

    Pn = []
    Ps = []

    for B in B_map:
        Pn.append(np.trapezoid(B[north] * weights[north], lat_grid[north]) / np.trapezoid(weights[north], lat_grid[north]))
        Ps.append(np.trapezoid(B[south] * weights[south], lat_grid[south]) / np.trapezoid(weights[south], lat_grid[south]))

    return np.array(Pn), np.array(Ps)


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def relative_rmse(a, b):
    denom = np.sqrt(np.mean(a ** 2)) + 1e-12
    return rmse(a, b) / denom


def print_metrics(lat_grid, time_grid, B_classical, B_qnn):
    B_rmse = rmse(B_classical, B_qnn)
    B_rrmse = relative_rmse(B_classical, B_qnn)

    D_classical = axial_dipole(lat_grid, B_classical)
    D_qnn = axial_dipole(lat_grid, B_qnn)

    D_rmse = rmse(D_classical, D_qnn)

    Pn_c, Ps_c = polar_field(lat_grid, B_classical)
    Pn_q, Ps_q = polar_field(lat_grid, B_qnn)

    Pn_rmse = rmse(Pn_c, Pn_q)
    Ps_rmse = rmse(Ps_c, Ps_q)

    print(f"Field RMSE          : {B_rmse:.6e}")
    print(f"Field relative RMSE : {B_rrmse:.6e}")
    print(f"Dipole RMSE         : {D_rmse:.6e}")
    print(f"North polar RMSE    : {Pn_rmse:.6e}")
    print(f"South polar RMSE    : {Ps_rmse:.6e}")
