"""
Classical finite-difference SFT solver.

Equation in normalized latitude lambda:

dB/dt =
- 1/cos(lambda) d/dlambda [u(lambda) B cos(lambda)]
+ eta/cos(lambda) d/dlambda [cos(lambda) dB/dlambda]
+ S(lambda,t)
- B/tau

This simple explicit solver is used as a reference benchmark for QNN-SFT.
"""

import numpy as np

from config import U0, ETA, TAU, INITIAL_FIELD_AMPLITUDE, BOUNDARY_VALUE


def meridional_flow(lat):
    """
    Simple poleward meridional flow profile in nondimensional form.
    """
    return U0 * np.sin(2.0 * lat)


def solve_classical_sft(
    lat_grid,
    time_grid,
    source_map,
    u0=U0,
    eta=ETA,
    tau=TAU,
):
    n_time = len(time_grid)
    n_lat = len(lat_grid)

    dlat = lat_grid[1] - lat_grid[0]
    dt = time_grid[1] - time_grid[0]

    B = np.zeros((n_time, n_lat))
    B[0, :] = INITIAL_FIELD_AMPLITUDE

    coslat = np.cos(lat_grid)
    coslat = np.clip(coslat, 1.0e-4, None)

    u = u0 * np.sin(2.0 * lat_grid)

    for n in range(n_time - 1):
        Bn = B[n].copy()

        # Boundary values
        Bn[0] = BOUNDARY_VALUE
        Bn[-1] = BOUNDARY_VALUE

        # Advective flux F = u B cos(lambda)
        F = u * Bn * coslat
        dF_dlat = np.zeros_like(Bn)
        dF_dlat[1:-1] = (F[2:] - F[:-2]) / (2.0 * dlat)
        dF_dlat[0] = (F[1] - F[0]) / dlat
        dF_dlat[-1] = (F[-1] - F[-2]) / dlat

        # Diffusion term d/dlambda [cos(lambda) dB/dlambda]
        dB_dlat = np.zeros_like(Bn)
        dB_dlat[1:-1] = (Bn[2:] - Bn[:-2]) / (2.0 * dlat)
        dB_dlat[0] = (Bn[1] - Bn[0]) / dlat
        dB_dlat[-1] = (Bn[-1] - Bn[-2]) / dlat

        Q = coslat * dB_dlat
        dQ_dlat = np.zeros_like(Bn)
        dQ_dlat[1:-1] = (Q[2:] - Q[:-2]) / (2.0 * dlat)
        dQ_dlat[0] = (Q[1] - Q[0]) / dlat
        dQ_dlat[-1] = (Q[-1] - Q[-2]) / dlat

        rhs = (
            -dF_dlat / coslat
            + eta * dQ_dlat / coslat
            + source_map[n]
            - Bn / tau
        )

        B[n + 1] = Bn + dt * rhs

        B[n + 1, 0] = BOUNDARY_VALUE
        B[n + 1, -1] = BOUNDARY_VALUE

    return B
