"""
Synthetic bipolar magnetic region source for the first QNN-SFT test.

This is a simplified Petrovay & Talafha-inspired source:
- butterfly migration from high to low latitude
- Joy's-law-like polarity separation
- no tilt quenching
- no latitude quenching
- no observational input
"""

import json
import numpy as np
from pathlib import Path

from config import (
    LAT_MIN_DEG,
    LAT_MAX_DEG,
    T_MIN,
    T_MAX,
    N_LAT,
    N_TIME,
    N_BMRS,
    SOURCE_SEED,
    CYCLE_AMP,
    LAT_START_DEG,
    LAT_END_DEG,
    BMR_SEPARATION_DEG,
    SIGMA_LAT_DEG,
    SIGMA_T,
    SOURCE_AMPLITUDE_SCALE,
)


def generate_bmrs(
    n_bmrs=N_BMRS,
    seed=SOURCE_SEED,
    cycle_amp=CYCLE_AMP,
):
    """
    Generate a list of synthetic BMRs.

    Each BMR has:
    - emergence latitude
    - emergence time
    - amplitude
    - polarity separation
    - Joy's-law tilt
    - Gaussian source widths
    """

    rng = np.random.default_rng(seed)
    regions = []

    for _ in range(n_bmrs):
        t0 = rng.uniform(T_MIN + 0.04, T_MAX - 0.04)

        phase = (t0 - T_MIN) / (T_MAX - T_MIN)
        lat_abs_deg = LAT_START_DEG * (1.0 - phase) + LAT_END_DEG * phase

        hemi = rng.choice([-1.0, 1.0])
        lat_deg = hemi * lat_abs_deg

        # Simple Joy's law-like tilt, signed by hemisphere.
        # This is intentionally simple for the first experiment.
        tilt_deg = hemi * 0.5 * np.sqrt(abs(lat_deg))

        amp = SOURCE_AMPLITUDE_SCALE * cycle_amp * rng.lognormal(mean=0.0, sigma=0.35)

        regions.append(
            {
                "lat_rad": float(np.deg2rad(lat_deg)),
                "time": float(t0),
                "amp": float(amp),
                "sep_rad": float(np.deg2rad(BMR_SEPARATION_DEG)),
                "tilt_rad": float(np.deg2rad(tilt_deg)),
                "sigma_lat_rad": float(np.deg2rad(SIGMA_LAT_DEG)),
                "sigma_t": float(SIGMA_T),
                "hemi": int(hemi),
            }
        )

    return regions


def source_from_regions_numpy(lat_grid, time_grid, regions):
    """
    Build S(lambda,t) on a numpy grid.

    Returns
    -------
    source_map : ndarray
        Shape (N_TIME, N_LAT)
    """

    lat = lat_grid[None, :]
    time = time_grid[:, None]

    S = np.zeros((len(time_grid), len(lat_grid)))

    for r in regions:
        lat0 = r["lat_rad"]
        t0 = r["time"]
        amp = r["amp"]
        sep = r["sep_rad"]
        tilt = r["tilt_rad"]
        sigma_lat = r["sigma_lat_rad"]
        sigma_t = r["sigma_t"]

        dlat = 0.5 * sep * np.sin(tilt)

        lat_plus = lat0 + dlat
        lat_minus = lat0 - dlat

        spatial = (
            np.exp(-0.5 * ((lat - lat_plus) / sigma_lat) ** 2)
            - np.exp(-0.5 * ((lat - lat_minus) / sigma_lat) ** 2)
        )

        temporal = np.exp(-0.5 * ((time - t0) / sigma_t) ** 2)

        S += amp * temporal * spatial

    # Remove global monopole from each time slice approximately.
    # This keeps net flux close to zero on the latitude grid.
    coslat = np.cos(lat_grid)
    weights = coslat / np.trapezoid(coslat, lat_grid)

    for n in range(S.shape[0]):
        mean_flux = np.trapezoid(S[n] * weights, lat_grid)
        S[n] -= mean_flux

    return S


def build_source_map(regions):
    lat_grid = np.deg2rad(np.linspace(LAT_MIN_DEG, LAT_MAX_DEG, N_LAT))
    time_grid = np.linspace(T_MIN, T_MAX, N_TIME)

    source_map = source_from_regions_numpy(lat_grid, time_grid, regions)
    return source_map, lat_grid, time_grid


def save_regions(regions, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(regions, f, indent=2)


def load_regions(path):
    with open(path, "r") as f:
        return json.load(f)
