import torch
torch.set_default_dtype(torch.float32)
print("CUDA available:", torch.cuda.is_available())
print("Torch device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

from src.source.synthetic_bmr_source import generate_bmrs, build_source_map, save_regions
from src.classical.finite_difference_sft import solve_classical_sft
from src.training.train_qnn import train_qnn_sft
from src.diagnostics.plots import make_all_plots
from src.diagnostics.metrics import print_metrics
from config import RESULTS_DIR, DATA_DIR


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "arrays").mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    (RESULTS_DIR / "models").mkdir(exist_ok=True)

    print("1) Generating synthetic BMR source...")
    regions = generate_bmrs()
    save_regions(regions, DATA_DIR / "synthetic_regions" / "regions.json")

    source_map, lat_grid, time_grid = build_source_map(regions)

    print("2) Solving classical finite-difference SFT...")
    B_classical = solve_classical_sft(
        lat_grid=lat_grid,
        time_grid=time_grid,
        source_map=source_map,
    )

    print("3) Training QNN-SFT model...")
    model, B_qnn, history = train_qnn_sft(
        lat_grid=lat_grid,
        time_grid=time_grid,
        source_map=source_map,
        B_reference=B_classical,
    )

    print("4) Saving arrays...")
    import numpy as np

    np.save(RESULTS_DIR / "arrays" / "lat_grid.npy", lat_grid)
    np.save(RESULTS_DIR / "arrays" / "time_grid.npy", time_grid)
    np.save(RESULTS_DIR / "arrays" / "source_map.npy", source_map)
    np.save(RESULTS_DIR / "arrays" / "B_classical.npy", B_classical)
    np.save(RESULTS_DIR / "arrays" / "B_qnn.npy", B_qnn)

    print("5) Making plots...")
    make_all_plots(
        lat_grid=lat_grid,
        time_grid=time_grid,
        source_map=source_map,
        B_classical=B_classical,
        B_qnn=B_qnn,
        history=history,
    )

    print("6) Metrics:")
    print_metrics(lat_grid, time_grid, B_classical, B_qnn)

    print("Done. Check results/figures and results/arrays.")


if __name__ == "__main__":
    main()
