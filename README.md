# qnn_sft

Minimal first-step project for solving the 1D solar surface flux transport equation using:

1. A synthetic bipolar magnetic region source.
2. A classical finite-difference SFT solver.
3. A hybrid quantum neural network SFT solver.
4. Diagnostics comparing the QNN solution with the classical reference.

This first version uses:
- No observational data.
- No tilt quenching.
- No latitude quenching.
- One normalized solar cycle, `t in [0, 1]`.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Outputs are saved in:

```text
results/figures/
results/arrays/
results/models/
```

## Main result figures

```text
source_butterfly.png
classical_sft_butterfly.png
qnn_sft_butterfly.png
qnn_minus_classical.png
polar_field.png
axial_dipole.png
training_loss.png
```
