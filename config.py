"""
Configuration for the first QNN-SFT experiment.

This version is deliberately simple:
- synthetic source only
- no observational data
- no nonlinearities
- 1D latitude-time SFT
"""
import torch

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# Latitude-time domain
LAT_MIN_DEG = -80.0
LAT_MAX_DEG = 80.0
T_MIN = 0.0
T_MAX = 1.0

# Classical solver grid
N_LAT = 181
N_TIME = 500

# SFT physical/nondimensional parameters
# In this first version, parameters are nondimensional.
U0 = 0.20
ETA = 0.002
TAU = 10.0

# Initial and boundary conditions
INITIAL_FIELD_AMPLITUDE = 0.0
BOUNDARY_VALUE = 0.0

# Synthetic BMR source
N_BMRS = 60
SOURCE_SEED = 42
CYCLE_AMP = 1.0
LAT_START_DEG = 35.0
LAT_END_DEG = 5.0
BMR_SEPARATION_DEG = 4.0
SIGMA_LAT_DEG = 5.0
SIGMA_T = 0.040
SOURCE_AMPLITUDE_SCALE = 0.20

# QNN model
N_QUBITS =4
N_Q_LAYERS = 4
READOUT_WIDTH = 32
# Spread of the initial variational weights. 0.01 (the original value) left the
# variational layers as a near-identity, so the quantum feature map barely
# trained; ~0.1 gives the entangling layers a real but gentle effect.
Q_WEIGHT_INIT_SCALE = 0.1

# Training
# EPOCHS is now an upper bound: early stopping (see below) normally ends the run
# much sooner. With the batched circuit each epoch is fast, so this is cheap.
EPOCHS = 3000
LR = 1.0e-3
N_COLLOCATION = 1000
N_IC = 181
N_BC = 200
N_REF = 8000

# Early stopping / LR schedule (monitored on a held-out reference set).
N_VAL = 4000
EARLY_STOP_PATIENCE = 400
LR_FACTOR = 0.5
LR_PATIENCE = 150
MIN_LR = 1.0e-5

# Loss weights
W_PDE = 5.0
W_IC = 5.0
W_BC = 2.0
W_REF = 20.0

# Field normalisation
# The SFT field amplitude is O(1e-3). Training directly on such tiny targets
# leaves the loss landscape almost flat, so the optimiser parks on the trivial
# near-zero solution (this is exactly what the first run did: rel. RMSE ~0.83,
# only ~17% better than predicting zero). We therefore train on the
# non-dimensional field b = B / FIELD_SCALE, where FIELD_SCALE is the RMS of the
# classical reference, computed at run time when left as None.
FIELD_SCALE = None

# Use weak reference loss for the first test.
# Later, set this to False to train with PDE + IC + BC only.
USE_REFERENCE_LOSS = True

# Device
#DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
