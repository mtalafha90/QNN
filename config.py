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

# Training
EPOCHS = 1200
LR = 5.0e-4
N_COLLOCATION = 1000
N_IC = 181
N_BC = 200
N_REF = 12000

# Loss weights
W_PDE = 5.0
W_IC = 5.0
W_BC = 2.0
W_REF = 20.0

# Use weak reference loss for the first test.
# Later, set this to False to train with PDE + IC + BC only.
USE_REFERENCE_LOSS = True

# Device
#DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
