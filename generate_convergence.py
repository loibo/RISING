import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from IPPy import metrics, operators, solvers
from IPPy import utilities as IPutils
from miscellaneous import data

# --- Configuration ---
IMG_SIZE = 256
CONVERGENCE_PATH = "./convergence_data/"
os.makedirs(CONVERGENCE_PATH, exist_ok=True)

# CT Operator config
START_ANGLE, END_ANGLE = 0, 180
N_ANGLES = 60
NOISE_LEVEL = 0.01

# Solver config
LAMBDA = 1e-1
IS_ITER = 200

# --- Load data + operator ---
gt_data = data.MayoDataset(data_path="../data/Mayo/train", data_shape=IMG_SIZE)

K = operators.CTProjector(
    img_shape=(IMG_SIZE, IMG_SIZE),
    angles=np.linspace(np.deg2rad(START_ANGLE), np.deg2rad(END_ANGLE), N_ANGLES),
    det_size=2 * IMG_SIZE,
    geometry="parallel",
)

# --- Initialize model and solver (for training) ---
solver = solvers.SGP(K)

# --- Loop
for i in range(len(gt_data)):
    print(f"Image number {i+1}...", end=" ")

    # Get x_true from data
    x_true = gt_data[i].unsqueeze(0)  # Shape (1, 1, nx, ny)

    with torch.no_grad():
        # Create test problem
        y = K(x_true)
        y_delta = y + IPutils.gaussian_noise(y, noise_level=NOISE_LEVEL)

        # Compute convergence solution
        x_IS, _ = solver(
            y_delta,
            lmbda=LAMBDA,
            maxiter=IS_ITER,
            starting_point=torch.zeros_like(x_true),
            x_true=x_true,
            verbose=False,
        )

    # Save computed solution on the given path
    IMG_PATH, IMG_N = gt_data.get_path(i)
    SAVE_PATH = os.path.join(CONVERGENCE_PATH, IMG_PATH)
    os.makedirs(SAVE_PATH, exist_ok=True)

    plt.imsave(os.path.join(SAVE_PATH, IMG_N), x_IS.squeeze(), cmap="gray")
    print(f"Done! SSIM = {metrics.SSIM(x_IS, x_true):0.4f}", end="\n")
