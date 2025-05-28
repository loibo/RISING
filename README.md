# RISING: A New Framework for Model-Based Few-View CT Image Reconstruction with Deep Learning
This repository contains the code for the paper "RISING: A new framework for model-based few-view CT image reconstruction with deep learning."

## Overview
This project implements the RISING (Regularization by ISTA-Net with learned Gradient) framework for few-view Computed Tomography (CT) image reconstruction. RISING combines a model-based iterative reconstruction approach with a deep learning component to achieve high-quality reconstructions from limited projection data.

The core idea is to train a U-Net model to map an initial reconstruction obtained from a few iterations of a model-based solver to a more converged solution. This leverages the strengths of both traditional iterative methods and data-driven deep learning.

## Repository Structure
The repository is organized as follows:

```
├── IPPy/                  # Inverse Problems in Python package (core functionalities)
├── miscellaneous/
│   ├── data.py            # Handles data loading (e.g., MayoDataset)
│   └── utilities.py       # Contains miscellaneous utility functions
├── generate_convergence.py# Script to pre-compute converged solutions for training
└── train.py               # Main script for training the RISING model
├── README.md              # This file
├── model_weights/         # Directory to save trained model weights (created during training)
└── convergence_data/      # Directory to save pre-computed convergence data (created by generate_convergence.py)
```

## Getting Started
### Prerequisites
To run the code, you'll need the following:

* Python 3.x
* PyTorch
* NumPy
* Matplotlib
* Astra Toolbox (for `IPPy.operators.CTProjector`)

You can install most of the Python dependencies using pip:

```bash
pip install torch numpy matplotlib
```

The `IPPy` package is a local dependency which can be downloaded from https://github.com/devangelista2/IPPy. Make sure it's accessible in your Python path. The `Astra Toolbox` might require a separate installation, please refer to their official documentation.

### Data Preparation
The code uses the Mayo Clinic CT dataset. Please download the dataset and adjust the `data_path` variable in `data.py` (and consequently `train.py` and `generate_convergence.py`) to point to your data directory. The `MayoDataset` class expects a specific directory structure as typically provided by the Mayo dataset.

### Steps to Run
1. **Generate Convergence Data (Optional but Recommended for Training):** The `train.py` script can either compute the "converged" solution (`x_IS`) on the fly during training or load it from pre-computed files. For faster training, it's recommended to pre-compute these solutions.
   Run the `generate_convergence.py` script to create the `convergence_data/` directory and populate it with the "converged" solutions:

   ```bash
   python generate_convergence.py
   ```

   This script iterates through your training data, generates noisy measurements, and then runs the SGP solver for a large number of iterations (`IS_ITER = 200`) to obtain a highly converged reconstruction. These reconstructions are then saved as image files.

   * **Configuration in generate_convergence.py:**
     * `IMG_SIZE`: Image resolution (default: `256`).
     * `CONVERGENCE_PATH`: Directory to save the pre-computed solutions (default: `./convergence_data/`).
     * `N_ANGLES`: Number of projection angles for the CT operator (default: `60`).
     * `NOISE_LEVEL`: Gaussian noise level added to projections (default: `0.01`).
     * `LAMBDA`: Regularization parameter for the SGP solver (default: `1e-1`).
     * `IS_ITER`: Number of iterations for the SGP solver to obtain the "converged" solution (default: `200`).

2. **Train the RISING Model:** After optionally generating the convergence data, you can train the RISING model by running `train.py`:
   
   ```bash
   python train.py
   ```

   * **Configuration in `train.py`:**
     * `device`: Specifies "cuda" if available, otherwise "cpu".
     * `IMG_SIZE`: Image resolution (default: `256`).
     * `N_ANGLES`, `NOISE_LEVEL`, `LAMBDA`: Similar to `generate_convergence.py`, these define the inverse problem.
     * `RIS_ITER`: Number of iterations for the SGP solver to obtain the initial reconstruction (`x_RIS`) during training (default: 5). This is the input to the UNet.
     * `GENERATE_CONVERGENCE`: Set to True if you want to compute `x_IS` on-the-fly, or `False` if you have pre-computed it using `generate_convergence.py` (and it will be loaded from the dataset).
     * `IS_ITER`: Number of iterations for on-the-fly `x_IS` computation (only relevant if `GENERATE_CONVERGENCE` is `True`).
     * `BATCH_SIZE`: Batch size for training (default: 4).
     * `N_EPOCHS`: Number of training epochs (default: 50).
     * `SAVE_EVERY_EPOCH`: Frequency for saving model weights (default: 5).
     * `WEIGHTS_PATH`: Path to save the trained model weights (default: `./model_weights/RISING.pt`).
  
    The training loop will print the current epoch, iteration, average loss (MSE), and SSIM. Model weights will be saved periodically in the `model_weights/` directory.

## Core Components
* `IPPy` Package: This package provides fundamental functionalities for inverse problems, including:
* `metrics`: Evaluation metrics like SSIM.
* `models`: Neural network architectures, specifically the UNet used in RISING.
* `operators`: Forward operators, such as CTProjector.
* `solvers`: Iterative solvers for inverse problems, including SGP (Scaled Gradient Projection).
* `miscellaneous/data.py`: Contains the MayoDataset class for loading and preprocessing the CT image data.
* `train.py`: Orchestrates the training process of the RISING model. It sets up the inverse problem, initializes the UNet, and trains it to map the output of a few SGP iterations to a more converged solution.
* `generate_convergence.py`: A utility script to pre-compute the "converged" solutions needed for training, which serve as the ground truth targets for the UNet.

## Citation
If you use this code or the RISING framework in your research, please cite the original paper:

```bibtex
@article{Evangelista2023RISING,
title={RISING: A new framework for model-based few-view CT image reconstruction with deep learning},
author={Evangelista, Davide and Morotti, Enrico and Loli Piccolomini, Elena},
journal={COMPUTERIZED MEDICAL IMAGING AND GRAPHICS},
volume={103},
pages={1--8},
year={2023},
publisher={Elsevier}
}
```