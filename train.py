import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch import nn

from IPPy import metrics, models, operators, solvers
from IPPy import utilities as IPutils
from miscellaneous import data

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}.")

IMG_SIZE = 256

# CT Operator config
START_ANGLE, END_ANGLE = 0, 180
N_ANGLES = 60
NOISE_LEVEL = 0.01

# Solver config
LAMBDA = 1e-1
RIS_ITER = 5

GENERATE_CONVERGENCE = True
IS_ITER = 10

# Train config
BATCH_SIZE = 4
N_EPOCHS = 50
SAVE_EVERY_EPOCH = 5
WEIGHTS_PATH = "./model_weights/RISING.pt"


# --- Load data + operator ---
train_data = data.MayoDataset(data_path="../data/Mayo/train", data_shape=IMG_SIZE)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

K = operators.CTProjector(
    img_shape=(IMG_SIZE, IMG_SIZE),
    angles=np.linspace(np.deg2rad(START_ANGLE), np.deg2rad(END_ANGLE), N_ANGLES),
    det_size=2 * IMG_SIZE,
    geometry="parallel",
)

# --- Initialize model and solver (for training) ---
model = models.UNet(
    ch_in=1,
    ch_out=1,
    middle_ch=[64, 128, 256, 512],
    n_layers_per_block=2,
    down_layers=("ResDownBlock", "ResDownBlock", "ResDownBlock"),
    up_layers=("ResUpBlock", "ResUpBlock", "ResUpBlock"),
    final_activation="relu",
).to(device)

solver = solvers.SGP(K)

# --- Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    ssim_loss = 0.0

    for i, x in enumerate(train_loader):
        # Get convergence if pre-computed, otherwise just keep x
        if not GENERATE_CONVERGENCE:
            x, x_IS = x

        # Create test problem + compute RIS solution
        with torch.no_grad():
            y = K(x)
            y_delta = y + IPutils.gaussian_noise(y, noise_level=NOISE_LEVEL)

            x_RIS, _ = solver(
                y_delta,
                lmbda=LAMBDA,
                maxiter=RIS_ITER,
                starting_point=torch.zeros_like(x),
                x_true=x,
                verbose=False,
            )

        # Compute convergence solution if needed
        with torch.no_grad():
            if GENERATE_CONVERGENCE:
                x_IS, _ = solver(
                    y_delta,
                    lmbda=LAMBDA,
                    maxiter=IS_ITER,
                    starting_point=torch.zeros_like(x),
                    x_true=x,
                    verbose=False,
                )

        # Send data to device
        x = x.to(device)
        x_RIS = x_RIS.to(device)
        x_IS = x_IS.to(device)

        # Apply neural network model + loss + update
        optimizer.zero_grad()

        x_RISING = model(x_RIS)
        loss = loss_fn(x_RISING, x_IS)
        loss.backward()
        optimizer.step()

        # Printing
        epoch_loss += loss.item()
        ssim_loss += metrics.SSIM(x_RISING.cpu().detach(), x.cpu().detach())
        print(
            f"Epoch {epoch+1} | It. {i+1} | Avg Loss: {epoch_loss / (i+1):0.4f} | Avg SSIM: {ssim_loss / (i+1):0.4f}.",
        )
    print()

    # Saving model after a few epochs
    if (epoch + 1) % SAVE_EVERY_EPOCH == 0 or (epoch + 1) == N_EPOCHS:
        torch.save(model.state_dict(), WEIGHTS_PATH)
