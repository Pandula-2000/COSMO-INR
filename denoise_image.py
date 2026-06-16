import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import argparse
import json
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from modules import utils
from modules.models1 import INR
from torch.utils.tensorboard import SummaryWriter

device_number = 0

parser = argparse.ArgumentParser(description='COSMO')

"""
Model
Raised coseine * e^(j.C.x) <-- COSMO
[T], [C]
Params: 1 + 1
"""

image = "kodim20"

NITERS = 500
STR = 5
MODEL = "BandRC"
LR = 9e-4

HIDDEN_LAYERS = 4

# -----------------------------------------------------------------------------
# DYNAMIC ACTIVATION PARAMETER CONFIGURATION
# -----------------------------------------------------------------------------
# Define any number of activation parameters and their respective ranges here.
# The keys will be used directly as labels in the plots and text logs.
parameter_ranges_dict = {
    'T': [0, 10],
    'C': [0, 3]
}

# Automatically derive the number of activation parameters
ACT_PARM = len(parameter_ranges_dict)


# Shared Parameters
parser.add_argument('--device_number', type=int, default=device_number, help='GPU device index')
parser.add_argument('--image', type=str, default=image, help='Image basename in Data/kodak (without extension)')
parser.add_argument('--model', type=str, default=MODEL, help='Default model name for run metadata and --inr_model')
parser.add_argument('--hidden_layers', type=int, default=HIDDEN_LAYERS, help='Number of hidden layers')
parser.add_argument('--parameter_ranges_dict', type=str, default=None, help='JSON dict for activation parameter ranges')
parser.add_argument('--input',type=str, default=None, help='Input image path (overrides --image)')
parser.add_argument('--inr_model',type=str, default=None, help='[gauss, mfn, relu, siren, wire, wire2d, ffn, incode]')
parser.add_argument('--lr',type=float, default=LR, help='Learning rate')
parser.add_argument('--using_schedular', type=bool, default=True, help='Whether to use schedular')
parser.add_argument('--scheduler_b', type=float, default=0.1, help='Learning rate scheduler')
parser.add_argument('--maxpoints', type=int, default=256*256, help='Batch size')
parser.add_argument('--niters', type=int, default=NITERS, help='Number if iterations')
parser.add_argument('--steps_til_summary', type=int, default=STR, help='Number of steps till summary visualization')
parser.add_argument('--tb_hist_interval', type=int, default=100, help='TensorBoard histogram logging interval in steps (<=0 disables)')
parser.add_argument('--tau', type=float, default=40.0, help='Photon noise')
parser.add_argument('--noise_snr', type=float, default=2.0, help='Readout noise')

# INCODE Parameters
parser.add_argument('--a_coef',type=float, default=0.1993, help='a coeficient')
parser.add_argument('--b_coef',type=float, default=0.0196, help='b coeficient')
parser.add_argument('--c_coef',type=float, default=0.0588, help='c coeficient')
parser.add_argument('--d_coef',type=float, default=0.0269, help='d coeficient')


args = parser.parse_args()

if args.input is None:
    args.input = f"Data/Images/kodak/{args.image}.png"
    #args.input = f"data/Images/img_patches/{args.image}.png"
    # args.input = f"data/Images/{args.image}.png"

if args.inr_model is None:

    args.inr_model = args.model

if args.parameter_ranges_dict is not None:
    try:
        parameter_ranges_dict = json.loads(args.parameter_ranges_dict)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --parameter_ranges_dict JSON: {exc}") from exc

try:
    parameter_ranges_dict = {
        str(name): [float(bounds[0]), float(bounds[1])]
        for name, bounds in parameter_ranges_dict.items()
    }
except (AttributeError, TypeError, ValueError, IndexError) as exc:
    raise ValueError("--parameter_ranges_dict must map names to [min, max] pairs.") from exc

ACT_PARM = len(parameter_ranges_dict)
run_name = f"{args.model}_iters{args.niters}_lr{args.lr}_layers{args.hidden_layers}_img{args.image}_128"

if torch.cuda.is_available():
    torch.cuda.set_device(args.device_number)
    device = torch.device(f"cuda:{args.device_number}")
else:
    device = torch.device("cpu")

time_array = torch.zeros(args.niters, device=device)

save_name = f"{run_name}_{args.inr_model}"
save_path = f"Results/image_denoising/{save_name}"
os.makedirs(save_path, exist_ok=True)

# Initialize TensorBoard Writer
tb_log_dir = f"runs/image_denoising/{save_name}"
writer = SummaryWriter(log_dir=tb_log_dir)
# -----------------------------------------------

def log_tensorboard_histograms(tb_writer, model, step, activation_consts=None):
    """Log parameter and gradient histograms for TensorBoard."""
    for name, param in model.named_parameters():
        tag_name = name.replace('.', '/')
        if 'bias' in name.lower():
            param_group = 'Biases'
        elif 'weight' in name.lower():
            param_group = 'Weights'
        else:
            param_group = 'Parameters'

        tb_writer.add_histogram(f'{param_group}/{tag_name}', param.detach().float().cpu(), step)
        if param.grad is not None:
            tb_writer.add_histogram(f'Gradients/{tag_name}', param.grad.detach().float().cpu(), step)

    if activation_consts is not None and activation_consts.size > 0:
        tb_writer.add_histogram(
            'ActivationParams/RawConsts',
            torch.from_numpy(activation_consts).float(),
            step
        )

## Loading Data
import cv2
im_RGB_gt = utils.normalize(plt.imread(args.input).astype(np.float32), True)
im = im_RGB_gt
H, W, _ = im.shape
np.random.seed(0)
im_noisy = utils.measure(im, args.noise_snr, args.tau).astype(np.float32)
im_noisy_gt = utils.normalize(im_noisy, True).astype(np.float32)
noisy_im = im_noisy_gt

### Model Configurations
print(f"Running Model: {args.inr_model}")

### Harmonizer Configurations
MLP_configs={
    'task': 'denoising',
    'model': 'resnet34',
    'truncated_layer':5,
    'in_channels': 64,
    'hidden_channels': [64, 32, (ACT_PARM)*(args.hidden_layers+1)],
    'mlp_bias':0.3120,
    'activation_layer': nn.SiLU,
    'GT': torch.tensor(im_noisy_gt).to(device)[None,...].permute(0, 3, 1, 2),
    
    # Pass just the list of ranges to the INR class for unpacking
    'param_ranges': list(parameter_ranges_dict.values()) 
}

### Model Configurations
model =INR(args.inr_model).run(in_features=2,
                                out_features=3,
                                hidden_features=128,#256,
                                hidden_layers=args.hidden_layers,
                                activation_parameters=ACT_PARM,
                                MLP_configs=MLP_configs
                               ).to(device)

## Training Code
init_time = time.time()

# Optimizer setup
if args.inr_model in ['wire', 'COSMOV3', 'COSMOV4']:
    args.lr = args.lr * min(1, args.maxpoints / (H * W))
optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
scheduler = lr_scheduler.LambdaLR(optim, lambda x: args.scheduler_b ** min(x / args.niters, 1))

# Initialize lists for PSNR and MSE values
consts, psnr_values = [], []
mse_array = torch.zeros(args.niters, device=device)

# Initialize best loss values
best_loss = torch.tensor(float('inf'))
best_summary_loss = float('inf')  

# Generate coordinate grid
coords = utils.get_coords(H, W, dim=2)[None, ...]
gt_noisy = torch.tensor(im_noisy).reshape(H * W, 3)[None, ...].to(device)
gt_clean = torch.tensor(im).reshape(H * W, 3)[None, ...].to(device)
rec = torch.zeros_like(gt_noisy)


pbar = tqdm(range(args.niters))
for step in pbar:
    indices = torch.randperm(H*W)

    for b_idx in range(0, H*W, args.maxpoints):
        b_indices = indices[b_idx:min(H*W, b_idx+args.maxpoints)]
        b_coords = coords[:, b_indices, ...].to(device)
        b_indices = b_indices.to(device)

        if args.inr_model == 'incode':
            model_output, coef = model(b_coords)
        else:
            model_output = model(b_coords)

        with torch.no_grad():
            rec[:, b_indices, :] = model_output

        output_loss = ((model_output - gt_noisy[:, b_indices, :])**2).mean()

        if args.inr_model == 'incode':
            a_coef, b_coef, c_coef, d_coef = coef[0]
            reg_loss = args.a_coef * torch.relu(-a_coef) + \
                       args.b_coef * torch.relu(-b_coef) + \
                       args.c_coef * torch.relu(-c_coef) + \
                       args.d_coef * torch.relu(-d_coef)
            loss = output_loss + reg_loss
        else:
            loss = output_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

    time_array[step] = time.time() - init_time
    
    with torch.no_grad():
        mse_array[step] = ((gt_clean - rec)**2).mean().item()
        psnr = -10*torch.log10(mse_array[step])
        psnr_values.append(psnr.item())
        
        # Extract the current step's constants before appending to the list
        current_consts = model.prior.consts.detach().cpu().numpy()
        consts.append(current_consts)
        
        # ---> INSERTION 2: Real-time TensorBoard Logging
        writer.add_scalar('Metrics/MSE_Loss', mse_array[step].item(), step)
        writer.add_scalar('Metrics/PSNR', psnr.item(), step)

        if args.tb_hist_interval > 0 and (step % args.tb_hist_interval == 0 or step == args.niters - 1):
            log_tensorboard_histograms(writer, model, step, activation_consts=current_consts)
        
        if current_consts.size > 0:
            coef_flat = current_consts.flatten()
            try:
                # Split the flattened array into ACT_PARM chunks.
                # Each chunk contains the values for all hidden layers for that specific parameter.
                raw_splits = np.split(coef_flat, ACT_PARM)
                
                for idx, (p_name, p_range) in enumerate(parameter_ranges_dict.items()):
                    for layer_idx, val in enumerate(raw_splits[idx]):
                        # Apply the identical sigmoid scaling used in your plotting logic
                        scaled_val = (1 / (1 + np.exp(-val))) * (p_range[1] - p_range[0]) + p_range[0]
                        
                        # Groups plots by parameter (e.g., "Params_T") and labels lines by layer
                        writer.add_scalar(f'Params_{p_name}/Layer_{layer_idx+1}', scaled_val, step)
            except ValueError:
                # Safely bypass if shapes mismatch during the very first initialization step
                pass
        # -----------------------------------------------

        pbar.set_description(f"step {step+1}/{args.niters} | PSNR {psnr:.2f} dB")
        scheduler.step()
    imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()

    if (mse_array[step] < best_loss) or (step == 0):
        best_loss = mse_array[step]
        best_flat_img = rec
        best_img = imrec
        best_epoch = step

    if step % args.steps_til_summary == 0:
        current_mse = mse_array[step].item()
        if current_mse < best_summary_loss:
            best_summary_loss = current_mse
            weight_save_path = os.path.join(save_path, f"{args.inr_model}_best_weights.pth")
            torch.save(model.state_dict(), weight_save_path)


print(f"Final PSNR: {psnr_values[-1]}")

def get_np_psnr(image1, image2):
    loss = ((image1.astype(np.float32) - image2.astype(np.float32))**2).mean()
    return -10*np.log10(loss)

def get_np_loss(image1, image2):
    loss = ((image1.astype(np.float32) - image2.astype(np.float32))**2).mean()
    return loss

rec_loss = get_np_loss(best_img, im)
rec_psnr = get_np_psnr(best_img, im)
print(f"Loss: {rec_loss:.8f}, PSNR: {rec_psnr:.5f}")


# -----------------------------------------------------------------------------
# DYNAMIC PLOTTING LOGIC
# -----------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

num_top_plots = 3
num_bottom_plots = 1 + ACT_PARM
total_cols = num_top_plots * num_bottom_plots

fig = plt.figure(figsize=(max(num_top_plots, num_bottom_plots) * 6, 12), dpi=500)
gs = gridspec.GridSpec(2, total_cols, figure=fig)

top_span = total_cols // num_top_plots

# 1. Ground Truth Image
ax_gt = fig.add_subplot(gs[0, 0:top_span])
ax_gt.imshow(np.clip(im, 0, 1))
ax_gt.set_title("Ground Truth Image")
ax_gt.axis('off')

# 2. Noisy Image
ax_noisy = fig.add_subplot(gs[0, top_span:2*top_span])
ax_noisy.imshow(np.clip(noisy_im, 0, 1))
noisy_psnr = -10*np.log10(((noisy_im - im)**2).mean())
ax_noisy.set_title(f"Noisy Image (PSNR: {noisy_psnr:.2f} dB)")
ax_noisy.axis('off')

# 3. Reconstructed Image
ax_img = fig.add_subplot(gs[0, 2*top_span:3*top_span])
ax_img.imshow(np.clip(best_img, 0, 1))
ax_img.set_title(f"Reconstructed Image (PSNR: {rec_psnr:.2f} dB)")
ax_img.axis('off')

bottom_span = total_cols // num_bottom_plots

# 4. Loss Curve
ax_loss = fig.add_subplot(gs[1, 0:bottom_span])
ax_loss.plot(range(args.niters), mse_array.cpu().numpy(), color='#1f77b4') 
ax_loss.set_title("Loss Convergence (MSE)")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("MSE Loss")
ax_loss.grid(True, linestyle="--", alpha=0.6)

# 5. Dynamic Activation Parameters
processed_params = {}
if len(consts) > 0:
    coef = np.array(consts).squeeze()
    if coef.ndim > 2:
        coef = coef.reshape(coef.shape[0], -1)
    
    raw_splits = np.split(coef, ACT_PARM, axis=1)
    for idx, (p_name, p_range) in enumerate(parameter_ranges_dict.items()):
        scaled_data = sigmoid(raw_splits[idx]) * (p_range[1] - p_range[0]) + p_range[0]
        processed_params[p_name] = scaled_data
        
        start_col = (idx + 1) * bottom_span
        end_col = start_col + bottom_span
        ax_p = fig.add_subplot(gs[1, start_col:end_col])
        
        for i in range(scaled_data.shape[1]):
            ax_p.plot(range(args.niters), scaled_data[:, i], label=f"{p_name}_{i+1}")
            
        final_vals = np.round(scaled_data[-1], 3)
        vals_str = " ".join(map(str, final_vals))
        ax_p.set_title(f"{p_name} Params\n(Final: {vals_str})")
        ax_p.set_xlabel("Epochs")
        ax_p.set_ylabel("Values")
        ax_p.legend(fontsize='small')
else:
    ax_params = fig.add_subplot(gs[1, bottom_span:2*bottom_span])
    ax_params.set_title("Activation Params (N/A)")
    ax_params.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_path, "training_summary.png"), dpi=500, bbox_inches='tight')
plt.close()


# -----------------------------------------------------------------------------
# DYNAMIC TEXT & DATA LOGGING
# -----------------------------------------------------------------------------
total_train_time = time.time() - init_time
avg_epoch_time = total_train_time / args.niters

# Calculate Trainable Model Parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate Max GPU Memory Consumption and Fetch GPU Model
if torch.cuda.is_available():
    max_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    gpu_model_name = torch.cuda.get_device_name(device)
else:
    max_gpu_memory_mb = 0.0
    gpu_model_name = "CPU (CUDA not available)"

# 1. Save Text Summary File
txt_save_path = os.path.join(save_path, "model_summary.txt")
with open(txt_save_path, "w") as f:
    f.write("=== COSMO Training Summary ===\n")
    f.write(f"Model Name:      {args.inr_model}\n")
    f.write(f"Input Image:     {args.input}\n")
    f.write(f"Learning Rate:   {args.lr}\n")
    f.write(f"Iterations:      {args.niters}\n")
    f.write(f"Batch Size:      {args.maxpoints}\n\n")
    
    f.write("=== Hardware & Architecture Benchmarks ===\n")
    f.write(f"GPU Model:               {gpu_model_name}\n")
    f.write(f"Total Trainable Params:  {total_params:,}\n")
    f.write(f"Peak GPU Memory Used:    {max_gpu_memory_mb:.2f} MB\n")
    f.write(f"Total Time:              {total_train_time:.2f} seconds\n")
    f.write(f"Time/Epoch:              {avg_epoch_time:.5f} seconds\n\n")
    
    f.write("=== Final Evaluation Metrics ===\n")
    f.write(f"Best Loss (MSE): {rec_loss:.8f}\n")
    f.write(f"Best PSNR:       {rec_psnr:.4f} dB\n")
    f.write(f"Final PSNR:      {psnr_values[-1]:.4f} dB\n\n")
    
    f.write("=== Converged Activation Parameters ===\n")
    if processed_params:
        num_tracked_layers = list(processed_params.values())[0].shape[1]
        for i in range(num_tracked_layers):
            f.write(f"Layer {i+1}:\n")
            for p_name, p_data in processed_params.items():
                f.write(f"  {p_name:<7} = {p_data[-1, i]:.4f}\n")
            f.write("\n")
    else:
        f.write("No activation parameters tracked for this configuration.\n")

# 2. Save Arrays to NPZ Archive
npz_save_path = os.path.join(save_path, "training_arrays.npz")
arrays_to_save = {
    'psnr_array': np.array(psnr_values),
    'loss_array': mse_array.cpu().numpy(),
    'time_array': time_array.cpu().numpy()
}

for p_name, p_data in processed_params.items():
    arrays_to_save[f'{p_name.lower()}_params'] = p_data

np.savez_compressed(npz_save_path, **arrays_to_save)

# Flush and close writer
writer.close()
# -----------------------------------------------

print(f"Summary text file saved to {txt_save_path}")
print(f"Training arrays saved to {npz_save_path}")
print(f"Results successfully saved to {save_path}")
