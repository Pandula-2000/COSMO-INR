import os
import cv2
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

parser = argparse.ArgumentParser(description='COSMO SR')
parser.add_argument('--device_number', type=int, default=device_number, help='GPU device index')
parser.add_argument('--image', type=str, default="kodim20", help='Image basename in Data/kodak (without extension)')
parser.add_argument('--model', type=str, default="BandRC", help='Default model name for run metadata and --inr_model')
parser.add_argument('--hidden_layers', type=int, default=4, help='Number of hidden layers')
parser.add_argument('--parameter_ranges_dict', type=str, default=None, help='JSON dict for activation parameter ranges')
parser.add_argument('--input',type=str, default=None, help='Input image path (overrides --image)')
parser.add_argument('--inr_model',type=str, default=None, help='[gauss, mfn, relu, siren, wire, wire2d, ffn, incode]')
parser.add_argument('--lr',type=float, default=9e-4, help='Learning rate')
parser.add_argument('--using_schedular', type=bool, default=True, help='Whether to use schedular')
parser.add_argument('--scheduler_b', type=float, default=0.1, help='Learning rate scheduler')
parser.add_argument('--maxpoints', type=int, default=256*256, help='Batch size')
parser.add_argument('--niters', type=int, default=500, help='Number if iterations')
parser.add_argument('--steps_til_summary', type=int, default=5, help='Number of steps till summary visualization')
parser.add_argument('--tb_hist_interval', type=int, default=100, help='TensorBoard histogram logging interval in steps (<=0 disables)')
parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor for super-resolution')

# INCODE Parameters
parser.add_argument('--a_coef',type=float, default=0.1993, help='a coeficient')
parser.add_argument('--b_coef',type=float, default=0.0196, help='b coeficient')
parser.add_argument('--c_coef',type=float, default=0.0588, help='c coeficient')
parser.add_argument('--d_coef',type=float, default=0.0269, help='d coeficient')

args = parser.parse_args()

parameter_ranges_dict = {
    'T': [0, 10],
    'C': [0, 3]
}

if args.input is None:
    args.input = f"Data/Images/kodak/{args.image}.png"
if args.inr_model is None:
    args.inr_model = args.model

ACT_PARM = len(parameter_ranges_dict)
run_name = f"{args.model}_iters{args.niters}_lr{args.lr}_layers{args.hidden_layers}_img{args.image}_128"

if torch.cuda.is_available():
    torch.cuda.set_device(args.device_number)
    device = torch.device(f"cuda:{args.device_number}")
else:
    device = torch.device("cpu")

time_array = torch.zeros(args.niters, device=device)

save_name = f"{run_name}_{args.inr_model}"
save_path = f"Results/image_sr/{save_name}"
os.makedirs(save_path, exist_ok=True)

tb_log_dir = f"runs/image_sr/{save_name}"
writer = SummaryWriter(log_dir=tb_log_dir)

im_hr = utils.normalize(plt.imread(args.input).astype(np.float32), True)
im_lr = cv2.resize(im_hr, None, fx=1/args.upscale_factor, fy=1/args.upscale_factor, interpolation=cv2.INTER_AREA)
H_hr, W_hr, _ = im_hr.shape
H_lr, W_lr, _ = im_lr.shape

MLP_configs={
    'task': 'image',
    'model': 'resnet34',
    'truncated_layer':5,
    'in_channels': 64,
    'hidden_channels': [64, 32, (ACT_PARM)*(args.hidden_layers+1)],
    'mlp_bias':0.3120,
    'activation_layer': nn.SiLU,
    'GT': torch.tensor(im_lr).to(device)[None,...].permute(0, 3, 1, 2),
    'param_ranges': list(parameter_ranges_dict.values()) 
}

model = INR(args.inr_model).run(in_features=2,
                                out_features=3,
                                hidden_features=128,
                                hidden_layers=args.hidden_layers,
                                activation_parameters=ACT_PARM,
                                MLP_configs=MLP_configs
                               ).to(device)

init_time = time.time()
if args.inr_model in ['wire', 'COSMOV3', 'COSMOV4']:
    args.lr = args.lr * min(1, args.maxpoints / (H_lr * W_lr))
optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
scheduler = lr_scheduler.LambdaLR(optim, lambda x: args.scheduler_b ** min(x / args.niters, 1))

consts, psnr_values_lr = [], []
mse_array = torch.zeros(args.niters, device=device)

coords_lr = utils.get_coords(H_lr, W_lr, dim=2)[None, ...]
coords_hr = utils.get_coords(H_hr, W_hr, dim=2)[None, ...]

gt_lr = torch.tensor(im_lr).reshape(H_lr * W_lr, 3)[None, ...].to(device)
gt_hr = torch.tensor(im_hr).reshape(H_hr * W_hr, 3)[None, ...].to(device)

rec_lr = torch.zeros_like(gt_lr)
rec_hr = torch.zeros_like(gt_hr)

pbar = tqdm(range(args.niters))
for step in pbar:
    indices = torch.randperm(H_lr*W_lr)

    for b_idx in range(0, H_lr*W_lr, args.maxpoints):
        b_indices = indices[b_idx:min(H_lr*W_lr, b_idx+args.maxpoints)]
        b_coords = coords_lr[:, b_indices, ...].to(device)
        b_indices = b_indices.to(device)

        if args.inr_model == 'incode':
            model_output, coef = model(b_coords)
        else:
            model_output = model(b_coords)

        with torch.no_grad():
            rec_lr[:, b_indices, :] = model_output

        output_loss = ((model_output - gt_lr[:, b_indices, :])**2).mean()

        if args.inr_model == 'incode':
            a_coef, b_coef, c_coef, d_coef = coef[0]
            reg_loss = args.a_coef * torch.relu(-a_coef) + args.b_coef * torch.relu(-b_coef) + args.c_coef * torch.relu(-c_coef) + args.d_coef * torch.relu(-d_coef)
            loss = output_loss + reg_loss
        else:
            loss = output_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

    time_array[step] = time.time() - init_time
    
    with torch.no_grad():
        mse_array[step] = ((gt_lr - rec_lr)**2).mean().item()
        psnr = -10*torch.log10(mse_array[step])
        psnr_values_lr.append(psnr.item())
        
        current_consts = model.prior.consts.detach().cpu().numpy()
        consts.append(current_consts)
        
        writer.add_scalar('Metrics/MSE_Loss_LR', mse_array[step].item(), step)
        writer.add_scalar('Metrics/PSNR_LR', psnr.item(), step)

        pbar.set_description(f"step {step+1}/{args.niters} | PSNR LR {psnr:.2f} dB")
        scheduler.step()

# End of training: Evaluate HR image
with torch.no_grad():
    indices_hr = torch.randperm(H_hr*W_hr)
    for b_idx in range(0, H_hr*W_hr, args.maxpoints):
        b_indices_hr = indices_hr[b_idx:min(H_hr*W_hr, b_idx+args.maxpoints)]
        b_coords_hr = coords_hr[:, b_indices_hr, ...].to(device)
        b_indices_hr = b_indices_hr.to(device)

        if args.inr_model == 'incode':
            model_eval, _ = model(b_coords_hr)  
        else:
            model_eval = model(b_coords_hr) 
            
        rec_hr[:, b_indices_hr, :] = model_eval

    loss_hr = ((gt_hr - rec_hr)**2).mean().item()
    rec_psnr_hr = -10*np.log10(loss_hr)
    rec_psnr_lr = psnr_values_lr[-1]

best_img_hr = rec_hr[0, ...].reshape(H_hr, W_hr, 3).detach().cpu().numpy()
best_img_lr = rec_lr[0, ...].reshape(H_lr, W_lr, 3).detach().cpu().numpy()

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

num_top_plots = 4
num_bottom_plots = 1 + ACT_PARM
total_cols = num_top_plots * num_bottom_plots

fig = plt.figure(figsize=(max(num_top_plots, num_bottom_plots) * 4, 8), dpi=300)
gs = gridspec.GridSpec(2, total_cols, figure=fig)

top_span = total_cols // num_top_plots

# 1. Ground Truth HR Image
ax_gt_hr = fig.add_subplot(gs[0, 0:top_span])
ax_gt_hr.imshow(np.clip(im_hr, 0, 1))
ax_gt_hr.set_title("GT HR Image")
ax_gt_hr.axis('off')

# 2. GT LR Image
ax_gt_lr = fig.add_subplot(gs[0, top_span:2*top_span])
ax_gt_lr.imshow(np.clip(im_lr, 0, 1))
ax_gt_lr.set_title("GT LR Image")
ax_gt_lr.axis('off')

# 3. Reconstructed LR Image
ax_img_lr = fig.add_subplot(gs[0, 2*top_span:3*top_span])
ax_img_lr.imshow(np.clip(best_img_lr, 0, 1))
ax_img_lr.set_title(f"Reconstructed LR (PSNR: {rec_psnr_lr:.2f} dB)")
ax_img_lr.axis('off')

# 4. Reconstructed HR Image
ax_img_hr = fig.add_subplot(gs[0, 3*top_span:4*top_span])
ax_img_hr.imshow(np.clip(best_img_hr, 0, 1))
ax_img_hr.set_title(f"Reconstructed HR (PSNR: {rec_psnr_hr:.2f} dB)")
ax_img_hr.axis('off')


bottom_span = total_cols // num_bottom_plots

# 5. Loss Curve
ax_loss = fig.add_subplot(gs[1, 0:bottom_span])
ax_loss.plot(range(args.niters), mse_array.cpu().numpy(), color='#1f77b4') 
ax_loss.set_title("Loss Convergence (LR MSE)")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("MSE Loss")
ax_loss.grid(True, linestyle="--", alpha=0.6)

# 6. Dynamic Activation Parameters
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

# Save arrays
npz_save_path = os.path.join(save_path, "training_arrays.npz")
arrays_to_save = {
    'psnr_array_lr': np.array(psnr_values_lr),
    'loss_array': mse_array.cpu().numpy(),
    'time_array': time_array.cpu().numpy()
}
for p_name, p_data in processed_params.items():
    arrays_to_save[f'{p_name.lower()}_params'] = p_data
np.savez_compressed(npz_save_path, **arrays_to_save)
writer.close()
print(f"Results successfully saved to {save_path}")
