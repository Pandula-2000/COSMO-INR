import skimage
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import cv2
from scipy import io
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
from modules import utils
from modules import models
!pip install -q pytorch-msssim # Install this library if error pops up (Uncomment and run)
from pytorch_msssim import ssim
from modules.models1 import INR

torch.cuda.set_device(7)

parser = argparse.ArgumentParser(description='INCODE')

run_name = f"GPU7"
image = "kodim20"

# Shared Parameters
parser.add_argument('--input',type=str, default=f"data/Images/kodak/{image}.png", help='Input image path')
parser.add_argument('--inr_model',type=str, default='COSMOV4', help='[gauss, mfn, relu, siren, wire, wire2d, ffn, incode]')
parser.add_argument('--lr',type=float, default=9e-4, help='Learning rate')
parser.add_argument('--using_schedular', type=bool, default=True, help='Whether to use schedular')
parser.add_argument('--scheduler_b', type=float, default=0.1, help='Learning rate scheduler')
parser.add_argument('--maxpoints', type=int, default=256*256, help='Batch size')
parser.add_argument('--niters', type=int, default=4500, help='Number if iterations')
parser.add_argument('--steps_til_summary', type=int, default=50, help='Number of steps till summary visualization')

# INCODE Parameters
parser.add_argument('--a_coef',type=float, default=0.1993, help='a coeficient')
parser.add_argument('--b_coef',type=float, default=0.0196, help='b coeficient')
parser.add_argument('--c_coef',type=float, default=0.0588, help='c coeficient')
parser.add_argument('--d_coef',type=float, default=0.0269, help='d coeficient')

args = parser.parse_args(args=[])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_array = torch.zeros(args.niters, device=device)

save_name = f"{run_name}_{args.inr_model}"
save_path = f"Results/{save_name}"
os.makedirs(save_path, exist_ok=True)

im_RGB_gt = utils.normalize(plt.imread(args.input).astype(np.float32), True)
im = im_RGB_gt
H, W, _ = im.shape
print(f"Image size(H,W): ({H} x {W})")

# --- Define Positional Encodings ---
# Frequency Encoding
pos_encode_freq = {'type':'frequency', 'use_nyquist': True, 'mapping_input': int(max(H, W)/3)}
# Gaussian Encoding
pos_encode_gaus = {'type':'gaussian', 'scale_B': 10, 'mapping_input': 256}
# No Encoding
pos_encode_no = {'type': None}

# --- Model Configurations ---
print(f"Running Model: {args.inr_model}")

act_prams=2
hidden_lay=3
MLP_configs={'task': 'image',
             'model': 'resnet34',
             'truncated_layer':5,
             'in_channels': 64,
             #'hidden_channels': [64, 32, 4],
             'hidden_channels': [64, 32, act_prams*(hidden_lay+1)],
             'mlp_bias':0.3120,
             'activation_layer': nn.SiLU,
             'GT': torch.tensor(im).to(device)[None,...].permute(0, 3, 1, 2),
             'T_range': [0, 10],
             'c_range': [0, 3],
            }

model =INR(args.inr_model).run(in_features=2,
                                out_features=3,
                                hidden_features=256,
                                hidden_layers=3,
                                #first_omega_0=30.0,
                                #hidden_omega_0=30.0,
                                #pos_encode_configs=pos_encode_freq,
                                #ffn_type='relu',
                                MLP_configs = MLP_configs
                               ).to(device)

# --- Training Loop ---
init_time = time.time() #timearray setup

# Optimizer setup
if args.inr_model == 'wire' or args.inr_model == 'BandRC':
    args.lr = args.lr * min(1, args.maxpoints / (H * W))
optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
scheduler = lr_scheduler.LambdaLR(optim, lambda x: args.scheduler_b ** min(x / args.niters, 1))

# Initialize lists for PSNR and MSE values
psnr_values = []
mse_array = torch.zeros(args.niters, device=device)
# Initialize best loss value as positive infinity
best_loss = torch.tensor(float('inf'))
# Generate coordinate grid
coords = utils.get_coords(H, W, dim=2)[None, ...]
# Convert input image to a tensor and reshape
gt = torch.tensor(im).reshape(H * W, 3)[None, ...].to(device)
# Initialize a tensor for reconstructed data
rec = torch.zeros_like(gt)

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

        output_loss = ((model_output - gt[:, b_indices, :])**2).mean()

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
        mse_array[step] = ((gt - rec)**2).mean().item()
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
print(rec_loss, rec_psnr)

weight_path = os.path.join(save_path, f"iters_{len(psnr_values)}_psnr_{max(psnr_values):.2f}.pth")
torch.save({'state_dict': model.state_dict(),
            'best_epoch': best_epoch,
            'gt': im_RGB_gt,
            'rec': best_img,
            #'consts_array': np.array(consts),
            'time_array': time_array.detach().cpu().numpy(),
            'mse_array': mse_array.detach().cpu().numpy(),
            'psnr_vals_hsv_domain': np.array(psnr_values)
            }, weight_path)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].set_title('Ground Truth')
axes[0].imshow(im_RGB_gt)
axes[0].axis('off')
axes[1].set_title(f"RGB_PSNR= {rec_psnr:.3f} ")
axes[1].imshow(best_img)
axes[1].axis('off')

plt.savefig(f"{save_path}/comparison.png")
# iters_{len(psnr_values)}_psnr_{max(psnr_values):.2f}_

plt.show()


# Print maximum PSNR achieved during training
print('--------------------')
print('Max PSNR:', max(psnr_values))
print('--------------------')
print(best_epoch)


def get_sigmoid_projected_params(A, T_range=[0,10], c_range=[0,3]):
    """
    Given unconstrained parameters A, project them via sigmoid to desired ranges.
    Args:
        A: tensor of shape (Tunable_params,)
        T_range: list, desired range for time parameter t0
        c_range: list, desired range for concentration parameter c0
    Returns:
        t0: projected time parameter in T range [0, 10]
        c0: projected concentration parameter in C range [0, 3]
    """
    A_ = A.view(4,2)
    t0, c0 = torch.unbind(A_, dim=1)
    t0 = torch.sigmoid(t0) * (T_range[1] - T_range[0]) + T_range[0]
    c0 = torch.sigmoid(c0) * (c_range[1] - c_range[0]) + c_range[0]
    return t0, c0


final_coverged_A = np.array(consts)[-1,:,:]
t0, c0 = get_sigmoid_projected_params(torch.tensor(final_coverged_A), T_range=MLP_configs['T_range'], c_range=MLP_configs['c_range'])

print("Final projected parameters after training:")
print("t0:", t0)
print("c0:", c0)



