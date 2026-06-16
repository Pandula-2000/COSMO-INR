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
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from modules import utils
from modules.models1 import INR
from torch.utils.tensorboard import SummaryWriter

device_number = 0

parser = argparse.ArgumentParser(description='COSMO Inpainting')
parser.add_argument('--device_number', type=int, default=device_number, help='GPU device index')
parser.add_argument('--image', type=str, default="celtic_spiral_knot", help='Image basename (without extension)')
parser.add_argument('--model', type=str, default="BandRC", help='Default model name for run metadata and --inr_model')
parser.add_argument('--hidden_layers', type=int, default=4, help='Number of hidden layers')
parser.add_argument('--parameter_ranges_dict', type=str, default=None, help='JSON dict for activation parameter ranges')
parser.add_argument('--input',type=str, default='./Data/Images/celtic_spiral_knot.jpg', help='Input image path (overrides --image)')
parser.add_argument('--inr_model',type=str, default=None, help='[gauss, mfn, relu, siren, wire, wire2d, ffn, incode]')
parser.add_argument('--lr',type=float, default=9e-4, help='Learning rate')
parser.add_argument('--using_schedular', type=bool, default=True, help='Whether to use schedular')
parser.add_argument('--scheduler_b', type=float, default=0.25, help='Learning rate scheduler')
parser.add_argument('--maxpoints', type=int, default=8192, help='Batch size')
parser.add_argument('--niters', type=int, default=500, help='Number if iterations')
parser.add_argument('--steps_til_summary', type=int, default=20, help='Number of steps till summary visualization')
parser.add_argument('--tb_hist_interval', type=int, default=100, help='TensorBoard histogram logging interval in steps (<=0 disables)')
parser.add_argument('--sampling_ratio', type=float, default=0.2, help='The percentage of pixels used for training')

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
    args.input = f"Data/Images/{args.image}.jpg"
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
save_path = f"Results/image_inpainting/{save_name}"
os.makedirs(save_path, exist_ok=True)

tb_log_dir = f"runs/image_inpainting/{save_name}"
writer = SummaryWriter(log_dir=tb_log_dir)

im = np.array(Image.open(args.input))
im = torch.from_numpy(im).float()/255.0
H, W, C = im.shape
pixel_count = H * W
sampled_pixel_count = int(pixel_count * args.sampling_ratio)

img_mask, img_train = utils.build_train_data(im, sampled_pixel_count)
args.maxpoints = int(min(len(img_mask), args.maxpoints))

train_dataset = TensorDataset(img_mask, img_train)
train_dataloader = DataLoader(train_dataset, batch_size=args.maxpoints, shuffle=True, pin_memory=True)

MLP_configs={
    'task': 'inpainting',
    'model': 'resnet34',
    'truncated_layer':5,
    'in_channels': 64,
    'hidden_channels': [64, 32, (ACT_PARM)*(args.hidden_layers+1)],
    'mlp_bias':0.3120,
    'activation_layer': nn.SiLU,
    'GT': img_train[None, ...].permute(0, 2, 1).to(device),
    'param_ranges': list(parameter_ranges_dict.values()) 
}

model = INR(args.inr_model).run(in_features=2,
                                out_features=3,
                                hidden_features=128,
                                hidden_layers=args.hidden_layers,
                                activation_parameters=ACT_PARM,
                                MLP_configs=MLP_configs
                               ).to(device)

def test(test_model, b_size=args.maxpoints):
    img_mask_full, img_eval_full = utils.build_eval_data(im)
    test_dataset_full = TensorDataset(img_mask_full, img_eval_full)
    test_dataloader_full = DataLoader(test_dataset_full, batch_size=b_size, shuffle=False, pin_memory=True)
    
    with torch.no_grad():
        predictions = []
        for batch in test_dataloader_full:
            inputs, _ = batch
            inputs = inputs.to(device)
            if args.inr_model == 'incode':
                prediction, _ = test_model(inputs)
            else:
                prediction = test_model(inputs)
            predictions.append(prediction)

        predicted_image = torch.cat(predictions).cpu().numpy()
        predicted_image = predicted_image.reshape((H, W, C)).astype(np.float32)
        return predicted_image

init_time = time.time()
if args.inr_model in ['wire', 'COSMOV3', 'COSMOV4']:
    args.lr = args.lr * min(1, args.maxpoints / (H * W))
optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
scheduler = lr_scheduler.LambdaLR(optim, lambda x: args.scheduler_b ** min(x / args.niters, 1))

consts, psnr_train_values, psnr_test_values = [], [], []
mse_array = torch.zeros(args.niters, device=device)

pbar = tqdm(range(args.niters))
for step in pbar:
    loss_values = []
    psnr_batch_values = []
    
    for batch in train_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        if args.inr_model == 'incode':
            model_output, coef = model(inputs)  
        else:
            model_output = model(inputs) 

        output_loss = ((model_output - targets)**2).mean()
        
        if args.inr_model == 'incode':
            a_coef, b_coef, c_coef, d_coef = coef[0]  
            reg_loss = args.a_coef * torch.relu(-a_coef) + args.b_coef * torch.relu(-b_coef) + args.c_coef * torch.relu(-c_coef) + args.d_coef * torch.relu(-d_coef)
            loss = output_loss + reg_loss 
        else: 
            loss = output_loss
        
        loss_values.append(output_loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            batch_psnr = -10*torch.log10(output_loss)
            psnr_batch_values.append(batch_psnr.item())
    
    avg_loss = np.mean(loss_values)
    avg_psnr = np.mean(psnr_batch_values)
    
    time_array[step] = time.time() - init_time
    
    with torch.no_grad():
        mse_array[step] = avg_loss
        psnr_train_values.append(avg_psnr)
        
        current_consts = model.prior.consts.detach().cpu().numpy()
        consts.append(current_consts)
        
        writer.add_scalar('Metrics/MSE_Loss_Train', avg_loss, step)
        writer.add_scalar('Metrics/PSNR_Train', avg_psnr, step)

        pbar.set_description(f"step {step+1}/{args.niters} | PSNR Train {avg_psnr:.2f} dB")
        scheduler.step()

# End of training: Evaluate Full image
best_img = test(model)
mse_loss_test = ((im.numpy() - best_img)**2).mean()
rec_psnr_test = -10*np.log10(mse_loss_test)
rec_psnr_train = psnr_train_values[-1]

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

num_top_plots = 3
num_bottom_plots = 1 + ACT_PARM
total_cols = num_top_plots * num_bottom_plots

fig = plt.figure(figsize=(max(num_top_plots, num_bottom_plots) * 4, 8), dpi=300)
gs = gridspec.GridSpec(2, total_cols, figure=fig)

top_span = total_cols // num_top_plots

# 1. Ground Truth Image
ax_gt = fig.add_subplot(gs[0, 0:top_span])
ax_gt.imshow(np.clip(im.numpy(), 0, 1))
ax_gt.set_title("Ground Truth Image")
ax_gt.axis('off')

# 2. Masked Image visualization
# Create a dummy image showing the sampled points
im_masked = np.zeros_like(im.numpy())
mask_coords = img_mask.numpy()
x_coords = (mask_coords[:, 0] * H).astype(int)
y_coords = (mask_coords[:, 1] * W).astype(int)
im_masked[x_coords, y_coords] = im.numpy()[x_coords, y_coords]

ax_masked = fig.add_subplot(gs[0, top_span:2*top_span])
ax_masked.imshow(np.clip(im_masked, 0, 1))
ax_masked.set_title(f"Sampled Points ({args.sampling_ratio*100}%)")
ax_masked.axis('off')

# 3. Reconstructed Image
ax_img = fig.add_subplot(gs[0, 2*top_span:3*top_span])
ax_img.imshow(np.clip(best_img, 0, 1))
ax_img.set_title(f"Reconstructed (PSNR: {rec_psnr_test:.2f} dB)")
ax_img.axis('off')

bottom_span = total_cols // num_bottom_plots

# 4. Loss Curve
ax_loss = fig.add_subplot(gs[1, 0:bottom_span])
ax_loss.plot(range(args.niters), mse_array.cpu().numpy(), color='#1f77b4') 
ax_loss.set_title("Loss Convergence (Train MSE)")
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

# Save arrays
npz_save_path = os.path.join(save_path, "training_arrays.npz")
arrays_to_save = {
    'psnr_array_train': np.array(psnr_train_values),
    'loss_array': mse_array.cpu().numpy(),
    'time_array': time_array.cpu().numpy()
}
for p_name, p_data in processed_params.items():
    arrays_to_save[f'{p_name.lower()}_params'] = p_data
np.savez_compressed(npz_save_path, **arrays_to_save)
writer.close()
print(f"Results successfully saved to {save_path}")
