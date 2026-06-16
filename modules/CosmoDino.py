import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torchvision.models.video as video
from torchvision.transforms import v2
import torchaudio
from modules.utils import normalize, measure, psnr


"""
Model
Raised_coseine * cos(Cx+theta)
Params: 1 + 2
"""

class MLP(torch.nn.Sequential):
    '''
    Args:
        in_channels (int): Number of input channels or features.
        hidden_channels (list of int): List of hidden layer sizes. The last element is the output size.
        mlp_bias (float): Value for initializing bias terms in linear layers.
        activation_layer (torch.nn.Module, optional): Activation function applied between hidden layers. Default is SiLU.
        bias (bool, optional): If True, the linear layers include bias terms. Default is True.
        dropout (float, optional): Dropout probability applied after the last hidden layer. Default is 0.0 (no dropout).
    '''
    def __init__(self, MLP_configs, bias=True, dropout = 0.0):
        super().__init__()

        in_channels=MLP_configs['in_channels']
        hidden_channels=MLP_configs['hidden_channels']
        self.mlp_bias=MLP_configs['mlp_bias']
        activation_layer=MLP_configs['activation_layer']

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if MLP_configs['task'] == 'denoising':
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.001)
            torch.nn.init.constant_(m.bias, self.mlp_bias)

    def forward(self, x):
        self.consts = self.layers(x)
        return self.consts


class RaisedCosineImpulseResponseLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False,
                 beta0=0.5, eps=1e-8, out_real=False):
        super().__init__()

        self.beta0 = nn.Parameter(torch.tensor(beta0, dtype=torch.float), requires_grad=False)
        self.eps = eps
        self.is_first = is_first
        self.out_real = out_real

        self.in_features = in_features
        self.out_features = out_features

        dtype = torch.float if self.is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        nn.init.uniform_(self.linear.weight, -1/self.in_features, 1/self.in_features)  # Initialize weights (new method)

    def forward(self, input, t0, c0):
        input = input.to(next(self.parameters()).device)  # Move input to correct device
        lin = self.linear(input)

        if not self.is_first:
            lin = lin / torch.abs(lin + self.eps)  # Normalize

        f1 = (1 / t0) * torch.sinc(lin / t0) * torch.cos(torch.pi * self.beta0 * lin / t0)
        f2 = 1 - (2 * self.beta0 * lin / t0) ** 2 + self.eps
        theta = 2 * torch.pi * c0 * lin * 1j

        rc = (f1 / f2)
        out = rc * torch.exp(theta)

        if not self.is_first:
            out = out / torch.abs(out + self.eps)

        return out.real if self.out_real else out

# 1. Load the Model
REPO_DIR = "dinov3"

dinov3_vits16_model_path = "DinoPreTrainModels/pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
dinov3_vith16plus_model_path = "DinoPreTrainModels/pretrain/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
dinov3_vits16plus_model_path = "DinoPreTrainModels/pretrain/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
dinov3_vitl16_sat_model_path = "DinoPreTrainModels/pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
dinov3_vitb16_model_path = "DinoPreTrainModels/pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
dinov3_vitl16_lvd_model_path = "DinoPreTrainModels/pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

# 2. Define the exact transform pipeline for LVD-1689M models
def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    # resize = v2.Resize((resize_size, resize_size), antialias=True)
    resize = to_tensor
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225)
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, activation_parameters, out_features, MLP_configs):
        super().__init__()

        self.img = make_transform()(MLP_configs['GT'].cuda())  # Add batch dimension

        self.prior = MLP(MLP_configs)
        self.param_ranges = MLP_configs['param_ranges']
        self.hidden_layers = hidden_layers
        self.activation_parameters = activation_parameters

        # 1. Load DINOv3 and freeze it
        repo_dir = MLP_configs.get('repo_dir', 'dinov3')
        dino_variant = MLP_configs.get('model', 'dinov3_vits16')

        self.feature_extractor = torch.hub.load(
                                REPO_DIR, 
                                'dinov3_vith16plus', 
                                source='local', 
                                weights=dinov3_vith16plus_model_path        # NOTE: Custom path to the pre-trained weights file
                            ).cuda().eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.nonlin = RaisedCosineImpulseResponseLayer
        dtype = torch.cfloat

        self.net = nn.ModuleList()
        self.net.append(self.nonlin(in_features, hidden_features, is_first=True))
        for _ in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features))

        self.final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)

    def forward(self, coords):
        # 2. Extract the [CLS] token without tracking gradients
        with torch.no_grad():
            # print(self.img.shape)
            features = self.feature_extractor.forward_features(self.img)
            cls_token = features['x_norm_clstoken'] # Shape: [1, D]
        # print("Extracted CLS token shape:", cls_token.shape)
        # 3. Feed directly into the prior (replaces GAP)
        coef = self.prior(cls_token).view(self.hidden_layers + 1, self.activation_parameters)
        
        raw_params = torch.unbind(coef, dim=1)

        projected_params = []
        for i, param_tensor in enumerate(raw_params):
            p_min, p_max = self.param_ranges[i]
            proj = torch.sigmoid(param_tensor) * (p_max - p_min) + p_min
            projected_params.append(proj)

        output = coords
        for id, lyr in enumerate(self.net):
            layer_args = [p[id] for p in projected_params]
            output = lyr(output, *layer_args) 
            
        output = self.final_linear(output).real
        return nn.Sigmoid()(output)
    



