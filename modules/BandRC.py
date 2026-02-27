import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torchvision.models.video as video
import torchaudio
from modules.utils import normalize, measure, psnr


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

class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,MLP_configs
                 ):
        super().__init__()

        self.img = MLP_configs['GT'].cuda()

        self.prior = MLP(MLP_configs)
        self.T_range = MLP_configs['T_range']
        self.c_range = MLP_configs['c_range']

        model_ft = getattr(models, MLP_configs['model'])(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model_ft.children())[:MLP_configs['truncated_layer']])

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.nonlin = RaisedCosineImpulseResponseLayer

        dtype = torch.cfloat

        self.net = nn.ModuleList()  # Use ModuleList for proper CUDA movement
        self.net.append(self.nonlin(in_features, hidden_features, is_first=True))
        for _ in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features))

        self.final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)

    def forward(self, coords):
        extracted_features = self.feature_extractor(self.img)

        gap = self.gap(extracted_features.view(extracted_features.size(0), extracted_features.size(1), -1))
        coef = self.prior(gap[..., 0]).view(4,2)
        t0, c0 = torch.unbind(coef, dim=1)

        # ------------------------------- Sigmoid Projection ---------------------------
        t0 = torch.sigmoid(t0) * (self.T_range[1] - self.T_range[0]) + self.T_range[0]
        c0 = torch.sigmoid(c0) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        # ------------------------------------------------------------------------------

        output = coords
        for id, lyr in enumerate(self.net):
            output = lyr(output, t0[id], c0[id])
        output = self.final_linear(output).real

        return nn.Sigmoid()(output)