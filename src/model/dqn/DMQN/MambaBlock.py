import torch
from torch import nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0, batch_first=True, d_conv = 4, expand = 2):
        super(MambaBlock).__init__()
        self.d_state = 4 * input_size
        self.d_conv = d_conv
        self.expand = expand
        self.d_model = input_size
        self.model = nn.Sequential(*[Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.d_model, # Model dimension d_model
            d_state=self.d_state,  # SSM state expansion factor
            d_conv=self.d_conv,    # Local convolution width
            expand=self.expand,    # Block expansion factor
        ).to("cuda") for _ in range(num_layers)], nn.Linear(input_size, hidden_size))
    def forward(self, x):
        y = self.model(x)
        return y