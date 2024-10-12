import torch
from torch import nn
from .base import DQNModuleBase, DQN
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state = 16, d_conv = 4, expand = 2):
        super(MambaBlock).__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_model = d_model
        self.model = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.d_model, # Model dimension d_model
            d_state=self.d_state,  # SSM state expansion factor
            d_conv=self.d_conv,    # Local convolution width
            expand=self.expand,    # Block expansion factor
        ).to("cuda")
    def forward(self, x):
        y = self.model(x)
        return y