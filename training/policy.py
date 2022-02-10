from typing import Union

import torch
import torch.nn as nn

class Policy(nn.Module):

    def __init__(
            self,
            state_size : int,
            action_size : int,
            hid_layers : list = [1024]*3,
            ) -> None:
        super(self, Policy).__init__()

    def forward(self, x):
        pass