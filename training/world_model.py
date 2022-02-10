from typing import Union

import torch
import torch.nn as nn

class WorldModel(nn.Module):

    def __init__(
            self,
            state_size : int,
            hid_layers : list = [1024]*3,
            ) -> None:
        super(self, WorldModel).__init__()

    def forward(self, x):
        pass