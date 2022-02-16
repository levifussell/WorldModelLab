from typing import Union, Callable

import torch
import torch.nn as nn

class WorldModel(nn.Module):

    def __init__(
            self,
            state_size : int,
            hid_layers : list,
            fn_pre_process_state : Callable,
            fn_post_process_state : Callable,
            ) -> None:
        super(self, WorldModel).__init__()

        self.fn_pre_process_sate = fn_pre_process_state
        self.fn_post_process_sate = fn_post_process_state
        self.model = None #nn.Sequential()

    def forward(
            self, 
            state_start : torch.tensor,
            actions : torch.tensor,
            ) -> torch.tensor:
        """
        Forward pass for the world model.
        :param state_start: batch of start states (N, F) N = num batches, F = features.
        :param actions: batch of actions (N, W, D) N = num batches, W = window, D = features.
        """

        window = actions.shape[1]

        pre_state_start = self.fn_pre_process_batch(state_start)

        x = torch.cat([pre_state_start, actions[:,0]], dim=-1)

        pred_states = [self.fn_post_process_batch(self.model(x))]

        for i in range(1, window):
            
            pre_state_pred = self.fn_pre_process_batch(pred_states[-1])

            x = torch.cat([pre_state_pred, actions[:, i:i+1]], dim=-1)

            pred_states.append(self.fn_post_process_batch(self.model(x)))

        pred_states = torch.cat(pred_states, axis=1)

        return pred_states