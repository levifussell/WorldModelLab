from typing import Union, Callable
from grpc import Call

import torch
import torch.nn as nn

class WorldModel(nn.Module):

    def __init__(
            self,
            state_size: int,
            action_size: int,
            hid_layers: list,
            fn_pre_process_state: Callable,
            fn_post_process_state: Callable,
            fn_pre_process_action: Callable,
            final_layer_scale: float = 0.1,
            ) -> None:
        super(WorldModel, self).__init__()

        self.fn_pre_process_state = fn_pre_process_state
        self.fn_post_process_state = fn_post_process_state
        self.fn_pre_process_action = fn_pre_process_action

        layers = [state_size + action_size]
        layers.extend(hid_layers)

        self.model = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            self.model.append(nn.Linear(l1, l2))
            self.model.append(nn.ELU())

        self.model.append(nn.Linear(layers[-1], state_size))
        self.model[-1].weight.data.mul_(final_layer_scale)
        self.model[-1].bias.data.mul_(0.0)

        self.model = nn.Sequential(*self.model)

    def forward(
            self, 
            state_start : torch.tensor,
            actions : torch.tensor,
            ) -> torch.tensor:
        """
        Forward pass for the world model.

        :param state_start: batch of start states (N, F) N = num batches, F = features.
        :param actions: batch of actions (N, W, D) N = num batches, W = window, D = features.

        :return: batch of future predictions (N, W, D), shifted one timestep forward.
        """

        if len(actions.shape) == 2: #IGNORE: SPECIAL CASE FOR TORCHSUMMARY.
            actions = actions.unsqueeze(1)

        window = actions.shape[1]

        pre_state_start = self.fn_pre_process_state(state_start)
        pre_action = self.fn_pre_process_action(actions[:,0])

        x = torch.cat([pre_state_start, pre_action], dim=-1)

        pred_states = [self.fn_post_process_state(state_start + self.model(x))]

        for i in range(1, window):
            
            pre_state_pred = self.fn_pre_process_state(pred_states[-1])
            pre_action = self.fn_pre_process_action(actions[:, i])

            x = torch.cat([pre_state_pred, pre_action], dim=-1)

            pred_states.append(self.fn_post_process_state(pred_states[-1] + self.model(x)))

        pred_states = torch.cat([t.unsqueeze(1) for t in pred_states], dim=1)

        return pred_states