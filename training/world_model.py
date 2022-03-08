from typing import Union, Callable
from grpc import Call

import torch
import torch.nn as nn

from utils.normalizer import Normalizer

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

        self.normalizer_state = Normalizer(state_size)
        self.normalizer_action = Normalizer(action_size)
        self.normalizer_state_delta = Normalizer(state_size)

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

    def _state_pre_process(self, state: torch.tensor) -> torch.tensor:
        return self.normalizer_state(self.fn_pre_process_state(state))

    def _state_integrate_and_post_process(self, state_from: torch.tensor, state_delta: torch.tensor) -> torch.tensor:
        """
        Integrates the world model predictions forward.

        :param state_from: the state the world model is predicting from.
        :param state_delta: the change in state the world model has predicted.
        :return: the predicted next state.
        """
        return self.fn_post_process_state(state_from + self.normalizer_state_delta.denormalize(state_delta))
        
    def _action_pre_process(self, action: torch.tensor) -> torch.tensor:
        return self.normalizer_action(self.fn_pre_process_action(action))

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

        pre_state_start = self._state_pre_process(state_start)
        pre_action = self._action_pre_process(actions[:,0])

        x = torch.cat([pre_state_start, pre_action], dim=-1)

        pred_resids = [self.model(x)]
        pred_states = [self._state_integrate_and_post_process(state_from=state_start, state_delta=pred_resids[-1])]

        for i in range(1, window):
            
            pre_state_pred = self._state_pre_process(pred_states[-1])
            pre_action = self._action_pre_process(actions[:, i])

            x = torch.cat([pre_state_pred, pre_action], dim=-1)

            pred_resids.append(self.model(x))
            pred_states.append(self._state_integrate_and_post_process(state_from=pred_states[-1], state_delta=pred_resids[-1]))

        pred_resids = torch.cat([t.unsqueeze(1) for t in pred_resids], dim=1)
        pred_states = torch.cat([t.unsqueeze(1) for t in pred_states], dim=1)

        assert pred_states.shape == pred_states.shape

        return pred_states, pred_resids