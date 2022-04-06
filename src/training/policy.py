from typing import Callable, Union, Tuple

import torch
import torch.nn as nn
from .world_model import WorldModel

# from ..utils.normalizer import Normalizer
from ..utils.normalizer_batch import NormalizerBatch

class Policy(nn.Module):

    def __init__(
            self,
            input_size: int,
            action_size: int,
            hid_layers: list,
            fn_combine_state_and_goal: Callable,
            fn_post_process_action: Callable,
            final_layer_scale: float = 0.1,
            ) -> None:
        super(Policy, self).__init__()

        self.action_size = action_size

        self.normalizer_state_and_goal = NormalizerBatch(input_size)

        self.fn_combine_state_and_goal = fn_combine_state_and_goal
        self.fn_post_process_action = fn_post_process_action

        layers = [input_size]
        layers.extend(hid_layers)

        self.model = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            self.model.append(nn.Linear(l1, l2))
            self.model.append(nn.ELU())

        self.model.append(nn.Linear(layers[-1], action_size))
        self.model[-1].weight.data.mul_(final_layer_scale)
        self.model[-1].bias.data.mul_(0.0)

        self.model = nn.Sequential(*self.model)

        self.device = 'cpu'

    def _state_and_goal_preprocess(
            self,
            state: torch.tensor,
            goal: torch.tensor,
            ) -> torch.tensor:
        return self.normalizer_state_and_goal(self.fn_combine_state_and_goal(state, goal))

    def forward(
            self, 
            state: torch.tensor,
            goal: torch.tensor,
            ) -> torch.tensor:
        """
        Compute the policy actions given the state and goal.

        :param state: state batch.
        :param state: goal batch.
        :return: action batch.
        """

        x = self._state_and_goal_preprocess(state, goal)
        return self.model(x)

    def forward_world_model(
            self,
            state_start: torch.tensor, 
            goals: torch.tensor,
            world_model: Callable, 
            policy_noise: float,
            ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute the policy predictions through a world model for a certain rollout length (window).

        :param state_start: batch of start states (N, F) N = batch size, F = num features.
        :param goals: batch of goals (N, W, F) N = batch size, W = window, F = num features.
        :param world_model: world_model to step through.
        :param policy_noise: std of guassian noise to add to the policy actions.

        :return: tuple of batch of predictions (N, W, F) [predicted state tensor, actions tensor]
        """

        window = goals.shape[1]

        act = self(state_start, goals[:,0])
        act += torch.randn(act.shape).to(act.device) * policy_noise

        state_next, _ = world_model.forward(state_start, act.unsqueeze(1))
        state_next = state_next.squeeze()

        pred_states = [state_next]
        pred_actions = [act]

        for i in range(1, window):

            state_curr = pred_states[-1]

            act = self(state_curr, goals[:, i])
            act += torch.randn(act.shape).to(act.device) * policy_noise

            state_next, _ = world_model.forward(state_curr, act.unsqueeze(1))
            state_next = state_next.squeeze()

            pred_states.append(state_next)
            pred_actions.append(act)

        pred_states = torch.cat([t.unsqueeze(1) for t in pred_states], dim=1)
        pred_actions = torch.cat([t.unsqueeze(1) for t in pred_actions], dim=1)

        return pred_states, pred_actions

    def save_to_path(self, filepath: str) -> None:

        torch.save(self.state_dict(), f=filepath)

    def load_from_path(self, filepath: str) -> None:

        state_dict = torch.load(filepath, map_location='cpu')
        self.load_state_dict(state_dict)

    def to(self, device, *args, **kwargs):

        self.device = device

        return super().to(device, *args, **kwargs)

