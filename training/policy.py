from typing import Callable, Union, Tuple

import torch
import torch.nn as nn

class Policy(nn.Module):

    def __init__(
            self,
            input_size : int,
            action_size : int,
            hid_layers : list,
            fn_combine_state_and_goal : Callable,
            fn_post_process_action : Callable,
            ) -> None:
        super(Policy, self).__init__()

        self.fn_combine_state_and_goal = fn_combine_state_and_goal
        self.fn_post_process_action = fn_post_process_action

        layers = [input_size]
        layers.extend(hid_layers)

        self.model = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            self.model.append(nn.Linear(l1, l2))
            self.model.append(nn.ELU())

        self.model.append(nn.Linear(layers[-1], action_size))

        self.model = nn.Sequential(*self.model)

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

        x = self.fn_combine_state_and_goal(state, goal)
        return self.model(x)

    def forward_world_model(
            self,
            state_start: torch.tensor, 
            goals: torch.tensor,
            world_model: Callable, 
            ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute the policy predictions through a world model for a certain rollout length (window).

        :param state_start: batch of start states (N, F) N = batch size, F = num features.
        :param goals: batch of goals (N, W, F) N = batch size, W = window, F = num features.
        :param world_model: world_model to step through.

        :return: tuple of batch of predictions (N, W, F) [predicted state tensor, actions tensor]
        """

        window = goals.shape[1]

        act = self(state_start, goals[:,0])

        state_next = world_model.forward(state_start, act.unsqueeze(1)).squeeze()

        pred_states = [state_next]
        pred_actions = [act]

        for i in range(1, window):

            state_curr = pred_states[-1]

            act = self(state_curr, goals[:, i])

            state_next = world_model.forward(state_curr, act.unsqueeze(1)).squeeze()

            pred_states.append(state_next)
            pred_actions.append(act)

        pred_states = torch.cat([t.unsqueeze(1) for t in pred_states], dim=1)
        pred_actions = torch.cat([t.unsqueeze(1) for t in pred_actions], dim=1)

        return pred_states, pred_actions