from typing import Callable, Union, Tuple

import torch
import torch.nn as nn

class Policy(nn.Module):

    def __init__(
            self,
            state_size : int,
            action_size : int,
            hid_layers : list,
            fn_combine_state_and_goal : Callable,
            fn_post_process_action : Callable,
            ) -> None:
        super(self, Policy).__init__()

        self.fn_combine_state_and_goal = fn_combine_state_and_goal
        self.fn_post_process_action = fn_post_process_action

        self.model = None # TODO.

    def forward(
            self, 
            state : torch.tensor,
            goal : torch.tensor,
            ) -> torch.tensor:
        """
        Compute the policy actions given the state.

        :param state: state batch.
        :param state: goal batch.
        :return: action batch.
        """

        x = self.fn_combine_state_and_goal(state, goal)
        pre_act = self.model(x)
        return self.fn_post_process_action(pre_act)

    def forward_world_model(
            self,
            state_start : torch.tensor, 
            goals : torch.tensor,
            world_model : Callable, 
            ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute the policy predictions through a world model for a certain rollout length (window).

        :param state_start: batch of start states (N, F) N = batch size, F = num features.
        :param goals: batch of goals (N, W, F) N = batch size, W = window, F = num features.
        :param world_model: world_model to step through.
        :return: [predicted state tensor, actions tensor]
        """

        window = goals.shape[1]

        act = self.forward(state_start, goals[:,:1])

        state_next = world_model(state_start, act)

        pred_states = [state_next(state_next)]
        pred_actions = [act]

        for i in range(1, window):

            state_curr = pred_states[-1]
            act = self.forward(state_curr, goals[:, i:i+1])

            state_next = world_model(state_curr, act)

            pred_states.append(state_next)
            pred_actions.appned(act)

        pred_states = torch.cat(pred_states, dim=-1)
        pred_actions = torch.cat(pred_actions, dim=-1)

        return pred_states, pred_actions