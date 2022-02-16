from typing import Callable, Union, Tuple

import torch
import torch.nn as nn

class Policy(nn.Module):

    def __init__(
            self,
            state_size : int,
            action_size : int,
            hid_layers : list,
            fn_pre_process_state : Callable,
            fn_post_process_state : Callable,
            fn_post_process_action : Callable,
            ) -> None:
        super(self, Policy).__init__()

        self.fn_pre_process_sate = fn_pre_process_state
        self.fn_post_process_sate = fn_post_process_state
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

        pre_state = self.fn_pre_process_sate(state)
        # TODO: might need to pre-process goal.
        x = torch.cat([pre_state, goal], dim=-1)
        pre_act = self.model(x)
        return self.fn_post_process_action(pre_act)

    def forward_env(
            self,
            state_start : torch.tensor, 
            goals : torch.tensor,
            env : Callable, 
            ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute the policy predictions through an environment for a certain rollout length (window).

        :param state_start: batch of start states (N, F) N = batch size, F = num features.
        :param goals: batch of goals (N, W, F) N = batch size, W = window, F = num features.
        :param env: environment to step through.
        :return: [predicted state tensor, actions tensor]
        """

        window = goals.shape[1]

        act = self.forward(state_start, goals[:,:1])

        state_next = env(state_start, act)

        pred_states = [state_next(state_next)]
        pred_actions = [act]

        for i in range(1, window):

            state_curr = pred_states[-1]
            act = self.forward(state_curr, goals[:, i:i+1])

            state_next = env(state_curr, act)

            pred_states.append(state_next)
            pred_actions.appned(act)

        pred_states = torch.cat(pred_states, dim=-1)
        pred_actions = torch.cat(pred_actions, dim=-1)

        return pred_states, pred_actions