from typing import Callable, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

def grad_layer_norm(parameters: Iterable[torch.Tensor]) -> torch.Tensor:

    all_grad_norms = []

    params = [p for p in parameters if p.grad is not None]

    for p in params:

        grad_norm = torch.norm(p.grad, p=2) + 1e-8

        p.grad.detach().div_(grad_norm)

        all_grad_norms.append(grad_norm)

    return torch.mean(torch.cat([t.view(1,-1) for t in all_grad_norms]))

def compute_per_rollout_wm_gradient(
        world_model: nn.Module, 
        world_model_opt: opt.Optimizer, 
        data: Tuple[torch.Tensor, torch.Tensor],
        loss_func: Callable
        ) -> torch.Tensor:
    """
    Computes the grad norm per rollout for the world model

    :return: list of gradients per rollout.
    """

    window = data[0].shape[1]

    window_grads = []

    for w in range(1, window):

        world_model_opt.zero_grad()

        # GET MODEL PREDICTIONS

        W_pred_state, W_pred_resids = world_model(
                state_start=data[0][:,0],
                actions=data[1][:,:w],
                )

        # COMPUTE LOSS

        wm_loss = loss_func(W_pred_state, data[0][:,1:(w+1)], W_pred_resids)

        # COMPUTE GRAD NORM

        wm_loss.backward()

        wm_grad_norm = nn.utils.clip_grad.clip_grad_norm_(world_model.parameters(), max_norm=1000.0)

        window_grads.append(wm_grad_norm)

    window_grads = torch.FloatTensor(window_grads)

    return window_grads

def compute_per_rollout_po_gradient(
        policy: nn.Module, 
        policy_opt: opt.Optimizer, 
        world_model: nn.Module,
        world_model_opt: opt.Optimizer,
        data: Tuple[torch.Tensor, torch.Tensor],
        loss_func: Callable
        ) -> torch.Tensor:
    """
    Computes the grad norm per rollout for the policy

    :return: list of gradients per rollout.
    """

    window = data[0].shape[1]

    window_grads = []

    for w in range(1, window):

        world_model_opt.zero_grad()
        policy_opt.zero_grad()

        # GET MODEL PREDICTIONS

        P_state, P_action = policy.forward_world_model(
                state_start=data[0][:,0], 
                goals=data[1][:,:w],
                world_model=world_model, 
                policy_noise=0.0,
                )

        # COMPUTE LOSS

        po_reward_loss = loss_func(P_state, data[1][:, 1:(w+1)], P_action)

        # COMPUTE GRAD NORM

        po_reward_loss.backward()

        po_grad_norm = nn.utils.clip_grad.clip_grad_norm_(policy.parameters(), max_norm=1000.0)

        window_grads.append(po_grad_norm)

    window_grads = torch.FloatTensor(window_grads)

    return window_grads