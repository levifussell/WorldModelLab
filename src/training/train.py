import os
import sys

curdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(curdir))

import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt

from typing import Callable, Tuple

from .env_collector import EnvCollector
from .policy import Policy
from .world_model import WorldModel

from ..utils.utils import grad_layer_norm, compute_per_rollout_wm_gradient, compute_per_rollout_po_gradient

DEFAULT_TRAIN_ARGS = {

        # general.

    'seed'                      : 1234,
    'device'                    : 'cuda',
    'logdir'                    : 'runs/',
    'epochs'                    : 5000,
    'max_buffer_size'           : 4096*32,

        # env.

    'env_steps_per_train'       : 8192,
    'env_max_steps'             : 512,

        # world model.

    'wm_lr'                     : 1e-3,
    'wm_max_grad_norm'          : 1.0,
    'wm_max_grad_skip'          : 20.0,

    'wm_train_samples'          : 4096,
    'wm_minibatch'              : 1024,

    'wm_hid_units'              : 1024,
    'wm_hid_layers'             : 3,
    'wm_window'                 : 4,

    'wm_l1_reg'                 : 0.01,
    'wm_l2_reg'                 : 0.001,

        # policy.

    'po_lr'                     : 1e-3,
    'po_max_grad_norm'          : 1.0,
    'po_max_grad_skip'          : 20.0,

    'po_wm_exploration'         : 0.01,
    'po_env_exploration'        : 0.1,

    'po_train_samples'          : 4096,
    'po_minibatch'              : 1024,

    'po_hid_units'              : 1024,
    'po_hid_layers'             : 3,
    'po_window'                 : 32,

    'po_l1_reg'                 : 0.01,
    'po_l2_reg'                 : 0.001,

}

class TrainArgs:

    def __init__(self, args_dict : dict) -> None:
        for k in args_dict.keys():
            self.__setattr__(k, args_dict[k])
    def __str__(self) -> str:
        text = ""
        for k in self.__dict__.keys():
            text += f"{k}={self.__dict__[k]}, "        
        return text

def train_step(

    env_collector               : EnvCollector,

    policy                      : Policy,
    policy_opt                  : opt.Optimizer,
    policy_opt_sched            : opt.lr_scheduler._LRScheduler,

    world_model                 : WorldModel,
    world_model_opt             : opt.Optimizer,
    world_model_opt_sched       : opt.lr_scheduler._LRScheduler,

    reward_func                 : Callable,

    train_args                  : TrainArgs,

    no_train                    : bool = False,

) -> None:
    """
    Performs training step of the policy and world model.

    :param env_collector:   interface for sampling buffer.

    :param policy:  policy model.
    :param policy_opt:  policy model optimizer.
    :param policy_opt_sched:  policy learning rate scheduler.

    :param world_model: world model.
    :param world_model_opt: world model optimizer.
    :param world_model_opt_sched: world model learning rate scheduler.

    :param reward_func: function that computes the reward of the state.
    
    :param train_args:  object for storing the training arguments.
    """

    stats = {
        'wm_loss_avg'                       : [],
        'wm_loss_diff_avg'                  : [],
        'wm_loss_l1_reg_avg'                : [],
        'wm_loss_l2_reg_avg'                : [],
        'wm_grad_norm_avg'                  : [],
        'wm_pred_resid_avg'                 : [],

        'po_loss_avg'                       : [],
        'po_loss_reward_avg'                : [],
        'po_loss_l1_reg_avg'                : [],
        'po_loss_l2_reg_avg'                : [],
        'po_grad_norm_avg'                  : [],
        'po_grnd_loss_reward_avg'           : [],
        'po_grnd_wm_loss_reward_diff_avg'   : [],
    }

    if train_args.deep_stats:

        for w in range(train_args.wm_window - 1):
            stats[f'wm_grad_rollout_{w}_avg'] = []

        for w in range(train_args.po_window - 1):
            stats[f'po_grad_rollout_{w}_avg'] = []

    """Train World Model."""

    world_model.train()

    def wm_loss_func(W_pred, W_targ, W_resids):

        assert W_pred.shape == W_targ.shape

        diff_loss = torch.mean(torch.sum(torch.abs(W_pred - W_targ), dim=-1))
        l1_reg_loss = train_args.wm_l1_reg * torch.mean(torch.sum(torch.abs(W_resids), dim=-1))
        l2_reg_loss = train_args.wm_l2_reg * torch.mean(torch.sum(torch.square(W_resids), dim=-1))

        loss_total = diff_loss + l1_reg_loss + l2_reg_loss

        return loss_total, (diff_loss, l1_reg_loss, l2_reg_loss)

        # collect buffer samples.

    samples_iterator = env_collector.sample_buffer(
                            nsamples=train_args.wm_train_samples,
                            batchsize=train_args.wm_minibatch,
                            window_size=train_args.wm_window)

    for (B_state, B_goal, B_act) in samples_iterator:

        # B_* : (N, T, F) where N = batch size, T = time window, F = features

            # track deep stats.

        if train_args.deep_stats:

            wm_per_rollout_grads = compute_per_rollout_wm_gradient(
                world_model=world_model,
                world_model_opt=world_model_opt,
                data=(B_state, B_act),
                loss_func=lambda x,y,z : wm_loss_func(x,y,z)[0]
            )

            assert train_args.wm_window -1 == wm_per_rollout_grads.shape[0]

            for w in range(train_args.wm_window - 1):
                stats[f'wm_grad_rollout_{w}_avg'].append(wm_per_rollout_grads[w])

        world_model_opt.zero_grad()

            # get world model predictions.

        W_pred_state, W_pred_resids = world_model(
                state_start=B_state[:,0],
                actions=B_act[:,:-1],
                )

            # train the world model.
        
        wm_loss, (wm_diff_loss, wm_l1_reg_loss, wm_l2_reg_loss) = wm_loss_func(W_pred_state, B_state[:,1:], W_pred_resids)

        wm_loss.backward()

        grad_skip = False

        if train_args.wm_max_grad_norm > 0:

            wm_grad = nn.utils.clip_grad_norm_(
                world_model.parameters(),
                max_norm=train_args.wm_max_grad_norm
            )
            # wm_grad = grad_layer_norm(world_model.parameters())

            if wm_grad > train_args.wm_max_grad_skip:
                grad_skip = True

            stats['wm_grad_norm_avg'].append(wm_grad.cpu().item())

        if no_train:
            print("WORLD MODEL TRAIN DISABLED")
        else:
            if not grad_skip:
                world_model_opt.step()
            else:
                print("!! WORLD MODEL GRADIENT SKIPPED.")

            # compile stats.

        stats['wm_loss_avg'].append(wm_loss.cpu().item())
        stats['wm_loss_diff_avg'].append(wm_diff_loss.cpu().item())
        stats['wm_loss_l1_reg_avg'].append(wm_l1_reg_loss.cpu().item())
        stats['wm_loss_l2_reg_avg'].append(wm_l2_reg_loss.cpu().item())
        stats['wm_pred_resid_avg'].append(torch.mean(torch.norm(W_pred_resids, p=2, dim=-1)).cpu().item())


    """Train Policy."""

    world_model.eval()
    policy.train()

    def po_loss_func(P_states, P_goals, P_actions):

        rew_batch_size = np.prod(P_states.shape[:2])
        window = P_states.shape[1]

        po_reward_loss = -1.0 * torch.mean(torch.sum(reward_func(
                            state=P_states.reshape(rew_batch_size, -1), 
                            goal=P_goals.reshape(rew_batch_size, -1),
                            act=P_actions.reshape(rew_batch_size, -1),
                            ).reshape(-1, window), dim=-1))

        po_l1_reg_loss = train_args.po_l1_reg * torch.mean(torch.sum(torch.abs(P_actions), dim=-1))
        po_l2_reg_loss = train_args.po_l2_reg * torch.mean(torch.sum(torch.square(P_actions), dim=-1))

        po_loss = po_reward_loss + po_l1_reg_loss + po_l2_reg_loss

        return po_loss, (po_reward_loss, po_l1_reg_loss, po_l2_reg_loss)

        # collect buffer samples.

    samples_iterator = env_collector.sample_buffer(
                            nsamples=train_args.po_train_samples,
                            batchsize=train_args.po_minibatch,
                            window_size=train_args.po_window)

    for (B_state, B_goal, B_act) in samples_iterator:

        # B_* : (N, T, F) where N = batch size, T = time window, F = features

            # track deep stats.

        if train_args.deep_stats:

            po_per_rollout_grads = compute_per_rollout_po_gradient(
                policy=policy,
                policy_opt=policy_opt,
                world_model=world_model,
                world_model_opt=world_model_opt,
                data=(B_state, B_goal, B_act),
                loss_func=lambda x,y,z : po_loss_func(x,y,z)[0]
            )

            assert train_args.po_window -1 == po_per_rollout_grads.shape[0]

            for w in range(train_args.po_window - 1):
                stats[f'po_grad_rollout_{w}_avg'].append(po_per_rollout_grads[w])

        world_model_opt.zero_grad()
        policy_opt.zero_grad()

        P_state, P_action = policy.forward_world_model(
                state_start=B_state[:,0], 
                goals=B_goal[:,:-1],
                world_model=world_model, 
                policy_noise=train_args.po_wm_exploration,
                )

            # train the policy.

        # rew_batch_size = np.prod(P_state.shape[:2])
        # window = P_state.shape[1]

        # # po_reward_loss = -1.0 * torch.mean(reward_func(
        # #                     state=P_state.view(rew_batch_size, -1), 
        # #                     goal=B_goal[:,1:].reshape(rew_batch_size, -1),
        # #                     act=P_action.view(rew_batch_size, -1),
        # #                     ))

        # po_reward_loss = -1.0 * torch.mean(torch.sum(reward_func(
        #                     state=P_state.view(rew_batch_size, -1), 
        #                     goal=B_goal[:,1:].reshape(rew_batch_size, -1),
        #                     act=P_action.view(rew_batch_size, -1),
        #                     ).reshape(-1, window), dim=-1))

        # po_l1_reg_loss = train_args.po_l1_reg * torch.mean(torch.sum(torch.abs(P_action), dim=-1))
        # po_l2_reg_loss = train_args.po_l2_reg * torch.mean(torch.sum(torch.square(P_action), dim=-1))

        # po_loss = po_reward_loss + po_l1_reg_loss + po_l2_reg_loss

        po_loss, (po_reward_loss, po_l1_reg_loss, po_l2_reg_loss) = po_loss_func(P_state, B_goal[:, 1:], P_action)

        po_loss.backward()

        grad_skip = False

        if train_args.po_max_grad_norm > 0:

            po_grad = nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=train_args.po_max_grad_norm
            )
            # po_grad = grad_layer_norm(policy.parameters())

            if po_grad > train_args.po_max_grad_skip:
                grad_skip = True

            stats['po_grad_norm_avg'].append(po_grad.cpu().item())

        if no_train:
            print("POLICY TRAIN DISABLED")
        else:
            if not grad_skip:
                policy_opt.step()
            else:
                print("!! POLICY GRADIENT SKIPPED.")

            # ground-truth policy.

        # po_grnd_reward_loss = -1.0 * torch.mean(reward_func(
        #                     state=B_state[:,1:].reshape(rew_batch_size, -1), 
        #                     goal=B_goal[:,1:].reshape(rew_batch_size, -1),
        #                     act=B_act[:,:-1].reshape(rew_batch_size, -1),
        #                     ))

        # po_grnd_reward_loss = -1.0 * torch.mean(torch.sum(reward_func(
        #                     state=B_state[:,1:].reshape(rew_batch_size, -1), 
        #                     goal=B_goal[:,1:].reshape(rew_batch_size, -1),
        #                     act=B_act[:,:-1].reshape(rew_batch_size, -1),
        #                     ).reshape(-1, window), dim=-1))

        po_grnd_loss, (po_grnd_reward_loss, po_grnd_l1_loss, po_grnd_l2_loss) = po_loss_func(B_state[:,1:], B_goal[:,1:], B_act[:,:-1])

            # compile stats.

        stats['po_loss_avg'].append(po_loss.cpu().item())
        stats['po_loss_reward_avg'].append(po_reward_loss.cpu().item())
        stats['po_loss_l1_reg_avg'].append(po_l1_reg_loss.cpu().item())
        stats['po_loss_l2_reg_avg'].append(po_l2_reg_loss.cpu().item())

        stats['po_grnd_loss_reward_avg'].append(po_grnd_reward_loss.cpu().item())
        stats['po_grnd_wm_loss_reward_diff_avg'].append(stats['po_grnd_loss_reward_avg'][-1] - stats['po_loss_reward_avg'][-1])

    """ Post Tran"""

    policy_opt_sched.step()
    world_model_opt_sched.step()

    stats['po_lr'] = policy_opt_sched.get_last_lr()[0]
    stats['wm_lr'] = world_model_opt_sched.get_last_lr()[0]

    def get_model_weight_info(model: nn.Module) -> Tuple[float, float]:

        weight_mag = 0
        weight_max = float('-inf')
        weight_min = float('+inf')
        bias_mag = 0
        bias_max = float('-inf')
        bias_min = float('+inf')
        layer_count = 0

        for n,p in model.named_parameters():
            pr = p.data.cpu()
            if 'weight' in n:
                weight_mag += torch.mean(pr).item()
                weight_max = max(weight_max, torch.max(pr).item())
                weight_min = min(weight_min, torch.min(pr).item())
                layer_count += 1 
            elif 'bias' in n:
                bias_mag += torch.mean(p.data.cpu()).item()
                bias_max = max(bias_max, torch.max(pr).item())
                bias_min = min(bias_min, torch.min(pr).item())

        avg_weight_scale =  weight_mag / layer_count
        avg_bias_scale =  bias_mag / layer_count

        return (avg_weight_scale, weight_max, weight_min, avg_bias_scale, bias_max, bias_min)

    ( stats['po_weight_scale'], stats['po_weight_max'], stats['po_weight_min'], 
        stats['po_bias_scale'], stats['po_bias_max'], stats['po_bias_min']
        ) = get_model_weight_info(policy)

    ( stats['wm_weight_scale'], stats['wm_weight_max'], stats['wm_weight_min'], 
        stats['wm_bias_scale'], stats['wm_bias_max'], stats['wm_bias_min']
        ) = get_model_weight_info(world_model)

    """Compute Stats"""

    for k in stats.keys():

        if isinstance(stats[k], list) and 'avg' in k:

            stats[k] = np.mean(stats[k])

    return stats
