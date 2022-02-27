import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt

from typing import Callable

from .env_collector import EnvCollector
from .policy import Policy
from .world_model import WorldModel

DEFAULT_TRAIN_ARGS = {

        # general.

    'device'                    : 'cuda',
    'logdir'                    : 'runs/',
    'epochs'                    : 2000, #100,
    'max_buffer_size'           : 4096*32,

        # env.

    'env_steps_per_train'       : 8192,

        # world model.

    'wm_lr'                     : 1e-3,
    'wm_max_grad_norm'          : 10.0,
    'wm_max_grad_skip'          : 20.0,

    'wm_train_samples'          : 4096,
    'wm_minibatch'              : 1024, #128,

    'wm_hid_units'              : 1024,
    'wm_hid_layers'             : 3,
    'wm_window'                 : 4,

        # policy.

    'po_lr'                     : 1e-4,
    'po_max_grad_norm'          : 10.0,
    'po_max_grad_skip'          : 20.0,

    'po_wm_exploration'         : 0.01, #0.05,
    'po_env_exploration'        : 0.1,

    'po_train_samples'          : 4096,
    'po_minibatch'              : 1024, #256,

    'po_hid_units'              : 1024,
    'po_hid_layers'             : 3,
    'po_window'                 : 64,

}

class TrainArgs:

    def __init__(self, args_dict : dict) -> None:
        for k in args_dict.keys():
            self.__setattr__(k, args_dict[k])

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
        'wm_loss_avg'           : [],
        'wm_grad_norm_avg'      : [],

        'po_loss_avg'           : [],
        'po_grad_norm_avg'      : [],
    }

    """Train World Model."""

        # collect buffer samples.

    samples_iterator = env_collector.sample_buffer(
                            nsamples=train_args.wm_train_samples,
                            batchsize=train_args.wm_minibatch,
                            window_size=train_args.wm_window)

    for (B_state, B_goal, B_act) in samples_iterator:

        # B_* : (N, T, F) where N = batch size, T = time window, F = features

        world_model_opt.zero_grad()

            # get world model predictions.

        W_pred_state = world_model(
                state_start=B_state[:,0],
                actions=B_act[:,:-1],
                )

            # train the world model.
        
        def wm_loss(W_pred, W_targ):

            assert W_pred.shape == W_targ.shape

            return torch.mean(torch.sum(torch.abs(W_pred - W_targ), dim=-1))

        wm_loss = wm_loss(W_pred_state, B_state[:,1:])

        wm_loss.backward()

        grad_skip = False

        if train_args.wm_max_grad_norm > 0:

            wm_grad = nn.utils.clip_grad_norm_(
                world_model.parameters(),
                max_norm=train_args.wm_max_grad_norm
            )

            if wm_grad > train_args.wm_max_grad_skip:
                grad_skip = True

            stats['wm_grad_norm_avg'].append(wm_grad.cpu().item())

        if not grad_skip:
            world_model_opt.step()
        else:
            print("WORLD MODEL GRADIENT SKIPPED.")

            # compile stats.

        stats['wm_loss_avg'].append(wm_loss.cpu().item())


    """Train Policy."""

        # collect buffer samples.

    samples_iterator = env_collector.sample_buffer(
                            nsamples=train_args.po_train_samples,
                            batchsize=train_args.po_minibatch,
                            window_size=train_args.po_window)

    for (B_state, B_goal, B_act) in samples_iterator:

        # B_* : (N, T, F) where N = batch size, T = time window, F = features

        world_model_opt.zero_grad()
        policy_opt.zero_grad()

        P_state, P_action = policy.forward_world_model(
                state_start=B_state[:,0], 
                goals=B_goal,
                world_model=world_model, 
                policy_noise=train_args.po_wm_exploration,
                )

            # train the policy.

        rew_batch_size = np.prod(P_state.shape[:2])

        po_rewards = reward_func(
                            state=P_state.view(rew_batch_size, -1), 
                            goal=B_goal.view(rew_batch_size, -1),
                            act=P_action.view(rew_batch_size, -1),
                            )

        po_loss = -1.0 * torch.mean(po_rewards)
        # po_loss = -1.0 * torch.mean(torch.sum(po_rewards.reshape(P_state.shape[0], P_state.shape[1]), dim=-1))

        po_loss.backward()

        grad_skip = False

        if train_args.po_max_grad_norm > 0:

            po_grad = nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=train_args.po_max_grad_norm
            )

            if po_grad > train_args.po_max_grad_skip:
                grad_skip = True

            stats['po_grad_norm_avg'].append(po_grad.cpu().item())

        if not grad_skip:
            policy_opt.step()
        else:
            print("POLICY GRADIENT SKIPPED.")

            # compile stats.

        stats['po_loss_avg'].append(po_loss.cpu().item())

    """ Post Tran"""

    policy_opt_sched.step()
    world_model_opt_sched.step()

    stats['po_lr'] = policy_opt_sched.get_last_lr()[0]
    stats['wm_lr'] = world_model_opt_sched.get_last_lr()[0]

    """Compute Stats"""

    for k in stats.keys():

        if isinstance(stats[k], list) and 'avg' in k:

            stats[k] = np.mean(stats[k])

    return stats

