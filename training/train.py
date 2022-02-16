import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt

from typing import Callable

from env_collector import EnvCollector
from policy import Policy
from world_model import WorldModel

DEFAULT_TRAIN_ARGS = {

        # general.

    'epochs'                    : 100000,
    'max_buffer_size'           : 512 * 1024,

        # env.

    'env_steps_per_train'       : 4096,

        # world model.

    'wm_lr'                     : 1e-3,
    'wm_max_grad_norm'          : 10.0,

    'wm_train_samples'          : 4096,
    'wm_minibatch'              : 512,

    'wm_hid_units'              : 1024,
    'wm_hid_layers'             : 3,
    'wm_window'                 : 8,

        # policy.

    'po_lr'                     : 1e-4,
    'po_max_grad_norm'          : 10.0,

    'po_train_samples'          : 4096,
    'po_minibatch'              : 512,

    'po_hid_units'              : 1024,
    'po_hid_layers'             : 3,
    'po_window'                 : 32,

}

class TrainArgs:

    def __init__(self, args_dict : dict) -> None:
        for k in args_dict.keys():
            self.__setattr__(k, args_dict[k])

def train_step(

    env_collector               : EnvCollector,

    policy                      : Policy,
    policy_opt                  : opt.Optimizer,
    world_model                 : WorldModel,
    world_model_opt             : opt.Optimizer,

    reward_func                 : Callable,

    train_args                  : TrainArgs,

) -> None:
    """
    Performs training step of the policy and world model.

    :param fn_pre_process_state:    convert state from global to local.
    :param fn_post_process_state:   convert state from local to global.
    :param fn_post_process_action:  convert action output of policy.

    :param env_collector:   interface for sampling buffer.

    :param policy:  policy model.
    :param policy_opt:  policy model optimizer.
    :param world_model: world model.
    :param world_model_opt: world model optimizer.

    :param reward_func: function that computes the reward of the state.
    
    :param train_args:  object for storing the training arguments.
    """

    stats = {
        'wm_loss_avg'           : [],
        'wm_grad_norm_avg'      : [],

        'po_loss_avg'           : [],
        'po_grad_norm_avg'      : [],
    }

    """Env Collector Pre-stuff"""

    if not env_collector.is_parallel:
        env_collector.collect(train_args.env_steps_per_train)

    """Train World Model."""

        # collect buffer samples.

    samples_iterator = env_collector.sample_buffer(
                            nsamples=train_args.wm_train_samples,
                            batchsize=train_args.wm_minibatch,
                            window_size=train_args.wm_window)

    for (B_state, B_act, B_goal) in samples_iterator:

        # B_* : (N, T, F) where N = batch size, T = time window, F = features

        world_model_opt.zero_grad()

            # get world model predictions.

        W_pred_state = world_model(
                state_start=B_state[:,:1],
                actions=B_act,
                )

            # train the world model.
        
        def wm_loss(W_pred, W_targ):

            assert W_pred.shape == W_targ.shape

            pass # TODO.

        wm_loss = wm_loss(W_pred_state, B_state)

        wm_loss.backward()

        if train_args.wm_max_grad_norm > 0:

            wm_grad = nn.utils.clip_grad_norm_(
                world_model.parameters(),
                max_norm=train_args.wm_max_grad_norm
            )

            stats['wm_grad_norm'].append(wm_grad)

        world_model_opt.step()

            # compile stats.

        stats['wm_loss_avg'].append(po_loss.item())


    """Train Policy."""

        # collect buffer samples.

    samples_iterator = env_collector.sample_buffer(
                            nsamples=train_args.po_train_samples,
                            batchsize=train_args.po_minibatch,
                            window_size=train_args.po_window)

    for (B_state, B_act, B_goal) in samples_iterator:

        # B_* : (N, T, F) where N = batch size, T = time window, F = features

        world_model_opt.zero_grad()
        policy_opt.zero_grad()

        P_state, P_action = policy.forward_env(
                state_start=B_state[:,:1], 
                goals=B_goal,
                env=world_model, 
        )

            # train the policy.

        po_loss = -1.0 * reward_func(
                            state=P_state, 
                            goal=B_goal,
                            act=P_action,
                            )

        po_loss.backward()

        if train_args.po_max_grad_norm > 0:

            po_grad = nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=train_args.po_max_grad_norm
            )

            stats['po_grad_norm'].append(po_grad)

        policy_opt.step()

            # compile stats.

        stats['po_loss_avg'].append(wm_loss.item())


    """Copy Policy"""

        # move the most recently trained policy to the gym.

    env_collector.copy_current_policy(policy)


    """Compute Stats"""

    for k in stats.keys():

        if isinstance(stats[k], list) and 'avg' in k:

            stats[k] = np.mean(stats[k])

    return stats

