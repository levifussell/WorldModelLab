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

        # world model.

    'wm_lr'                     : 1e-3,
    'wm_max_grad_norm'          : 10.0,

    'wm_hid_units'              : 1024,
    'wm_hid_layers'             : 3,
    'wm_window'                 : 8,

        # policy.

    'po_lr'                     : 1e-4,
    'po_max_grad_norm'          : 10.0,

    'po_hid_units'              : 1024,
    'po_hid_layers'             : 3,
    'po_window'                 : 32,

}

class TrainArgs:

    def __init__(self, args_dict : dict) -> None:
        for k in args_dict.keys():
            self.__setattr__(k, args_dict[k])

def train_step(

    fn_pre_process_state        : Callable,
    fn_post_process_state       : Callable,
    fn_post_process_action      : Callable,

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

    """COLLECT SAMPLES FROM BUFFER."""

    samples = env_collector.sample_buffer()

    batches = None # somehow get these from the samples.

    stats = {
        'wm_loss_avg' : [],
        'po_loss_avg' : [],
    }

    """Train World Model."""

    for Bg in batches:

        # Bg : (N, T, F) where N = batch size, T = time window, F = features

        world_model_opt.zero_grad()
        policy_opt.zero_grad()

            # get world model predictions.
        Wg = world_model(
                Bg,
                fn_pre_process_state,
                fn_post_process_state)


            # train the world model.
        
        def wm_loss(W_pred, W_targ):

            assert W_pred.shape == W_targ.shape

            pass # TODO.

        wm_loss = wm_loss(Wg, Bg[:,:-1])

        wm_loss.backward()

        if train_args.wm_max_grad_norm > 0:

            nn.utils.clip_grad_norm_(
                world_model.parameters(),
                max_norm=train_args.wm_max_grad_norm
            )

        world_model_opt.step()


    """Train Policy."""

    for Bg in batches:

        # Bg : (N, T, F) where N = batch size, T = time window, F = features

        Pg = policy(
                Bg, 
                world_model, 
                fn_pre_process_state,
                fn_post_process_state,
                fn_post_process_action,
        )

            # train the policy.

        po_loss = -1.0 * reward_func(Pg)

        po_loss.backward()

        if train_args.po_max_grad_norm > 0:

            nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=train_args.po_max_grad_norm
            )

        policy_opt.step()

        """Compile Stats."""

        stats['wm_loss_avg'].append(po_loss.item())
        stats['po_loss_avg'].append(wm_loss.item())

    """Compute Stats"""

    for k in stats.keys():

        if isinstance(stats[k], list) and 'avg' in k:

            stats[k] = np.mean(stats[k])

    return stats

