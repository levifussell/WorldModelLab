from fileinput import filename
import os
import platform
from datetime import datetime
from typing import Callable

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from PIL import Image
from src.env.goal_env import GoalEnvDataProcessor

from src.training.policy import Policy
from src.training.world_model import WorldModel
from src.training.env_collector import EnvCollector
from src.training.buffer import Buffer
from src.training.train import train_step, TrainArgs, DEFAULT_TRAIN_ARGS

from src.env.reacher_dm_control_env import ReacherGoalEnv, ReacherGoalEnvDataProcessor
from src.env.reacher_train_args import REACHER_TRAIN_ARGS
from src.env.cartpole_balance_dm_control_env import CartpoleBalanceGoalEnv
from src.env.cartpole_balance_train_args import CARTPOLE_BALANCE_TRAIN_ARGS

def run(
    f_env: Callable = None, 
    f_env_data_processor: Callable = None, 
    train_args: dict = DEFAULT_TRAIN_ARGS,
):

    train_args = TrainArgs(train_args)
    print("ARGS: \n" +str(train_args))

    if train_args.deep_stats:
        print("!! WARNING: TRACKING DEEP STATS. This will slow down training.")

    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    """ Build Env Data Processor """

    env_data_processor: GoalEnvDataProcessor = f_env_data_processor()

    """ Get Environment/Gym Data """

    ff_env = lambda : f_env(data_processor=env_data_processor, render=train_args.save_renders)

    temp_env = ff_env()

    state_size = temp_env.state_size
    goal_size = temp_env.goal_size
    act_size = temp_env.action_size
    po_input_size = temp_env.policy_input_size

    del temp_env

    """ Logging """

    date_str = datetime.now().strftime('%b%d_%H-%M-%S')
    experiment_hash = date_str + '_' + platform.uname().node
    log_path = os.path.join(train_args.logdir, experiment_hash)

    writer = SummaryWriter(log_path)
    writer.add_text(tag="args", text_string=str(train_args))

    """ Build Policy """

    policy_kwargs = dict(
                input_size=po_input_size,
                action_size=act_size,
                hid_layers=[train_args.po_hid_units] * train_args.po_hid_layers,
                )

    policy = Policy(
                input_size=po_input_size,
                action_size=act_size,
                hid_layers=[train_args.po_hid_units] * train_args.po_hid_layers,
                # fn_combine_state_and_goal=env.preprocess_state_and_goal_for_policy,
                # fn_combine_state_and_goal=None,
                fn_combine_state_and_goal=env_data_processor.preprocess_state_and_goal_for_policy,
                fn_post_process_action=lambda x : x,
                ).to(train_args.device)

    policy_opt = opt.RAdam(
                    params=policy.parameters(), 
                    lr=train_args.po_lr)
                
    policy_opt_sched = opt.lr_scheduler.LambdaLR(
                    optimizer=policy_opt,
                    lr_lambda=lambda epoch: 1 - epoch / train_args.epochs)

    # summary(policy, input_size=[(state_size,), (goal_size,)])

    """ Build World Model"""

    world_model = WorldModel(
                state_input_size=state_size,
                action_size=act_size,
                hid_layers=[train_args.wm_hid_units] * train_args.wm_hid_layers,
                # fn_pre_process_state=env.preprocess_state_for_world_model,
                # fn_post_process_state=env.postprocess_state_for_world_model,
                # fn_pre_process_state=None,
                # fn_post_process_state=None,
                fn_pre_process_state=env_data_processor.preprocess_state_for_world_model,
                fn_post_process_state=env_data_processor.postprocess_state_for_world_model,
                fn_pre_process_action=lambda x: x,
                activation=train_args.wm_activation,
                use_spectral_normalization=train_args.wm_use_spectral_norm,
                ).to(train_args.device)

    world_model_opt = opt.RAdam(
                    params=world_model.parameters(), 
                    lr=train_args.wm_lr)

    world_model_opt_sched = opt.lr_scheduler.LambdaLR(
                    optimizer=world_model_opt,
                    lr_lambda=lambda epoch: 1 - epoch / train_args.epochs)

    # summary(world_model, input_size=[(state_size,), (act_size,)])

    """ Build EnvCollector """

    buffer = Buffer(
                state_size=state_size,
                goal_size=goal_size,
                act_size=act_size,
                max_len=train_args.max_buffer_size,
                )

    env_collector = EnvCollector(
                        f_env_builder=ff_env,
                        env_data_processor=env_data_processor,
                        policy=policy,
                        policy_kwargs=policy_kwargs,
                        buffer=buffer,
                        min_env_steps=max(train_args.wm_window, train_args.po_window),
                        max_env_steps=train_args.env_max_steps,
                        exploration_std=train_args.po_env_exploration,
                        normalizer_wm_state=world_model.normalizer_state,
                        normalizer_wm_state_delta=world_model.normalizer_state_delta,
                        normalizer_wm_action=world_model.normalizer_action,
                        normalizer_po_state_and_goal=policy.normalizer_state_and_goal,
                        n_workers=4,
                        collect_device='cpu',
                        train_device=train_args.device
                        )

    """ Warm Up """

        # move the policy to the gym.
    env_collector.copy_current_policy(policy)

    print(f"BUFFER SIZE: {len(buffer)}")

    print("WARMING UP NORMALIZER AND BUFFER...")
    # env_collector.warmup_normalizer(warmup_steps=train_args.env_steps_per_train)
    env_collector.collect_nsteps(train_args.env_steps_per_train, use_policy=False, update_normalizers=True)
    print("...WARMUP COMPLETE.")

    """ Train Loop """

    best_po_loss = float('+inf')
    best_avg_rew = float('-inf')

    for e in range(train_args.epochs):

        # collect environment.

        # n_steps, n_eps, returns, _ = env_collector.collect(train_args.env_steps_per_train)
        env_collect_start_sec = time.perf_counter()
        n_steps, n_eps, returns = env_collector.collect_nsteps(train_args.env_steps_per_train, use_policy=True, update_normalizers=False)
        env_collet_total_sec = time.perf_counter() - env_collect_start_sec

        print(f"COLLECTED {n_steps} STEPS")
        print(f"COLLECTED {n_eps} EPISODES")
        print(f"TIME {env_collect_start_sec}s")
        print(f"BUFFER {env_collector.buffer.percent_filled}\% FILLED")

        # train.

        result = train_step(
            env_collector=env_collector,

            policy=policy,
            policy_opt=policy_opt,
            policy_opt_sched=policy_opt_sched,

            world_model=world_model,
            world_model_opt=world_model_opt,
            world_model_opt_sched=world_model_opt_sched,

            reward_func=env_data_processor.compute_reward,
            train_args=train_args,
            )

        # copy policy to environment.

        env_collector.copy_current_policy(policy)

        # LOGGING

        return_avg = torch.mean(returns).item()

        print(f"EPOCH {e}: po-loss = {result['po_loss_avg']}, wm-loss = {result['wm_loss_avg']}, return = {return_avg}")

        writer.add_scalar('rewards/rew_mean', return_avg, global_step=e)
        writer.add_scalar('rewards/rew_std', torch.std(returns).item(), global_step=e)
        writer.add_scalar('rewards/rew_min', torch.min(returns).item(), global_step=e)
        writer.add_scalar('rewards/rew_max', torch.max(returns).item(), global_step=e)

        writer.add_scalar('loss/policy', result['po_loss_avg'], global_step=e)
        writer.add_scalar('loss/policy_l1_reg', result['po_loss_l1_reg_avg'], global_step=e)
        writer.add_scalar('loss/policy_l2_reg', result['po_loss_l2_reg_avg'], global_step=e)
        writer.add_scalar('loss/policy_loss_reward_wm', result['po_loss_reward_avg'], global_step=e)
        writer.add_scalar('loss/policy_loss_reward_grnd', result['po_grnd_loss_reward_avg'], global_step=e)
        writer.add_scalar('loss/policy_loss_reward_wm_to_grnd_diff', result['po_grnd_wm_loss_reward_diff_avg'], global_step=e)

        writer.add_scalar('loss/world_model', result['wm_loss_avg'], global_step=e)
        writer.add_scalar('loss/world_model_diff', result['wm_loss_diff_avg'], global_step=e)
        writer.add_scalar('loss/world_model_l1_reg', result['wm_loss_l1_reg_avg'], global_step=e)
        writer.add_scalar('loss/world_model_l2_reg', result['wm_loss_l2_reg_avg'], global_step=e)

        writer.add_scalar('grad/policy', result['po_grad_norm_avg'], global_step=e)
        writer.add_scalar('grad/world_model', result['wm_grad_norm_avg'], global_step=e)

        writer.add_scalar('learning/policy_lr', result['po_lr'], global_step=e)
        writer.add_scalar('learning/world_model_lr', result['wm_lr'], global_step=e)

        writer.add_scalar('inputs/state_scale_mean', env_collector.normalizer_wm_state.accum_mean.norm(p=2), global_step=e)
        writer.add_scalar('inputs/state_scale_std', env_collector.normalizer_wm_state.accum_std.norm(p=2), global_step=e)

        writer.add_scalar('inputs/state_delta_mean', env_collector.normalizer_wm_state_delta.accum_mean.norm(p=2), global_step=e)
        writer.add_scalar('inputs/state_delta_std', env_collector.normalizer_wm_state_delta.accum_std.norm(p=2), global_step=e)

        writer.add_scalar('inputs/act_scale_mean', env_collector.normalizer_wm_action.accum_mean.norm(p=2), global_step=e)
        writer.add_scalar('inputs/act_scale_std', env_collector.normalizer_wm_action.accum_std.norm(p=2), global_step=e)

        writer.add_scalar('inputs/state_and_goal_mean', env_collector.normalizer_po_state_and_goal.accum_mean.norm(p=2), global_step=e)
        writer.add_scalar('inputs/state_and_goal_std', env_collector.normalizer_po_state_and_goal.accum_std.norm(p=2), global_step=e)

        writer.add_scalar('models/po_weight_scale', result['po_weight_scale'], global_step=e)
        writer.add_scalar('models/po_weight_max', result['po_weight_max'], global_step=e)
        writer.add_scalar('models/po_weight_min', result['po_weight_min'], global_step=e)
        writer.add_scalar('models/po_bias_scale', result['po_bias_scale'], global_step=e)
        writer.add_scalar('models/po_bias_max', result['po_bias_max'], global_step=e)
        writer.add_scalar('models/po_bias_min', result['po_bias_min'], global_step=e)

        writer.add_scalar('models/wm_weight_scale', result['wm_weight_scale'], global_step=e)
        writer.add_scalar('models/wm_weight_max', result['wm_weight_max'], global_step=e)
        writer.add_scalar('models/wm_weight_min', result['wm_weight_min'], global_step=e)
        writer.add_scalar('models/wm_bias_scale', result['wm_bias_scale'], global_step=e)
        writer.add_scalar('models/wm_bias_max', result['wm_bias_max'], global_step=e)
        writer.add_scalar('models/wm_bias_min', result['wm_bias_min'], global_step=e)
        writer.add_scalar('models/wm_pred_residuals_avg', result['wm_pred_resid_avg'], global_step=e)

        writer.add_scalar('buffer/perc_buffer_filled', env_collector.buffer.percent_filled, global_step=e)
        writer.add_scalar('buffer/nsteps_collected', n_steps, global_step=e)
        writer.add_scalar('buffer/neps_collected', n_eps, global_step=e)

        writer.add_scalar('time/collection_time', env_collet_total_sec, global_step=e)
        writer.add_scalar('time/episodes_per_second', float(n_eps) / float(env_collet_total_sec), global_step=e)
        writer.add_scalar('time/seconds_per_episode', float(env_collet_total_sec) / float(n_eps), global_step=e)
        writer.add_scalar('time/steps_per_second', float(n_steps) / float(env_collet_total_sec), global_step=e)

        if train_args.deep_stats:

            for w in range(train_args.wm_window - 1):
                writer.add_scalar(f'deepstats/wm_grad_rollout_{w}_avg', result[f'wm_grad_rollout_{w}_avg'], global_step=e)

            for w in range(train_args.po_window - 1):
                writer.add_scalar(f'deepstats/po_grad_rollout_{w}_avg', result[f'po_grad_rollout_{w}_avg'], global_step=e)

        # SAVING

        if return_avg > best_avg_rew:

            best_avg_rew = return_avg

            filepath = os.path.join("runs", "models")
            os.makedirs(filepath, exist_ok=True)

            policy.save_to_path(filepath=os.path.join(filepath, f"best_{train_args.name}_rew_policy.pth"))

            print("## BEST REWARD POLICY SAVED.")

            # #TEMP ---
            # # Reload the policy and determine that it matches.

            # temp_policy = Policy(
            #         input_size=po_input_size,
            #         action_size=act_size,
            #         hid_layers=[train_args.po_hid_units] * train_args.po_hid_layers,
            #         fn_combine_state_and_goal=env.preprocess_state_and_goal_for_policy,
            #         fn_post_process_action=lambda x : x,
            #         )
            # temp_policy.load_from_path(filepath=os.path.join(filepath, f"best_{train_args.name}_rew_policy.pth"))

            # for p,q in zip(env_collector.current_policy.parameters(), temp_policy.parameters()):

            #     assert torch.sum(torch.abs(p.cpu().data - q.cpu().data)) == 0
            # #--------

        if result['po_loss_avg'] < best_po_loss:

            best_po_loss = result['po_loss_avg']

            filepath = os.path.join("runs", "models")
            os.makedirs(filepath, exist_ok=True)

            policy.save_to_path(filepath=os.path.join(filepath, f"best_{train_args.name}_loss_policy.pth"))

            print("## BEST LOSS POLICY SAVED.")

        # SAVE RENDERS.

        # if train_args.save_renders:

        #     print("COLLECTING RENDER.")

        #     n_steps, n_eps, returns, frame_renders = env_collector.collect(train_args.env_max_steps)

        #     im_dir = os.path.join(log_path, "frame_renders", f"epoch_{e}")
        #     os.makedirs(im_dir, exist_ok=True)

        #     for i,f in enumerate(frame_renders):

        #         im = Image.fromarray(f)
        #         im.save(os.path.join(im_dir, f"frame_{i}.jpg"))

if __name__ == "__main__":

    run(ReacherGoalEnv, ReacherGoalEnvDataProcessor, REACHER_TRAIN_ARGS)
    # run(CartpoleBalanceGoalEnv(), CARTPOLE_BALANCE_TRAIN_ARGS)
    # run(CartpoleBalanceGoalEnv(), REACHER_TRAIN_ARGS)
