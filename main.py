import os
import platform
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from training.policy import Policy
from training.world_model import WorldModel
from training.env_collector import EnvCollector
from training.buffer import Buffer
from training.train import train_step, TrainArgs, DEFAULT_TRAIN_ARGS

from env.reacher_dm_control_env import ReacherGoalEnv

def run(
    train_args = DEFAULT_TRAIN_ARGS,
):

    train_args = TrainArgs(train_args)

    """ Setup Environment/Gym """

    env = ReacherGoalEnv()

    state_size = env.state_size
    goal_size = env.goal_size
    act_size = env.action_size
    po_input_size = env.policy_input_size

    """ Build EnvCollector """

    buffer = Buffer(
                state_size=state_size,
                goal_size=goal_size,
                act_size=act_size,
                max_len=train_args.max_buffer_size,
                )

    env_collector = EnvCollector(
                        env=env,
                        buffer=buffer,
                        min_env_steps=max(train_args.wm_window, train_args.po_window),
                        exploration_std=train_args.po_env_exploration,
                        is_parallel=False, # TODO: True for multiprocessing
                        collect_device='cpu',
                        train_device=train_args.device
                        )

    """ Logging """


    date_str = datetime.now().strftime('%b%d_%H-%M-%S')
    experiment_hash = date_str + '_' + platform.uname().node
    log_path = os.path.join(train_args.logdir, experiment_hash)

    writer = SummaryWriter(log_path)

    """ Build Policy """

    policy = Policy(
                input_size=po_input_size,
                action_size=act_size,
                hid_layers=[train_args.po_hid_units] * train_args.po_hid_layers,
                fn_combine_state_and_goal=env.preprocess_state_and_goal_for_policy,
                fn_post_process_action=lambda x : x,
                ).to(train_args.device)

    policy_opt = opt.RAdam(
                    params=policy.parameters(), 
                    lr=train_args.po_lr)

    summary(policy, input_size=[(state_size,), (goal_size,)])

    """ Build World Model"""

    world_model = WorldModel(
                state_size=state_size,
                action_size=act_size,
                hid_layers=[train_args.wm_hid_units] * train_args.wm_hid_layers,
                fn_pre_process_state=env.preprocess_state_for_world_model,
                fn_post_process_state=env.postprocess_state_for_world_model,
                ).to(train_args.device)

    world_model_opt = opt.RAdam(
                    params=world_model.parameters(), 
                    lr=train_args.wm_lr)

    summary(world_model, input_size=[(state_size,), (act_size,)])

    """ Train Loop """

        # move the policy to the gym.
    env_collector.copy_current_policy(policy)
        # start the gym process.
    # env_collector.start_gym_process()

    best_po_loss = float('inf')

    for e in range(train_args.epochs):

        # collect environment.

        if not env_collector.is_parallel:
            n_steps = env_collector.collect(train_args.env_steps_per_train)

            print(f"COLLECTED {n_steps} STEPS")
            print(f"BUFFER {env_collector.buffer.percent_filled}\% FILLED")

        # train.

        result = train_step(
            env_collector=env_collector,
            policy=policy,
            policy_opt=policy_opt,
            world_model=world_model,
            world_model_opt=world_model_opt,
            reward_func=env.reward,
            train_args=train_args,
            )

        # copy policy to environment.

        env_collector.copy_current_policy(policy)

        # LOGGING

        print(f"EPOCH {e}: po-loss = {result['po_loss_avg']}, wm-loss = {result['wm_loss_avg']}")

        writer.add_scalar('loss/policy', result['po_loss_avg'], global_step=e)
        writer.add_scalar('loss/world_model', result['wm_loss_avg'], global_step=e)

        writer.add_scalar('grad/policy', result['po_grad_norm_avg'], global_step=e)
        writer.add_scalar('grad/world_model', result['wm_grad_norm_avg'], global_step=e)

        # SAVING

        if result['po_loss_avg'] < best_po_loss:

            best_po_loss = result['po_loss_avg']

            filepath = os.path.join("runs", "models")
            os.makedirs(filepath, exist_ok=True)

            torch.save(policy.state_dict(), f=os.path.join(filepath, "best_policy.pth"))

            print("BEST POLICY SAVED.")

if __name__ == "__main__":

    run()