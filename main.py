import torch
import torch.nn as nn
import torch.optim as opt

from training.policy import Policy
from training.world_model import WorldModel
from training.env_collector import EnvCollector
from training.buffer import Buffer
from training.train import train_step, TrainArgs, DEFAULT_TRAIN_ARGS

from env.goal_env import IdentityGoalEnv

def run(
    train_args = DEFAULT_TRAIN_ARGS,
):

    train_args = TrainArgs(train_args)

    """ Setup Environment/Gym """

    env = None

    state_size = None

    goal_size = None

    act_size = None

    fn_pre_process_state = lambda x : x
    fn_post_process_state = lambda x : x
    fn_post_process_action = lambda x : x

    reward_func = None

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
                        is_parallel=False, # TODO: True for multiprocessing
                        )

    """ Build Policy """

    policy = Policy(
                state_size=state_size,
                action_size=act_size,
                hid_layers=[train_args.po_hid_units] * train_args.po_hid_layers,
                fn_pre_process_state=fn_pre_process_state,
                fn_post_process_state=fn_post_process_state,
                fn_post_process_action=fn_post_process_action,
                )

    policy_opt = opt.RAdam(
                    parameters=policy.parameters(), 
                    lr=train_args.po_lr)

    """ Build World Model"""

    world_model = WorldModel(
                state_size=state_size,
                hid_layers=[train_args.wm_hid_units] * train_args.wm_hid_layers,
                fn_pre_process_state=fn_pre_process_state,
                fn_post_process_state=fn_post_process_state,
                )

    world_model_opt = opt.RAdam(
                    parameters=world_model.parameters(), 
                    lr=train_args.wm_lr)

    """ Train Loop """

        # move the policy to the gym.
    env_collector.copy_current_policy(policy)
        # start the gym process.
    # env_collector.start_gym_process()

    for e in train_args.epochs:

        result = train_step(
            env_collector=env_collector,
            policy=policy,
            policy_opt=policy_opt,
            world_model=world_model,
            world_model_opt=world_model_opt,
            reward_func=reward_func,
            train_args=train_args,
            )

        # move the most recently trained policy to the gym.

        env_collector.copy_current_policy(policy)

        # TODO: some logging here.

if __name__ == "__main__":

    run()