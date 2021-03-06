from typing import Tuple, Union
import numpy as np
import torch
from torchsummary import summary

from dm_control import mujoco
from dm_control.rl import control
from dm_control import suite
from dm_control import viewer
from dm_control.utils import rewards

from src.training.train import TrainArgs, DEFAULT_TRAIN_ARGS
from src.training.policy import Policy
from src.training.buffer import Buffer
from src.training.env_collector import EnvCollector

from src.env.cartpole_balance_dm_control_env import CartpoleBalanceGoalEnv
from src.env.cartpole_balance_train_args import CARTPOLE_BALANCE_TRAIN_ARGS

from enum import Enum


class InferencePolicyType(Enum):
    DETERMINISTIC = 0
    NOISY = 1
    RANDOM = 2


if __name__ == "__main__":

    # test-specific args.
    policy_type = InferencePolicyType.DETERMINISTIC
    run_bounded_ep = True
    nsteps = 0

    # env
    train_args = TrainArgs(CARTPOLE_BALANCE_TRAIN_ARGS)
    env = CartpoleBalanceGoalEnv()

    state_size = env.state_size
    goal_size = env.goal_size
    act_size = env.action_size
    po_input_size = env.policy_input_size

    po_filename = f"runs/models/best_{train_args.name}_rew_policy.pth"

    policy = Policy(
        input_size=po_input_size,
        action_size=act_size,
        hid_layers=[train_args.po_hid_units] * train_args.po_hid_layers,
        fn_combine_state_and_goal=env.preprocess_state_and_goal_for_policy,
        fn_post_process_action=lambda x: x,
    ).to(train_args.device)

    policy.load_from_path(po_filename)

    summary(policy, input_size=[(state_size,), (goal_size,)])

    for n, p in policy.named_parameters():
        print(n, p)

    policy.cpu()


    def env_step(timestep):
        global nsteps

        # action = np.random.uniform(act_spec.minimum, act_spec.maximum, act_spec.shape)

        curr_state, curr_goal = env.get_curr_global_state_and_goal()

        curr_state = torch.FloatTensor(curr_state)
        curr_goal = torch.FloatTensor(curr_goal)

        with torch.no_grad():

            action = policy.forward(
                state=curr_state,
                goal=curr_goal)

            if policy_type == InferencePolicyType.DETERMINISTIC:

                pass

            elif policy_type == InferencePolicyType.NOISY:

                action += torch.randn(action.shape) * train_args.po_env_exploration

            elif policy_type == InferencePolicyType.RANDOM:

                action = torch.randn(action.shape) * train_args.po_env_exploration

            # action += torch.randn(action.shape) * 0.5
            # action = torch.randn(action.shape) * 0.1

        nsteps += 1

        reward = env.reward(curr_state, curr_goal, action)

        print("------------")
        print(f"\t ENV REWARD = {timestep.reward}")
        print(f"\t WM REWARD = {reward}")
        print(f"\t GOAL = {curr_goal}")
        print(f"\t STEPS = {nsteps}")
        print("------------")

        if timestep.last():
            nsteps = 0

        return action


    viewer.launch(env._env_gym, policy=env_step)