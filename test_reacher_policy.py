from typing import Tuple, Union
import numpy as np
import torch
from torchsummary import summary

from dm_control import mujoco
from dm_control.rl import control
from dm_control import suite
from dm_control import viewer
from dm_control.utils import rewards
from dm_control.suite.reacher import easy as build_reacher_task

from training.train import TrainArgs, DEFAULT_TRAIN_ARGS
from training.policy import Policy
from training.buffer import Buffer
from training.env_collector import EnvCollector

from env.reacher_dm_control_env import ReacherGoalEnv
from env.reacher_train_args import REACHER_TRAIN_ARGS
from env.cartpole_balance_dm_control_env import CartpoleBalanceGoalEnv
from env.cartpole_balance_train_args import CARTPOLE_BALANCE_TRAIN_ARGS

from enum import Enum

class InferencePolicyType(Enum):

    DETERMINISTIC = 0
    NOISY = 1
    RANDOM = 2

if __name__ == "__main__":

    train_args = TrainArgs(REACHER_TRAIN_ARGS)

    # test-specific args.

    policy_type = InferencePolicyType.DETERMINISTIC
    run_bounded_ep = True
    nsteps = 0

    # env

    env = ReacherGoalEnv()

    # train_args = TrainArgs(CARTPOLE_BALANCE_TRAIN_ARGS)
    # env = CartpoleBalanceGoalEnv()

    state_size = env.state_size
    goal_size = env.goal_size
    act_size = env.action_size
    po_input_size = env.policy_input_size

    # po_filename = f"runs/models/best_{train_args.name}_rew_policy.pth"
    po_filename = f"runs/models/complete_{train_args.name}_rew_policy.pth"

    policy = Policy(
                input_size=po_input_size,
                action_size=act_size,
                hid_layers=[train_args.po_hid_units] * train_args.po_hid_layers,
                fn_combine_state_and_goal=env.preprocess_state_and_goal_for_policy,
                fn_post_process_action=lambda x : x,
                ).to('cuda')

    policy.load_from_path(po_filename)

    summary(policy, input_size=[(state_size,), (goal_size,)])

    for n,p in policy.named_parameters():
        print(n, p)

    policy.cpu()

    # def env_step(timestep):
    #     global nsteps

    #     # action = np.random.uniform(act_spec.minimum, act_spec.maximum, act_spec.shape)

    #     curr_state, curr_goal = env.get_curr_global_state_and_goal()

    #     curr_state = torch.FloatTensor(curr_state)
    #     curr_goal = torch.FloatTensor(curr_goal)
        
    #     with torch.no_grad():

    #         action = policy.forward(
    #                     state=curr_state,
    #                     goal=curr_goal)

    #         if policy_type == InferencePolicyType.DETERMINISTIC:

    #             pass

    #         elif policy_type == InferencePolicyType.NOISY:

    #             action += torch.randn(action.shape) * train_args.po_env_exploration

    #         elif policy_type == InferencePolicyType.RANDOM:

    #             action = torch.randn(action.shape) * train_args.po_env_exploration

    #         # action += torch.randn(action.shape) * 0.5
    #         # action = torch.randn(action.shape) * 0.1

    #     nsteps += 1

    #     reward = env.reward(curr_state, curr_goal, action)

    #     print("------------")
    #     print(f"\t ENV REWARD = {timestep.reward}")
    #     print(f"\t WM REWARD = {reward}")
    #     print(f"\t GOAL = {curr_goal}")
    #     print(f"\t STEPS = {nsteps}")
    #     print("------------")

    #     if timestep.last():
    #         nsteps = 0

    #     return action

    # viewer.launch(env._env_gym, policy=env_step)

    # USING CUSTOM ENV LOOP

    # eps = 10

    # for e in range(eps):

    #     done = False

    #     states = []
    #     goals = []
    #     acts = []
    #     rewards = []

    #     state, goal = env.reset()

    #     state = torch.FloatTensor(state)
    #     goal = torch.FloatTensor(goal)

    #     states.append(state)

    #     goals.append(goal)

    #     state_and_goal = env.preprocess_state_and_goal_for_policy(state=state, goal=goal)

    #     while not done:

    #         with torch.no_grad():

    #             act = policy(
    #                         state=state,
    #                         goal=goal,
    #                         ).cpu()

    #             # act += torch.randn(act.shape).to(act.device) * self.exploration_std

    #         acts.append(act)

    #         rewards.append(env.reward(state=state, goal=goal, act=act))

    #         next_state, goal, done, info = env.step(act)

    #         next_state = torch.FloatTensor(next_state)
    #         goal = torch.FloatTensor(goal)

    #         # TODO: environment specific. Check the environment termination behaviour.
    #         if train_args.env_max_steps > 0 and len(states) >= train_args.env_max_steps:
    #             done = True # TODO: make this a case-specific done.
    #             break

    #         states.append(next_state)

    #         goals.append(goal)

    #         state_and_goal = env.preprocess_state_and_goal_for_policy(state=next_state, goal=goal)

    #         state = next_state

    #     print(f"EPISODE COMPLETE. RETURN: {np.sum(rewards)}")

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
                        max_env_steps=train_args.env_max_steps,
                        exploration_std=train_args.po_env_exploration,
                        normalizer_state=None,
                        normalizer_state_delta=None,
                        normalizer_action=None,
                        normalizer_state_and_goal=None,
                        is_parallel=False, # TODO: True for multiprocessing
                        collect_device='cpu',
                        train_device=train_args.device
                        )
    env_collector.copy_current_policy(policy)

    n_steps, n_eps, returns = env_collector.collect(train_args.env_steps_per_train)

    print(f"COLLECTED {n_steps} STEPS")
    print(f"COLLECTED {n_eps} EPISODES")

    print(returns)
