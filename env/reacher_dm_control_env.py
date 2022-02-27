from typing import Tuple, Union
import numpy as np
import torch

from .goal_env import ControlSuiteGoalEnv

from dm_control import mujoco
from dm_control.rl import control
from dm_control import suite
from dm_control import viewer
from dm_control.utils import rewards
from dm_control.suite.reacher import easy as build_reacher_task

class ReacherGoalEnv(ControlSuiteGoalEnv):

    def __init__(self):
        super().__init__(task_build_func=build_reacher_task)

    def get_curr_global_state_and_goal(
            self,
            ) -> Tuple[np.array, np.array]:

        target_pos = self._physics.named.data.geom_xpos['target', :2]
        finger_pos = self._physics.named.data.geom_xpos['finger', :2]
        joint_angs = self._physics.position()
        joint_vels = self._physics.velocity()

        # TODO: this contains 'extra' info: the finger_pos, which is non local.
        #  we can only extract this from the environment with FK on the joints,
        #  so we'd have to run it through the MuJoCo physics which might be
        #  expensive. 
        state = np.concatenate([finger_pos, joint_angs, joint_vels], axis=-1)
        goal = np.concatenate([target_pos], axis=-1)

        return state, goal

    def preprocess_state_for_world_model(
            self, 
            state: Union[np.array, torch.tensor]
            ) -> np.array:
        return state

    def postprocess_state_for_world_model(
            self, 
            state: Union[np.array, torch.tensor]
            ) -> np.array:
        return state

    def preprocess_state_and_goal_for_policy(
            self,
            state: Union[np.array, torch.tensor],
            goal: Union[np.array, torch.tensor],
            ) -> np.array:

        target_pos = goal[...,:2]
        finger_pos = state[...,:2]
        joint_angs = state[...,2:4]
        joint_vels = state[...,4:]

        to_target = target_pos - finger_pos

        if isinstance(state, torch.Tensor):

            return torch.cat([joint_angs, to_target, joint_vels], dim=-1)

        elif isinstance(state, np.ndarray):

            return np.concatenate([joint_angs, to_target, joint_vels], axis=-1)

        else:
            raise Exception("Invalid state type.")

    def reward(
            self, 
            state: Union[np.array, torch.tensor], 
            goal: Union[np.array, torch.tensor],
            act: Union[np.array, torch.tensor],
            ) -> float:

        finger_pos = state[...,:2]
        radii = self._physics.named.model.geom_size[['target', 'finger'], 0].sum()

        if isinstance(state, torch.Tensor):

            finger_to_target_dist = torch.norm(goal[...,:2] - finger_pos, p=2, dim=-1)
            # return (finger_to_target_dist e= radii).float()
            # TODO: below is incorrect, it is a smoothed loss.
            # return torch.clamp(finger_to_target_dist, min=0, max=radii) #/ radii
            # return 1.0 / (finger_to_target_dist + 0.1) #/ radii
            return torch.exp(-10.0 * finger_to_target_dist) #/ radii


        elif isinstance(state, np.ndarray):

            finger_to_target_dist = np.linalg.norm(goal[...,:2] - finger_pos, axis=-1)
            return rewards.tolerance(finger_to_target_dist, (0, radii))

        else:
            raise Exception("Invalid state type.")

if __name__ == "__main__":

    print("Testing Reacher gym.")

    env = ReacherGoalEnv()

    act_size = env.action_size
    act_min = env.action_min
    act_max = env.action_max

    states = []

    def policy(timestep):

        # action = np.random.uniform(act_spec.minimum, act_spec.maximum, act_spec.shape)
        action = torch.rand(act_size) * (act_max - act_min) + act_min

        curr_state, curr_goal = env.get_curr_global_state_and_goal()
        reward = env.reward(curr_state, curr_goal, action)

        policy_state = env.preprocess_state_and_goal_for_policy(curr_state, curr_goal)

        flat_obs = control.flatten_observation(timestep.observation)['observations']

        print("------------")

        print(f"WORLD MODEL:")
        print(f"\t REWARD = {reward}")
        print(f"\t STATE = {policy_state}")
        # print(f"\t DONE = {done}")

        print(f"CONTROL SUITE:")
        print(f"\t REWARD = {timestep.reward}")
        print(f"\t STATE = {flat_obs}")
        print(f"\t DONE = {timestep.last()}")

        states.append(policy_state[np.newaxis,:])

        print(f"STATE MEAN: {np.mean(np.concatenate(states, axis=0), axis=0)}")
        print(f"STATE STD: {np.std(np.concatenate(states, axis=0), axis=0)}")

        if not timestep.first():
            assert timestep.reward == reward
        assert np.allclose(policy_state, flat_obs)

        print("------------")

        return action

    viewer.launch(env._env_gym, policy=policy)
