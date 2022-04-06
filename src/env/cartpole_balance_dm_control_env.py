from typing import Tuple, Union
import numpy as np
import torch

from env.control_suite_goal_env import ControlSuiteGoalEnv
from utils import rewards

from dm_control.suite.cartpole import balance as build_cartpole_balance
from dm_control.rl import control
from dm_control import viewer


# Cartpole source code:
# https://github.com/deepmind/dm_control/blob/4110221701c2666df21953b55e98e5c552485599/dm_control/suite/cartpole.py


class CartpoleBalanceGoalEnv(ControlSuiteGoalEnv):

    def __init__(self):
        super().__init__(task_build_func=build_cartpole_balance)

    def get_curr_global_state_and_goal(
            self,
            ) -> Tuple[np.array, np.array]:
        """
        Following Barto et al. (1983) - Neuronlike adaptive elements that can solve difficult learning control problems
        http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf
        The cart-pole has 4 state variables
        - position of the cart on the track
        - angle of the pole with the vertical
        - cart velocity
        - rate of change of the angle (angular velocity?)
        """
        positions = self._physics.bounded_position()  # 3D, (cart_position, cos(pole_angle), sin(pole_angle))
        velocities = self._physics.velocity()  # 2D, (cart_velocity??, pole_angular_velocity)
        state = np.hstack([positions, velocities])
        goal = np.array([])  # No goal/target to aim for
        return state, goal

    def preprocess_state_for_world_model(
            self,
            state: Union[np.array, torch.tensor]
            ) -> np.array:
        """
        Converts the global state into a local state for the world model.

        :param state: global state.
        :return: local world model state.
        """
        return state

    def postprocess_state_for_world_model(
            self,
            prev_global_state: Union[np.array, torch.tensor],
            state_delta: Union[np.array, torch.tensor],
            ) -> np.array:
        """
        Converts the local state from the world model into the global state.

        :param state: local world model state.
        :return: global state.
        """
        return prev_global_state + state_delta

    def compute_delta_state_world_model(
            self, 
            state_from: Union[np.array, torch.tensor],
            state_to: Union[np.array, torch.tensor],
            ) -> np.array:
        """
        Computes the difference between two states of the world model.
        In general, this will just be a Euclidean difference.

        :param state_from: state the world model came from.
        :param state_to: state the world model arrived at.
        :return: difference between two states.
        """
        return state_to - state_from

    def preprocess_state_and_goal_for_policy(
            self,
            state: Union[np.array, torch.tensor],
            goal: Union[np.array, torch.tensor],
            ) -> np.array:
        """
        Converts the global state and goal into a local input state for
            the policy.

        :param state: global state.
        :param goal: global goal.
        :return: local input state for the policy.
        """
        # As there is no goal for balancing the cartpole we cannot calculate relative to the goal
        return state

    def reward(
            self,
            state: np.array,
            goal: np.array,
            act: np.array,
            ) -> float:
        """
        Computes the reward given the state, goal, and action.

        Based on _get_reward method from the control suite source code
        https://github.com/deepmind/dm_control/blob/4110221701c2666df21953b55e98e5c552485599/dm_control/suite/cartpole.py#L204

        self._task.get_reward(self._physics)
        """
        cart_position = state[..., 0:1]
        pole_angle_cosine = state[..., 1:2]
        angular_vel = state[..., 4:5]

        upright = (pole_angle_cosine + 1) / 2  # ~ how upright is the pole

        centered = rewards.tolerance(cart_position, margin=2)
        centered = (1 + centered) / 2  # ~ how centered is the cart (= how close to 0 is the cart position)

        small_velocity = rewards.tolerance(angular_vel, margin=5).min()
        small_velocity = (1 + small_velocity) / 2  # ~ how still is the pole (= how close to 0 is the angular velocity)

        # TODO: in control suite they take rewards.tolerance()[0]
        #       why do they take [0]?
        #       Could be an error as np.where can return a tuple, but when a condition is given it doesn't
        #       Check some more? Compare performance of both?

        # This is what is in the control suite
        # small_control = rewards.tolerance(act, margin=1,
        #                                   value_at_margin=0,
        #                                   sigmoid='quadratic')[0]

        # What is wrong with this?
        small_control = rewards.tolerance(act, margin=1,
                                          value_at_margin=0,
                                          sigmoid='quadratic')

        small_control = (4 + small_control) / 5  # ~ how small are the necessary adjustments
                                                 # (= how close to 0 is the action/cart movement)

        return upright.mean() * small_control * small_velocity * centered


if __name__ == "__main__":

    print("Testing Cartpole gym.")

    env = CartpoleBalanceGoalEnv()

    act_size = env.action_size
    act_min = env.action_min
    act_max = env.action_max

    states = []

    def policy(timestep):

        # action = np.random.uniform(act_spec.minimum, act_spec.maximum, act_spec.shape)
        action = torch.rand(act_size) * (act_max - act_min) + act_min

        curr_state, curr_goal = env.get_curr_global_state_and_goal()

        # Seems this casting is now required to make things run
        curr_state = torch.tensor(curr_state).float()
        curr_goal = torch.tensor(curr_goal).float()
        action = action.float()

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

        # NOTE: the rewards do not match because the small_control term does not match (the other reward terms do match)
        #       the control suite version uses the previous action (just taken), we use the next action (about to take)
        #       it makes sense that both should be small, so we leave the reward as it is for now.
        # if not timestep.first():
        #     assert timestep.reward == reward
        assert np.allclose(policy_state, flat_obs)

        print("------------")

        return action

    viewer.launch(env._env_gym, policy=policy)