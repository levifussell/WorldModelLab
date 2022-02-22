import numpy as np

from abc import ABC, abstractmethod

class GoalEnv(ABC):
    """
    Abstract class.
    Wraps the gym env to allow for a 'goal' channel.
    """

    def __init__(
        self,
        env_gym,
        ):

        self.env_gym = env_gym

    @abstractmethod
    def _split_state_for_goal(
            self,
            state,
            ):
        """
        Splits the OpenAI Gym state into a 'state' and a 'goal'
        component.

        :param state: gym state.
        """
        # raise Exception("Not implemented.")
        ...

    @abstractmethod
    def reward(
            self,
            state,
            goal,
            act,
            ):
        """
        Computes the reward given the state, goal, and action.
        """
        # raise Exception("Not implemented.")
        ...

    def step(
            self,
            act,
            ):
        """
        Takes a step in the environment.

        :param act: gym action.
        """

        state, _, done, info = self.env_gym.step(act) # NOTE: '_' is reward, but we don't care about it.
        new_state, goal = self._split_state_for_goal(state)

        return new_state, goal, done, info


# NOTE:
#   The below isn't possible, unless the environment has
#       a 'compute reward' method built-in.
#       OpenAI Gyms don't. Not sure about DeepMind Control Suite.
# class IdentityGoalEnv(GoalEnv):
#     """
#     Turns any gym into a GoalEnv, assumes the goal channel is empty.
#     """

#     def _split_state_for_goal(self, state):
#         return state, np.array([])

#     def reward():