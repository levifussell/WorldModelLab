from typing import Callable, OrderedDict, Union, Tuple

import numpy as np
import torch

from abc import ABC, abstractmethod

class GoalEnv(ABC):
    """
    Abstract class.
    Wraps the env to allow for a 'goal' channel.
    """

    def __init__(
        self,
        env_gym,
        ) -> None:

        self._env_gym = env_gym

    @abstractmethod
    def get_curr_global_state_and_goal(
            self,
            ) -> Tuple[np.array, np.array]:
        """
        Computes the global state and global goal of the environment.

        :return: a tuple of the state and goal.
        """
        pass

    @abstractmethod
    def preprocess_state_for_world_model(
            self, 
            state: Union[np.array, torch.tensor]
            ) -> np.array:
        """
        Converts the global state into a local state for the world model.

        :param state: global state.
        :return: local world model state.
        """
        pass

    @abstractmethod
    def postprocess_state_for_world_model(
            self, 
            state: Union[np.array, torch.tensor]
            ) -> np.array:
        """
        Converts the local state from the world model into the global state.

        :param state: local world model state.
        :return: global state.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def reward(
            self,
            state: np.array,
            goal: np.array,
            act: np.array,
            ) -> float:
        """
        Computes the reward given the state, goal, and action.
        """
        pass

    @abstractmethod
    def step(
            self,
            act: np.array,
            ):
        """
        Takes a step in the environment.

        :param act: action.
        """
        pass

class ControlSuiteGoalEnv(GoalEnv):
    """
    Abstract class.
    Wraps the DeepMind Control Suite gyms.
    """

    def __init__(
            self,
            task_build_func: Callable,
            ) -> None:
        super().__init__(env_gym=task_build_func())

        self._physics = self._env_gym._physics
        self._task = self._env_gym.task

        self._env_gym._flat_observation = False  

    def action_spec(self):
        return self._env_gym.action_spec()

    def step(
            self,
            act: np.array,
            ):
        """
        Takes a step in the environment.

        :param act: gym action.
        """

        timestep = self._env_gym.step(act) 

        done = timestep.last()

        global_state, global_goal = self._get_global_state_and_goal()

        return global_state, global_goal, done