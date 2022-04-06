from typing import Callable, OrderedDict, Union, Tuple

import numpy as np
import torch

from abc import ABC, abstractclassmethod, abstractmethod

class GoalEnvDataProcessor(ABC):
    """
    Abstract class.
    Performs the post/pre-processing of the data.
    """

    @abstractmethod
    def preprocess_state_and_goal_for_policy(
            self,
            state: torch.Tensor,
            goal: torch.Tensor,
            ) -> torch.Tensor:
        """
        Converts the global state and goal into a local input state for
            the policy.

        :param state: global state.
        :param goal: global goal.
        :return: local input state for the policy.
        """
        pass

    @abstractmethod
    def preprocess_state_for_world_model(
            self, 
            state: torch.Tensor,
            ) -> torch.Tensor:
        """
        Converts the global state into a local state for the world model.

        :param state: global state.
        :return: local world model state.
        """
        pass

    @abstractmethod
    def postprocess_state_for_world_model(
            self, 
            prev_global_state: torch.Tensor,
            state_delta: torch.Tensor,
            ) -> torch.Tensor:
        """
        Converts the local state from the world model into the global state.

        :param prev_global_state: previous global state of the world model.
        :param state: local world model state.
        :return: global state.
        """
        pass

    @abstractmethod
    def compute_delta_state_world_model(
            self, 
            state_from: torch.Tensor,
            state_to: torch.Tensor,
            ) -> torch.Tensor:
        """
        Computes the difference between two states of the world model.
        In general, this will just be a Euclidean difference.

        :param state_from: state the world model came from.
        :param state_to: state the world model arrived at.
        :return: difference between two states.
        """
        pass

    @abstractmethod
    def compute_reward(
            self,
            state: torch.Tensor,
            goal: torch.Tensor,
            act: torch.Tensor,
            ) -> float:
        """
        Computes the reward given the state, goal, and action.
        """
        pass


class GoalEnv(ABC):
    """
    Abstract class.
    Wraps the env to allow for a 'goal' channel.
    """

    def __init__(
        self,
        env_gym,
        data_preprocessor: GoalEnvDataProcessor,
        render: bool = False
        ) -> None:

        self.render = render
        self.frames = []

        self._env_gym = env_gym
        self._data_processor = data_preprocessor

    @abstractmethod
    def get_curr_global_state_and_goal(
            self,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the global state and global goal of the environment.

        :return: a tuple of the state and goal.
        """
        pass

    @abstractmethod
    def step(
            self,
            act: torch.Tensor,
            ):
        """
        Takes a step in the environment.

        :param act: action.
        :return: env timestep.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the environment.

        :return: env timestep.
        """
        pass