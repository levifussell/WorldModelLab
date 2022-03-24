from typing import Callable, OrderedDict, Union, Tuple

import numpy as np
import torch

from abc import ABC, abstractclassmethod, abstractmethod

class GoalEnv(ABC):
    """
    Abstract class.
    Wraps the env to allow for a 'goal' channel.
    """

    def __init__(
        self,
        env_gym,
        render: bool = False
        ) -> None:

        self.render = render
        self.frames = []

        self._env_gym = env_gym

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
    def reward(
            self,
            state: torch.Tensor,
            goal: torch.Tensor,
            act: torch.Tensor,
            ) -> float:
        """
        Computes the reward given the state, goal, and action.
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

class ControlSuiteGoalEnv(GoalEnv):
    """
    Abstract class.
    Wraps the DeepMind Control Suite gyms.
    """

    def __init__(
            self,
            task_build_func: Callable,
            render: bool = False,
            ) -> None:
        """
        :param max_steps: maximium environment steps, usually for testing.
        """
        super().__init__(env_gym=task_build_func(), render=render)

        self._physics = self._env_gym._physics
        self._task = self._env_gym.task

        self._env_gym._flat_observation = False  

    @property
    def action_size(self):
        return self._env_gym.action_spec().shape[-1]

    @property
    def action_min(self):
        return self._env_gym.action_spec().minimum

    @property
    def action_max(self):
        return self._env_gym.action_spec().maximum

    @property
    def state_size(self):
        state, goal = self.get_curr_global_state_and_goal()
        return state.shape[-1]

    @property
    def goal_size(self):
        state, goal = self.get_curr_global_state_and_goal()
        return goal.shape[-1]

    @property
    def policy_input_size(self):
        state, goal = self.get_curr_global_state_and_goal()
        policy_input = self.preprocess_state_and_goal_for_policy(state, goal)
        return policy_input.shape[-1]

    def step(
            self,
            act: torch.Tensor,
            ):

        timestep = self._env_gym.step(act.flatten().numpy()) 

        done = timestep.last()

        global_state, global_goal = self.get_curr_global_state_and_goal()

        info = {}

        if self.render:
            self.frames.append(self._physics.render(camera_id=0, height=200, width=200))

        return global_state, global_goal, done, info

    def reset(self):

        timestep = self._env_gym.reset()

        # done = timestep.last()

        global_state, global_goal = self.get_curr_global_state_and_goal()

        if self.render:
            self.frames.clear()
            self.frames.append(self._physics.render(camera_id=0, height=200, width=200))

        return global_state, global_goal #, done