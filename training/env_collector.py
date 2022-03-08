from this import d
from typing import Tuple

import copy
import time

import numpy as np
import torch

from utils.normalizer import Normalizer

from .buffer import Buffer
from .buffer_iterator import BufferIterator

from env.goal_env import GoalEnv

class EnvCollector:

    def __init__(
            self, 
            env: GoalEnv, 
            buffer: Buffer,
            min_env_steps: int,
            exploration_std: float,
            normalizer_state: Normalizer,
            normalizer_state_delta: Normalizer,
            normalizer_action: Normalizer,
            normalizer_state_and_goal: Normalizer,
            max_env_steps: int = 0,
            collect_device: str = 'cpu',
            train_device: str = 'cuda',
            is_parallel: bool = True):
        """
        :param env: the environment to collect from.
        :param buffer: the buffer to store environment samples.
        :param min_env_steps: the minimum size of the window before termination is allowed.
        :param exploration_std: std of the gaussian noise added to the policy.
        :param collect_device:
        :param train_device:
        :param is_parallel: whether collection happens in parallel.
        """

        self.env = env
        self.buffer = buffer
        self.min_env_steps = min_env_steps
        self.max_env_steps = max_env_steps
        self.exploration_std = exploration_std
        self.collect_device = collect_device
        self.train_device = train_device
        self.is_parallel = is_parallel

        self.normalizer_state = normalizer_state
        self.normalizer_state_delta = normalizer_state_delta
        self.normalizer_action = normalizer_action
        self.normalizer_state_and_goal = normalizer_state_and_goal

        self.current_policy = None

    def copy_current_policy(self, policy):
        """
        Copies a policy to the gym's policy
        """

        # TODO: need a lock here so we don't write/read at same time.

        self.current_policy = copy.deepcopy(policy).to(self.collect_device)

    def start_gym_process(self):
        """
        Launches a gym in a separate process and samples from it.
        """

        # TODO: launch a separate process with the gym and collect it in the buffer.

        if not self.is_parallel:
            raise Exception("Collect is in parallel mode.")

        pass

    def sample_buffer(
        self,
        nsamples: int,
        batchsize: int,
        window_size: int
    ) -> torch.tensor:
        """
        Returns an iterator that gives batches for samples from a buffer.

        :param nsamples: number of samples from the buffer.
        :param batchsize: size of the minibatches in the samples.
        :param window_size: size of the windows in the samples.
        :return: iterator for batches of shape (N, T, F)
        """

        # TODO: need a lock here so we don't write/read at same time.

        data_sample = self.buffer.sample(
            nsamples=nsamples,
            window_size=window_size,
        )

        return BufferIterator(
            data=data_sample,
            batchsize=batchsize,
            device=self.train_device,
        )

    def warmup_normalizer(self, warmup_steps) -> int:

        n_steps = 0

        while n_steps < warmup_steps:

            done = False

            states = []
            dstates = []
            goals = []
            acts = []

            state, goal = self.env.reset()

            state = torch.FloatTensor(state)
            goal = torch.FloatTensor(goal)

            states.append(state)
            goals.append(goal)

            while not done:

                act = torch.randn(self.current_policy.action_size) * self.exploration_std
                acts.append(act)
                
                next_state, goal, done, info = self.env.step(act)

                next_state = torch.FloatTensor(next_state)
                goal = torch.FloatTensor(goal)
                
                dstates.append(next_state - state)

                # TODO: environment specific. Check the environment termination behaviour.
                if done and len(states) >= self.min_env_steps:
                    break

                # TODO: environment specific. Check the environment termination behaviour.
                if self.max_env_steps > 0 and len(states) >= self.max_env_steps:
                    done = True # TODO: make this a case-specific done.
                    break

                states.append(next_state)
                goals.append(goal)

                state = next_state

            n_steps += len(states)

        states = torch.cat([s.unsqueeze(0) for s in states], dim=0)
        dstates = torch.cat([s.unsqueeze(0) for s in dstates], dim=0)
        goals = torch.cat([g.unsqueeze(0) for g in goals], dim=0)
        acts = torch.cat([a.unsqueeze(0) for a in acts], dim=0)

        states_and_goals = self.env.preprocess_state_and_goal_for_policy(state=states, goal=goals)

        self.normalizer_state.warmup(states)
        self.normalizer_state_delta.warmup(dstates)
        self.normalizer_action.warmup(acts)
        self.normalizer_state_and_goal.warmup(states_and_goals)


    def collect(
        self,
        min_num_steps: int,
    ) -> Tuple[int, int, torch.Tensor]:
        """
        Manually collects from the environment. Used for non-parallel training.

        :param min_num_steps: minimum number of steps for the policy to collect.
        """

        if self.is_parallel:
            raise Exception("Collector is parallel. Shouldn't be doing manual collecting.")

        n_steps = 0
        n_eps = 0

        returns = []

        while n_steps < min_num_steps:

            done = False

            states = []
            goals = []
            acts = []
            rewards = []

            state, goal = self.env.reset()

            state = torch.FloatTensor(state)
            goal = torch.FloatTensor(goal)

            states.append(state)
            if self.normalizer_state is not None:
                self.normalizer_state += state

            goals.append(goal)

            state_and_goal = self.env.preprocess_state_and_goal_for_policy(state=state, goal=goal)
            if self.normalizer_state_and_goal is not None:
                self.normalizer_state_and_goal += state_and_goal

            while not done:

                with torch.no_grad():

                    act = self.current_policy(
                                state=state.to(self.collect_device),
                                goal=goal.to(self.collect_device),
                                ).cpu()

                    act += torch.randn(act.shape).to(act.device) * self.exploration_std

                acts.append(act)
                if self.normalizer_action is not None:
                    self.normalizer_action += act

                rewards.append(self.env.reward(state=state, goal=goal, act=act))

                next_state, goal, done, info = self.env.step(act)

                next_state = torch.FloatTensor(next_state)
                goal = torch.FloatTensor(goal)

                if self.normalizer_state_delta is not None:
                    self.normalizer_state_delta += (next_state - state)

                # TODO: environment specific. Check the environment termination behaviour.
                if done and len(states) >= self.min_env_steps:
                    break

                # TODO: environment specific. Check the environment termination behaviour.
                if self.max_env_steps > 0 and len(states) >= self.max_env_steps:
                    done = True # TODO: make this a case-specific done.
                    break

                states.append(next_state)
                if self.normalizer_state is not None:
                    self.normalizer_state += next_state

                goals.append(goal)

                state_and_goal = self.env.preprocess_state_and_goal_for_policy(state=next_state, goal=goal)
                if self.normalizer_state_and_goal is not None:
                    self.normalizer_state_and_goal += state_and_goal

                state = next_state

            self.buffer.add(
                state=torch.cat([s.unsqueeze(0) for s in states], dim=0),
                goal=torch.cat([g.unsqueeze(0) for g in goals], dim=0),
                act=torch.cat([a.unsqueeze(0) for a in acts], dim=0),
            )

            returns.append(np.sum(rewards))

            n_steps += len(states)
            n_eps += 1

        return n_steps, n_eps, torch.FloatTensor(returns)