import copy
import time

import torch

from buffer import Buffer
from buffer_iterator import BufferIterator

class EnvCollector:

    def __init__(
            self, 
            env, 
            buffer : Buffer,
            min_env_steps : int,
            is_parallel : bool = True):
        """
        :param env: the environment to collect from.
        :param buffer: the buffer to store environment samples.
        :param min_collection_window_size: the minimum size of the window before termination is allowed.
        :is_parallel: whether collection happens in parallel.
        """

        self.env = env
        self.buffer = buffer
        self.min_env_steps = min_env_steps
        self.is_parallel = is_parallel

        self.current_policy = None

    def copy_current_policy(self, policy):
        """
        Copies a policy to the gym's policy
        """

        # TODO: need a lock here so we don't write/read at same time.

        self.current_policy = copy.deepcopy(policy)

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
        )

    def collect(
        self,
        min_num_steps: int,
    ) -> None:
        """
        Manually collects from the environment. Used for non-parallel training.

        :param min_num_steps: minimum number of steps for the policy to collect.
        """

        if self.is_parallel:
            raise Exception("Collector is parallel. Shouldn't be doing manual collecting.")

        n_steps = 0

        while n_steps < min_num_steps:

            done = False

            states = []
            goals = []
            acts = []

            state, goal = self.env.reset()

            states.append(state)
            goals.append(goal)

            while not done:

                act = self.current_policy(
                            state=state,
                            goal=goal,
                            )

                acts.append(act)
                
                state, goal, done, info = self.env.step(act)

                if done and len(states) >= self.min_env_steps:
                    break

                states.append(state)
                goals.append(goal)

                time.sleep(0.001)

            self.buffer.add(
                state=states,
                goal=goals,
                act=acts,
            )

            time.sleep(0.001)

        pass
