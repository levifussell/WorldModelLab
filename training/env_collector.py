import copy

import torch

from buffer import Buffer

class EnvCollector:

    def __init__(self, gym, buffer):

        self.gym = gym
        self.buffer = buffer

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

        pass

    def sample_buffer(
        self
    ) -> torch.tensor:
        """
        Returns a batch sampled uniformly random from the buffer.

        :return: batch (N, T, F)
        """

        # TODO: need a lock here so we don't write/read at same time.

        pass
