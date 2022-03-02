import queue
from typing import Any, Tuple

import numpy as np
import torch

class Buffer:

    def __init__(
            self, 
            state_size: int,
            act_size: int,
            goal_size: int,
            max_len: int):
        
        self._max_len = max_len

        self.state = None
        self.act = None
        self.goal = None
        self.end_index = None

    def __len__(self):
        return self._max_len

    @property
    def curr_size(self):
        return self.state.shape[0] if self.state is not None else 0

    @property
    def percent_filled(self):
        return (float(self.curr_size) / float(len(self))) * 100

    def add(
            self,
            state : torch.tensor,
            goal : torch.tensor,
            act : torch.tensor,
            ):

        assert state.shape[0] == act.shape[0] and state.shape[0] == goal.shape[0]

        if self.state is None:

            self.state = torch.clone(state)
            self.goal = torch.clone(goal)
            self.act = torch.clone(act)

            self.end_index = [self.curr_size]

        else:

            self.state = torch.cat([self.state, state], dim=0)
            self.goal = torch.cat([self.goal, goal], dim=0)
            self.act = torch.cat([self.act, act], dim=0)
            self.end_index.append(self.curr_size)

            while self.end_index[-1] > self._max_len:

                oldest_index = self.end_index[0]

                self.state = self.state[oldest_index:]
                self.goal = self.goal[oldest_index:]
                self.act = self.act[oldest_index:]

                self.end_index = self.end_index[1:]
                for i in range(len(self.end_index)):
                    self.end_index[i] -= oldest_index

    def sample(
            self,
            nsamples: int,
            window_size: int,
            ) -> None:

        valid_indices = []
        
        for e_srt, e_end in zip(self.end_index[:-1], self.end_index[1:]):

            valid_indices.append(np.arange(e_srt, e_end - window_size + 1, step=1))

        valid_indices = np.concatenate(valid_indices)

        if len(valid_indices) < nsamples:
            raise Exception("Buffer is too small for number of requested samples.")

        selected_indices = valid_indices[np.random.permutation(len(valid_indices))[:nsamples]]

        sample_state = []
        sample_goal = []
        sample_act = []

        for s in selected_indices:

            sample_state.append(self.state[s:s+window_size].unsqueeze(0))
            sample_goal.append(self.goal[s:s+window_size].unsqueeze(0))
            sample_act.append(self.act[s:s+window_size].unsqueeze(0))

        sample_state = torch.cat(sample_state, axis=0)
        sample_goal = torch.cat(sample_goal, axis=0)
        sample_act = torch.cat(sample_act, axis=0)

        assert sample_state.shape[0] == nsamples
        assert sample_state.shape[1] == window_size

        assert sample_goal.shape[0] == nsamples
        assert sample_goal.shape[1] == window_size

        assert sample_act.shape[0] == nsamples
        assert sample_act.shape[1] == window_size

        return (
            sample_state,
            sample_goal,
            sample_act,
        )
        







