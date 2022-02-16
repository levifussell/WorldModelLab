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
            max_len):
        
        self._max_len = max_len

        self.data = {
            'state'         : None,
            'act'           : None,
            'goal'          : None,
            'end_index'     : None,
        }

    def __len__(self):
        return self._max_len

    def __getattribute__(self, __name: str) -> Any:

        if __name in self.data.keys():
            return self.data[__name]

        return None

    def curr_size(self):
        return self.shape[0]

    def add(
            self,
            state : np.array,
            goal : np.array,
            act : np.array,
            ):

        assert state.shape[0] == act.shape[0] and state.shape[0] == goal.state[0]

        if self.state == None:

            self.state = np.copy(state)
            self.goal = np.copy(goal)
            self.act = np.copy(act)

            self.end_index = [self.curr_size()]

        else:

            self.state = np.concatenate([self.state, state], axis=0)
            self.goal = np.concatenate([self.goal, goal], axis=0)
            self.act = np.concatenate([self.act, act], axis=0)
            self.end_index.append(self.curr_size())

            while self.end_index[-1] > self._max_len:

                oldest_index = self.end_index[0]

                self.state = self.state[oldest_index:]
                self.goal = self.goal[oldest_index:]
                self.act = self.act[oldest_index:]

                self.end_index = self.end_index[1:]
                for i in range(self.end_index):
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

            sample_state.append(self.state[s:s+window_size][:,np.newaxis])
            sample_goal.append(self.goal[s:s+window_size][:,np.newaxis])
            sample_act.append(self.act[s:s+window_size][:,np.newaxis])

        sample_state = np.concatenate(sample_state, axis=1)
        sample_goal = np.concatenate(sample_goal, axis=1)
        sample_act = np.concatenate(sample_act, axis=1)

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
        







