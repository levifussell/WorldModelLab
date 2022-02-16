from typing import Tuple

class BufferIterator:

    def __init__(
            self,
            data : Tuple,
            batchsize : int,
            ) -> None:

        self.data = data
        self.max_len = self.data[0].shape[0]
        self.batchsize = batchsize

    def __iter__(self):
        self.idx = 0

    def __next__(self):

        if self.idx >= self.max_len:
            raise StopIteration

        batch = (b[self.idx:self.idx + self.batchsize] for b in self.data)

        self.idx += self.batchsize

        return batch