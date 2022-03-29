from typing import Tuple

class BufferIterator:

    def __init__(
            self,
            data : Tuple,
            batchsize : int,
            device: str,
            ) -> None:

        self.data = data
        self.max_len = self.data[0].shape[0]
        self.batchsize = batchsize
        self.device = device

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):

        if self.idx >= self.max_len:
            raise StopIteration

        batch = [b[self.idx:(self.idx+self.batchsize)].to(self.device) for b in self.data]

        self.idx += self.batchsize

        return batch