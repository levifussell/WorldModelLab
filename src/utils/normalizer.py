from __future__ import annotations

import torch
import torch.nn as nn

class Normalizer(nn.Module):

    def __init__(self, length): #, accum_rate=0.99):
        super().__init__()

        self.n = 0
        self.accum_mean = nn.parameter.Parameter(torch.zeros(length), requires_grad=False)
        self._accum_S = torch.zeros(length)
        self.accum_std = nn.parameter.Parameter(torch.ones(length), requires_grad=False)

        # self.accum_rate = accum_rate

    def warmup(self, batch: torch.tensor):

        self.n += batch.shape[0]
        self.accum_mean.data = torch.mean(batch, 0)
        self.accum_std.data = torch.std(batch, 0)
        self._accum_S = torch.square(self.accum_std) * self.n

    def __iadd__(self, value: torch.tensor):

        self.n += 1

        if self.n == 1:

            self.accum_mean.data = torch.clone(value)

            self._accum_S = torch.clone(value)

            self.accum_std.data = torch.clone(value)

        else:

            old_mean = torch.clone(self.accum_mean)

            self.accum_mean.data = old_mean + (value - old_mean) / self.n

            self._accum_S += (value - old_mean) * (value - self.accum_mean)

            self.accum_std.data = torch.sqrt(self._accum_S / self.n)

        return self

    # def to(self, device, *args, **kwargs):
    #     self.device = device
        # return super().to(device, *args, **kwargs)

    # def __iadd__(self, batch: torch.tensor) -> Normalizer:

    #     if batch.shape[0] == 1:
    #         raise Exception("Must be a batch of data.")

    #     self.n += batch.shape[0]

    #     if self.accum_mean is None:

    #         self.accum_mean = torch.mean(batch, dim=0).to(self.device)
    #         self.accum_std = torch.std(batch, dim=0).to(self.device)

    #     else:

    #         self.accum_mean = self.accum_rate * self.accum_mean + (1 - self.accum_rate) * torch.mean(batch, dim=0).to(self.device)
    #         self.accum_std = self.accum_rate * self.accum_std + (1 - self.accum_rate) * torch.std(batch, dim=0).to(self.device)

    #     return self

    def __call__(self, x: torch.tensor) -> torch.tensor:

        if self.n == 0:
            return x
        else:
            return (x - self.accum_mean.to(x.device)) / self.accum_std.to(x.device)

    def denormalize(self, x: torch.tensor) -> torch.tensor:

        if self.n == 0:
            return x
        else:
            return x * self.accum_std.to(x.device) + self.accum_mean.to(x.device)