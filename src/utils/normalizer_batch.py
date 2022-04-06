from __future__ import annotations

import torch
import torch.nn as nn

class NormalizerBatch(nn.Module):
    """
    Simpler normalizer that just takes a running mean of batch inputs.
    """

    def __init__(self, length):
        super().__init__()

        self.n = 0
        self.accum_mean = nn.parameter.Parameter(torch.zeros(1, length), requires_grad=False)
        self.accum_mean_sqr = nn.parameter.Parameter(torch.zeros(1, length), requires_grad=False)
        self.accum_std = nn.parameter.Parameter(torch.ones(1, length), requires_grad=False)

    def __iadd__(self, value: torch.tensor):

        assert len(value.shape) > 1

        batchsize = value.shape[0]

        if self.n == 0:

            self.accum_mean.data = torch.mean(value, dim=0, keepdim=True)
            self.accum_mean_sqr.data = torch.mean(torch.square(value), dim=0, keepdim=True)

        else:

            new_n = self.n + batchsize
            w_old = float(self.n) / float(new_n)
            w_new = 1.0 / float(new_n)

            new_sum = torch.sum(value, dim=0, keepdim=True)
            new_sum_sqr = torch.sum(torch.square(value), dim=0, keepdim=True)

            self.accum_mean.data = w_old * self.accum_mean.data + w_new * new_sum.to(self.accum_mean.device)
            self.accum_mean_sqr.data = w_old * self.accum_mean_sqr.data + w_new * new_sum_sqr.to(self.accum_mean_sqr.device)

        var = torch.clamp(self.accum_mean_sqr.data - torch.square(self.accum_mean.data), min=0)
        self.accum_std.data = torch.sqrt(var)
        self.accum_std.data[self.accum_std < 1e-6] = 1.0

        if torch.any(torch.isnan(self.accum_std)):
            import pdb; pdb.set_trace()
            print("STAT")

        self.n += batchsize

        return self

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

    def descale(self, x: torch.tensor) -> torch.tensor:

        if self.n == 0:
            return x
        else:
            return x * self.accum_std.to(x.device)