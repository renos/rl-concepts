from typing import Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from typing import Tuple


class NormalizeEwma(nn.Module):
    def __init__(
        self,
        insize: int,
        norm_axes: int = 1,
        beta: float = 0.99,
        epsilon: float = 1e-2,
    ):
        super().__init__()

        # Params
        self.norm_axes = norm_axes
        self.beta = beta
        self.epsilon = epsilon

        # Parameters
        self.running_mean = nn.Parameter(th.zeros(insize), requires_grad=False)
        self.running_mean_sq = nn.Parameter(th.zeros(insize), requires_grad=False)
        self.debiasing_term = nn.Parameter(th.tensor(0.0), requires_grad=False)

    def running_mean_var(self) -> Tuple[th.Tensor, th.Tensor]:
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (mean_sq - mean**2).clamp(min=1e-2)
        return mean, var

    def forward(self, x: th.Tensor, training=False) -> th.Tensor:
        if self.training and training:
            x_detach = x.detach()
            batch_mean = x_detach.mean(dim=tuple(range(self.norm_axes)))
            batch_mean_sq = (x_detach**2).mean(dim=tuple(range(self.norm_axes)))

            weight = self.beta

            # self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            # self.running_mean_sq.mul_(weight).add_(batch_mean_sq * (1.0 - weight))
            # self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))
            self.running_mean.data = self.running_mean * weight + batch_mean * (
                1.0 - weight
            )
            self.running_mean_sq.data = (
                self.running_mean_sq * weight + batch_mean_sq * (1.0 - weight)
            )
            self.debiasing_term.data = self.debiasing_term * weight + 1.0 * (
                1.0 - weight
            )

        mean, var = self.running_mean_var()
        mean = mean[(None,) * self.norm_axes]
        var = var[(None,) * self.norm_axes]
        x = (x - mean) / th.sqrt(var)
        return x

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        mean, var = self.running_mean_var()
        mean = mean[(None,) * self.norm_axes]
        var = var[(None,) * self.norm_axes]
        x = x * th.sqrt(var) + mean
        return x


class ScaledMSEHead(nn.Module):
    def __init__(
        self,
        insize: int,
        outsize: int,
        init_scale: float = 0.1,
        norm_kwargs: Dict = {},
    ):
        super().__init__()

        # Layer
        self.linear = nn.Linear(insize, outsize)

        # Initialization
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)

        # Normalizer
        self.normalizer = NormalizeEwma(outsize, **norm_kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        value = self.linear(x)
        return value, self.denormalize(value)

    def normalize(self, x: th.Tensor) -> th.Tensor:
        return self.normalizer(x)

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        return self.normalizer.denormalize(x)

    def mse_loss(self, pred: th.Tensor, targ: th.Tensor) -> th.Tensor:
        norm_targ = self.normalizer(targ, training=True)
        return F.mse_loss(pred, norm_targ, reduction="none")


class CategoricalActionHead(nn.Module):
    def __init__(
        self,
        insize: int,
        num_actions: int,
        init_scale: float = 0.1,
    ):
        super().__init__()

        # Layer
        self.linear = nn.Linear(insize, num_actions)

        # Initialization
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.linear(x)
        return x
