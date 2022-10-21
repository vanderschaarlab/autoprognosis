# stdlib
import random
from typing import Any

# third party
import numpy as np
import torch
import torch.nn.functional as F


def enable_reproducible_results(seed: int = 0) -> None:
    """Set fixed seed for all the libraries"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
    """
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right   (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    """

    def sample(self, sample_shape: Any = torch.Size()) -> torch.Tensor:
        """Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical"""
        u = torch.empty(
            self.logits.size(),
            device=self.logits.device,
            dtype=self.logits.dtype,
        ).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape: Any = torch.Size()) -> torch.Tensor:
        """
        Gumbel-softmax resampling using the Straight-Through trick.
        https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        """
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """value is one-hot or relaxed"""
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            if value.shape != self.logits.shape:
                raise RuntimeError("log_prob failure")
        return -torch.sum(-value * F.log_softmax(self.logits, -1), -1)
