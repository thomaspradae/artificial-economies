from __future__ import annotations

import torch
from torch.optim import Optimizer


class DeepMindRMSprop(Optimizer):
    """Centered RMSProp update used by the released DeepMind DQN code.

    The epsilon is inside the square root:
    ``param -= lr * grad / sqrt(g2 - g**2 + eps)``.
    PyTorch's built-in centered RMSprop uses a different epsilon placement.
    """

    def __init__(self, params, lr: float = 2.5e-4, alpha: float = 0.95, eps: float = 0.01):
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0 <= alpha < 1:
            raise ValueError(f"invalid alpha: {alpha}")
        if eps < 0:
            raise ValueError(f"invalid epsilon: {eps}")
        super().__init__(params, {"lr": lr, "alpha": alpha, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            one_minus_alpha = 1.0 - alpha

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("DeepMindRMSprop does not support sparse gradients")

                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["grad_avg"] = torch.zeros_like(parameter)
                    state["grad_sq_avg"] = torch.zeros_like(parameter)

                grad_avg = state["grad_avg"]
                grad_sq_avg = state["grad_sq_avg"]
                grad_avg.mul_(alpha).add_(gradient, alpha=one_minus_alpha)
                grad_sq_avg.mul_(alpha).addcmul_(gradient, gradient, value=one_minus_alpha)

                denominator = grad_sq_avg.sub(grad_avg.square()).add(eps).sqrt()
                parameter.addcdiv_(gradient, denominator, value=-lr)

        return loss
