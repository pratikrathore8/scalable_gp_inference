from typing import TYPE_CHECKING

import numpy as np
import torch

from rlaopt.solvers import Solver

from .sdd_config import SDDConfig

if TYPE_CHECKING:
    from .kernel_linsys import KernelLinSys  # Import only for type hints


class SDD(Solver):
    def __init__(
        self,
        config: SDDConfig,
        system: "KernelLinSys",
        W_init: torch.Tensor,
        device: torch.device,
    ):
        self.system = system
        self.config = config
        self.precond_conifg = self.config.precond_config

        self._W = W_init.clone()
        self.device = device

        # Sampling probs
        self.probs = torch.ones(self.system.A.shape[0]) / self.system.A.shape[0]
        self.probs_cpu = self.probs.cpu().numpy()

        # Velocity Tensor
        self._V = self._W.clone()
        # Averaged iterate Tensor
        self._Y = self._W.clone()

    @property
    def W(self):
        # SDD outputs averaged iterate
        return self._Y

    def _get_precond(self):
        pass

    def _get_blk(self) -> torch.Tensor:
        try:
            blk = torch.multinomial(self.probs, self.config.blk_size, replacement=False)
        except RuntimeError as e:
            if "number of categories cannot exceed" not in str(e):
                raise e
            blk = np.random.choice(
                self.probs.shape[0],
                size=self.config.blk_size,
                replace=False,
                p=self.probs_cpu,
            )
            blk = torch.from_numpy(blk)
        return blk

    def _get_blk_grad(
        self,
        W: torch.Tensor,
        B: torch.Tensor,
        blk: torch.Tensor,
    ) -> torch.Tensor:

        # Compute the block gradient
        blk_grad = (
            self.system.A_row_oracle(blk) @ W + self.system.reg * W[blk, :] - B[blk, :]
        )

        return self.system.A.shape[0] / self.config.blk_size * blk_grad

    def _step(self):

        # Get mask
        mask = self.system.mask

        # If all components have converged, nothing to do
        if not mask.any():
            return

        # Sample coordinate blk
        blk = self._get_blk()

        # Blk gradient is evaluated at iterates + momentum
        eval_loc = self._W[:, mask] + self.config.m * self._V[:, mask]
        g = self._get_blk_grad(eval_loc, self.system.B[:, mask], blk)

        # Create update direction
        update = torch.zeros_like(self._W[:, mask])
        update[blk] = self.config.step_size * g

        # Velocity update
        self._V[:, mask] = self.config.m * self._V[:, mask] - update
        # Iterate update
        self._W[:, mask] += self._V[:, mask]
        # Average update
        self._Y[:, mask] = (
            self.config.theta * self._W[:, mask]
            + (1 - self.config.theta) * self._Y[:, mask]
        )
