from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F

from ..utils import logging


logger = logging.get_logger(__name__)


class OFTModule(torch.nn.Module):
    param_keys = ("R",)

    @classmethod
    def get_parametrizations_kwargs(
        cls, module_weight: torch.Tensor, state_dict: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, Any]:
        param = state_dict.get("R")
        if param is None:
            return kwargs
        if param.ndim == 2:
            return {**kwargs, "share_blocks": True, "reduction_rate": module_weight.shape[1] // param.shape[1]}
        elif param.ndim == 3:
            return {**kwargs, "share_blocks": False, "reduction_rate": param.shape[0]}
        else:
            raise ValueError(f"Invalid shape for parameter: {param.shape}")

    def __init__(
        self,
        weight: torch.Tensor,
        weight_type: Optional[str] = None,
        init_delta: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        reduction_rate: int = 4,
        epsilon: float = 5e-6,
        use_coft: bool = False,
        share_blocks: bool = False,
    ):
        """
        Parameters:
        OFT module to be registered as hook
            weight_type (`string`, *optional*): choose from [None, 'conv', '1d']
            init_delta (`string, *optional*`): init values for delta
            reduction_rate (`int`): Number of diagonal blocks used for orthogonal update matrix.
            scale (`float`): orthogonal update blend scale. This is intended to use during inference.
            use_coft (`bool`): Use constrained orthogonal finetuning.
            share_blocks (`bool`): Share parameters in diagonal blocks of orthogonal update matrix.
        """
        super().__init__()
        self.weight_type = weight_type
        self.reduction_rate = weight.shape[1] if weight_type == "conv" else reduction_rate
        self.use_coft = use_coft
        self.share_blocks = share_blocks

        weight = self._reshape_weight(weight)
        if weight.shape[1] % self.reduction_rate != 0:
            raise ValueError(
                f"Input feature dimension must be divisible by reduction rate, "
                f"but input feature dimension {weight.shape[1]} does not divide "
                f"reduction rate {self.reduction_rate}."
            )

        block_dim = weight.shape[1] // self.reduction_rate

        # Initialize to 0 (maps to identity after cayley transformation) for smooth tuning
        if init_delta is not None:
            self.R = torch.nn.Parameter(init_delta)
        elif self.share_blocks:
            self.R = torch.nn.Parameter(
                torch.zeros((block_dim, block_dim), device=weight.device, dtype=torch.float32)
            )
        else:
            self.R = torch.nn.Parameter(
                torch.zeros((self.reduction_rate, block_dim, block_dim), device=weight.device, dtype=torch.float32)
            )

        self.epsilon = epsilon * block_dim * block_dim
        self.scale = scale

    def _reshape_weight(
        self,
        updated_weight: torch.Tensor,
        original_weight: torch.Tensor = None,
        reverse: bool = False,
        **kwargs,
    ):
        if self.weight_type is not None:
            if self.weight_type == "conv":
                if not reverse:
                    return updated_weight.flatten(1, -1)
                else:
                    return updated_weight.reshape(-1, *original_weight.shape[1:])
            elif self.weight_type == "1d":
                if not reverse:
                    return updated_weight.unsqueeze(1)
                else:
                    return updated_weight.squeeze(1)
            else:
                raise ValueError(f"`weight_type` must be one of 'conv' or '1d, but is {self.weight_type}.")
        else:
            return updated_weight

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.use_coft:
            with torch.no_grad():
                self.R.copy_(project_norm(self.R, eps=self.epsilon))

        reshaped_weight = self._reshape_weight(weight)

        # Block-diagonal parametrization for orthogonal weight update
        orthogonal_update = block_diagonal(cayley(self.R), self.reduction_rate).t()

        # Slerp between original and updated weight if required
        if self.scale != 1.0:
            I_update = torch.eye(
                orthogonal_update.shape[-1], device=orthogonal_update.device, dtype=orthogonal_update.dtype
            )
            orthogonal_update = slerp(self.scale, I_update, orthogonal_update)

        updated_weight = reshaped_weight.to(orthogonal_update.dtype) @ orthogonal_update
        updated_weight = self._reshape_weight(updated_weight.to(weight.dtype), weight, reverse=True)
        return updated_weight


class SVDiffModule(torch.nn.Module):
    param_keys = ("delta",)

    @classmethod
    def get_parametrizations_kwargs(
        cls, module_weight: torch.Tensor, state_dict: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, Any]:
        return kwargs

    def __init__(
        self,
        weight: torch.Tensor,
        weight_type: Optional[str] = None,
        init_delta: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ):
        """
        Parameters:
        SVDiff module to be registered as hook
            weight (`torch.Tensor`, *required*): pre-trained weight
            weight_type (`string`, *optional*): choose from [None, 'conv', '1d']
            init_delta (`string, *optional*`): init values for delta
            scale (`float`): spectral shifts scale. This is intended to use during inference.
        """
        super().__init__()
        self.weight_type = weight_type
        weight = self._reshape_weight_for_svd(weight)
        # perform SVD
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        self.register_buffer("U", U.detach(), persistent=False)
        self.U.requires_grad = False
        self.register_buffer("S", S.detach(), persistent=False)
        self.S.requires_grad = False
        self.register_buffer("Vh", Vh.detach(), persistent=False)
        self.Vh.requires_grad = False
        # initialize to 0 for smooth tuning
        self.delta = torch.nn.Parameter(torch.zeros_like(S)) if init_delta is None else torch.nn.Parameter(init_delta)
        self.scale = scale

    # Copied from diffusers.models.parametrizations.OFTModule._reshape_weight_for_svd
    def _reshape_weight_for_svd(
        self,
        updated_weight: torch.Tensor,
        original_weight: torch.Tensor = None,
        reverse: bool = False,
        **kwargs,
    ):
        if self.weight_type is not None:
            if self.weight_type == "conv":
                if not reverse:
                    return updated_weight.flatten(1, -1)
                else:
                    return updated_weight.reshape(-1, *original_weight.shape[1:])
            elif self.weight_type == "1d":
                if not reverse:
                    return updated_weight.unsqueeze(0)
                else:
                    return updated_weight.squeeze(0)
            else:
                raise ValueError(f"`weight_type` must be one of 'conv' or '1d, but is {self.weight_type}.")
        else:
            return updated_weight

    def forward(self, weight: torch.Tensor):
        updated_weight = self.U @ torch.diag(F.relu(self.S + self.scale * self.delta)) @ self.Vh
        updated_weight = self._reshape_weight_for_svd(updated_weight, weight, reverse=True)
        return updated_weight


def cayley(data: torch.Tensor) -> torch.Tensor:
    # Ensure the input matrix is square
    if data.shape[-2] != data.shape[-1]:
        raise ValueError(
            "Cayley parametrization matrix must be square along "
            f"last two dimensions, but has shape {data.shape[-2:]}."
        )

    skew = 0.5 * (data - data.transpose(-2, -1))
    I_data = torch.eye(data.shape[-2], device=data.device).expand(*data.shape)

    # Perform the Cayley parametrization
    return (I_data + skew) @ torch.inverse(I_data - skew)


def block_diagonal(data: torch.Tensor, reduction_rate: int) -> torch.Tensor:
    if len(data.shape) == 2:
        # Create a list of R repeated block_count times
        blocks = [data] * reduction_rate
    else:
        # Create a list of R slices along the third dimension
        blocks = [data[i, ...] for i in range(reduction_rate)]

    return torch.block_diag(*blocks)


def project_norm(data: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if len(data.shape) == 2:
        norm = torch.norm(data)
        return eps * (data / norm) if norm <= eps else data

    eps = eps / torch.sqrt(torch.tensor(data.shape[0]))
    norm = torch.norm(data, dim=(-2, -1), keepdim=True)
    return torch.where(norm <= eps, data, eps * (data / norm))


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4"""
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res
