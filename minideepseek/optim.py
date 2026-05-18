from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from .config import OptimizerConfig


def _orthogonalize_update(
    update: torch.Tensor,
    steps: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Approximate Muon-style orthogonalization with Newton-Schulz iterations.

    这里实现的是一个“学习优先”的近似版 Muon：
    - 只对矩阵参数做正交化更新
    - 保留 Nesterov + Newton-Schulz 的核心思路
    - 不追求论文级内核效率
    """

    if update.ndim != 2:
        return update

    rows, cols = update.shape
    transposed = False
    work = update.float()
    if rows < cols:
        work = work.transpose(0, 1)
        transposed = True

    work = work / work.norm().clamp_min(eps)
    for _ in range(max(1, steps)):
        gram = work.transpose(0, 1) @ work
        work = 1.5 * work - 0.5 * work @ gram

    if transposed:
        work = work.transpose(0, 1)
    return work.to(dtype=update.dtype)


class MuonOptimizer(torch.optim.Optimizer):
    """
    Lightweight Muon approximation for matrix-like parameters.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        update_rms_target: float = 0.18,
        eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "update_rms_target": update_rms_target,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            update_rms_target = group["update_rms_target"]
            eps = group["eps"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                if grad.ndim != 2:
                    raise ValueError("MuonOptimizer only supports 2D parameters.")

                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = grad + (momentum * buf)
                update = _orthogonalize_update(update, steps=ns_steps, eps=eps)
                update = update / update.pow(2).mean().sqrt().clamp_min(eps)
                update = update * update_rms_target

                parameter.mul_(1.0 - (lr * weight_decay))
                parameter.add_(update, alpha=-lr)

        return loss


class HybridMuonAdamW:
    """
    Route matrix-heavy weights to Muon and keep the rest on AdamW.

    目标不是论文级严格一致，而是把“Muon + AdamW 混合优化”这条技术路线
    真正落到当前教学代码里，并保持训练稳定。
    """

    def __init__(self, model: torch.nn.Module, config: OptimizerConfig):
        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if self._should_use_adamw(name=name, parameter=parameter):
                adamw_params.append(parameter)
            else:
                muon_params.append(parameter)

        self.muon = (
            MuonOptimizer(
                muon_params,
                lr=config.lr,
                momentum=config.muon_momentum,
                weight_decay=config.weight_decay,
                ns_steps=config.muon_ns_steps,
                update_rms_target=config.muon_update_rms_target,
                eps=config.eps,
            )
            if muon_params
            else None
        )
        self.adamw = (
            torch.optim.AdamW(
                adamw_params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
            if adamw_params
            else None
        )
        self.param_groups: list[dict[str, Any]] = []
        for optimizer in (self.muon, self.adamw):
            if optimizer is not None:
                self.param_groups.extend(optimizer.param_groups)

    def _should_use_adamw(self, name: str, parameter: torch.nn.Parameter) -> bool:
        if parameter.ndim != 2:
            return True
        adamw_name_markers = (
            "embed_tokens",
            "lm_head",
            "mtp_heads",
            "norm",
            "router",
        )
        return any(marker in name for marker in adamw_name_markers)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "hybrid_muon",
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        muon_state = state_dict.get("muon")
        adamw_state = state_dict.get("adamw")
        if self.muon is not None and muon_state is not None:
            self.muon.load_state_dict(muon_state)
        if self.adamw is not None and adamw_state is not None:
            self.adamw.load_state_dict(adamw_state)


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer | HybridMuonAdamW:
    if config.kind == "hybrid_muon":
        return HybridMuonAdamW(model=model, config=config)
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def optimizer_summary(config: OptimizerConfig) -> dict[str, Any]:
    return asdict(config)
