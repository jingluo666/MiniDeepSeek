from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .layers import RMSNorm, SwiGLU, TransformerBlock


class SequentialMTPBlock(nn.Module):
    """
    Sequential Multi-Token Prediction block.

    相比简单独立头，这里让每个深度的 MTP 使用：
    - 上一层 MTP state
    - teacher-forced future token embedding
    这样更接近 sequential MTP 的训练路径。
    """

    def __init__(self, hidden_size: int, vocab_size: int, rms_norm_eps: float):
        super().__init__()
        self.state_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.token_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fuse_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.refine = SwiGLU(hidden_size=hidden_size, intermediate_size=hidden_size * 2)
        self.output_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, mtp_state: torch.Tensor, future_token_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat([self.state_norm(mtp_state), self.token_norm(future_token_embeds)], dim=-1)
        fused = self.fuse_proj(fused)
        mtp_state = mtp_state + self.refine(fused)
        logits = self.head(self.output_norm(mtp_state))
        return mtp_state, logits


class DenseTransformerLM(nn.Module):
    """
    Dense autoregressive language model.

    Dense 自回归语言模型。
    The design mirrors modern LLMs while keeping every component readable.
    结构参考现代 LLM，但每个部件都尽量保持透明可读。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer_ffn_types = self._build_layer_ffn_types(config)
        self.layer_attention_types = self._build_layer_attention_types(config)
        self.layer_residual_modes = self._build_layer_residual_modes(config)
        self.layer_moe_router_types = self._build_layer_moe_router_types(config)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    rope_base=config.rope_base,
                    rms_norm_eps=config.rms_norm_eps,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    use_sdpa=config.use_sdpa,
                    attention_type=self.layer_attention_types[layer_idx],
                    attention_window_size=config.attention_window_size,
                    attention_compression_stride=config.attention_compression_stride,
                    attention_compressed_top_k=config.attention_compressed_top_k,
                    attention_qk_norm=config.attention_qk_norm,
                    ffn_type=self.layer_ffn_types[layer_idx],
                    residual_mode=self.layer_residual_modes[layer_idx],
                    mhc_width_multiplier=config.mhc_width_multiplier,
                    moe_num_experts=config.moe_num_experts,
                    moe_top_k=config.moe_top_k,
                    moe_num_shared_experts=config.moe_num_shared_experts,
                    moe_normalize_topk_prob=config.moe_normalize_topk_prob,
                    moe_router_type=self.layer_moe_router_types[layer_idx],
                    swiglu_clamp_linear_max=config.swiglu_clamp_linear_max,
                    swiglu_clamp_gate_max=config.swiglu_clamp_gate_max,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList(
            [
                SequentialMTPBlock(
                    hidden_size=config.hidden_size,
                    vocab_size=config.vocab_size,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.mtp_depth)
            ]
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
            for mtp_head in self.mtp_heads:
                mtp_head.head.weight = self.embed_tokens.weight

        self._init_weights()

    def _build_layer_ffn_types(self, config: ModelConfig) -> list[str]:
        if config.ffn_type != "moe":
            return [config.ffn_type for _ in range(config.num_layers)]

        if config.moe_num_layers <= 0 or config.moe_num_layers >= config.num_layers:
            return ["moe" for _ in range(config.num_layers)]

        dense_prefix = config.num_layers - config.moe_num_layers
        return ["dense" if layer_idx < dense_prefix else "moe" for layer_idx in range(config.num_layers)]

    def _build_layer_attention_types(self, config: ModelConfig) -> list[str]:
        if config.attention_type not in {"compressed", "csa", "hca"}:
            return [config.attention_type for _ in range(config.num_layers)]

        if config.attention_type == "compressed":
            attention_types = ["csa" for _ in range(config.num_layers)]
        else:
            attention_types = [config.attention_type for _ in range(config.num_layers)]
        if config.num_layers >= 2:
            attention_types[0] = "sliding_window"
            attention_types[1] = "sliding_window"
        if config.attention_type == "compressed" and config.num_layers >= 4:
            for layer_idx in range(2, config.num_layers):
                attention_types[layer_idx] = "csa" if layer_idx % 2 == 0 else "hca"
        return attention_types

    def _build_layer_moe_router_types(self, config: ModelConfig) -> list[str]:
        router_types = ["learned" for _ in range(config.num_layers)]
        if config.ffn_type != "moe" or config.moe_hash_num_layers <= 0:
            return router_types
        moe_start = max(0, config.num_layers - config.moe_num_layers)
        hash_layers = min(config.moe_hash_num_layers, config.moe_num_layers)
        for layer_idx in range(moe_start, moe_start + hash_layers):
            router_types[layer_idx] = "hash"
        return router_types

    def _build_layer_residual_modes(self, config: ModelConfig) -> list[str]:
        if config.residual_mode != "mhc":
            return [config.residual_mode for _ in range(config.num_layers)]

        if config.mhc_num_layers <= 0 or config.mhc_num_layers >= config.num_layers:
            return ["mhc" for _ in range(config.num_layers)]

        dense_prefix = config.num_layers - config.mhc_num_layers
        return ["standard" if layer_idx < dense_prefix else "mhc" for layer_idx in range(config.num_layers)]

    def _init_weights(self) -> None:
        """
        Lightweight initialization close to common LLM defaults.

        一个接近常见 LLM 默认值的轻量初始化。
        """

        seen_weights: set[int] = set()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight_ptr = module.weight.data_ptr()
                if weight_ptr not in seen_weights:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    seen_weights.add(weight_ptr)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                weight_ptr = module.weight.data_ptr()
                if weight_ptr not in seen_weights:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    seen_weights.add(weight_ptr)

    def gradient_checkpointing_enable(self) -> None:
        self.config_gradient_checkpointing = True

    def _run_layer_with_optional_checkpoint(
        self,
        layer: TransformerBlock,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        hyper_states: torch.Tensor | None,
        token_ids: torch.Tensor,
        sample_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        use_checkpoint = bool(getattr(self, "config_gradient_checkpointing", False)) and self.training
        if not use_checkpoint:
            return layer(
                hidden_states,
                attention_mask=attention_mask,
                hyper_states=hyper_states,
                token_ids=token_ids,
                sample_ids=sample_ids,
            )

        empty = hidden_states.new_empty(0)
        packed_attention_mask = attention_mask if attention_mask is not None else empty
        packed_hyper = hyper_states if hyper_states is not None else empty
        packed_sample_ids = sample_ids if sample_ids is not None else token_ids.new_zeros(token_ids.shape)
        outputs = checkpoint(
            layer.forward_checkpoint,
            hidden_states,
            packed_attention_mask,
            packed_hyper,
            token_ids,
            packed_sample_ids,
            use_reentrant=False,
        )
        next_hidden, next_hyper, moe_aux, moe_entropy, router_fraction, expert_fraction, mhc_rms = outputs
        stats: dict[str, torch.Tensor] = {}
        if moe_aux.numel() > 0:
            stats["moe_aux_loss"] = moe_aux
        if moe_entropy.numel() > 0:
            stats["moe_router_entropy"] = moe_entropy
        if router_fraction.numel() > 0:
            stats["moe_router_fraction"] = router_fraction
        if expert_fraction.numel() > 0:
            stats["moe_expert_fraction"] = expert_fraction
        if mhc_rms.numel() > 0:
            stats["mhc_hyper_rms"] = mhc_rms
        return next_hidden, stats, next_hyper if next_hyper.numel() > 0 else None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        hyper_states = None
        moe_aux_losses = []
        moe_entropies = []
        moe_router_fractions = []
        moe_expert_fractions = []
        mhc_rms = []
        for layer in self.layers:
            hidden_states, layer_stats, hyper_states = self._run_layer_with_optional_checkpoint(
                layer,
                hidden_states,
                attention_mask,
                hyper_states,
                input_ids,
                sample_ids,
            )
            if "moe_aux_loss" in layer_stats:
                moe_aux_losses.append(layer_stats["moe_aux_loss"])
            if "moe_router_entropy" in layer_stats:
                moe_entropies.append(layer_stats["moe_router_entropy"])
            if "moe_router_fraction" in layer_stats:
                moe_router_fractions.append(layer_stats["moe_router_fraction"])
            if "moe_expert_fraction" in layer_stats:
                moe_expert_fractions.append(layer_stats["moe_expert_fraction"])
            if "mhc_hyper_rms" in layer_stats:
                mhc_rms.append(layer_stats["mhc_hyper_rms"])
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        output = {"logits": logits}
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = lm_loss
            output["lm_loss"] = lm_loss.detach()

            if self.mtp_heads:
                mtp_losses = []
                mtp_state = hidden_states
                for depth_idx, mtp_head in enumerate(self.mtp_heads, start=1):
                    mtp_inputs = self._build_mtp_inputs(input_ids=input_ids, labels=labels, offset=depth_idx)
                    future_token_embeds = self.embed_tokens(mtp_inputs)
                    mtp_state, mtp_logits = mtp_head(mtp_state, future_token_embeds)
                    mtp_labels = self._build_mtp_labels(labels=labels, offset=depth_idx)
                    mtp_loss = F.cross_entropy(
                        mtp_logits.view(-1, mtp_logits.shape[-1]),
                        mtp_labels.view(-1),
                        ignore_index=-100,
                    )
                    mtp_losses.append(mtp_loss)
                    output[f"mtp_loss_d{depth_idx}"] = mtp_loss.detach()

                mtp_loss = torch.stack(mtp_losses).mean()
                loss = loss + (self.config.mtp_loss_weight * mtp_loss)
                output["mtp_loss"] = mtp_loss.detach()

            if moe_aux_losses:
                moe_aux_loss = torch.stack(moe_aux_losses).mean()
                loss = loss + (self.config.moe_aux_loss_weight * moe_aux_loss)
                output["moe_aux_loss"] = moe_aux_loss.detach()
                output["moe_router_entropy"] = torch.stack(moe_entropies).mean().detach()

                router_fraction = torch.stack(moe_router_fractions).mean(dim=0)
                expert_fraction = torch.stack(moe_expert_fractions).mean(dim=0)
                output["moe_router_fraction_mean"] = router_fraction.mean().detach()
                output["moe_expert_fraction_mean"] = expert_fraction.mean().detach()
                output["moe_expert_fraction_max"] = expert_fraction.max().detach()
                output["moe_expert_fraction_min"] = expert_fraction.min().detach()

            if mhc_rms:
                output["mhc_hyper_rms"] = torch.stack(mhc_rms).mean().detach()

            output["loss"] = loss
        return output

    def _build_mtp_labels(self, labels: torch.Tensor, offset: int) -> torch.Tensor:
        mtp_labels = torch.full_like(labels, fill_value=-100)
        if offset >= labels.shape[1]:
            return mtp_labels
        shifted = labels[:, offset:]
        mtp_labels[:, :-offset] = shifted
        return mtp_labels

    def _build_mtp_inputs(self, input_ids: torch.Tensor, labels: torch.Tensor, offset: int) -> torch.Tensor:
        mtp_inputs = input_ids.clone()
        if offset >= input_ids.shape[1]:
            return mtp_inputs
        mtp_inputs[:, :-offset] = input_ids[:, offset:]
        mtp_inputs[:, -offset:] = input_ids[:, -1:].expand(-1, offset)
        masked_positions = self._build_mtp_labels(labels=labels, offset=offset).eq(-100)
        if masked_positions.any():
            mtp_inputs = mtp_inputs.masked_fill(masked_positions, 0)
        return mtp_inputs

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def summary(self) -> dict[str, int | dict[str, Any] | list[str]]:
        return {
            "num_parameters": self.num_parameters(),
            "model_config": asdict(self.config),
            "layer_ffn_types": self.layer_ffn_types,
            "layer_attention_types": self.layer_attention_types,
            "layer_residual_modes": self.layer_residual_modes,
            "layer_moe_router_types": self.layer_moe_router_types,
        }
