from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm 只用均方根缩放，不减去均值，形式更简单，
    也是很多现代 LLM 使用的归一化方式。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    SwiGLU 前馈层：
    gate = silu(Wg x), value = Wv x, output = Wo(gate * value)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        clamp_linear_max: float = 0.0,
        clamp_gate_max: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.clamp_linear_max = clamp_linear_max
        self.clamp_gate_max = clamp_gate_max

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value = self.up_proj(hidden_states)
        gate_input = self.gate_proj(hidden_states)
        if self.clamp_linear_max > 0:
            value = value.clamp(min=-self.clamp_linear_max, max=self.clamp_linear_max)
        if self.clamp_gate_max > 0:
            gate_input = gate_input.clamp(max=self.clamp_gate_max)
        gate = F.silu(gate_input)
        return self.down_proj(gate * value)


class DeepSeekMoEFFN(nn.Module):
    """
    Minimal but faithful DeepSeekMoE-style FFN.

    这个实现保留 DeepSeekMoE 的两个核心结构：
    - routed experts / 路由专家
    - shared experts / 共享专家

    The implementation is intentionally explicit rather than kernel-optimized
    so that routing behavior stays easy to inspect and extend.
    为了便于学习与后续扩展，这里优先保证路由过程清晰可读，而不是追求内核级优化。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_experts: int = 1,
        normalize_topk_prob: bool = True,
        router_type: str = "learned",
        clamp_linear_max: float = 0.0,
        clamp_gate_max: float = 0.0,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if num_shared_experts < 0:
            raise ValueError("num_shared_experts must be non-negative")

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.normalize_topk_prob = normalize_topk_prob
        self.router_type = router_type

        self.router = nn.Linear(hidden_size, num_experts, bias=False) if router_type == "learned" else None
        self.routed_experts = nn.ModuleList(
            [
                SwiGLU(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    clamp_linear_max=clamp_linear_max,
                    clamp_gate_max=clamp_gate_max,
                )
                for _ in range(num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                SwiGLU(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    clamp_linear_max=clamp_linear_max,
                    clamp_gate_max=clamp_gate_max,
                )
                for _ in range(num_shared_experts)
            ]
        )

    def _hash_router(
        self,
        flat_states: torch.Tensor,
        token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_ids is None:
            token_ids = torch.arange(flat_states.shape[0], device=flat_states.device, dtype=torch.long)
        else:
            token_ids = token_ids.reshape(-1).to(device=flat_states.device, dtype=torch.long)

        offsets = torch.arange(self.top_k, device=flat_states.device, dtype=torch.long)
        topk_indices = (token_ids.unsqueeze(-1) + offsets.unsqueeze(0)) % self.num_experts
        topk_weights = torch.full(
            (flat_states.shape[0], self.top_k),
            fill_value=1.0 / self.top_k,
            dtype=flat_states.dtype,
            device=flat_states.device,
        )
        router_probs = torch.zeros(
            flat_states.shape[0],
            self.num_experts,
            dtype=flat_states.dtype,
            device=flat_states.device,
        )
        router_probs.scatter_add_(-1, topk_indices, topk_weights)
        return router_probs, topk_weights, topk_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        if self.router_type == "hash":
            router_probs, topk_weights, topk_indices = self._hash_router(flat_states=flat_states, token_ids=token_ids)
        else:
            router_logits = self.router(flat_states)
            router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
            if self.normalize_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        routed_output = torch.zeros_like(flat_states)
        token_indices = torch.arange(flat_states.shape[0], device=flat_states.device)

        for expert_id, expert in enumerate(self.routed_experts):
            expert_mask = topk_indices == expert_id
            if not expert_mask.any():
                continue

            selected_tokens = token_indices.unsqueeze(-1).expand_as(expert_mask)[expert_mask]
            selected_weights = topk_weights[expert_mask].to(flat_states.dtype)
            expert_input = flat_states.index_select(0, selected_tokens)
            expert_output = expert(expert_input) * selected_weights.unsqueeze(-1)
            routed_output.index_add_(0, selected_tokens, expert_output)

        if self.shared_experts:
            shared_output = torch.zeros_like(flat_states)
            for expert in self.shared_experts:
                shared_output = shared_output + expert(flat_states)
        else:
            shared_output = torch.zeros_like(flat_states)

        combined_output = (shared_output + routed_output).view(*original_shape)

        expert_load = torch.bincount(topk_indices.reshape(-1), minlength=self.num_experts).to(router_probs.dtype)
        expert_fraction = expert_load / max(1, topk_indices.numel())
        router_fraction = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(expert_fraction * router_fraction)

        stats = {
            "moe_aux_loss": aux_loss,
            "moe_router_entropy": (-(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()),
            "moe_expert_fraction": expert_fraction.detach(),
            "moe_router_fraction": router_fraction.detach(),
            "moe_router_type": 1.0 if self.router_type == "hash" else 0.0,
        }
        return combined_output, stats


class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding cache.

    预先缓存 RoPE 的 cos / sin，避免每一步重复构造。
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(device=device),
            self.sin_cached[:seq_len].to(device=device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding to query/key.

    把旋转位置编码应用到 query / key 上。
    """

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)


def _repeat_kv(hidden_states: torch.Tensor, num_heads: int, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, num_key_value_groups, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_heads, seq_len, head_dim)


class CompressedAttentionBase(nn.Module):
    """
    Shared utilities for CSA/HCA-style approximate long-context attention.

    为 CSA / HCA 近似实现提供共用的压缩块构造与局部窗口工具。
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        attention_window_size: int,
        attention_compression_stride: int,
        attention_compressed_top_k: int,
        scale: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_window_size = attention_window_size
        self.attention_compression_stride = attention_compression_stride
        self.attention_compressed_top_k = attention_compressed_top_k
        self.scale = scale

    def _local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = query.shape[-2]
        device = query.device
        local_mask = self._build_window_mask(seq_len=seq_len, device=device)
        local_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            token_mask = attention_mask[:, None, None, :].bool()
        else:
            token_mask = None
        local_probs = _masked_softmax(local_scores, local_mask[None, None, :, :], query.dtype)
        if token_mask is not None:
            local_probs = local_probs * token_mask.to(local_probs.dtype)
            local_probs = local_probs / local_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.matmul(local_probs, value)

    def _build_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        distance = positions[:, None] - positions[None, :]
        return (distance >= 0) & (distance < self.attention_window_size)

    def _compress_blocks(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        sample_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch_size, _, seq_len, _ = key.shape
        stride = max(1, self.attention_compression_stride)
        num_blocks = math.ceil(seq_len / stride)
        pad_len = num_blocks * stride - seq_len
        if pad_len > 0:
            key_pad = F.pad(key, (0, 0, 0, pad_len))
            value_pad = F.pad(value, (0, 0, 0, pad_len))
            if attention_mask is not None:
                mask_pad = F.pad(attention_mask, (0, pad_len), value=0)
            else:
                mask_pad = None
            if sample_ids is not None:
                sample_ids_pad = F.pad(sample_ids, (0, pad_len), value=-1)
            else:
                sample_ids_pad = None
        else:
            key_pad = key
            value_pad = value
            mask_pad = attention_mask
            sample_ids_pad = sample_ids

        key_blocks = key_pad.view(batch_size, self.num_heads, num_blocks, stride, self.head_dim)
        value_blocks = value_pad.view(batch_size, self.num_heads, num_blocks, stride, self.head_dim)
        if sample_ids_pad is None:
            if mask_pad is not None:
                block_mask = mask_pad.view(batch_size, 1, num_blocks, stride, 1).to(key_blocks.dtype)
                denom = block_mask.sum(dim=3).clamp_min(1.0)
                compressed_key = (key_blocks * block_mask).sum(dim=3) / denom
                compressed_value = (value_blocks * block_mask).sum(dim=3) / denom
            else:
                compressed_key = key_blocks.mean(dim=3)
                compressed_value = value_blocks.mean(dim=3)
            return compressed_key, compressed_value, stride

        sample_blocks = sample_ids_pad.view(batch_size, num_blocks, stride)
        key_valid = mask_pad.view(batch_size, num_blocks, stride).bool() if mask_pad is not None else None
        token_positions = torch.arange(num_blocks * stride, device=key.device).view(num_blocks, stride)
        query_positions = torch.arange(seq_len, device=key.device)
        sample_match = sample_blocks[:, None, :, :] == sample_ids[:, :, None, None]
        causal_match = token_positions[None, None, :, :] <= query_positions[None, :, None, None]
        visibility = sample_match & causal_match
        if key_valid is not None:
            visibility = visibility & key_valid[:, None, :, :]
        block_mask = visibility[:, None, :, :, :].to(key_blocks.dtype)
        denom = block_mask.sum(dim=4).clamp_min(1.0)
        compressed_key = (key_blocks.unsqueeze(2) * block_mask.unsqueeze(-1)).sum(dim=4) / denom.unsqueeze(-1)
        compressed_value = (value_blocks.unsqueeze(2) * block_mask.unsqueeze(-1)).sum(dim=4) / denom.unsqueeze(-1)
        return compressed_key, compressed_value, stride


class CSAApproxAttention(CompressedAttentionBase):
    """
    CSA-inspired approximate attention.

    近似保留两个核心思想：
    - local sliding-window attention for nearby context
    - sparse selection over compressed distant memory blocks
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        index_query: torch.Tensor | None = None,
        index_key: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, seq_len, _ = query.shape
        device = query.device

        local_output = self._local_attention(query, key, value, attention_mask)
        if sample_ids is not None:
            sample_mask = sample_ids[:, None, :, None] == sample_ids[:, None, None, :]
            local_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            local_mask = self._build_window_mask(seq_len=seq_len, device=device)[None, None, :, :]
            local_probs = _masked_softmax(local_scores, local_mask & sample_mask, query.dtype)
            local_output = torch.matmul(local_probs, value)

        compressed_key, compressed_value, stride = self._compress_blocks(key, value, attention_mask, sample_ids=sample_ids)
        num_blocks = compressed_key.shape[-2]
        compressed_index_key = compressed_key if index_key is None else self._compress_blocks(index_key, index_key, attention_mask, sample_ids=sample_ids)[0]
        index_query = query if index_query is None else index_query
        if sample_ids is None:
            block_scores = torch.matmul(index_query, compressed_index_key.transpose(-2, -1)) * self.scale
        else:
            block_scores = (index_query.unsqueeze(-2) * compressed_index_key).sum(dim=-1) * self.scale
        query_block_ids = torch.arange(seq_len, device=device) // stride
        compressed_causal_mask = torch.arange(num_blocks, device=device)[None, :] <= query_block_ids[:, None]

        if self.attention_compressed_top_k < num_blocks:
            _, top_indices = torch.topk(
                block_scores,
                k=self.attention_compressed_top_k,
                dim=-1,
            )
            sparse_mask = torch.zeros_like(block_scores, dtype=torch.bool)
            sparse_mask.scatter_(-1, top_indices, True)
            compressed_mask = sparse_mask & compressed_causal_mask[None, None, :, :]
            block_scores = block_scores.masked_fill(~compressed_mask, torch.finfo(block_scores.dtype).min)
            compressed_probs = torch.softmax(block_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        else:
            compressed_probs = _masked_softmax(
                block_scores,
                compressed_causal_mask[None, None, :, :],
                query.dtype,
            )
        if sample_ids is None:
            compressed_output = torch.matmul(compressed_probs, compressed_value)
        else:
            compressed_output = (compressed_probs.unsqueeze(-1) * compressed_value).sum(dim=-2)
        return local_output + compressed_output


class HCAApproxAttention(CompressedAttentionBase):
    """
    HCA-inspired approximate attention.

    相比 CSA，这里做更重的压缩，并去掉 sparse top-k 选择，
    用一个更粗糙但更便宜的全压缩记忆通路来近似远程依赖。
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        index_query: torch.Tensor | None = None,
        index_key: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = query.shape[-2]
        device = query.device
        local_output = self._local_attention(query, key, value, attention_mask)
        if sample_ids is not None:
            sample_mask = sample_ids[:, None, :, None] == sample_ids[:, None, None, :]
            local_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            local_mask = self._build_window_mask(seq_len=seq_len, device=device)[None, None, :, :]
            local_probs = _masked_softmax(local_scores, local_mask & sample_mask, query.dtype)
            local_output = torch.matmul(local_probs, value)

        compressed_key, compressed_value, stride = self._compress_blocks(key, value, attention_mask, sample_ids=sample_ids)
        num_blocks = compressed_key.shape[-2]
        compressed_index_key = compressed_key if index_key is None else self._compress_blocks(index_key, index_key, attention_mask, sample_ids=sample_ids)[0]
        index_query = query if index_query is None else index_query
        if sample_ids is None:
            block_scores = torch.matmul(index_query, compressed_index_key.transpose(-2, -1)) * self.scale
        else:
            block_scores = (index_query.unsqueeze(-2) * compressed_index_key).sum(dim=-1) * self.scale
        query_block_ids = torch.arange(seq_len, device=device) // stride
        compressed_causal_mask = torch.arange(num_blocks, device=device)[None, :] <= query_block_ids[:, None]
        compressed_probs = _masked_softmax(
            block_scores,
            compressed_causal_mask[None, None, :, :],
            query.dtype,
        )
        if sample_ids is None:
            return local_output + torch.matmul(compressed_probs, compressed_value)
        return local_output + (compressed_probs.unsqueeze(-1) * compressed_value).sum(dim=-2)


class HyperResidualState(nn.Module):
    """
    Lightweight stateful approximation of manifold-constrained hyper-connections.

    这里不追求论文级一比一实现，而是保留两个核心学习点：
    - residual stream is widened into a parallel hyper stream
    - widened stream feeds back into the main stream every block
    """

    def __init__(self, hidden_size: int, width_multiplier: int, sinkhorn_steps: int = 3):
        super().__init__()
        if width_multiplier < 1:
            raise ValueError("width_multiplier must be >= 1")
        self.hidden_size = hidden_size
        self.width_multiplier = width_multiplier
        self.hyper_size = hidden_size * width_multiplier
        self.sinkhorn_steps = sinkhorn_steps
        self.expand = nn.Linear(hidden_size, self.hyper_size, bias=False)
        self.feedback = nn.Linear(self.hyper_size, hidden_size, bias=False)
        self.norm = RMSNorm(self.hyper_size)
        self.input_logits = nn.Parameter(torch.zeros(width_multiplier))
        self.output_logits = nn.Parameter(torch.zeros(width_multiplier))
        self.transition_logits = nn.Parameter(torch.zeros(width_multiplier, width_multiplier))

    def _constrain_transition(self) -> torch.Tensor:
        transition = torch.exp(self.transition_logits.float())
        for _ in range(self.sinkhorn_steps):
            transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            transition = transition / transition.sum(dim=-2, keepdim=True).clamp_min(1e-9)
        return transition.to(dtype=self.transition_logits.dtype)

    def init_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(hidden_states).view(*hidden_states.shape[:-1], self.width_multiplier, self.hidden_size)
        input_mix = torch.softmax(self.input_logits, dim=0).view(*([1] * (expanded.ndim - 2)), self.width_multiplier, 1)
        return (expanded * input_mix).reshape(*hidden_states.shape[:-1], self.hyper_size)

    def update(self, hyper_states: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hyper_states = hyper_states.view(*delta.shape[:-1], self.width_multiplier, self.hidden_size)
        delta_expanded = self.expand(delta).view(*delta.shape[:-1], self.width_multiplier, self.hidden_size)
        transition = self._constrain_transition()
        hyper_states = torch.einsum("ij,...jh->...ih", transition, hyper_states)
        input_mix = torch.softmax(self.input_logits, dim=0).view(*([1] * (delta_expanded.ndim - 2)), self.width_multiplier, 1)
        hyper_states = hyper_states + (delta_expanded * input_mix)
        normalized = self.norm(hyper_states.reshape(*delta.shape[:-1], self.hyper_size)).view(
            *delta.shape[:-1], self.width_multiplier, self.hidden_size
        )
        output_mix = torch.softmax(self.output_logits, dim=0).view(*([1] * (normalized.ndim - 2)), self.width_multiplier, 1)
        feedback_input = (normalized * output_mix).reshape(*delta.shape[:-1], self.hyper_size)
        feedback = self.feedback(feedback_input)
        hyper_states = hyper_states.reshape(*delta.shape[:-1], self.hyper_size)
        return hyper_states, feedback


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with grouped-query attention support.

    支持 GQA 的因果自注意力。
    This implementation keeps the math explicit for educational purposes.
    这里刻意把核心数学过程展开，便于学习每一步在做什么。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_base: float,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_sdpa: bool = False,
        attention_type: str = "causal",
        attention_window_size: int = 512,
        attention_compression_stride: int = 16,
        attention_compressed_top_k: int = 8,
        attention_qk_norm: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_sdpa = use_sdpa
        self.attention_type = attention_type
        self.attention_window_size = attention_window_size
        self.attention_compression_stride = attention_compression_stride
        self.attention_compressed_top_k = attention_compressed_top_k
        self.attention_qk_norm = attention_qk_norm

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.index_q_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.index_k_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.output_projection_groups = max(1, self.num_key_value_groups)
        group_hidden_size = hidden_size // self.output_projection_groups
        group_head_count = self.num_heads // self.output_projection_groups
        self.group_out_projs = nn.ModuleList(
            [nn.Linear(group_head_count * self.head_dim, group_hidden_size, bias=False) for _ in range(self.output_projection_groups)]
        )
        self.group_merge_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = attention_dropout
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
        )
        if attention_qk_norm:
            self.query_norm = RMSNorm(self.head_dim)
            self.key_norm = RMSNorm(self.head_dim)
        else:
            self.query_norm = None
            self.key_norm = None
        self.csa_attention = CSAApproxAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            attention_window_size=attention_window_size,
            attention_compression_stride=attention_compression_stride,
            attention_compressed_top_k=attention_compressed_top_k,
            scale=self.scale,
        )
        self.hca_attention = HCAApproxAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            attention_window_size=attention_window_size,
            attention_compression_stride=max(attention_compression_stride, attention_window_size),
            attention_compressed_top_k=attention_compressed_top_k,
            scale=self.scale,
        )

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def _run_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        index_query: torch.Tensor | None = None,
        index_key: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = query.shape[-2]
        if self.attention_type == "csa":
            return self.csa_attention(
                query,
                key,
                value,
                attention_mask,
                index_query=index_query,
                index_key=index_key,
                sample_ids=sample_ids,
            )
        if self.attention_type == "hca":
            return self.hca_attention(
                query,
                key,
                value,
                attention_mask,
                index_query=index_query,
                index_key=index_key,
                sample_ids=sample_ids,
            )

        if self.attention_type == "sliding_window":
            mask = self.csa_attention._build_window_mask(seq_len=seq_len, device=query.device)[None, None, :, :]
        else:
            mask = self._build_causal_mask(seq_len=seq_len, device=query.device)[None, None, :, :]

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        combined_mask = mask
        if sample_ids is not None:
            combined_mask = combined_mask & (sample_ids[:, None, :, None] == sample_ids[:, None, None, :])
        probs = _masked_softmax(scores, combined_mask, query.dtype)
        if attention_mask is not None and sample_ids is None:
            expanded = attention_mask[:, None, None, :].to(probs.dtype)
            probs = probs * expanded
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)
        return torch.matmul(probs, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        cos, sin = self.rope(seq_len=seq_len, device=hidden_states.device)
        query, key = apply_rotary(query, key, cos, sin)
        if self.query_norm is not None and self.key_norm is not None:
            query = self.query_norm(query)
            key = self.key_norm(key)

        key = _repeat_kv(key, num_heads=self.num_heads, num_key_value_groups=self.num_key_value_groups)
        value = _repeat_kv(value, num_heads=self.num_heads, num_key_value_groups=self.num_key_value_groups)
        index_query = self.index_q_proj(query)
        index_key = self.index_k_proj(key)

        if (
            self.attention_type == "causal"
            and self.use_sdpa
            and sample_ids is None
            and hasattr(F, "scaled_dot_product_attention")
        ):
            attn_mask = None
            if attention_mask is not None:
                expanded = attention_mask[:, None, None, :].to(dtype=query.dtype)
                attn_mask = (1.0 - expanded) * torch.finfo(query.dtype).min
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )
        else:
            attn_output = self._run_attention(
                query,
                key,
                value,
                attention_mask,
                index_query=index_query,
                index_key=index_key,
                sample_ids=sample_ids,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        if self.attention_type in {"csa", "hca"}:
            groups = []
            group_size = self.num_heads // self.output_projection_groups
            for group_idx, group_proj in enumerate(self.group_out_projs):
                group_states = attn_output[:, :, group_idx * group_size : (group_idx + 1) * group_size, :]
                group_states = group_states.reshape(batch_size, seq_len, group_size * self.head_dim)
                groups.append(group_proj(group_states))
            attn_output = torch.cat(groups, dim=-1)
            return self.group_merge_proj(self.dropout(attn_output))

        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(self.dropout(attn_output))


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block.

    预归一化 Transformer Block。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_base: float,
        rms_norm_eps: float,
        dropout: float,
        attention_dropout: float,
        use_sdpa: bool,
        attention_type: str = "causal",
        attention_window_size: int = 512,
        attention_compression_stride: int = 16,
        attention_compressed_top_k: int = 8,
        attention_qk_norm: bool = False,
        ffn_type: str = "dense",
        residual_mode: str = "standard",
        mhc_width_multiplier: int = 2,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_num_shared_experts: int = 1,
        moe_normalize_topk_prob: bool = True,
        moe_router_type: str = "learned",
        swiglu_clamp_linear_max: float = 0.0,
        swiglu_clamp_gate_max: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.residual_mode = residual_mode
        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sdpa=use_sdpa,
            attention_type=attention_type,
            attention_window_size=attention_window_size,
            attention_compression_stride=attention_compression_stride,
            attention_compressed_top_k=attention_compressed_top_k,
            attention_qk_norm=attention_qk_norm,
        )
        if residual_mode == "mhc":
            self.hyper_residual = HyperResidualState(
                hidden_size=hidden_size,
                width_multiplier=mhc_width_multiplier,
            )
        elif residual_mode == "standard":
            self.hyper_residual = None
        else:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")
        self.ffn_type = ffn_type
        if ffn_type == "dense":
            self.mlp = SwiGLU(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                clamp_linear_max=swiglu_clamp_linear_max,
                clamp_gate_max=swiglu_clamp_gate_max,
            )
        elif ffn_type == "moe":
            self.mlp = DeepSeekMoEFFN(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                num_shared_experts=moe_num_shared_experts,
                normalize_topk_prob=moe_normalize_topk_prob,
                router_type=moe_router_type,
                clamp_linear_max=swiglu_clamp_linear_max,
                clamp_gate_max=swiglu_clamp_gate_max,
            )
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hyper_states: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None]:
        attn_delta = self.attn(self.attn_norm(hidden_states), attention_mask=attention_mask, sample_ids=sample_ids)
        hidden_states = hidden_states + attn_delta
        if self.hyper_residual is not None:
            if hyper_states is None:
                hyper_states = self.hyper_residual.init_state(hidden_states)
            hyper_states, feedback = self.hyper_residual.update(hyper_states, attn_delta)
            hidden_states = hidden_states + feedback
        ffn_input = self.ffn_norm(hidden_states)
        stats: dict[str, Any] = {}
        if self.ffn_type == "moe":
            ffn_output, stats = self.mlp(ffn_input, token_ids=token_ids)
        else:
            ffn_output = self.mlp(ffn_input)
        hidden_states = hidden_states + ffn_output
        if self.hyper_residual is not None and hyper_states is not None:
            hyper_states, feedback = self.hyper_residual.update(hyper_states, ffn_output)
            hidden_states = hidden_states + feedback
            stats["mhc_hyper_rms"] = hyper_states.detach().float().pow(2).mean().sqrt()
        return hidden_states, stats, hyper_states

    def forward_checkpoint(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        hyper_states: torch.Tensor,
        token_ids: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_mask_or_none = attention_mask if attention_mask.numel() > 0 else None
        hyper_or_none = hyper_states if hyper_states.numel() > 0 else None
        hidden_states, stats, next_hyper = self.forward(
            hidden_states,
            attention_mask=attn_mask_or_none,
            hyper_states=hyper_or_none,
            token_ids=token_ids,
            sample_ids=sample_ids,
        )
        empty = hidden_states.new_empty(0)
        packed_hyper = next_hyper if next_hyper is not None else empty
        moe_aux = stats.get("moe_aux_loss", empty)
        moe_entropy = stats.get("moe_router_entropy", empty)
        router_fraction = stats.get("moe_router_fraction", empty)
        expert_fraction = stats.get("moe_expert_fraction", empty)
        mhc_rms = stats.get("mhc_hyper_rms", empty)
        return hidden_states, packed_hyper, moe_aux, moe_entropy, router_fraction, expert_fraction, mhc_rms
