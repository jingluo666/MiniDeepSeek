from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TokenizerConfig:
    """
    Tokenizer settings.

    分词器配置。
    """

    kind: str = "byte"
    tokenizer_path: str = "minideepseek/tokenizer.json"
    tokenizer_config_path: str = "minideepseek/tokenizer_config.json"
    add_bos: bool = True
    add_eos: bool = True


@dataclass
class DataConfig:
    """
    Data pipeline settings.

    数据处理配置。
    """

    train_path: str = ""
    valid_path: str = ""
    cache_dir: str = "dataset/.cache"
    max_seq_len: int = 2048
    num_workers: int = 2
    task: str = "pretrain"


@dataclass
class ModelConfig:
    """
    Dense Transformer baseline config.

    Dense Transformer 基线模型配置。
    """

    vocab_size: int = 260
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4
    max_position_embeddings: int = 16384
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    use_sdpa: bool = False
    attention_type: str = "causal"
    attention_window_size: int = 512
    attention_compression_stride: int = 16
    attention_compressed_top_k: int = 8
    attention_qk_norm: bool = False
    tie_word_embeddings: bool = True
    residual_mode: str = "standard"
    mhc_num_layers: int = 0
    mhc_width_multiplier: int = 2
    mtp_depth: int = 0
    mtp_loss_weight: float = 0.1
    ffn_type: str = "dense"
    moe_num_layers: int = 0
    moe_hash_num_layers: int = 0
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_num_shared_experts: int = 1
    moe_normalize_topk_prob: bool = True
    moe_aux_loss_weight: float = 0.0
    swiglu_clamp_linear_max: float = 0.0
    swiglu_clamp_gate_max: float = 0.0


@dataclass
class OptimizerConfig:
    """
    Optimizer and scheduler settings.

    优化器与学习率调度配置。
    """

    kind: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_update_rms_target: float = 0.18


@dataclass
class TrainingConfig:
    """
    Training runtime settings.

    训练运行时配置。
    """

    seed: int = 42
    output_dir: str = "outputs/pretrain_dense"
    train_steps: int = 1000
    eval_interval: int = 100
    log_interval: int = 10
    save_interval: int = 200
    batch_size: int = 2
    grad_accum_steps: int = 16
    amp_dtype: str = "bfloat16"
    device: str = "cuda"
    compile_model: bool = False
    gradient_checkpointing: bool = False
    resume_from: str = ""
    keep_last_k_checkpoints: int = 3
    show_progress_bar: bool = True


def _section(payload: dict[str, Any], key: str, cls: Any) -> Any:
    return cls(**payload.get(key, {}))


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON experiment file into dataclasses.

    将 JSON 实验配置文件读入 dataclass，避免命令行参数过多。
    """

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "tokenizer": _section(payload, "tokenizer", TokenizerConfig),
        "data": _section(payload, "data", DataConfig),
        "model": _section(payload, "model", ModelConfig),
        "optimizer": _section(payload, "optimizer", OptimizerConfig),
        "training": _section(payload, "training", TrainingConfig),
        "raw": payload,
    }
