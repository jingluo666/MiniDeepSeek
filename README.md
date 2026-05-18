# MiniDeepSeek

![MiniDeepSeek Logo](./MiniDeepSeek_logo.png)

This repository contains **MiniDeepSeek**, a lightweight project for learning and hands-on practice with large language models (LLMs).  
Its primary goal is to study and reproduce the core technical ideas of **DeepSeek-V4**. The implementation is informed by the DeepSeek-V4 technical report and also draws on selected design ideas from the open-source learning project [miniMind](https://github.com/jingyaogong/minimind/tree/master).

The project has been built and iterated in a **vibe coding** style, emphasizing a learning process centered on implementation, understanding, and experimentation.  
Rather than aiming for a full industrial-grade reproduction of a large-scale model system, this project focuses on keeping the codebase as clear, compact, and runnable as possible, so that the barrier to learning LLM internals remains low.

We hope learners will be able to:

- start learning LLMs by understanding the code line by line;
- understand the core components and training logic of modern large language models;
- validate their own ideas quickly through minimal implementations;
- freely modify, extend, and experiment on top of the project.

The goal of this project is to help more people genuinely understand how LLMs work internally, rather than stopping at API-level usage.

## Project Layout

```text
github_release_learning/
├── minideepseek/
├── scripts/
├── requirements.txt
├── README.md
└── README.zh-CN.md
```

## Environment Setup

Install dependencies first:

```bash
pip install -r requirements.txt
```

## Training

> The training commands documented here currently cover **pretraining only**. We will expand this section in future updates.  
> Although the repository already contains some `SFT`-related scripts, they are not yet presented here as an official training entry point. If you are new to this repository, it is best to first run through the complete pretraining workflow below.

The recommended workflow is:

1. Preprocess the raw dataset in `jsonl` format into cached files that can be consumed directly by training.
2. Prepare a local training configuration file that clearly defines the model, data, and training hyperparameters.
3. Choose one of the following launch modes:
   - train from scratch;
   - run a V4-route verification stage from an existing checkpoint.

If this is your first time working with this repository, you can think of the full pipeline as:

`raw jsonl text -> tokenizer encoding -> pretraining cache artifacts -> DataLoader sequence packing -> model training -> logs and checkpoints`

### 1. Preprocess pretraining data

The purpose of this step is to convert raw text into tokenized cache artifacts that the training script can load efficiently.  
The training script does **not** read the original `jsonl` text directly. Instead, it relies on the `.bin`, `.json`, and sample-boundary index files generated here.

Assume your pretraining dataset file is:

```text
dataset/pretrain_t2t.jsonl
```

Each line should be a JSON object containing at least a `text` field, for example:

```json
{"text": "Today we will study a minimal language model training project."}
{"text": "The goal of MiniDeepSeek is to help learners understand the internal mechanics of LLMs."}
```

Run:

```bash
python3 scripts/preprocess_pretrain.py \
  --input dataset/pretrain_t2t.jsonl \
  --output-prefix dataset/.cache/pretrain_t2t
```

Internally, this script mainly performs the following operations:

1. Reads the `jsonl` file line by line.
2. Encodes the `text` field into token IDs using the repository tokenizer.
3. Writes all token IDs sequentially into a contiguous binary file.
4. Records the end offset of each sample in the token stream for sample-level masking.
5. Saves metadata so that the training script can automatically locate the generated artifacts.

It will generate:

- `dataset/.cache/pretrain_t2t.bin`
- `dataset/.cache/pretrain_t2t.json`
- `dataset/.cache/pretrain_t2t_sample_index.bin`

These files serve different purposes:

- `pretrain_t2t.bin`: the actual training payload, containing the token IDs stored as a contiguous stream.
- `pretrain_t2t.json`: dataset metadata, including the number of samples, the number of tokens, the `bin` path, and the `sample_index` path.
- `pretrain_t2t_sample_index.bin`: sample boundary offsets used for sample-level masking, which helps prevent incorrect cross-sample modeling after packing.

If you only want to verify that preprocessing completed successfully, open `dataset/.cache/pretrain_t2t.json` and check the following fields:

- `task` should be `pretrain`
- `num_samples` should be greater than 0
- `num_tokens` should be greater than 0
- `bin_path` and `sample_index_path` should point to the newly generated files

### 2. Create a local training configuration file

The purpose of this step is to define exactly how the training run should be executed in a JSON file, rather than hard-coding model structure, step counts, batch size, and other hyperparameters inside the scripts.

For example, create:

```text
local_pretrain_v4_route_24g.json
```

with the following content:

```json
{
  "tokenizer": {
    "kind": "hf",
    "tokenizer_path": "minideepseek/tokenizer.json",
    "tokenizer_config_path": "minideepseek/tokenizer_config.json",
    "add_bos": false,
    "add_eos": false
  },
  "data": {
    "cache_dir": "dataset/.cache",
    "max_seq_len": 512,
    "num_workers": 2,
    "task": "pretrain"
  },
  "model": {
    "vocab_size": 6400,
    "hidden_size": 1024,
    "intermediate_size": 2816,
    "num_layers": 24,
    "num_heads": 16,
    "num_kv_heads": 4,
    "max_position_embeddings": 16384,
    "rope_base": 10000.0,
    "rms_norm_eps": 1e-05,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "use_sdpa": false,
    "attention_type": "compressed",
    "attention_window_size": 256,
    "attention_compression_stride": 16,
    "attention_compressed_top_k": 8,
    "attention_qk_norm": true,
    "tie_word_embeddings": true,
    "residual_mode": "mhc",
    "mhc_num_layers": 8,
    "mhc_width_multiplier": 2,
    "mtp_depth": 1,
    "mtp_loss_weight": 0.2,
    "ffn_type": "moe",
    "moe_num_layers": 8,
    "moe_hash_num_layers": 2,
    "moe_num_experts": 4,
    "moe_top_k": 1,
    "moe_num_shared_experts": 1,
    "moe_normalize_topk_prob": true,
    "moe_aux_loss_weight": 0.01,
    "swiglu_clamp_linear_max": 10.0,
    "swiglu_clamp_gate_max": 10.0
  },
  "optimizer": {
    "kind": "hybrid_muon",
    "lr": 0.000015,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-08,
    "warmup_steps": 100,
    "min_lr_ratio": 0.1,
    "grad_clip": 1.0,
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "muon_update_rms_target": 0.18
  },
  "training": {
    "seed": 42,
    "output_dir": "outputs/pretrain_v4_route_350m_seq512_24g",
    "train_steps": 1000,
    "eval_interval": 200,
    "log_interval": 10,
    "save_interval": 200,
    "batch_size": 1,
    "grad_accum_steps": 16,
    "amp_dtype": "bfloat16",
    "device": "cuda",
    "compile_model": false,
    "gradient_checkpointing": false,
    "resume_from": "",
    "keep_last_k_checkpoints": 3,
    "show_progress_bar": true
  }
}
```

Key configuration groups to pay attention to:

- `tokenizer`
  - Defines the tokenizer used in both preprocessing and training.
  - It should generally remain consistent with the preprocessing stage; otherwise, token boundaries will differ.
- `data`
  - `cache_dir`: directory containing cached dataset artifacts.
  - `max_seq_len`: sequence length used when slicing training samples.
  - `task`: should be explicitly set to `pretrain`.
- `model`
  - Defines the model architecture, such as layer count, hidden size, attention type, and whether `MoE`, `MTP`, and related components are enabled.
- `optimizer`
  - Defines the optimization strategy, including learning rate, warmup, weight decay, and gradient clipping.
- `training`
  - `output_dir`: output directory for logs and checkpoints.
  - `train_steps`: total number of training steps for this run.
  - `batch_size`: micro-batch size per forward/backward pass.
  - `grad_accum_steps`: number of gradient accumulation steps.
  - `batch_size * grad_accum_steps` can be interpreted as the effective batch size.
  - `resume_from`: whether to resume from an existing checkpoint; leave it empty to start from scratch.

If this is your first end-to-end run, prioritize checking these four fields:

- `data.max_seq_len`: determines the sequence length of each training sample
- `training.output_dir`: determines where logs and checkpoints will be written
- `training.train_steps`: determines how long the run will last
- `training.resume_from`: determines whether the run starts cold or continues from an existing checkpoint

You can think of the first two steps as:

- Step 1 generates the training data cache.
- Step 2 defines the training rules.

### 3. Run one-click preprocessing and training verification

If you already have a usable warmup checkpoint, for example one produced by a Dense + MTP stage, you can run the following one-click script:

```bash
python3 scripts/run_pretrain_v4_route_with_preprocess.py \
  --config local_pretrain_v4_route_24g.json \
  --resume-from outputs/pretrain_dense_350m_mtp_d1_seq2048_stability/checkpoints/step_0010000.pt
```

This script will automatically:

1. check whether the current preprocessing artifacts already include sample-level boundary indices;
2. rebuild preprocessing artifacts if necessary;
3. launch the current V4-route learning training run.

Additional notes:

- `--resume-from` can point directly to a `.pt` checkpoint.
- If you pass a `.json` file that records the latest checkpoint, the script will resolve the actual checkpoint path automatically.
- If you only want to verify preprocessing and do not want to start training immediately, add `--skip-train`.

For example, to validate preprocessing only:

```bash
python3 scripts/run_pretrain_v4_route_with_preprocess.py \
  --config local_pretrain_v4_route_24g.json \
  --skip-train
```

### 4. Direct training command

If you prefer to run preprocessing and training as separate steps, you can invoke the training script directly after preparing the dataset:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_pretrain.py \
  --config local_pretrain_v4_route_24g.json \
  --train-meta dataset/.cache/pretrain_t2t.json \
  --resume-from outputs/pretrain_dense_350m_mtp_d1_seq2048_stability/checkpoints/step_0010000.pt
```

The command-line arguments mean:

- `--config`: the training configuration file that defines model structure and training hyperparameters.
- `--train-meta`: the metadata file produced in Step 1. The training script reads it to locate the `.bin` data and the `sample_index`.
- `--resume-from`: resumes training from an existing checkpoint. If you want to train from scratch, remove this argument or leave `training.resume_from` empty in the config.

In other words, to **train from scratch**, the command can be simplified to:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_pretrain.py \
  --config local_pretrain_v4_route_24g.json \
  --train-meta dataset/.cache/pretrain_t2t.json
```

Once training starts, the overall flow is roughly:

1. Load the configuration file.
2. Resolve the preprocessed token data from `train-meta`.
3. Build `PackedPretrainDataset`.
4. Slice training sequences according to `max_seq_len`.
5. Construct the model, optimizer, and trainer.
6. Enter the training loop, periodically printing logs and saving checkpoints.

Training outputs are typically written to `training.output_dir` in the config, and usually include:

- `model_summary.json`: a summary of the model structure
- `train_log.jsonl`: appended step-by-step training logs
- `checkpoints/`: periodically saved checkpoints
- `latest_checkpoint.json`: an index file pointing to the most recent checkpoint

## Acknowledgements and References

This project borrows from **miniMind** in both engineering organization and its learning-oriented design philosophy.

The datasets and tokenizer configuration used in the experiments are **both derived from the datasets and tokenizer configuration open-sourced by miniMind**.

If you would like a fuller understanding of the data sources and training background, we also recommend referring to the miniMind project.
