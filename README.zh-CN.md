![MiniDeepSeek Logo](./MiniDeepSeek_logo.png)

本仓库包含一个面向大语言模型（LLM）技术学习与实践的轻量级项目 **MiniDeepSeek**。项目以学习和复现 **DeepSeek-V4** 的核心技术思想为主要目标，参考了 DeepSeek-V4 技术报告，并借鉴开源学习项目 [miniMind](https://github.com/jingyaogong/minimind/tree/master) 的部分实现思路进行开发。项目整体采用 **vibe coding** 的方式进行快速搭建、修改与迭代，强调“边实现、边理解、边实验”的学习过程。本项目并不追求完整复现工业级大模型系统，而是希望通过尽可能清晰、简洁、可运行的代码结构，降低大语言模型的学习门槛。

我们希望学习者能够：

- 从理解每一行代码开始学习 LLM；
- 理解现代大语言模型的核心模块与训练逻辑；
- 通过最小化实现快速验证自己的想法；
- 在项目基础上自由修改、扩展并开展进一步实验。

这个项目的目标是帮助更多人真正理解 LLM 的内部机制，而不仅仅停留在调用 API 的层面。

## 目录结构

```text
github_release_learning/
├── minideepseek/
├── scripts/
├── requirements.txt
├── README.md
└── README.zh-CN.md
```

## 环境安装

先安装依赖：

```bash
pip install -r requirements.txt
```

## 启动训练

> 当前启动命令只覆盖**预训练（pretrain）**阶段，后续我们将及时更新。  
> 虽然仓库里已经包含部分 `SFT` 相关脚本，但我们暂不把它作为正式训练入口来介绍；如果你是第一次学习这个仓库，建议先完整跑通下面的预训练流程。

训练顺序为：

1. 先把原始数据集 `jsonl` 文本数据预处理成训练可直接读取的缓存文件
2. 再准备一个本地训练配置文件，明确模型、数据和训练参数
3. 最后选择一种启动方式：
   - 从零开始直接训练
   - 基于已有 checkpoint 做一次 V4 路线验证训练

如果你第一次接触这个仓库，可以把整条链路理解成：

`原始文本 jsonl -> tokenizer 编码 -> 预训练缓存文件 -> DataLoader 切分序列 -> 模型训练 -> 日志与 checkpoint`

### 1. 预处理预训练数据

这一步的目的，是把原始文本转换成训练脚本可以高效读取的 token 缓存。  
训练脚本**不会直接读取原始 `jsonl` 文本**，而是依赖这一步生成的 `.bin`、`.json` 和样本边界索引文件。

假设你的预训练数据文件是：

```text
dataset/pretrain_t2t.jsonl
```

其中每一行应当是一个 JSON 对象，并至少包含一个 `text` 字段，例如：

```json
{"text": "今天我们来学习一个最小化语言模型训练项目。"}
{"text": "MiniDeepSeek 的目标是帮助学习者理解 LLM 的内部机制。"}
```

执行：

```bash
python3 scripts/preprocess_pretrain.py \
  --input dataset/pretrain_t2t.jsonl \
  --output-prefix dataset/.cache/pretrain_t2t
```

这个脚本内部主要做了几件事：

1. 逐行读取 `jsonl`
2. 使用仓库里的 tokenizer 将 `text` 编码成 token id 
3. 把所有 token 按顺序写入一个连续的二进制文件
4. 记录每条样本在 token 流中的结束位置，供 sample-level mask 使用
5. 保存一份元信息，供训练脚本自动找到数据文件

会生成：

- `dataset/.cache/pretrain_t2t.bin`
- `dataset/.cache/pretrain_t2t.json`
- `dataset/.cache/pretrain_t2t_sample_index.bin`

这三个文件的作用分别是：

- `pretrain_t2t.bin`：真正用于训练的数据本体，里面是连续写入的 token id
- `pretrain_t2t.json`：数据元信息，记录样本数、token 数、`bin` 路径、`sample_index` 路径等
- `pretrain_t2t_sample_index.bin`：记录每条样本的边界位置，训练时可用于 sample-level mask，避免 pack 后错误跨样本建模

如果你只想先确认预处理是否成功，可以打开 `dataset/.cache/pretrain_t2t.json`，重点检查：

- `task` 是否为 `pretrain`
- `num_samples` 是否大于 0
- `num_tokens` 是否大于 0
- `bin_path` 和 `sample_index_path` 是否指向刚生成的文件

### 2. 新建一个本地训练配置文件

这一步的目的，是把“本次训练要怎么跑”明确写成一个 JSON 配置，避免把模型结构、训练步数、batch size 等超参数硬编码在脚本里。

例如创建：

```text
local_pretrain_v4_route_24g.json
```

内容如下：

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

几组重点字段的解释：

- `tokenizer`
  - 定义训练和预处理使用的 tokenizer
  - 一般应与预处理阶段保持一致，否则 token 边界会不一致
- `data`
  - `cache_dir`：缓存数据目录
  - `max_seq_len`：训练时每个样本切出的序列长度
  - `task`：这里应明确写成 `pretrain`
- `model`
  - 定义模型结构，例如层数、隐藏维度、注意力类型，以及是否启用 `MoE`、`MTP` 等
- `optimizer`
  - 定义学习率、warmup、weight decay、梯度裁剪等优化策略
- `training`
  - `output_dir`：训练输出目录
  - `train_steps`：本次训练总步数
  - `batch_size`：单次前向/反向的 micro batch size
  - `grad_accum_steps`：梯度累计步数
  - `batch_size * grad_accum_steps` 可以理解为有效 batch size
  - `resume_from`：是否从已有 checkpoint 恢复；留空表示从零开始

如果你是第一次跑通流程，建议优先确认下面 4 个字段：

- `data.max_seq_len`：决定每个训练样本的序列长度
- `training.output_dir`：决定日志和 checkpoint 输出到哪里
- `training.train_steps`：决定这次只跑多长时间
- `training.resume_from`：决定是冷启动还是接着已有 checkpoint 继续训练

可以把前两步简单理解成：

- 第 1 步预处理生成的是“训练数据缓存”
- 第 2 步配置文件定义的是“训练规则”

### 3. 一键执行“预处理 + 训练验证”

如果你已经有一个可用的 warmup checkpoint，例如 Dense + MTP 阶段训练得到的 checkpoint，那么可以直接运行下面的一键脚本：

```bash
python3 scripts/run_pretrain_v4_route_with_preprocess.py \
  --config local_pretrain_v4_route_24g.json \
  --resume-from outputs/pretrain_dense_350m_mtp_d1_seq2048_stability/checkpoints/step_0010000.pt
```

这个脚本会自动：

1. 检查当前预处理产物是否已经包含 sample-level 边界索引
2. 如有必要则自动重做预处理
3. 启动当前 V4 路线学习版训练

补充说明：

- `--resume-from` 可以直接传一个 `.pt` checkpoint
- 如果传入的是一个记录最新 checkpoint 的 `.json` 文件，脚本也会自动解析出真正的 checkpoint 路径
- 如果你只想检查预处理是否能跑通，不想立即开始训练，可以额外加 `--skip-train`

例如，只检查预处理：

```bash
python3 scripts/run_pretrain_v4_route_with_preprocess.py \
  --config local_pretrain_v4_route_24g.json \
  --skip-train
```

### 4. 直接训练命令

如果你想把“预处理”和“训练”拆开执行，在处理好数据集后，可以调用训练脚本：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_pretrain.py \
  --config local_pretrain_v4_route_24g.json \
  --train-meta dataset/.cache/pretrain_t2t.json \
  --resume-from outputs/pretrain_dense_350m_mtp_d1_seq2048_stability/checkpoints/step_0010000.pt
```

这条命令里各参数的作用是：

- `--config`：训练配置文件，定义模型结构和训练超参数
- `--train-meta`：第 1 步预处理产生的元信息文件，训练脚本会从里面自动找到 `.bin` 数据和 `sample_index`
- `--resume-from`：从已有 checkpoint 继续训练；如果你要从零开始，可以去掉这个参数，或者把配置文件中的 `training.resume_from` 留空

也就是说，**从零开始训练**时，命令可以简化为：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_pretrain.py \
  --config local_pretrain_v4_route_24g.json \
  --train-meta dataset/.cache/pretrain_t2t.json
```

训练启动后，整体流程大致是：

1. 读取配置文件
2. 根据 `train-meta` 找到预处理后的 token 数据
3. 构建 `PackedPretrainDataset`
4. 按 `max_seq_len` 切分训练序列
5. 构建模型、优化器和 Trainer
6. 开始执行训练循环，并定期打印日志、保存 checkpoint

训练输出通常会写到配置文件的 `training.output_dir`，其中一般会包含：

- `model_summary.json`：模型结构摘要
- `train_log.jsonl`：逐步追加的训练日志
- `checkpoints/`：阶段性保存的 checkpoint
- `latest_checkpoint.json`：最近一次 checkpoint 的索引文件

## 致谢与参考

本项目在工程组织方式和学习导向上，**借鉴并参考了 miniMind 项目**。

同时，本项目实验中使用到的数据集和tokenizer，**均来自 miniMind 所开源的数据集和tokenizer配置**。

如果你想完整理解数据来源和训练背景，也建议同时参考 miniMind 项目。
