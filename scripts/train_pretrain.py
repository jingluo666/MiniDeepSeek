#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable when running `python scripts/...`.
# 当使用 `python scripts/...` 直接执行脚本时，主动把项目根目录加入 sys.path，
# 避免出现 `ModuleNotFoundError: minideepseek`。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from minideepseek.config import load_experiment_config
from minideepseek.data import PackedPretrainDataset, load_pretrain_meta
from minideepseek.models import DenseTransformerLM
from minideepseek.trainer import Trainer
from minideepseek.utils import ensure_dir, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a dense baseline for autoregressive pretraining.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--train-bin", default="", help="Tokenized training .bin file.")
    parser.add_argument("--train-meta", default="", help="Optional preprocessing meta json. If set, train-bin can be omitted.")
    parser.add_argument("--valid-bin", default="", help="Optional tokenized validation .bin file.")
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional checkpoint path that overrides training.resume_from in the config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    if args.resume_from:
        config["training"].resume_from = args.resume_from
    set_seed(config["training"].seed)

    train_bin_path = args.train_bin
    pretrain_meta = {}
    if args.train_meta:
        pretrain_meta = load_pretrain_meta(args.train_meta)
        train_bin_path = str(pretrain_meta["bin_path"])
    if not train_bin_path:
        raise ValueError("Either --train-bin or --train-meta must be provided.")

    model = DenseTransformerLM(config["model"])
    output_dir = ensure_dir(config["training"].output_dir)
    save_json(output_dir / "model_summary.json", model.summary())

    sample_index_path = str(pretrain_meta.get("sample_index_path", "")) if pretrain_meta else ""
    train_dataset = PackedPretrainDataset(
        train_bin_path,
        seq_len=config["data"].max_seq_len,
        sample_index_path=sample_index_path,
    )
    valid_dataset = (
        PackedPretrainDataset(
            args.valid_bin,
            seq_len=config["data"].max_seq_len,
            sample_index_path=sample_index_path,
        )
        if args.valid_bin
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].batch_size,
        shuffle=True,
        num_workers=config["data"].num_workers,
        pin_memory=True,
    )
    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=config["training"].batch_size,
            shuffle=False,
            num_workers=config["data"].num_workers,
            pin_memory=True,
        )
        if valid_dataset
        else None
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer_config=config["optimizer"],
        training_config=config["training"],
        run_metadata={
            "task": "pretrain",
            "train_bin": train_bin_path,
            "valid_bin": args.valid_bin,
            "seq_len": config["data"].max_seq_len,
            "dataset_meta": pretrain_meta,
            "num_train_sequences": len(train_dataset),
            "num_valid_sequences": len(valid_dataset) if valid_dataset else 0,
            "global_batch_size": config["training"].batch_size * config["training"].grad_accum_steps,
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
