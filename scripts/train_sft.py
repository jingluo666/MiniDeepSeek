#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

# Ensure the project root is importable when running `python scripts/...`.
# 当使用 `python scripts/...` 直接执行脚本时，主动把项目根目录加入 sys.path，
# 避免出现 `ModuleNotFoundError: minideepseek`。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from minideepseek.config import load_experiment_config
from minideepseek.data import IndexedSFTDataset, sft_collate_fn
from minideepseek.models import DenseTransformerLM
from minideepseek.tokenizer import build_tokenizer
from minideepseek.trainer import Trainer
from minideepseek.utils import ensure_dir, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a dense baseline on supervised fine-tuning data.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--train-meta", required=True, help="Path to preprocessed SFT meta json.")
    parser.add_argument("--valid-meta", default="", help="Optional validation meta json.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    set_seed(config["training"].seed)
    tokenizer = build_tokenizer(config["tokenizer"])

    model = DenseTransformerLM(config["model"])
    output_dir = ensure_dir(config["training"].output_dir)
    save_json(output_dir / "model_summary.json", model.summary())

    train_dataset = IndexedSFTDataset(args.train_meta, pad_id=tokenizer.pad_id)
    valid_dataset = IndexedSFTDataset(args.valid_meta, pad_id=tokenizer.pad_id) if args.valid_meta else None
    collate = partial(sft_collate_fn, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].batch_size,
        shuffle=True,
        num_workers=config["data"].num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=config["training"].batch_size,
            shuffle=False,
            num_workers=config["data"].num_workers,
            pin_memory=True,
            collate_fn=collate,
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
            "task": "sft",
            "train_meta": args.train_meta,
            "valid_meta": args.valid_meta,
            "global_batch_size": config["training"].batch_size * config["training"].grad_accum_steps,
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
