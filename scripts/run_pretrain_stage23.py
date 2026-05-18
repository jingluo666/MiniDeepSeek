#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Ensure the project root is importable when running `python scripts/...`.
# 当使用 `python scripts/...` 直接执行脚本时，主动把项目根目录加入 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_latest_checkpoint(latest_json_path: str | Path) -> str:
    """
    Read the latest checkpoint pointer written by a previous stage.

    读取上一阶段写出的最新 checkpoint 指针文件。
    """

    payload = json.loads(Path(latest_json_path).read_text(encoding="utf-8"))
    checkpoint_path = payload.get("latest_checkpoint", "")
    if not checkpoint_path:
        raise ValueError(f"No latest_checkpoint found in {latest_json_path}")
    return checkpoint_path


def run_stage(config_path: str, train_meta: str, resume_from: str) -> None:
    """
    Launch one training stage and stream logs to the terminal.

    启动一个训练阶段，并把日志直接透传到终端。
    """

    command = [
        sys.executable,
        "scripts/train_pretrain.py",
        "--config",
        config_path,
        "--train-meta",
        train_meta,
        "--resume-from",
        resume_from,
    ]
    print(f"\n=== Running stage with config: {config_path} ===", flush=True)
    print(f"Resume from: {resume_from}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stage 2 and stage 3 of the 350M pretraining curriculum with automatic checkpoint handoff."
    )
    parser.add_argument(
        "--train-meta",
        default="dataset/.cache/pretrain_t2t.json",
        help="Preprocessed pretraining meta json.",
    )
    parser.add_argument(
        "--stage1-latest",
        default="outputs/pretrain_dense_350m_stage1_seq512/latest_checkpoint.json",
        help="Path to stage 1 latest_checkpoint.json.",
    )
    parser.add_argument(
        "--stage2-config",
        default="configs/pretrain_dense_350m_seq1024.json",
        help="Config for stage 2.",
    )
    parser.add_argument(
        "--stage3-config",
        default="configs/pretrain_dense_350m_seq2048.json",
        help="Config for stage 3.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    stage1_checkpoint = load_latest_checkpoint(args.stage1_latest)
    run_stage(args.stage2_config, args.train_meta, stage1_checkpoint)

    stage2_latest_json = Path("outputs/pretrain_dense_350m_stage2_seq1024/latest_checkpoint.json")
    stage2_checkpoint = load_latest_checkpoint(stage2_latest_json)
    run_stage(args.stage3_config, args.train_meta, stage2_checkpoint)

    print("\nAll remaining curriculum stages finished successfully.", flush=True)


if __name__ == "__main__":
    main()
