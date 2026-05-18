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
        description="Run the Dense+MTP stability stage and then automatically hand off to the Hybrid MoE+MTP stage."
    )
    parser.add_argument(
        "--train-meta",
        default="dataset/.cache/pretrain_t2t.json",
        help="Preprocessed pretraining meta json.",
    )
    parser.add_argument(
        "--dense-latest",
        default="outputs/pretrain_dense_350m_stage3_seq2048/latest_checkpoint.json",
        help="Dense baseline checkpoint pointer used to start the Dense+MTP stage.",
    )
    parser.add_argument(
        "--mtp-config",
        default="configs/pretrain_dense_350m_mtp_d1_seq2048.json",
        help="Config for the Dense+MTP stability stage.",
    )
    parser.add_argument(
        "--mtp-latest",
        default="outputs/pretrain_dense_350m_mtp_d1_seq2048_stability/latest_checkpoint.json",
        help="Output checkpoint pointer expected from the Dense+MTP stage.",
    )
    parser.add_argument(
        "--hybrid-config",
        default="configs/pretrain_hybrid_moe_mtp_350m_seq1024.json",
        help="Config for the Hybrid Dense/MoE + MTP stage.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dense_checkpoint = load_latest_checkpoint(args.dense_latest)
    run_stage(args.mtp_config, args.train_meta, dense_checkpoint)

    mtp_checkpoint = load_latest_checkpoint(args.mtp_latest)
    run_stage(args.hybrid_config, args.train_meta, mtp_checkpoint)

    print("\nDense+MTP -> Hybrid MoE+MTP mainline finished successfully.", flush=True)


if __name__ == "__main__":
    main()
