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

from minideepseek.config import TokenizerConfig
from minideepseek.data import preprocess_pretrain_jsonl
from minideepseek.tokenizer import build_tokenizer


def count_jsonl_lines(path: str | Path) -> int:
    """
    Count lines in a large jsonl file efficiently.

    高效统计大 jsonl 文件的总行数，用于进度条总量估计。
    """

    count = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess pretraining jsonl into flat token stream.")
    parser.add_argument("--input", required=True, help="Path to a jsonl file containing {'text': ...} rows.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix without extension.")
    parser.add_argument("--tokenizer-kind", default="hf", choices=["hf", "byte"], help="Tokenizer backend.")
    parser.add_argument("--tokenizer-path", default="minideepseek/tokenizer.json", help="Path to tokenizer.json.")
    parser.add_argument(
        "--tokenizer-config-path",
        default="minideepseek/tokenizer_config.json",
        help="Path to tokenizer_config.json.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar and periodic progress logs.",
    )
    parser.add_argument(
        "--progress-update-interval",
        type=int,
        default=1000,
        help="How often to refresh detailed progress statistics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = build_tokenizer(
        TokenizerConfig(
            kind=args.tokenizer_kind,
            tokenizer_path=args.tokenizer_path,
            tokenizer_config_path=args.tokenizer_config_path,
            add_bos=False,
            add_eos=False,
        )
    )
    total_lines = count_jsonl_lines(args.input)
    print(
        f"Preparing to preprocess {total_lines} samples from {args.input} using {tokenizer.__class__.__name__}.",
        flush=True,
    )
    meta = preprocess_pretrain_jsonl(
        args.input,
        args.output_prefix,
        tokenizer,
        total_lines=total_lines,
        show_progress=not args.no_progress,
        progress_update_interval=args.progress_update_interval,
    )
    print(meta)


if __name__ == "__main__":
    main()
