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
from minideepseek.data import preprocess_sft_jsonl
from minideepseek.tokenizer import build_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess conversation jsonl for SFT.")
    parser.add_argument("--input", required=True, help="Path to conversation jsonl.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix without extension.")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sample length after tokenization.")
    parser.add_argument("--tokenizer-kind", default="hf", choices=["hf", "byte"], help="Tokenizer backend.")
    parser.add_argument("--tokenizer-path", default="minideepseek/tokenizer.json", help="Path to tokenizer.json.")
    parser.add_argument(
        "--tokenizer-config-path",
        default="minideepseek/tokenizer_config.json",
        help="Path to tokenizer_config.json.",
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
    meta = preprocess_sft_jsonl(args.input, args.output_prefix, tokenizer, max_seq_len=args.max_seq_len)
    print({"meta_path": f"{args.output_prefix}.json", "num_samples": meta["num_samples"]})


if __name__ == "__main__":
    main()
