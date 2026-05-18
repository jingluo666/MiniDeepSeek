#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_latest_checkpoint(latest_json_path: str | Path) -> str:
    payload = load_json(latest_json_path)
    checkpoint_path = payload.get("latest_checkpoint", "")
    if not checkpoint_path:
        raise ValueError(f"No latest_checkpoint found in {latest_json_path}")
    return checkpoint_path


def meta_has_sample_index(meta_path: str | Path) -> bool:
    path = Path(meta_path)
    if not path.exists():
        return False
    payload = load_json(path)
    sample_index_path = payload.get("sample_index_path", "")
    if not sample_index_path:
        return False
    return Path(sample_index_path).exists()


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def ensure_preprocessed_meta(args: argparse.Namespace, env: dict[str, str]) -> str:
    train_meta_path = Path(args.train_meta)
    if not args.force_preprocess and meta_has_sample_index(train_meta_path):
        print(f"Using existing preprocessed meta with sample index: {train_meta_path}", flush=True)
        return str(train_meta_path)

    print(
        "Pretraining cache is missing sample-level index metadata, or force refresh was requested. "
        "Rebuilding preprocessing artifacts now.",
        flush=True,
    )
    preprocess_command = [
        sys.executable,
        "scripts/preprocess_pretrain.py",
        "--input",
        args.input,
        "--output-prefix",
        args.output_prefix,
    ]
    if args.no_progress:
        preprocess_command.append("--no-progress")
    preprocess_command.extend(
        [
            "--tokenizer-kind",
            args.tokenizer_kind,
            "--tokenizer-path",
            args.tokenizer_path,
            "--tokenizer-config-path",
            args.tokenizer_config_path,
            "--progress-update-interval",
            str(args.progress_update_interval),
        ]
    )
    run_command(preprocess_command, env=env)
    return str(train_meta_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-click verification script for the current MiniDeepseek V4 learning route: "
            "preprocess pretraining data with sample-level boundaries and then launch the V4-route training stage."
        )
    )
    parser.add_argument("--input", default="dataset/pretrain_t2t.jsonl", help="Input pretraining jsonl.")
    parser.add_argument(
        "--output-prefix",
        default="dataset/.cache/pretrain_t2t",
        help="Output prefix for preprocessed cache artifacts.",
    )
    parser.add_argument(
        "--train-meta",
        default="dataset/.cache/pretrain_t2t.json",
        help="Expected preprocessing meta json written from --output-prefix.",
    )
    parser.add_argument(
        "--config",
        default="configs/pretrain_v4_route_350m_seq512_24g.json",
        help="Training config used to verify the current code and data-path changes.",
    )
    parser.add_argument(
        "--resume-from",
        default="outputs/pretrain_dense_350m_mtp_d1_seq2048_stability/checkpoints/step_0010000.pt",
        help="Checkpoint used to warm-start the V4 route validation run.",
    )
    parser.add_argument("--tokenizer-kind", default="hf", choices=["hf", "byte"], help="Tokenizer backend.")
    parser.add_argument("--tokenizer-path", default="minideepseek/tokenizer.json", help="Path to tokenizer.json.")
    parser.add_argument(
        "--tokenizer-config-path",
        default="minideepseek/tokenizer_config.json",
        help="Path to tokenizer_config.json.",
    )
    parser.add_argument(
        "--progress-update-interval",
        type=int,
        default=1000,
        help="How often preprocessing prints detailed progress stats.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Always rebuild the pretraining cache even if sample-level metadata already exists.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable preprocessing progress display.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only run preprocessing checks/build and stop before training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_meta = ensure_preprocessed_meta(args, env=env)

    payload = load_json(train_meta)
    print(
        json.dumps(
            {
                "event": "pretrain_meta_ready",
                "train_meta": train_meta,
                "bin_path": payload.get("bin_path", ""),
                "sample_index_path": payload.get("sample_index_path", ""),
                "num_samples": payload.get("num_samples", 0),
                "num_tokens": payload.get("num_tokens", 0),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.skip_train:
        print("\nSkipping training as requested.", flush=True)
        return

    resume_from = args.resume_from
    if resume_from.endswith(".json"):
        resume_from = load_latest_checkpoint(resume_from)

    train_command = [
        sys.executable,
        "scripts/train_pretrain.py",
        "--config",
        args.config,
        "--train-meta",
        train_meta,
        "--resume-from",
        resume_from,
    ]
    print(
        f"\nLaunching V4-route verification run with config: {args.config}\n"
        f"Resume from: {resume_from}",
        flush=True,
    )
    run_command(train_command, env=env)

    print("\nPreprocess + V4-route verification run finished successfully.", flush=True)


if __name__ == "__main__":
    main()
