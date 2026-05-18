from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {} if default is None else default
    return json.loads(file_path.read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """
    Append one structured log record to a jsonl file.

    向 jsonl 文件追加一条结构化日志，方便后续画图或复盘。
    """

    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    """
    Seed Python and torch if available.

    设置随机种子，尽量保证实验可复现。
    """

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_ratio: float) -> float:
    """
    Standard warmup + cosine decay scheduler.

    标准的 warmup + cosine 衰减学习率。
    """

    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


class SimpleTimer:
    """
    Small helper for throughput logging.

    轻量计时器，用于打印吞吐与训练速度。
    """

    def __init__(self) -> None:
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time


def format_seconds(seconds: float) -> str:
    """
    Convert raw seconds into a compact human-readable string.

    将秒数格式化成更容易阅读的字符串，便于显示 ETA。
    """

    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def estimate_finish_time(remaining_seconds: float) -> str:
    """
    Estimate the wall-clock finish time in UTC.

    估算训练完成的 UTC 时间，方便判断还要等多久。
    """

    finish_ts = datetime.now(timezone.utc).timestamp() + max(0.0, remaining_seconds)
    finish_dt = datetime.fromtimestamp(finish_ts, tz=timezone.utc)
    return finish_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
