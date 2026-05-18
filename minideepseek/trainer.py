from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import OptimizerConfig, TrainingConfig
from .optim import build_optimizer, optimizer_summary
from .utils import (
    SimpleTimer,
    append_jsonl,
    cosine_lr,
    ensure_dir,
    estimate_finish_time,
    format_seconds,
    load_json,
    save_json,
)


def resolve_amp_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[name.lower()]


class Trainer:
    """
    Minimal reusable trainer for pretraining and SFT.

    一个尽量通用的最小训练器，可同时服务预训练和 SFT。
    It intentionally exposes the main training steps instead of hiding them.
    它刻意不把训练过程封装得过深，方便学习每个步骤。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        run_metadata: dict | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.output_dir = ensure_dir(training_config.output_dir)
        self.run_metadata = run_metadata or {}
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.train_log_path = self.output_dir / "train_log.jsonl"
        self.eval_log_path = self.output_dir / "eval_log.jsonl"
        self.latest_checkpoint_path = self.output_dir / "latest_checkpoint.json"
        self.checkpoint_index_path = self.output_dir / "checkpoint_index.json"
        self.progress_bar = None

        self.device = self._resolve_device(training_config.device)
        self.model.to(self.device)
        if training_config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if training_config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.optimizer = build_optimizer(self.model, optimizer_config)

        amp_dtype = resolve_amp_dtype(training_config.amp_dtype)
        use_amp = self.device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
        self.autocast = (
            torch.autocast(device_type=self.device.type, dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        )
        if use_amp and amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)

        self.global_step = 0
        self.total_tokens_seen = 0
        self.resume_metadata: dict[str, Any] = {}
        self._maybe_resume()
        self.stage_start_global_step = self.global_step
        self.stage_start_total_tokens_seen = self.total_tokens_seen
        self.stage_target_global_step = self.stage_start_global_step + self.training_config.train_steps

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _create_progress_bar(self) -> None:
        """
        Create an optional tqdm progress bar for step-level training progress.

        创建可选的 tqdm 训练进度条，用 step 作为主要进度单位。
        """

        if not self.training_config.show_progress_bar:
            self.progress_bar = None
            return
        try:
            from tqdm import tqdm

            self.progress_bar = tqdm(
                total=self.training_config.train_steps,
                initial=0,
                desc="Training",
                unit="step",
                mininterval=0.5,
                dynamic_ncols=True,
            )
        except ImportError:
            self.progress_bar = None

    def _set_lr(self, step: int) -> float:
        lr = cosine_lr(
            step=step,
            total_steps=self.training_config.train_steps,
            warmup_steps=self.optimizer_config.warmup_steps,
            base_lr=self.optimizer_config.lr,
            min_lr_ratio=self.optimizer_config.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _grad_norm(self) -> float:
        total = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            total += grad.float().pow(2).sum().item()
        return total ** 0.5

    def _extract_scalar_metrics(self, output: dict[str, torch.Tensor | float | int]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in output.items():
            if key in {"logits", "loss"}:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                metrics[key] = float(value.detach().float().item())
            elif isinstance(value, (float, int)):
                metrics[key] = float(value)
        return metrics

    def _categorize_metrics(self, metrics: dict[str, float]) -> dict[str, dict[str, float]]:
        categorized = {
            "losses": {},
            "mtp": {},
            "moe": {},
            "other_metrics": {},
        }
        for key, value in metrics.items():
            if key.startswith("mtp_"):
                categorized["mtp"][key] = round(value, 6)
            elif key.startswith("moe_"):
                categorized["moe"][key] = round(value, 6)
            elif key.endswith("_loss") or key == "lm_loss":
                categorized["losses"][key] = round(value, 6)
            else:
                categorized["other_metrics"][key] = round(value, 6)
        return {key: value for key, value in categorized.items() if value}

    def _resolve_resume_path(self, resume_from: str) -> tuple[Path, dict[str, Any]]:
        path = Path(resume_from).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Resume path does not exist: {resume_from}. "
                "Please pass a checkpoint `.pt` file or a `latest_checkpoint.json` file."
            )

        if path.suffix == ".json":
            payload = load_json(path)
            checkpoint_path = payload.get("latest_checkpoint", "")
            if not checkpoint_path:
                raise ValueError(f"No `latest_checkpoint` field found in {path}")
            resolved = Path(checkpoint_path).expanduser()
            if not resolved.is_absolute():
                cwd_candidate = resolved.resolve()
                if cwd_candidate.exists():
                    resolved = cwd_candidate
                else:
                    resolved = (path.parent / resolved).resolve()
            if not resolved.exists():
                raise FileNotFoundError(
                    f"`latest_checkpoint.json` points to a missing file: {resolved}"
                )
            return resolved, {
                "resume_source": str(path),
                "resolved_checkpoint": str(resolved),
                "resume_mode": "latest_checkpoint_json",
            }

        return path, {
            "resume_source": str(path),
            "resolved_checkpoint": str(path),
            "resume_mode": "checkpoint_pt",
        }

    def _save_checkpoint(self, last_train_log: dict | None = None) -> None:
        checkpoint_name = f"step_{self.global_step:07d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "total_tokens_seen": self.total_tokens_seen,
                "training_config": vars(self.training_config),
                "optimizer_config": vars(self.optimizer_config),
                "run_metadata": self.run_metadata,
            },
            checkpoint_path,
        )
        latest_payload = {
            "latest_checkpoint": str(checkpoint_path),
            "global_step": self.global_step,
            "total_tokens_seen": self.total_tokens_seen,
            "last_train_log": last_train_log or {},
        }
        save_json(self.latest_checkpoint_path, latest_payload)

        checkpoint_index = load_json(self.checkpoint_index_path, default={"checkpoints": []})
        entries = checkpoint_index.get("checkpoints", [])
        entries.append(
            {
                "step": self.global_step,
                "path": str(checkpoint_path),
                "total_tokens_seen": self.total_tokens_seen,
            }
        )
        keep_k = self.training_config.keep_last_k_checkpoints
        if keep_k > 0 and len(entries) > keep_k:
            stale_entries = entries[:-keep_k]
            for stale in stale_entries:
                stale_path = Path(stale["path"])
                if stale_path.exists():
                    stale_path.unlink()
            entries = entries[-keep_k:]
        save_json(self.checkpoint_index_path, {"checkpoints": entries})

    def _maybe_resume(self) -> None:
        resume_from = self.training_config.resume_from
        if not resume_from:
            return
        checkpoint_path, metadata = self._resolve_resume_path(resume_from)
        payload = torch.load(checkpoint_path, map_location="cpu")
        model_load_mode = "strict_resume"
        model_missing_keys: list[str] = []
        model_unexpected_keys: list[str] = []
        optimizer_loaded = False

        try:
            self.model.load_state_dict(payload["model"], strict=True)
        except RuntimeError:
            incompat = self.model.load_state_dict(payload["model"], strict=False)
            model_missing_keys = list(incompat.missing_keys)
            model_unexpected_keys = list(incompat.unexpected_keys)
            model_load_mode = "warm_start_non_strict"

        if model_load_mode == "strict_resume":
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
                optimizer_loaded = True
            except ValueError:
                optimizer_loaded = False

        self.global_step = payload.get("global_step", 0)
        self.total_tokens_seen = payload.get("total_tokens_seen", 0)
        self.resume_metadata = {
            **metadata,
            "model_load_mode": model_load_mode,
            "optimizer_loaded": optimizer_loaded,
            "missing_key_count": len(model_missing_keys),
            "unexpected_key_count": len(model_unexpected_keys),
            "missing_keys_preview": model_missing_keys[:8],
            "unexpected_keys_preview": model_unexpected_keys[:8],
            "resumed_global_step": self.global_step,
            "resumed_total_tokens_seen": self.total_tokens_seen,
        }
        print(
            json.dumps(
                {
                    "event": "resume",
                    **self.resume_metadata,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        for batch in self.valid_loader or []:
            batch = self._move_batch(batch)
            with self.autocast:
                output = self.model(**batch)
            losses.append(output["loss"].detach().float())
        self.model.train()
        if not losses:
            return float("nan")
        return torch.stack(losses).mean().item()

    def train(self) -> None:
        metadata_path = self.output_dir / "run_config.json"
        save_json(
            metadata_path,
            {
                "training": vars(self.training_config),
                "optimizer": vars(self.optimizer_config),
                "optimizer_runtime": optimizer_summary(self.optimizer_config),
                "run_metadata": self.run_metadata,
                "resume_metadata": self.resume_metadata,
                "stage_start_global_step": self.stage_start_global_step,
                "stage_start_total_tokens_seen": self.stage_start_total_tokens_seen,
                "stage_target_global_step": self.stage_target_global_step,
            },
        )

        timer = SimpleTimer()
        self.model.train()
        train_iter = iter(self.train_loader)
        self._create_progress_bar()
        train_log: dict = {}

        while self.global_step < self.stage_target_global_step:
            self.optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            metric_sums: dict[str, float] = {}
            tokens_this_step = 0
            current_stage_step = self.global_step - self.stage_start_global_step
            for _ in range(self.training_config.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                batch = self._move_batch(batch)
                if "attention_mask" in batch:
                    tokens_this_step += int(batch["attention_mask"].sum().item())
                else:
                    tokens_this_step += int(batch["input_ids"].numel())
                with self.autocast:
                    output = self.model(**batch)
                    loss = output["loss"] / self.training_config.grad_accum_steps

                micro_losses.append(loss.detach().float())
                for key, value in self._extract_scalar_metrics(output).items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            grad_norm = self._grad_norm()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimizer_config.grad_clip)

            lr = self._set_lr(current_stage_step)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.global_step += 1
            self.total_tokens_seen += tokens_this_step
            mean_loss = torch.stack(micro_losses).sum().item()
            elapsed_sec = max(timer.elapsed(), 1e-6)
            stage_tokens_seen = self.total_tokens_seen - self.stage_start_total_tokens_seen
            tokens_per_sec = stage_tokens_seen / elapsed_sec
            completed_stage_step = self.global_step - self.stage_start_global_step
            steps_per_sec = completed_stage_step / elapsed_sec
            remaining_steps = max(0, self.stage_target_global_step - self.global_step)
            eta_seconds = remaining_steps / max(steps_per_sec, 1e-9)
            train_log = {
                "progress": {
                    "step": self.global_step,
                    "stage_step": completed_stage_step,
                    "total_steps": self.training_config.train_steps,
                    "progress_pct": round(100.0 * completed_stage_step / max(1, self.training_config.train_steps), 2),
                },
                "optimization": {
                    "loss": round(mean_loss, 6),
                    "lr": lr,
                    "grad_norm": round(grad_norm, 6),
                },
                "throughput": {
                    "step_tokens": tokens_this_step,
                    "stage_tokens_seen": stage_tokens_seen,
                    "total_tokens_seen": self.total_tokens_seen,
                    "tokens_per_sec": round(tokens_per_sec, 2),
                    "steps_per_sec": round(steps_per_sec, 4),
                },
                "runtime": {
                    "elapsed_sec": round(elapsed_sec, 2),
                    "eta_sec": round(eta_seconds, 2),
                    "eta_hms": format_seconds(eta_seconds),
                    "eta_finish_utc": estimate_finish_time(eta_seconds),
                    "device": str(self.device),
                },
            }
            if metric_sums:
                averaged_metrics = {
                    key: value / self.training_config.grad_accum_steps for key, value in sorted(metric_sums.items())
                }
                train_log.update(self._categorize_metrics(averaged_metrics))

            if self.progress_bar is not None:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    lr=f"{lr:.2e}",
                    toks=f"{tokens_per_sec:.0f}/s",
                    eta=format_seconds(eta_seconds),
                )

            if self.global_step % self.training_config.log_interval == 0:
                append_jsonl(self.train_log_path, train_log)
                print(json.dumps(train_log, ensure_ascii=False), flush=True)

            if self.valid_loader and self.global_step % self.training_config.eval_interval == 0:
                eval_loss = self.evaluate()
                eval_log = {
                    "step": self.global_step,
                    "total_steps": self.training_config.train_steps,
                    "eval_loss": round(eval_loss, 6),
                    "total_tokens_seen": self.total_tokens_seen,
                    "elapsed_sec": round(timer.elapsed(), 2),
                }
                append_jsonl(self.eval_log_path, eval_log)
                print(json.dumps(eval_log, ensure_ascii=False), flush=True)

            if self.global_step % self.training_config.save_interval == 0:
                self._save_checkpoint(last_train_log=train_log)

        if self.progress_bar is not None:
            self.progress_bar.close()
        self._save_checkpoint(last_train_log=train_log)
