from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset

from .tokenizer import TokenizerLike
from .utils import ensure_dir, save_json

UINT32 = struct.Struct("<I")
UINT64 = struct.Struct("<Q")


def write_u32_file(path: str | Path, values: Iterable[int]) -> int:
    """
    Write token ids as a contiguous little-endian uint32 stream.

    将 token id 按连续的小端 uint32 写入二进制文件。
    This format is easy to memory-map later and does not depend on numpy.
    这个格式后续可以直接 mmap，且不依赖 numpy。
    """

    count = 0
    with Path(path).open("wb") as handle:
        for value in values:
            handle.write(UINT32.pack(int(value)))
            count += 1
    return count


def write_u64_file(path: str | Path, values: Iterable[int]) -> int:
    count = 0
    with Path(path).open("wb") as handle:
        for value in values:
            handle.write(UINT64.pack(int(value)))
            count += 1
    return count


def preprocess_pretrain_jsonl(
    input_path: str | Path,
    output_prefix: str | Path,
    tokenizer: TokenizerLike,
    total_lines: int | None = None,
    show_progress: bool = False,
    progress_update_interval: int = 1000,
) -> dict[str, int | str]:
    """
    Convert {"text": "..."} jsonl into a flat token stream.

    将 {"text": "..."} 格式的 jsonl 转为平铺 token 流，适合 next-token pretraining。
    """

    source = Path(input_path)
    prefix = Path(output_prefix)
    ensure_dir(prefix.parent)
    bin_path = prefix.with_suffix(".bin")
    meta_path = prefix.with_suffix(".json")
    sample_index_path = prefix.with_name(prefix.name + "_sample_index.bin")

    total_samples = 0
    total_tokens = 0
    sample_end_offsets: list[int] = []
    progress_bar = None

    if show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=total_lines,
                desc="Preprocessing pretrain data",
                unit="samples",
                mininterval=0.5,
            )
        except ImportError:
            progress_bar = None

    def iterator() -> Iterable[int]:
        nonlocal total_samples, total_tokens
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                text = payload.get("text", "")
                token_ids = tokenizer.encode(text)
                total_samples += 1
                total_tokens += len(token_ids)
                sample_end_offsets.append(total_tokens)
                if progress_bar is not None:
                    progress_bar.update(1)
                    if total_samples % progress_update_interval == 0:
                        progress_bar.set_postfix(
                            tokens=total_tokens,
                            avg_len=round(total_tokens / max(total_samples, 1), 2),
                        )
                elif show_progress and total_samples % progress_update_interval == 0:
                    print(
                        json.dumps(
                            {
                                "samples_processed": total_samples,
                                "tokens_processed": total_tokens,
                                "avg_tokens_per_sample": round(total_tokens / max(total_samples, 1), 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                for token_id in token_ids:
                    yield token_id

    written = write_u32_file(bin_path, iterator())
    written_sample_index = write_u64_file(sample_index_path, sample_end_offsets)
    if progress_bar is not None:
        progress_bar.close()
    meta = {
        "task": "pretrain",
        "input_path": str(source),
        "bin_path": str(bin_path),
        "num_samples": total_samples,
        "num_tokens": total_tokens,
        "written_values": written,
        "sample_index_path": str(sample_index_path),
        "written_sample_index_values": written_sample_index,
        "tokenizer": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
    }
    save_json(meta_path, meta)
    return meta


def load_pretrain_meta(meta_path: str | Path) -> dict[str, int | str]:
    """
    Load pretraining preprocessing metadata.

    读取预训练预处理阶段保存的元信息。
    """

    return json.loads(Path(meta_path).read_text(encoding="utf-8"))


@dataclass
class ConversationExample:
    input_ids: list[int]
    labels: list[int]


def _render_conversation_turn(role: str, content: str) -> str:
    """
    Serialize one dialogue turn into an explicit template.

    将单轮对话序列化成清晰模板，便于后续 SFT 和 RL 阶段共用。
    """

    return f"<|{role}|>\n{content.strip()}\n"


def build_sft_example(conversations: list[dict[str, str]], tokenizer: TokenizerLike) -> ConversationExample:
    """
    Build an SFT sample with selective loss masking.

    构建带选择性 loss mask 的 SFT 样本。
    User / system / tool tokens use label -100, assistant tokens participate in loss.
    `user/system/tool` 部分标签为 -100，不参与损失；只有 assistant 响应参与监督。
    """

    input_ids: list[int] = []
    labels: list[int] = []

    if getattr(tokenizer, "add_bos", False):
        input_ids.append(tokenizer.bos_id)
        labels.append(-100)

    for turn in conversations:
        role = turn["role"]
        content = turn.get("content", "")
        rendered = _render_conversation_turn(role, content)
        # Important:
        # 重要说明：
        # We intentionally tokenize every rendered turn with the same tokenizer
        # that is used in pretraining. This keeps token boundaries consistent
        # across pretrain and SFT.
        # 这里必须使用与预训练完全相同的 tokenizer 来编码每一轮模板化文本，
        # 这样预训练和 SFT 阶段的 token 边界才是一致的。
        turn_ids = tokenizer.encode(rendered)
        input_ids.extend(turn_ids)
        if role == "assistant":
            labels.extend(turn_ids)
        else:
            labels.extend([-100] * len(turn_ids))

    if getattr(tokenizer, "add_eos", False):
        input_ids.append(tokenizer.eos_id)
        labels.append(tokenizer.eos_id)

    return ConversationExample(input_ids=input_ids, labels=labels)


def preprocess_sft_jsonl(
    input_path: str | Path,
    output_prefix: str | Path,
    tokenizer: TokenizerLike,
    max_seq_len: int,
) -> dict[str, int | str]:
    """
    Convert conversation jsonl into indexed binary records.

    将对话 jsonl 转为带索引的二进制记录，便于后续多次训练复用。
    """

    source = Path(input_path)
    prefix = Path(output_prefix)
    ensure_dir(prefix.parent)
    input_bin = prefix.with_name(prefix.name + "_input_ids.bin")
    label_bin = prefix.with_name(prefix.name + "_labels.bin")
    index_json = prefix.with_suffix(".json")

    offsets: list[dict[str, int]] = []
    token_cursor = 0
    sample_count = 0

    with source.open("r", encoding="utf-8") as reader, input_bin.open("wb") as xh, label_bin.open("wb") as yh:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            conversations = payload.get("conversations", [])
            example = build_sft_example(conversations, tokenizer)
            if len(example.input_ids) > max_seq_len:
                example = ConversationExample(
                    input_ids=example.input_ids[:max_seq_len],
                    labels=example.labels[:max_seq_len],
                )
            for value in example.input_ids:
                xh.write(UINT32.pack(int(value)))
            for value in example.labels:
                # Store labels as signed int32 by reusing struct packing from Python int.
                yh.write(struct.pack("<i", int(value)))
            offsets.append({"offset": token_cursor, "length": len(example.input_ids)})
            token_cursor += len(example.input_ids)
            sample_count += 1

    meta = {
        "task": "sft",
        "input_path": str(source),
        "input_bin": str(input_bin),
        "label_bin": str(label_bin),
        "index": offsets,
        "num_samples": sample_count,
        "max_seq_len": max_seq_len,
    }
    save_json(index_json, meta)
    return meta


class PackedPretrainDataset(Dataset):
    """
    Sample fixed-length windows from a flat token stream.

    从平铺 token 流中切出固定长度窗口，是最常见的预训练数据读取方式之一。
    """

    def __init__(self, bin_path: str | Path, seq_len: int, sample_index_path: str | Path = ""):
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.file = self.bin_path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.num_tokens = self.mapping.size() // 4
        self.num_sequences = max(0, (self.num_tokens - 1) // self.seq_len)
        self.sample_index_path = Path(sample_index_path) if sample_index_path else None
        self.sample_index_file = None
        self.sample_index_mapping = None
        self.num_samples = 0
        if self.sample_index_path and self.sample_index_path.exists():
            self.sample_index_file = self.sample_index_path.open("rb")
            self.sample_index_mapping = mmap.mmap(self.sample_index_file.fileno(), 0, access=mmap.ACCESS_READ)
            self.num_samples = self.sample_index_mapping.size() // 8

    def __len__(self) -> int:
        return self.num_sequences

    def _sample_end_offset(self, sample_idx: int) -> int:
        if self.sample_index_mapping is None:
            raise ValueError("sample_index_mapping is not available")
        return UINT64.unpack_from(self.sample_index_mapping, sample_idx * 8)[0]

    def _find_sample_index(self, token_pos: int) -> int:
        if self.sample_index_mapping is None:
            return 0
        left = 0
        right = self.num_samples
        while left < right:
            mid = (left + right) // 2
            if self._sample_end_offset(mid) <= token_pos:
                left = mid + 1
            else:
                right = mid
        return min(left, max(0, self.num_samples - 1))

    def _build_sample_ids(self, start: int) -> torch.Tensor:
        if self.sample_index_mapping is None:
            return torch.zeros(self.seq_len, dtype=torch.long)
        sample_ids = []
        sample_idx = self._find_sample_index(start)
        sample_end = self._sample_end_offset(sample_idx)
        for pos in range(start, start + self.seq_len):
            while pos >= sample_end and sample_idx + 1 < self.num_samples:
                sample_idx += 1
                sample_end = self._sample_end_offset(sample_idx)
            sample_ids.append(sample_idx)
        return torch.tensor(sample_ids, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.seq_len
        token_ids = []
        for pos in range(start, start + self.seq_len + 1):
            byte_pos = pos * 4
            token_ids.append(UINT32.unpack_from(self.mapping, byte_pos)[0])
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)
        sample_ids = self._build_sample_ids(start)
        return {"input_ids": x, "labels": y, "sample_ids": sample_ids}


class IndexedSFTDataset(Dataset):
    """
    Load variable-length SFT samples from indexed binary files.

    从索引二进制文件中读取变长 SFT 样本。
    """

    def __init__(self, meta_path: str | Path, pad_id: int):
        payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.index = payload["index"]
        self.pad_id = pad_id
        self.input_file = Path(payload["input_bin"]).open("rb")
        self.label_file = Path(payload["label_bin"]).open("rb")
        self.input_mmap = mmap.mmap(self.input_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.label_mmap = mmap.mmap(self.label_file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.index[idx]
        offset = item["offset"]
        length = item["length"]
        input_ids = []
        labels = []
        for pos in range(offset, offset + length):
            input_ids.append(UINT32.unpack_from(self.input_mmap, pos * 4)[0])
            labels.append(struct.unpack_from("<i", self.label_mmap, pos * 4)[0])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sft_collate_fn(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """
    Pad SFT samples to a common length.

    将一批变长 SFT 样本补齐到统一长度。
    """

    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_rows = []
    label_rows = []
    attention_rows = []
    for item in batch:
        x = item["input_ids"]
        y = item["labels"]
        pad_len = max_len - x.shape[0]
        input_rows.append(torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        label_rows.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_rows.append(torch.cat([torch.ones_like(x), torch.zeros((pad_len,), dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_rows),
        "labels": torch.stack(label_rows),
        "attention_mask": torch.stack(attention_rows),
    }
