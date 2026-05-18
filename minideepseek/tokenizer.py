from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import TokenizerConfig


class TokenizerLike(Protocol):
    """
    Minimal tokenizer protocol used by this project.

    本项目使用的最小 tokenizer 协议。
    Any tokenizer only needs encode / decode / vocab_size / special token ids.
    任何 tokenizer 只要满足 encode / decode / vocab_size / 特殊 token id 即可接入。
    """

    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...


@dataclass(frozen=True)
class SpecialTokens:
    """
    Built-in special token ids for the byte fallback tokenizer.

    这是 byte fallback tokenizer 的内置特殊 token 编号。
    """

    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    unk_id: int = 259


class ByteTokenizer:
    """
    A tiny tokenizer that maps UTF-8 bytes directly to token ids.

    一个教学友好的极简分词器：直接把 UTF-8 字节映射为 token。
    It is kept as a fallback because it has almost zero hidden behavior.
    保留它作为 fallback，是因为它几乎没有隐藏行为，适合教学。
    """

    def __init__(self, add_bos: bool = True, add_eos: bool = True):
        self.special = SpecialTokens()
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size = 260

    @property
    def pad_id(self) -> int:
        return self.special.pad_id

    @property
    def bos_id(self) -> int:
        return self.special.bos_id

    @property
    def eos_id(self) -> int:
        return self.special.eos_id

    @property
    def unk_id(self) -> int:
        return self.special.unk_id

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        if self.add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(text.encode("utf-8"))
        if self.add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        payload = bytearray()
        for token_id in token_ids:
            if 0 <= token_id <= 255:
                payload.append(token_id)
        return payload.decode("utf-8", errors="replace")


class HuggingFaceTokenizer:
    """
    Thin wrapper around a Hugging Face `tokenizers` fast tokenizer.

    这是对 Hugging Face `tokenizers` fast tokenizer 的薄封装。
    We keep the wrapper small so the preprocessing / training code can stay framework-light.
    封装尽量保持轻量，这样预处理与训练代码仍然足够透明。
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        tokenizer_config_path: str | Path | None = None,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "Using tokenizer.json requires the `tokenizers` package. "
                "Please install it from requirements.txt."
            ) from exc

        self.tokenizer_path = Path(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        config_payload: dict = {}
        if tokenizer_config_path and Path(tokenizer_config_path).exists():
            config_payload = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8"))

        self.add_bos = config_payload.get("add_bos_token", False) if add_bos is None else add_bos
        self.add_eos = config_payload.get("add_eos_token", False) if add_eos is None else add_eos

        self.special_token_to_id: dict[str, int] = {}
        self.id_to_special_token: dict[int, str] = {}
        self._load_special_tokens(config_payload)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def _load_special_tokens(self, config_payload: dict) -> None:
        decoder = config_payload.get("added_tokens_decoder", {})
        for token_id_text, token_spec in decoder.items():
            token_id = int(token_id_text)
            token_text = token_spec.get("content")
            self.special_token_to_id[token_text] = token_id
            self.id_to_special_token[token_id] = token_text

        self.pad_id = self._resolve_special_id(config_payload.get("pad_token"))
        self.bos_id = self._resolve_special_id(config_payload.get("bos_token"))
        self.eos_id = self._resolve_special_id(config_payload.get("eos_token"))
        self.unk_id = self._resolve_special_id(config_payload.get("unk_token"))

        # If the config does not explicitly declare ids, ask the tokenizer object.
        if self.pad_id is None:
            self.pad_id = self.tokenizer.token_to_id("<|endoftext|>")
        if self.bos_id is None:
            self.bos_id = self.tokenizer.token_to_id("<|im_start|>")
        if self.eos_id is None:
            self.eos_id = self.tokenizer.token_to_id("<|im_end|>")
        if self.unk_id is None:
            self.unk_id = self.tokenizer.token_to_id("<|endoftext|>")

    def _resolve_special_id(self, token_text: str | None) -> int | None:
        if token_text is None:
            return None
        return self.special_token_to_id.get(token_text, self.tokenizer.token_to_id(token_text))

    def encode(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text).ids
        if self.add_bos and self.bos_id is not None:
            token_ids = [self.bos_id] + token_ids
        if self.add_eos and self.eos_id is not None:
            token_ids = token_ids + [self.eos_id]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def build_tokenizer(config: TokenizerConfig) -> TokenizerLike:
    """
    Build a tokenizer from config.

    根据配置构建 tokenizer。
    Strategy:
    策略：
    1. If `kind == "hf"`, load the uploaded tokenizer files directly.
    2. 如果 `kind == "hf"`，直接加载你上传的 tokenizer 文件。
    3. Otherwise fall back to the byte tokenizer.
    4. 否则回退到 byte tokenizer。
    """

    kind = config.kind.lower()
    if kind in {"hf", "huggingface", "fast"}:
        return HuggingFaceTokenizer(
            tokenizer_path=config.tokenizer_path,
            tokenizer_config_path=config.tokenizer_config_path,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
        )
    return ByteTokenizer(add_bos=config.add_bos, add_eos=config.add_eos)
