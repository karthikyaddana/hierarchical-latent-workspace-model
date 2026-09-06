from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler


def find_split(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / (split + ".jsonl"),
        data_dir / "data" / (split + ".jsonl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(data_dir.rglob(split + ".jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError("could not uniquely locate %s.jsonl under %s" % (split, data_dir))
    return matches[0]


class TeacherPairDataset(Dataset):
    def __init__(self, path: Path, limit: Optional[int] = None) -> None:
        self.path = path
        self.rows: List[Dict[str, Any]] = []
        seen_ids = set()
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                for key in ("id", "prompt", "chosen", "rejected"):
                    if not str(value.get(key) or "").strip():
                        raise ValueError("%s:%d missing %s" % (path, line_number, key))
                if value["chosen"].strip() == value["rejected"].strip():
                    raise ValueError("%s:%d chosen and rejected are identical" % (path, line_number))
                if value["id"] in seen_ids:
                    raise ValueError("duplicate id %s in %s" % (value["id"], path))
                seen_ids.add(value["id"])
                self.rows.append(value)
                if limit is not None and len(self.rows) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]

    @property
    def programmatic_indices(self) -> List[int]:
        return [index for index, row in enumerate(self.rows) if row.get("task_type")]


class DeterministicDistributedBatchSampler(Sampler[List[int]]):
    """Resume the exact global sample order without serializing a DataLoader iterator."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        start_microstep: int,
        total_microsteps: int,
        seed: int,
        rank: int,
        world_size: int,
        allowed_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0 or world_size <= 0:
            raise ValueError("dataset size, batch size and world size must be positive")
        if not 0 <= start_microstep <= total_microsteps:
            raise ValueError("invalid microstep range")
        self.pool = list(allowed_indices) if allowed_indices is not None else list(range(dataset_size))
        if not self.pool:
            raise ValueError("sampler pool is empty")
        self.batch_size = int(batch_size)
        self.start_microstep = int(start_microstep)
        self.total_microsteps = int(total_microsteps)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __len__(self) -> int:
        return self.total_microsteps - self.start_microstep

    def __iter__(self):
        cache: Dict[int, List[int]] = {}
        pool_size = len(self.pool)
        for microstep in range(self.start_microstep, self.total_microsteps):
            indices: List[int] = []
            for offset in range(self.batch_size):
                global_position = (
                    (microstep * self.world_size + self.rank) * self.batch_size + offset
                )
                epoch = global_position // pool_size
                if epoch not in cache:
                    generator = torch.Generator().manual_seed(self.seed + epoch)
                    permutation = torch.randperm(pool_size, generator=generator).tolist()
                    cache = {epoch: [self.pool[index] for index in permutation]}
                indices.append(cache[epoch][global_position % pool_size])
            yield indices


def _chat_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    parts = [str(row["prompt"]).strip()]
    context = [str(value).strip() for value in row.get("context") or [] if str(value).strip()]
    constraints = [
        str(value).strip() for value in row.get("constraints") or [] if str(value).strip()
    ]
    if context:
        parts.append("Context:\n- " + "\n- ".join(context))
    if constraints:
        parts.append("Constraints:\n- " + "\n- ".join(constraints))
    message = "\n\n".join(parts)
    messages = [{"role": "user", "content": message}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _pad(rows: Sequence[Sequence[int]], pad_id: int, length: int) -> tuple[Tensor, Tensor]:
    ids: List[List[int]] = []
    masks: List[List[int]] = []
    for row in rows:
        clipped = list(row[:length])
        padding = length - len(clipped)
        ids.append(clipped + [pad_id] * padding)
        masks.append([1] * len(clipped) + [0] * padding)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


class HLWM8BCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_prompt_tokens: int = 384,
        max_answer_tokens: int = 192,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_answer_tokens = int(max_answer_tokens)
        self.pad_id = int(tokenizer.pad_token_id)
        self.eos_id = int(tokenizer.eos_token_id)

    def _answer_ids(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        return ids[: max(1, self.max_answer_tokens - 1)] + [self.eos_id]

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        prompt_texts = [_chat_prompt(self.tokenizer, row) for row in rows]
        prompt_rows = [
            self.tokenizer.encode(text, add_special_tokens=False)[-self.max_prompt_tokens :]
            for text in prompt_texts
        ]
        chosen_rows = [self._answer_ids(str(row["chosen"])) for row in rows]
        rejected_rows = [self._answer_ids(str(row["rejected"])) for row in rows]
        prompt_ids, prompt_mask = _pad(prompt_rows, self.pad_id, self.max_prompt_tokens)
        chosen_ids, chosen_mask = _pad(chosen_rows, self.pad_id, self.max_answer_tokens)
        rejected_ids, rejected_mask = _pad(rejected_rows, self.pad_id, self.max_answer_tokens)

        full_rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        full_length = self.max_prompt_tokens + self.max_answer_tokens
        for prompt, chosen in zip(prompt_rows, chosen_rows):
            combined = prompt + chosen
            labels = [-100] * len(prompt) + chosen
            padding = full_length - len(combined)
            full_rows.append(combined + [self.pad_id] * padding)
            label_rows.append(labels + [-100] * padding)
        full_ids = torch.tensor(full_rows, dtype=torch.long)
        labels = torch.tensor(label_rows, dtype=torch.long)
        full_mask = (full_ids != self.pad_id).long()
        difficulty = torch.tensor(
            [1.0 if int(row.get("difficulty") or 3) >= 3 else 0.0 for row in rows],
            dtype=torch.float32,
        )
        return {
            "ids": [str(row["id"]) for row in rows],
            "task_types": [str(row.get("task_type") or "") for row in rows],
            "prompt_texts": prompt_texts,
            "chosen_texts": [str(row["chosen"]) for row in rows],
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "full_input_ids": full_ids,
            "full_attention_mask": full_mask,
            "labels": labels,
            "chosen_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "difficulty_targets": difficulty,
        }


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
