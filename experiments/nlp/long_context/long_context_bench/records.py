from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class DataRecord:
    """A text-level causal prediction example.

    `prompt` is visible context. `target` is the sequence whose next-token loss
    is measured. Token labels are constructed at runtime as:
    `[-100] * len(prompt_ids) + target_ids`.
    """

    id: str
    dataset: str
    split: str
    prompt: str
    target: str
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "DataRecord":
        return cls(
            id=str(obj["id"]),
            dataset=str(obj.get("dataset", "")),
            split=str(obj.get("split", "")),
            prompt=str(obj.get("prompt", "")),
            target=str(obj.get("target", "")),
            answers=[str(x) for x in obj.get("answers", [])],
            metadata=dict(obj.get("metadata", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str | Path) -> Iterator[DataRecord]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield DataRecord.from_json(json.loads(line))


def write_jsonl(path: str | Path, records: Iterable[DataRecord]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n

