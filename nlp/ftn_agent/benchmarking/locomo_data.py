"""LoCoMo-10 loading + rendering.

Schema (snap-research/locomo `data/locomo10.json`): a list of samples, each
  { "sample_id", "qa": [{question, answer|adversarial_answer, evidence,
    category}], "conversation": { "speaker_a", "speaker_b",
    "session_<i>": [{speaker, dia_id, text, (img_url, blip_caption)}],
    "session_<i>_date_time": str, ... } }
Categories (paper): 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
5 adversarial (dropped — no usable ground truth; see README).

Corrected answer key: the dial481/locomo-audit repo is merged best-effort by
(sample_id, question) — `apply_corrections` reports how many gold answers it
replaced so the merge is auditable.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                  4: "single-hop", 5: "adversarial"}

BENCH_ROOT = Path(os.environ.get("FTN_AGENT_STATE", "/home/dev/ftn_agent_state")) / "benchmarking"
LOCOMO_JSON = BENCH_ROOT / "locomo" / "data" / "locomo10.json"
AUDIT_DIR = BENCH_ROOT / "locomo-audit"


@dataclass
class Session:
    key: str                      # "session_3"
    index: int                    # 3
    date_time: str
    turns: list[dict]


@dataclass
class QA:
    question: str
    answer: str
    category: int
    evidence: list = field(default_factory=list)
    corrected: bool = False

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, str(self.category))


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa: list[QA]


def load_locomo(path: Path = LOCOMO_JSON, drop_adversarial: bool = True) -> list[Conversation]:
    data = json.loads(Path(path).read_text())
    convs = []
    for sample in data:
        c = sample["conversation"]
        sessions = []
        for key in c:
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(c[key], list):
                idx = int(key.split("_")[1])
                sessions.append(Session(key=key, index=idx,
                                        date_time=str(c.get(f"{key}_date_time", "")),
                                        turns=c[key]))
        sessions.sort(key=lambda s: s.index)
        qa = []
        for q in sample.get("qa", []):
            cat = int(q.get("category", -1))
            if drop_adversarial and cat == 5:
                continue
            ans = q.get("answer", q.get("adversarial_answer", ""))
            qa.append(QA(question=str(q.get("question", "")).strip(),
                         answer=str(ans).strip(), category=cat,
                         evidence=q.get("evidence", [])))
        convs.append(Conversation(sample_id=str(sample.get("sample_id", len(convs))),
                                  speaker_a=c.get("speaker_a", "A"),
                                  speaker_b=c.get("speaker_b", "B"),
                                  sessions=sessions, qa=qa))
    return convs


def apply_corrections(convs: list[Conversation], audit_dir: Path = AUDIT_DIR,
                      log=print) -> int:
    """Merge the audit's corrected gold answers from errors.json (156 entries;
    120 change the gold — WRONG_CITATION entries have correct_answer ==
    golden_answer and are no-ops). question_id is `locomo_<conv-order>_qa<idx>`,
    so we match by conversation ORDER in the file + normalized question text.
    Returns #gold answers actually changed."""
    errors_p = Path(audit_dir) / "errors.json"
    if not errors_p.exists():
        log("corrections: errors.json missing — using original answer key")
        return 0
    norm = lambda s: " ".join(str(s).lower().split())
    fixes: dict[tuple, dict] = {}
    for e in json.loads(errors_p.read_text()):
        m = re.match(r"locomo_(\d+)_qa\d+", e.get("question_id", ""))
        if m and e.get("correct_answer") is not None:
            fixes[(int(m.group(1)), norm(e["question"]))] = e
    n = matched = 0
    for ci, conv in enumerate(convs):
        for qa in conv.qa:
            e = fixes.get((ci, norm(qa.question)))
            if e is None:
                continue
            matched += 1
            new = str(e["correct_answer"]).strip()
            if norm(new) != norm(qa.answer):
                qa.answer, qa.corrected = new, True
                n += 1
    log(f"corrections: matched {matched}/{len(fixes)} audit entries, "
        f"changed {n} gold answers")
    return n


# --- rendering ----------------------------------------------------------------

def render_turn(t: dict) -> str:
    txt = str(t.get("text", "")).strip()
    cap = t.get("blip_caption")
    if cap:
        txt = f"{txt} [shared an image: {cap}]".strip()
    return f'{t.get("speaker", "?")}: {txt}'


def render_session(s: Session) -> str:
    head = f"--- Session {s.index} ({s.date_time}) ---"
    return "\n".join([head] + [render_turn(t) for t in s.turns])


def render_conversation(conv: Conversation) -> str:
    head = (f"Conversation between {conv.speaker_a} and {conv.speaker_b}, "
            f"over {len(conv.sessions)} sessions:")
    return "\n\n".join([head] + [render_session(s) for s in conv.sessions])


if __name__ == "__main__":
    convs = load_locomo()
    apply_corrections(convs)
    for c in convs:
        toks = sum(len(render_session(s).split()) for s in c.sessions)
        cats = {}
        for q in c.qa:
            cats[q.category_name] = cats.get(q.category_name, 0) + 1
        print(f"{c.sample_id}: {len(c.sessions)} sessions, ~{toks} words, "
              f"{len(c.qa)} questions {cats}")
    print(f"TOTAL: {sum(len(c.qa) for c in convs)} questions, "
          f"{sum(len(c.sessions) for c in convs)} sessions")
