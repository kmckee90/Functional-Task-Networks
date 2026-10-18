"""Question-conditioned mask retrieval for the chat LoRA bank.

The archive is learned from training questions only. A route owns a set of
questions and a repaired mask. Inference never sees the task key: it compares
the user's question with every stored route and either retrieves a mask or
returns a miss to the ordinary Bayes search.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn.functional as F

from spatial import kwta


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[0-9]+")


def words(text):
    """Preserve code identifiers, while also exposing their components."""
    out = set()
    for match in _WORD.findall(text.casefold()):
        out.add(match)
        if "_" in match:
            out.update(part for part in match.split("_") if len(part) > 1)
    return out


class InputRoutingArchive:
    def __init__(self, cfg, tokenizer, model=None):
        self.cfg, self.tokenizer, self.model = cfg, tokenizer, model
        self.routes = {}  # task -> questions, lexical counts, mask logits
        self.repair_round = 0
        self._docs = {}
        self._semantic = {}

    def state_dict(self):
        return {"version": 1, "H": self.cfg.H, "k": self.cfg.k,
                "repair_round": self.repair_round,
                "routes": self.routes}

    def load_state_dict(self, state):
        if state.get("H") != self.cfg.H or state.get("k") != self.cfg.k:
            raise ValueError("routing archive bank shape differs from checkpoint")
        routes = state.get("routes", {})
        for route in routes.values():
            if route["logits"].numel() != self.cfg.H:
                raise ValueError("routing archive logit size differs from bank")
        self.routes = routes
        self.repair_round = int(state.get("repair_round", 0))
        self._reindex()

    def upsert(self, task, questions, logits, lexical_questions=None):
        unique = list(dict.fromkeys(q.strip() for q in questions if q.strip()))
        if not unique:
            return
        corpus = lexical_questions if lexical_questions is not None else unique
        counts = Counter(t for q in corpus for t in words(q))
        self.routes[task] = {"questions": unique,
                             "logits": logits.detach().float().cpu().clone(),
                             "terms": dict(counts), "doc_count": len(corpus)}
        self._reindex()

    def _reindex(self):
        self._docs = {}
        for task, route in self.routes.items():
            self._docs[task] = Counter(route.get("terms") or
                                       Counter(t for q in route["questions"] for t in words(q)))
        self._semantic = {}

    def _semantic_vector(self, question):
        if self.model is None:
            return None
        ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        if not ids:
            return None
        emb = self.model.get_input_embeddings()
        with torch.no_grad():
            vectors = emb(torch.tensor(ids[:256], device=emb.weight.device))
            # Mean pooling keeps semantic context while avoiding a second LLM pass.
            pooled = F.normalize(vectors.float().mean(0), dim=0)
        return pooled.cpu()

    def _semantic_route(self, task):
        if task not in self._semantic:
            questions = self.routes[task]["questions"]
            # Several exemplars allow topics within a task to vary.
            chosen = questions[:self.cfg.input_route_semantic_examples]
            vecs = [self._semantic_vector(q) for q in chosen]
            vecs = [v for v in vecs if v is not None]
            self._semantic[task] = torch.stack(vecs) if vecs else None
        return self._semantic[task]

    def scores(self, question, semantic_weight=None):
        if not self.routes:
            return {}
        query = words(question)
        qvec = self._semantic_vector(question)
        n = len(self.routes)
        semantic_weight = (self.cfg.input_route_semantic_weight if semantic_weight is None
                           else float(semantic_weight))
        scores = {}
        for task in self.routes:
            count = self._docs[task]
            others = Counter()
            for other, docs in self._docs.items():
                if other != task:
                    others.update(docs)
            here_n = max(self.routes[task].get("doc_count", len(self.routes[task]["questions"])), 1)
            other_n = max(sum(r.get("doc_count", len(r["questions"])) for t, r in self.routes.items()
                              if t != task), 1)
            evidence = []
            if n > 1:
                for term in query:
                    if count[term]:
                        # Smoothed Bernoulli log odds across routes. A rare,
                        # route-specific identifier can outweigh generic words.
                        odds = math.log((count[term] + 0.5) / (here_n + 1)) - \
                               math.log((others[term] + 0.5) / (other_n + 1))
                        evidence.append(max(odds, 0.0))
            evidence.sort(reverse=True)
            lexical = sum(evidence[:4])
            semantic = 0.0
            prototypes = self._semantic_route(task) if qvec is not None else None
            if prototypes is not None:
                semantic = float((prototypes @ qvec).max())
            # Shared code vocabulary is still informative: require at least
            # 3x the smoothed document frequency outside this task, rather than
            # absolute exclusivity (which rejected FTN shared with FTNDDPG).
            anchors = [term for term in query
                       if count[term] >= self.cfg.input_route_anchor_min_docs
                       and (others[term] == 0 or
                            (count[term] + 0.5) / (here_n + 1) >=
                            3.0 * (others[term] + 0.5) / (other_n + 1))]
            scores[task] = {"score": lexical + semantic_weight * semantic,
                            "lexical": lexical, "semantic": semantic,
                            "anchors": anchors}
        return scores

    def match(self, question, *, k=None, margin=None, semantic_weight=None):
        scores = self.scores(question, semantic_weight)
        if not scores:
            return None, {"reason": "empty archive"}
        ranked = sorted(scores, key=lambda name: scores[name]["score"], reverse=True)
        best = ranked[0]
        gap = (scores[best]["score"] - scores[ranked[1]]["score"]
               if len(ranked) > 1 else float("inf"))
        needed = self.cfg.input_route_margin if margin is None else float(margin)
        info = {"task": best, "margin": gap, "scores": scores}
        semantic_gap = (scores[best]["semantic"] - scores[ranked[1]]["semantic"]
                        if len(ranked) > 1 else 1.0)
        supported = (bool(scores[best]["anchors"]) or
                     (scores[best]["semantic"] >= self.cfg.input_route_semantic_floor and
                      semantic_gap >= self.cfg.input_route_semantic_gap))
        if gap < needed or not supported:
            info["reason"] = "uncertain"
            return None, info
        logits = self.routes[best]["logits"]
        mask = kwta(logits, self.cfg.k if k is None else int(k))
        info["reason"] = "matched"
        return mask, info
