"""Support-set construction: identifiability + greedy selection (design section 6)."""

from __future__ import annotations

import numpy as np

from . import algorithms as A
from . import generator as G
from . import seeds as S


def outputs_all(graph, algorithms: list[int]) -> dict[int, np.ndarray]:
    return {a: A.final_values(graph, a) for a in algorithms}


def consistent_algorithms(graphs: list, target: int, candidates: list[int]) -> list[int]:
    """Algorithms in `candidates` whose complete final outputs match `target` on every graph."""
    alive = list(candidates)
    for g in graphs:
        tgt = A.final_values(g, target)
        alive = [a for a in alive if a == target or np.array_equal(A.final_values(g, a), tgt)]
    return alive


def is_identifiable(graphs: list, target: int, family: list[int]) -> bool:
    return consistent_algorithms(graphs, target, family) == [target]


def greedy_identifiable_support(run_seed: int, fold: int, target: int, family: list[int],
                                support_id: int, size: int = 16, pool_size: int = 256,
                                max_pool_tries: int = 20) -> list:
    """Greedy selection per design 6.4. `family` includes the target; alternatives are
    the rest. Returns `size` Graphs; raises if no identifiable set found."""
    alts = [a for a in family if a != target]
    for pool_try in range(max_pool_tries):
        pool = [G.generate("train", run_seed, S.STREAM_SUPPORT_POOL, fold, target, support_id, pool_try, i)
                for i in range(pool_size)]
        outs = [{a: A.final_values(g, a) for a in family} for g in pool]
        # disagree[i][a] = alternative a disagrees with target on pool graph i
        disagree = [{a: not np.array_equal(o[a], o[target]) for a in alts} for o in outs]
        n_diff = [{a: int(np.count_nonzero(o[a] != o[target])) for a in alts} for o in outs]

        remaining = set(alts)
        selected: list[int] = []
        while remaining and len(selected) < size:
            best, best_key = None, None
            for i in range(pool_size):
                if i in selected:
                    continue
                elim = sum(1 for a in remaining if disagree[i][a])
                tie = sum(n_diff[i][a] for a in remaining)
                key = (elim, tie)
                if best_key is None or key > best_key:
                    best, best_key = i, key
            if best is None or best_key[0] == 0:
                break  # nothing eliminates anything further
            selected.append(best)
            remaining -= {a for a in remaining if disagree[best][a]}
        if remaining:
            continue  # reject pool, regenerate
        # Fill remaining slots by largest mean nodewise disagreement from alternatives.
        rest = sorted((i for i in range(pool_size) if i not in selected),
                      key=lambda i: -sum(n_diff[i][a] for a in alts) / max(1, len(alts)))
        selected.extend(rest[: size - len(selected)])
        graphs = [pool[i] for i in selected]
        assert is_identifiable(graphs, target, family)
        return graphs
    raise RuntimeError(f"no identifiable support set for target={target} after {max_pool_tries} pools")


def random_support(run_seed: int, fold: int, target: int, support_id: int, size: int = 16) -> list:
    """Randomly sampled, potentially ambiguous support set (reported separately)."""
    return [G.generate("train", run_seed, S.STREAM_SUPPORT_POOL, fold, target, support_id, 999, i)
            for i in range(size)]
