"""Behavioral / temporal adjacency graph over centroids, with Dijkstra planning.

An edge ``i -> j`` exists if the agent was ever observed transitioning *membership*
directly from centroid ``i`` to centroid ``j`` (consecutive timesteps, nothing in
between). Edge weight counts how often that transition happened; the planner uses
``cost = 1 / weight`` so frequently-traversed (reliable) transitions are cheaper.
"""

from __future__ import annotations

import heapq


class BehavioralGraph:
    def __init__(self):
        self.nodes: set[int] = set()
        self.edges: dict[tuple[int, int], float] = {}  # (i, j) -> weight (directed)

    def add_transition(self, i: int, j: int) -> None:
        if i < 0 or j < 0 or i == j:
            return
        self.nodes.add(i)
        self.nodes.add(j)
        self.edges[(i, j)] = self.edges.get((i, j), 0.0) + 1.0

    def build_from_memberships(self, ids) -> "BehavioralGraph":
        """Stream a sequence of per-step membership ids and add an edge on every
        direct membership change. `ids` may contain -1 (no centroid) which breaks
        the chain. Episode boundaries should be passed as -1 to avoid spurious
        cross-episode edges."""
        prev = -1
        for cur in ids:
            cur = int(cur)
            if cur >= 0 and prev >= 0 and cur != prev:
                self.add_transition(prev, cur)
            prev = cur
        return self

    def neighbors(self, i: int):
        for (a, b), w in self.edges.items():
            if a == i:
                yield b, w

    def undirected_edge_list(self) -> list[tuple[int, int, float]]:
        """Merged (i, j, weight) with i < j -- for rendering."""
        merged: dict[tuple[int, int], float] = {}
        for (a, b), w in self.edges.items():
            key = (min(a, b), max(a, b))
            merged[key] = merged.get(key, 0.0) + w
        return [(a, b, w) for (a, b), w in merged.items()]

    def dijkstra(self, src: int, dst: int) -> list[int] | None:
        """Shortest path (list of node ids, inclusive) over directed edges with
        cost 1/weight, or None if unreachable."""
        if src == dst:
            return [src]
        if src not in self.nodes or dst not in self.nodes:
            return None
        dist: dict[int, float] = {src: 0.0}
        prev: dict[int, int] = {}
        pq: list[tuple[float, int]] = [(0.0, src)]
        visited: set[int] = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break
            for v, w in self.neighbors(u):
                nd = d + 1.0 / max(w, 1e-6)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dst not in prev and dst != src:
            return None
        # Reconstruct.
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def to_dict(self) -> dict:
        return {"nodes": sorted(self.nodes),
                "edges": {f"{a},{b}": w for (a, b), w in self.edges.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "BehavioralGraph":
        g = cls()
        g.nodes = set(d["nodes"])
        for key, w in d["edges"].items():
            a, b = key.split(",")
            g.edges[(int(a), int(b))] = w
        return g

    def reachable_from(self, src: int) -> set[int]:
        """All nodes reachable from src (for picking a valid random goal)."""
        seen = {src}
        stack = [src]
        while stack:
            u = stack.pop()
            for v, _ in self.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        seen.discard(src)
        return seen
