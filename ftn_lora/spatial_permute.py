"""Deterministic, display-only consolidation of a shared task-mask layout."""

import numpy as np


def _geometry(shape, radius, roundness):
    """Undirected grid edges: four-neighbor adjacency plus a circular kernel."""
    height, width = shape
    grid = np.arange(height * width).reshape(shape)
    offsets = []
    for dy in range(min(radius, height - 1) + 1):
        for dx in range(-min(radius, width - 1), min(radius, width - 1) + 1):
            distance2 = dy * dy + dx * dx
            if (dy == 0 and dx <= 0) or not 0 < distance2 <= radius * radius:
                continue
            offsets.append((dy, dx, np.exp(-2 * distance2 / radius**2)))
    # Match the circular term's interior weight to the four-neighbor term.
    total = sum(weight for _, _, weight in offsets)
    left, right, weights = [], [], []
    for dy, dx, weight in offsets:
        x0, x1 = max(0, -dx), min(width, width - dx)
        a = grid[:height - dy, x0:x1].ravel()
        b = grid[dy:, x0 + dx:x1 + dx].ravel()
        w = float(dy * dy + dx * dx == 1) + roundness * 2 * weight / total
        if w:
            left.append(a)
            right.append(b)
            weights.append(np.full(a.size, w))
    if not left:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    return np.concatenate(left), np.concatenate(right), np.concatenate(weights)


def _random_blob_order(bits, shape, rng):
    """Place small membership groups around random free sites, then fill blanks."""
    n = bits.shape[1]
    yy, xx = np.indices(shape)
    coordinates = np.column_stack((yy.ravel(), xx.ravel()))
    occupied = bits.any(axis=0)
    _, labels = np.unique(bits.T, axis=0, return_inverse=True)
    chunks = []
    # Split large signatures so even one task can start as several blobs.
    chunk_size = max(1, n // 16)
    for label in np.unique(labels[occupied]):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        chunks.extend(members[i:i + chunk_size] for i in range(0, len(members), chunk_size))
    order = np.empty(n, dtype=int)
    free = np.ones(n, dtype=bool)
    for index in rng.permutation(len(chunks)):
        members = chunks[index]
        available = np.flatnonzero(free)
        origin = coordinates[rng.choice(available)]
        distance = ((coordinates[available] - origin)**2).sum(axis=1)
        distance = distance + rng.random(len(available)) * .5
        sites = available[np.argpartition(distance, len(members) - 1)[:len(members)]]
        order[sites] = members
        free[sites] = False
    order[free] = rng.permutation(np.flatnonzero(~occupied))
    return order


def spatial_permutation(masks, shape, previous=None, sweeps=8, *,
                        roundness=2.0, center=0.4, radius=3, temperature=0.05,
                        initialization="default", seed=0):
    """Return display-position -> original-position indices for flattened grids.

    Maximize mean per-task active-neighbor affinity (four-neighbor plus a
    distance-weighted circular kernel), minus the union's squared distance
    from the canvas center. Complete membership signatures move together, so
    all intersections survive. Disconnected and irregular task regions are
    allowed. No wraparound. Set roundness=center=temperature=0 for adjacency
    alone. ``sweeps`` sets the budget (eight proposals per pixel per sweep).

    Seeds include identity, previous, serpentine grouping, and central sectors.
    The first half of the budget anneals; the rest greedily refines the best
    layout found. Return the best observed layout, never worse than the seeds
    under this combined objective. A private RNG leaves training untouched.
    ``random_blobs`` instead starts from scattered compact membership groups,
    or exclusively from ``previous`` on later calls. It never selects the
    sector/serpentine seeds, allowing the same layout to evolve over task blocks.
    """
    if initialization not in ("default", "random_blobs"):
        raise ValueError("initialization must be default or random_blobs")
    if (not isinstance(radius, (int, np.integer)) or radius < 1
            or not isinstance(sweeps, (int, np.integer)) or sweeps < 0):
        raise ValueError("radius must be a positive integer; sweeps a nonnegative integer")
    if any(not np.isfinite(v) or v < 0 for v in (roundness, center, temperature)):
        raise ValueError("roundness, center, and temperature must be finite and nonnegative")
    bits = np.asarray(masks) > 0
    height, width = shape
    n = height * width
    if height < 1 or width < 1 or bits.ndim != 2 or bits.shape[1] != n:
        raise ValueError("masks must have shape (tasks, height * width) on a nonempty grid")
    identity = np.arange(n)
    if previous is not None:
        previous = np.asarray(previous)
        if previous.shape != (n,) or not np.array_equal(np.sort(previous), identity):
            raise ValueError("previous must be a permutation of grid positions")
    if n < 2 or not bits.shape[0]:
        return identity
    signatures = bits.T.astype(float) / np.sqrt(bits.shape[0])
    occupied = bits.any(axis=0).astype(float)
    grid = identity.reshape(shape)
    left, right, weights = _geometry(shape, radius, roundness)
    yy, xx = np.indices(shape, dtype=float)
    yy -= (height - 1) / 2
    xx -= (width - 1) / 2
    distance2 = (xx**2 + yy**2).ravel()
    central_cost = center * distance2 / max(distance2.max(), 1)

    def score(order):
        affinity = np.einsum('ij,ij->i', signatures[order[left]], signatures[order[right]])
        return weights @ affinity - central_cost @ occupied[order]

    grouped = np.lexsort(bits[::-1])
    snake = grid.copy()
    snake[1::2] = snake[1::2, ::-1]
    candidate = identity.copy()
    candidate[snake.ravel()] = grouped
    candidates = [identity, candidate]
    # Pack the union into the closest pixels and place signature groups in
    # angular sectors. This supplies a compact seed without imposing topology.
    active = grouped[occupied[grouped] > 0]
    central = np.argsort(distance2, kind="stable")
    sites = central[:len(active)]
    sites = sites[np.argsort(np.arctan2(yy.ravel()[sites], xx.ravel()[sites]), kind="stable")]
    packed = identity.copy()
    packed[sites] = active
    packed[central[len(active):]] = grouped[occupied[grouped] == 0]
    candidates.append(packed)
    if previous is not None:
        candidates.append(previous)
    if initialization == "random_blobs":
        order = (previous.copy() if previous is not None else
                 _random_blob_order(bits, shape, np.random.default_rng(seed)))
    else:
        order = max(candidates, key=score).copy()
    best = order.copy()
    best_score = current_score = score(order)

    neighbors = [[] for _ in range(n)]
    neighbor_weights = [[] for _ in range(n)]
    for a, b, w in zip(left, right, weights):
        neighbors[a].append(b)
        neighbors[b].append(a)
        neighbor_weights[a].append(w)
        neighbor_weights[b].append(w)
    neighbors = [np.array(row, dtype=int) for row in neighbors]
    neighbor_weights = [np.array(row) for row in neighbor_weights]

    def fields():
        return np.array([w @ signatures[order[ns]]
                         for ns, w in zip(neighbors, neighbor_weights)])

    field = fields()
    rng = np.random.default_rng(seed)
    anneal_sweeps = sweeps // 2 if temperature else 0
    for sweep in range(sweeps):
        if sweep == anneal_sweeps:
            order = best.copy()
            current_score = best_score
            field = fields()
        heat = temperature * (0.02 ** (sweep / max(anneal_sweeps - 1, 1))) if sweep < anneal_sweeps else 0
        for a in rng.permutation(n):
            local = rng.choice(neighbors[a], size=4) if len(neighbors[a]) else []
            for b in [*local, *rng.integers(n, size=4)]:
                if a == b:
                    continue
                difference = signatures[order[b]] - signatures[order[a]]
                # Exclude the a-b edge: swapping its endpoints cannot change it.
                mutual = neighbor_weights[a][neighbors[a] == b].sum()
                delta = difference @ (field[a] - field[b]) - mutual * (difference @ difference)
                delta -= (occupied[order[b]] - occupied[order[a]]) * (central_cost[a] - central_cost[b])
                if delta > 1e-12 or (heat and delta < -1e-12 and rng.random() < np.exp(delta / heat)):
                    field[neighbors[a]] += neighbor_weights[a][:, None] * difference
                    field[neighbors[b]] -= neighbor_weights[b][:, None] * difference
                    order[a], order[b] = order[b], order[a]
                    current_score += delta
                    if current_score > best_score + 1e-10:
                        best_score = current_score
                        best = order.copy()
    return best
