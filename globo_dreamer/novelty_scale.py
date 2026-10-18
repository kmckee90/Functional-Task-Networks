"""Causal, checkpointable scaling of nonnegative ensemble standard deviation.

Input adapts slowly; objective scale never decreases, so improved agreement
does not get amplified back into a constant curiosity reward. No centering:
zero disagreement always means zero input and zero reward.
"""
import math


class NoveltyScale:
    def __init__(self):
        self.second = .1 ** 2
        self.reward_scale = .1
        self.count = 0

    def transform(self, raw):
        raw = float(raw)
        if not math.isfinite(raw) or raw < 0:
            raise ValueError('Disagreement must be finite and nonnegative')
        scale = max(.001, math.sqrt(self.second))
        # Compute using PAST statistics, before updating with this sample.
        feedback = math.tanh(raw / (3 * scale))
        reward = .1 * min(raw / self.reward_scale, 3.)
        # Winsorize the statistics too: one outlier cannot ruin future scaling.
        capped = min(raw, 3 * scale)
        self.second = .999 * self.second + .001 * capped ** 2
        self.reward_scale = max(self.reward_scale, math.sqrt(self.second))
        self.count += 1
        return feedback, reward

    def save(self):
        return vars(self).copy()

    def load(self, state):
        self.__dict__.update(state)
