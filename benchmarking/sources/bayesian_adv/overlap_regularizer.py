"""Training-search preference for separating from, or reusing, the entry mask."""
import math
import torch


class OverlapRegularizer:
    def configure_overlap(self, mode="bump", weight=0.0, threshold=0.5, temperature=0.2):
        if mode not in ("bump", "gated"):
            raise ValueError("overlap mode must be 'bump' or 'gated'")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("overlap lambda must be finite and non-negative")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("overlap threshold must be in [0, 1]")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("overlap temperature must be finite and positive")
        self.overlap_mode = mode
        self.overlap_lambda = weight
        self.overlap_threshold = threshold
        self.overlap_temperature = temperature

    def _overlap_reference(self):
        # No RNG, graph, persistent archive, or additional checkpoint state.
        if self.overlap_lambda == 0 or not torch.any(self.m):
            return None
        return self.m.detach().clone()

    def _overlap_penalty(self, mask, reference):
        if reference is None:
            return mask.new_zeros(())
        # Normalize by total reference activity, not self.k: stacked models
        # have K active atoms per block. Soft search masks also stay in [0,1].
        overlap = (mask * reference).sum() / reference.sum()
        if self.overlap_mode == "bump":
            cost = overlap * (1 - overlap)
        else:
            cost = overlap * torch.sigmoid(
                (self.overlap_threshold - overlap) / self.overlap_temperature)
        return self.overlap_lambda * cost

    def _with_overlap(self, loss, mask, reference):
        # Preserve the exact original objective/graph when disabled.
        return loss if reference is None else loss + self._overlap_penalty(mask, reference)
