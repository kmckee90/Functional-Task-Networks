"""Root mask-search regularizers: output entropy and mean logits on seen atoms."""
import math
import torch
import torch.nn.functional as F
from projects.bayesian_adv.overlap_regularizer import OverlapRegularizer as _Shared


class OverlapRegularizer(_Shared):
    def configure_entropy(self, coefficient=0.0):
        if not math.isfinite(coefficient):
            raise ValueError("reconfig_entropy_coef must be finite")
        self.reconfig_entropy_coef = coefficient

    def _output_entropy(self, pred):
        # Use fp32 for low-precision logits; retain fp64 if supplied.
        logits = pred.float() if pred.dtype in (torch.float16, torch.bfloat16) else pred
        if self.binary:
            # Independent Bernoulli outputs, averaged over examples and cells.
            return (logits.sigmoid() * F.softplus(-logits)
                    + (-logits).sigmoid() * F.softplus(logits)).mean()
        logp = F.log_softmax(logits, dim=-1)
        return -(logp.exp() * logp).sum(dim=-1).mean()

    def _reconfig_task_loss(self, pred, y):
        loss = self.loss(pred, y)
        coef = self.reconfig_entropy_coef
        if coef == 0 or not (self.classification or self.binary):
            return loss
        return loss + coef * self._output_entropy(pred)

    def _regularized_search(self, reference):
        return reference is not None or (self.reconfig_entropy_coef != 0 and
                                         (self.classification or self.binary))

    def configure_overlap(self, mode="seen_logits", weight=0.0, threshold=0.5, temperature=0.2):
        # Old selectors remain load/config compatible, but all now mean the
        # requested linear seen-atom cost. Threshold/temperature are unused.
        if mode not in ("seen_logits", "seen_mass", "bump", "gated"):
            raise ValueError("overlap mode must be seen_logits (legacy selectors accepted)")
        super().configure_overlap("bump", weight, threshold, temperature)
        self.overlap_mode = "seen_logits"
        if not hasattr(self, "train_mask_archive"):
            self.register_buffer("train_mask_archive", torch.empty((0, 0), dtype=torch.bool))
        self._archive_last_index = None

    @torch.no_grad()
    def record_training_mask(self):
        """Call after a weight update, never from reconfiguration/evaluation.

        Keep all distinct hard masks, even with lambda=0 or replay disabled.
        Boolean storage is independent of the bounded data replay buffer.
        """
        mask = self.m.detach().reshape(-1) > 0.5
        if not mask.any():
            return False
        archive = self.train_mask_archive
        if not archive.numel():
            self.train_mask_archive = mask.unsqueeze(0).clone()
            self._archive_last_index = 0
            return True
        if archive.shape[1] != mask.numel():
            raise ValueError("Training mask width changed without migrating the archive")
        last = self._archive_last_index
        if last is not None and last < len(archive) and torch.equal(archive[last], mask):
            return False
        matches = (archive == mask).all(dim=1).nonzero().flatten()
        if len(matches):
            self._archive_last_index = int(matches[0])
            return False
        self.train_mask_archive = torch.cat([archive, mask.unsqueeze(0)])
        self._archive_last_index = len(archive)
        return True

    def _overlap_reference(self, length=None):
        if self.overlap_lambda == 0 or not self.train_mask_archive.numel():
            return None
        reference = self.train_mask_archive[:, :length] if length is not None else self.train_mask_archive
        seen = reference.any(dim=0)
        return seen.detach().to(dtype=self.m.dtype).clone() if seen.any() else None

    def _overlap_penalty(self, logits, reference):
        if reference is None:
            return logits.new_zeros(())
        # Arithmetic mean over ALL mask positions, including unseen zeros.
        return self.overlap_lambda * (logits.reshape(-1) * reference).mean()

    def copy_overlap_archive_from(self, other):
        """Preserve masks when FTN expands its bank; new atoms were inactive."""
        archive = other.train_mask_archive
        self.train_mask_archive = torch.zeros((len(archive), self.m.numel()), dtype=torch.bool, device=self.m.device)
        if archive.numel():
            self.train_mask_archive[:, :archive.shape[1]].copy_(archive)
        self._archive_last_index = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "train_mask_archive"
        if key in state_dict:
            saved = state_dict[key]
            if saved.ndim != 2 or (saved.numel() and saved.shape[1] != self.m.numel()):
                raise ValueError("Invalid training mask archive shape")
            self.train_mask_archive = torch.empty(saved.shape, dtype=torch.bool, device=self.m.device)
        else:
            # Old checkpoints have no historical archive; do not invent one.
            state_dict[key] = torch.empty((0, 0), dtype=torch.bool, device=self.m.device)
            self.train_mask_archive = torch.empty_like(state_dict[key])
        self._archive_last_index = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

__all__ = ["OverlapRegularizer"]
