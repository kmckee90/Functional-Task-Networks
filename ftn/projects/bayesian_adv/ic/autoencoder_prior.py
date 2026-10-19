"""A shared easy-task FTN mask gates a separate predictor-logit FTN.

The Base FTN autoencodes x with reactive reconfiguration and no replay. Its
hard mask also gates an independent Base FTN trained to predict the Bayesian
predictor's posterior logits. At evaluation, a few unlabeled inputs refit the
autoencoder mask, then the second FTN emits a prior for the hard predictor.
Neither auxiliary network receives labels from an old domain again.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from ftn import FTN, kwta


class AutoencoderLogitPrior:
    def __init__(self, config, input_size, logit_size, groups=1):
        self.config = config
        self.groups = groups
        side = config.AE_SQUARE_DIM
        hidden = side * side
        common = dict(input_size=input_size, hidden_size=hidden,
                      square_dim=side, classification=False,
                      layers=config.AE_LAYERS, inner_size=config.AE_INNER_DIM,
                      dropout_p=0.0, k=config.AE_K,
                      reconfig_steps=config.AE_RECONFIG_STEPS,
                      reconfig_lr=config.AE_RECONFIG_LR,
                      reconfig_sample_size=config.T,
                      reconfig_algorithm=config.AE_RECONFIG_ALGORITHM)
        self.autoencoder = FTN(output_size=input_size, **common)
        self.logit_net = FTN(output_size=logit_size, **common)
        self.a_optimizer = torch.optim.Adam(self.autoencoder.parameters(),
                                             lr=config.AE_TRAIN_LR)
        self.l_optimizer = torch.optim.Adam(self.logit_net.parameters(),
                                             lr=config.AE_LOGIT_LR)
        self.r_mean = None
        self.r_var = 0.0
        self.r_since = 0
        self.r_warmup = int(np.ceil(1.0 / (1.0 - config.AE_REACTIVE_EMA)))
        self.cache = []
        self.cached_rows = 0
        self.train_masks = {}
        self.train_targets = {}
        self.stats = dict(a_changepoints=0, evals=0)
        self.a_train_losses = []
        self.l_train_losses = []
        self.a_eval_losses = []
        self.a_mask_recalls = []
        self.logit_cosines = []
        self.logit_topk_recalls = []
        self.a_cross_domain_overlaps = []
        self.l_fresh_fractions = []
        self.teacher_query_accs = []
        self.direct_query_accs = []
        self.by_task = defaultdict(lambda: defaultdict(list))
        self.last_eval_logits = None

    def _normalize(self, logits):
        rows = logits.reshape(self.groups, -1)
        rows = rows - rows.mean(-1, keepdim=True)
        rows = rows / rows.std(-1, keepdim=True).clamp_min(1e-5)
        return rows.clamp(-self.config.AE_LOGIT_CLIP,
                          self.config.AE_LOGIT_CLIP).flatten() * self.config.AE_LOGIT_SCALE

    def train_step(self, x, y=None):
        """Base + Reactive_Low + NoReplay on the input reconstruction task."""
        c = self.config
        with torch.no_grad():
            detector_loss = self.autoencoder.loss(self.autoencoder(x), x).item()
        sd = float(np.sqrt(max(self.r_var, 0.0)))
        changepoint = self.r_mean is None or (
            self.r_since >= self.r_warmup and
            detector_loss > self.r_mean + c.AE_REACTIVE_SD * sd)
        if changepoint:
            self.autoencoder.reconfigure_changepoint(x, x)
            self.stats["a_changepoints"] += 1
            self.r_since = 0
        prediction = self.autoencoder(x)
        loss = self.autoencoder.loss(prediction, x)
        value = loss.item()
        self.a_optimizer.zero_grad()
        loss.backward()
        self.a_optimizer.step()
        self.a_train_losses.append(value)
        if changepoint:
            self.r_mean, self.r_var = value, 0.0
        else:
            diff = value - self.r_mean
            alpha = c.AE_REACTIVE_EMA
            self.r_mean = alpha * self.r_mean + (1 - alpha) * value
            self.r_var = alpha * self.r_var + (1 - alpha) * diff * diff
        self.r_since += 1

        if self.cached_rows < c.AE_LOGIT_CACHE_ROWS:
            take = min(c.AE_LOGIT_CACHE_ROWS - self.cached_rows, x.shape[0])
            self.cache.append((x[:take].detach().clone(),
                               None if y is None else y[:take].detach().clone()))
            self.cached_rows += take

    def train_block_end(self, predictor, task):
        """Distill the current predictor posterior under the borrowed mask."""
        if not self.cache:
            return
        c = self.config
        x = torch.cat([pair[0] for pair in self.cache])
        y = (torch.cat([pair[1] for pair in self.cache])
             if self.cache[0][1] is not None else None)
        a = self.autoencoder.m.detach().clone()
        if c.AE_TEACHER_MODE == "cold_mask":
            if y is None:
                raise ValueError("cold-mask teacher needs labeled current-task data")
            saved_predictor = predictor.get_reconfig_state()
            try:
                predictor.cold_reconfigure(
                    x, y, steps=c.AE_TEACHER_STEPS, lr=c.AE_TEACHER_LR,
                    sample_size=None, temperature=1.0,
                    restarts=c.AE_TEACHER_RESTARTS, take_best=True)
                teacher_mask = predictor.m.detach().clone()
            finally:
                predictor.set_reconfig_state(saved_predictor)
            target = torch.where(teacher_mask > 0.5,
                                 torch.full_like(teacher_mask, c.AE_LOGIT_SCALE),
                                 torch.full_like(teacher_mask, -c.AE_LOGIT_SCALE))
        elif c.AE_TEACHER_MODE == "posterior":
            target = self._normalize(predictor.mask_logits.detach())
            teacher_mask = self._topk(predictor, target)
        else:
            raise ValueError("AE_TEACHER_MODE must be posterior or cold_mask")
        for old_task, old_mask in self.train_masks.items():
            if old_task != task:
                self.a_cross_domain_overlaps.append(
                    ((a * old_mask).sum() / a.sum().clamp_min(1)).item())
        frozen = torch.zeros_like(a, dtype=torch.bool)
        if c.AE_LOGIT_FREEZE_OLD:
            for old_task, old_mask in self.train_masks.items():
                if old_task != task:
                    frozen |= old_mask > 0.5
        self.l_fresh_fractions.append(
            ((a * (~frozen)).sum() / a.sum().clamp_min(1)).item())
        self.train_masks[task] = a
        self.train_targets[task] = target.detach().clone()
        self.l_optimizer.state = defaultdict(dict)
        weight = 1.0 + (c.AE_LOGIT_POSITIVE_WEIGHT - 1.0) * teacher_mask
        for _ in range(c.AE_LOGIT_TRAIN_STEPS):
            rows = torch.randint(0, x.shape[0],
                                 (min(c.AE_LOGIT_BATCH_SIZE, x.shape[0]),),
                                 device=x.device)
            pred = self.logit_net(x[rows], m=a)
            loss = ((pred - target.unsqueeze(0)).square() * weight).mean()
            self.l_optimizer.zero_grad()
            loss.backward()
            if c.AE_LOGIT_FREEZE_OLD:
                for parameter in (*self.logit_net.parallel.weights,
                                  *self.logit_net.parallel.biases):
                    parameter.grad[frozen] = 0
                self.logit_net.linear.weight.grad[:, frozen] = 0
            self.l_optimizer.step()
            self.l_train_losses.append(loss.item())
        self.cache = []
        self.cached_rows = 0

    def _topk(self, predictor, logits):
        if hasattr(predictor, "_topk_mask"):
            return predictor._topk_mask(logits)
        return kwta(logits, predictor.k)

    def eval_reconfigure(self, predictor, x, y, task=None):
        """Infer a from x->x, predict L under a, use L only at evaluation."""
        c = self.config
        xs, ys = x[:c.IC_SUPPORT_SIZE], y[:c.IC_SUPPORT_SIZE]
        if c.AE_EVAL_STRATEGY == "zero":
            self.last_eval_logits = None
            return predictor.cold_reconfigure(xs, ys, **c.EVAL_FTN_PARS)

        saved_a = self.autoencoder.get_reconfig_state()
        try:
            self.autoencoder.cold_reconfigure(
                xs, xs, steps=c.AE_EVAL_RECONFIG_STEPS,
                lr=c.AE_EVAL_RECONFIG_LR, sample_size=None,
                temperature=c.AE_EVAL_TEMPERATURE,
                restarts=c.AE_EVAL_RESTARTS, take_best=True)
            a = self.autoencoder.m.detach().clone()
            with torch.no_grad():
                self.a_eval_losses.append(
                    self.autoencoder.loss(self.autoencoder(xs, m=a), xs).item())
                logits = self.logit_net(xs, m=a).mean(0).detach()
            self.stats["evals"] += 1
            self.last_eval_logits = logits
            if task in self.train_masks:
                old_a = self.train_masks[task]
                overlap = ((a * old_a).sum() /
                           old_a.sum().clamp_min(1)).item()
                self.a_mask_recalls.append(overlap)
                self.by_task[task]["a_mask_recall"].append(overlap)
                old_l = self.train_targets[task]
                cosine = F.cosine_similarity(
                    logits.unsqueeze(0), old_l.unsqueeze(0)).item()
                self.logit_cosines.append(cosine)
                self.by_task[task]["logit_cosine"].append(cosine)
                selected = self._topk(predictor, logits)
                target = self._topk(predictor, old_l)
                recall = ((selected * target).sum() /
                          target.sum().clamp_min(1)).item()
                self.logit_topk_recalls.append(recall)
                self.by_task[task]["logit_topk_recall"].append(recall)

            if c.AE_EVAL_STRATEGY == "direct":
                predictor.m.copy_(self._topk(predictor, logits))
                return predictor.m
            saved_l = predictor.mask_logits.detach().clone()
            try:
                predictor.mask_logits.copy_(logits)
                predictor._posterior_update(
                    xs, ys, steps=c.AE_PRIOR_STEPS,
                    lr=c.EVAL_RECONFIG_LR, sample_size=None,
                    kl_weight=c.AE_PRIOR_KL,
                    temperature=c.AE_PRIOR_TEMPERATURE,
                    prior_decay=1.0, stochastic_mask=False,
                    changepoint=False)
            finally:
                predictor.mask_logits.copy_(saved_l)
            return predictor.m
        finally:
            self.autoencoder.set_reconfig_state(saved_a)

    def record_query(self, predictor, x, y, task):
        """Diagnostics only: score old teacher and predicted masks on query."""
        if task not in self.train_targets or self.last_eval_logits is None:
            return
        with torch.no_grad():
            teacher = self._topk(predictor, self.train_targets[task])
            direct = self._topk(predictor, self.last_eval_logits)
            target = y.argmax(-1)
            teacher_acc = (predictor(x, m=teacher).argmax(-1) == target).float().mean().item()
            direct_acc = (predictor(x, m=direct).argmax(-1) == target).float().mean().item()
        self.teacher_query_accs.append(teacher_acc)
        self.direct_query_accs.append(direct_acc)
        self.by_task[task]["teacher_query_acc"].append(teacher_acc)
        self.by_task[task]["direct_query_acc"].append(direct_acc)

    def metrics(self):
        result = {f"ae_{key}": value for key, value in self.stats.items()}
        for key, values in (
            ("ae_train_mse", self.a_train_losses),
            ("ae_logit_train_mse", self.l_train_losses),
            ("ae_eval_mse", self.a_eval_losses),
            ("ae_mask_recall", self.a_mask_recalls),
            ("ae_logit_cosine", self.logit_cosines),
            ("ae_logit_topk_recall", self.logit_topk_recalls),
            ("ae_cross_domain_mask_overlap", self.a_cross_domain_overlaps),
            ("ae_logit_fresh_fraction", self.l_fresh_fractions),
            ("ae_teacher_query_acc", self.teacher_query_accs),
            ("ae_direct_query_acc", self.direct_query_accs),
        ):
            result[key] = float(np.mean(values)) if values else None
        for task, fields in self.by_task.items():
            for key, values in fields.items():
                result[f"ae_t{task}_{key}"] = float(np.mean(values))
        return result
