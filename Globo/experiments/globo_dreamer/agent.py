"""DreamerV3 with density feedback, action-conditioned disagreement, two values.

Uses the pinned upstream RSSM/optimizer/return implementation. The world reward
head learns only real survival reward. Intrinsic rewards are evaluated separately
in imagination; no synthetic transition is inserted into real replay.
"""
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3.agent import Agent as Dreamer, imag_loss, repl_loss, sg, sample, concat, prefix


def prediction_error_loss(error_predictions, selection, observed_error, valid):
    selected = (error_predictions * selection).sum(-1)
    return (((selected - sg(observed_error)) ** 2) * valid).sum() / jnp.maximum(valid.sum(), 1)


class Disagreement(nj.Module):
    def __init__(self, count, actions):
        self.actions = actions
        self.proj = nn.Linear(64, name='projection')
        self.heads = [embodied.jax.MLPHead(
            elements.Space(np.float32, (72,)), 'mse', units=128, layers=1,
            act='silu', norm='rms', name=f'head{i}') for i in range(count)]

    def __call__(self, features):
        x = nn.cast(sg(features))
        x = self.proj(x)
        shape = x.shape[:-1]
        state = jnp.broadcast_to(x[..., None, :], (*shape, self.actions, x.shape[-1]))
        actions = jnp.broadcast_to(jnp.eye(self.actions, dtype=x.dtype), (*shape, self.actions, self.actions))
        inp = jnp.concatenate([state, actions], -1)
        return jnp.stack([head(inp, inp.ndim - 1).pred() for head in self.heads], 0).astype(jnp.float32)


class Agent(Dreamer):
    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        self.model_only = config.explore.get('model_only', False)
        self.novelty_only = config.explore.get('novelty_only', False)
        scalar = elements.Space(np.float32, ())
        self.ensemble = Disagreement(config.explore.ensemble, int(act_space['action'].classes), name='ensemble')
        self.density = embodied.jax.MLPHead(scalar, 'mse', units=128, layers=1, name='density')
        self.use_error = config.explore.get('prediction_error', 0.) > 0
        if self.use_error:
            self.error_head = embodied.jax.MLPHead(
                elements.Space(np.float32, (int(act_space['action'].classes),)),
                'mse', units=128, layers=1, name='error_head')
        self.ival = embodied.jax.MLPHead(scalar, **config.value, name='ival')
        self.slowival = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name='slowival'),
            source=self.ival, **config.slowvalue)
        self.iretnorm = embodied.jax.Normalize(**config.retnorm, name='iretnorm')
        self.ivalnorm = embodied.jax.Normalize(**config.valnorm, name='ivalnorm')
        self.iadvnorm = embodied.jax.Normalize(**config.advnorm, name='iadvnorm')
        self.modules += [self.ensemble, self.density, self.ival]
        if self.use_error:
            self.modules += [self.error_head]
        if self.model_only:
            self.modules = [self.enc, self.dyn, self.dec, self.rew, self.con,
                            self.ensemble, self.density]
            if self.use_error:
                self.modules += [self.error_head]
        if self.novelty_only:
            self.modules = [self.enc,self.dyn,self.dec,self.rew,self.con,
                            self.pol,self.val,self.ensemble]
        self.opt = embodied.jax.Optimizer(self.modules, self._make_opt(**config.opt),
                                          summary_depth=1, name='opt')

    @property
    def policy_keys(self):
        return '^(enc|dyn|dec|pol|density|ensemble)/'

    def policy_input(self, feat, density=None):
        x = self.feat2tensor(feat)
        disagreement = jnp.sqrt(self.ensemble(x).var(0).mean(-1) + 1e-8)
        if self.novelty_only:
            return x, disagreement
        if density is None:
            density = self.density(sg(x), x.ndim - 1).pred()[..., None]
        extra = jnp.concatenate([jnp.clip(density, 0, 1), jnp.clip(disagreement, 0, 1)], -1)
        if not self.config.explore.feedback:
            extra = jnp.zeros_like(extra)
        return jnp.concatenate([x, sg(extra)], -1), disagreement

    def policy(self, carry, obs, mode='train'):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        reset = obs['is_first']
        enc_carry, _, tokens = self.enc(enc_carry, obs, reset, training=False, single=True)
        dyn_carry, _, feat = self.dyn.observe(dyn_carry, tokens, prevact, reset, training=False, single=True)
        inp, disagreement = self.policy_input(feat, obs['density'])
        if self.model_only:
            # Placeholder replaced by the independently learning density explorer.
            action = {'action': jnp.zeros(reset.shape, jnp.int32)}
        else:
            policy = self.pol(inp, 1)
            action = sample(policy) if mode == 'train' else jax.tree.map(lambda d: d.pred(), policy)
        out = dict(uncertainty=disagreement.mean(-1))
        if self.config.explore.get('reward_feedback',False):
            out['action_disagreement']=disagreement
        return (enc_carry, dyn_carry, dec_carry, action), action, out

    @property
    def ext_space(self):
        spaces=dict(uncertainty=elements.Space(np.float32), **super().ext_space)
        if self.config.explore.get('reward_feedback',False):
            spaces['action_disagreement']=elements.Space(np.float32,(int(self.act_space['action'].classes),))
        return spaces

    def train(self, carry, data):
        if self.model_only:
            if self.config.replay_context:
                raise ValueError('Model-only training requires replay_context=0')
            carry, obs, prevact, _ = self._apply_replay_context(carry, data)
            metrics, (carry, entries, outs, mets) = self.opt(
                self.loss, carry, obs, prevact, training=True, has_aux=True)
            metrics.update(mets)
            carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
            return carry, {}, metrics
        result = super().train(carry, data)
        if not self.novelty_only:self.slowival.update()
        return result

    def report(self, carry, data):
        if self.model_only or self.novelty_only:
            from .world_probe import ProbeAgent
            return ProbeAgent.report(self, carry, data)
        return super().report(carry, data)

    def loss(self, carry, obs, prevact, training):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs['is_first']
        B, T = reset.shape
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, losses, repfeat, metrics = self.dyn.loss(
            dyn_carry, tokens, prevact, reset, training)
        losses = dict(losses)
        dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)
        real = self.feat2tensor(repfeat)
        losses['rew'] = self.rew(sg(real, skip=self.config.reward_grad), 2).loss(obs['reward'])
        cont = (~obs['is_terminal']).astype(jnp.float32)
        if self.config.contdisc:
            cont *= 1 - 1 / self.config.horizon
        losses['con'] = self.con(real, 2).loss(cont)
        for key, recon in recons.items():
            value = obs[key].astype(jnp.float32)
            if self.obs_space[key].dtype == np.uint8:
                value /= 255
            losses[key] = recon.loss(sg(value))
        density_loss = (jnp.zeros((B,T),jnp.float32) if self.novelty_only else
            self.density(sg(real), 2).loss(obs['density'][..., 0]))
        # Predict stable next-image block means and bodily signals, given action.
        img = obs['image'].astype(jnp.float32) / 255
        pixels = img.reshape(B, T, 8, 8, 8, 8, 3).mean((3, 5, 6)).reshape(B, T, 64)
        target = sg(jnp.concatenate([pixels, obs['body'][..., :8]], -1))
        forecasts = self.ensemble(real[:, :-1])
        action = prevact['action'][:, 1:]
        selection = jax.nn.one_hot(action, forecasts.shape[-2])
        predictions = (forecasts * selection[None, ..., None]).sum(-2)
        valid = (~reset[:, 1:]).astype(jnp.float32)
        bootstrap = jax.random.bernoulli(nj.seed(), .8, predictions.shape[:-1]).astype(jnp.float32)
        weights = bootstrap * valid[None]
        errors = (predictions - target[None, :, 1:]) ** 2
        ensemble_loss = (errors.mean(-1) * weights).sum() / jnp.maximum(weights.sum(), 1)
        # Expected one-step prediction error, conditioned on state and action.
        # Targets use the ensemble MEAN (not average per-head error, which would
        # double-count disagreement). Both target and RSSM features are detached.
        error_loss = jnp.float32(0.)
        observed_error = jnp.sqrt(((predictions.mean(0) - target[:, 1:]) ** 2).mean(-1) + 1e-8)
        if self.use_error:
            error_predictions = self.error_head(sg(real[:, :-1]), 2).pred()
            error_loss = prediction_error_loss(error_predictions, selection, observed_error, valid)

        if self.model_only:
            total = sum(self.scales[k] * v.mean() for k, v in losses.items())
            total += density_loss.mean() + ensemble_loss + error_loss
            metrics = dict(metrics)
            metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
            metrics.update({'loss/ensemble': ensemble_loss, 'loss/density': density_loss.mean(),
                'loss/error_head': error_loss, 'explore/observed_prediction_error': observed_error.mean(),
                'explore/disagreement': jnp.sqrt(predictions.var(0).mean(-1)+1e-8).mean()})
            return total, ((enc_carry,dyn_carry,dec_carry),
                (enc_entries,dyn_entries,dec_entries),dict(tokens=tokens,repfeat=repfeat,losses=losses),metrics)

        K = min(self.config.imag_last or T, T)
        H = self.config.imag_length
        starts = self.dyn.starts(dyn_entries, dyn_carry, K)
        policyfn = lambda feat: sample(self.pol(self.policy_input(feat)[0], 1))
        _, imagined, previous = self.dyn.imagine(starts, policyfn, H, training)
        first = jax.tree.map(lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
        imagined = concat([sg(first), sg(imagined)], 1)
        last = policyfn(jax.tree.map(lambda x: x[:, -1], imagined))
        actions = concat([previous, jax.tree.map(lambda x: x[:, None], last)], 1)
        x = self.feat2tensor(imagined)
        policy_in, uncertainty = self.policy_input(imagined)
        chosen = jnp.take_along_axis(uncertainty, actions['action'][..., None], -1)[..., 0]
        nov = (jnp.zeros_like(chosen) if self.novelty_only else
            jnp.clip(self.density(sg(x), 2).pred(), 0, 1))
        disagreement_reward = jnp.clip(chosen, 0, 1) ** self.config.explore.get('disagreement_power', 1.)
        predicted_error = jnp.zeros_like(chosen)
        if self.use_error:
            predicted_error = jnp.take_along_axis(self.error_head(sg(x), 2).pred(),
                actions['action'][..., None], -1)[..., 0]
            predicted_error = jnp.clip(predicted_error, 0, 1)
        ir = (self.config.explore.density * nov
              + self.config.explore.disagreement * disagreement_reward
              + self.config.explore.get('prediction_error', 0.) * predicted_error)
        policy = self.pol(policy_in, 2)
        con = self.con(x, 2).prob(1)
        if self.novelty_only:
            # Upstream Dreamer actor/critic learns imagined disagreement returns.
            # Neither density nor external reward enters this actor objective.
            icon = jnp.clip(con/(1-1/self.config.horizon),0,1) if self.config.contdisc else con
            actor_critic, _, acmetrics = imag_loss(actions, disagreement_reward, icon,
                policy, self.val(x,2), self.slowval(x,2),
                self.retnorm,self.valnorm,self.advnorm,update=training,
                contdisc=False,horizon=self.config.explore.horizon,**self.config.imag_loss)
            total = sum(self.scales[k]*v.mean() for k,v in losses.items())+ensemble_loss
            total += actor_critic['value'].mean()+actor_critic['policy'].mean()
            metrics=dict(metrics)
            metrics.update({f'loss/{k}':v.mean() for k,v in losses.items()})
            metrics.update(prefix(acmetrics,'novelty_actor'))
            metrics.update({'loss/ensemble':ensemble_loss,
                'loss/novelty_value':actor_critic['value'].mean(),
                'loss/novelty_policy':actor_critic['policy'].mean(),
                'explore/disagreement':chosen.mean(),
                'explore/novelty_reward':disagreement_reward.mean(),
                'explore/observed_prediction_error':observed_error.mean(),
                'explore/dreamer_action_fraction':jnp.float32(1.),
                'explore/density_reward_weight':jnp.float32(0.),
                'explore/survival_actor_weight':jnp.float32(0.)})
            return total, ((enc_carry,dyn_carry,dec_carry),
                (enc_entries,dyn_entries,dec_entries),dict(tokens=tokens,repfeat=repfeat,losses=losses),metrics)
        ext, extout, extmets = imag_loss(actions, self.rew(x, 2).pred(), con,
            policy, self.val(x, 2), self.slowval(x, 2),
            self.retnorm, self.valnorm, self.advnorm, update=training,
            contdisc=self.config.contdisc, horizon=self.config.horizon, **self.config.imag_loss)
        # Recover continuation probability before applying the shorter intrinsic discount.
        icon = jnp.clip(con / (1 - 1 / self.config.horizon), 0, 1) if self.config.contdisc else con
        ikw = dict(self.config.imag_loss)
        ikw['actent'] = 0.
        intr, _, imets = imag_loss(actions, ir, icon, policy,
            self.ival(x, 2), self.slowival(x, 2), self.iretnorm, self.ivalnorm,
            self.iadvnorm, update=training, contdisc=False,
            horizon=self.config.explore.horizon, **ikw)
        # Anchor survival value to actual deaths, using upstream Dreamer's
        # replay-return objective. This was omitted in the first pilot.
        if self.config.repval_loss:
            replay_input = sg(real[:, -K:])
            replay_losses, _, _ = repl_loss(
                obs['is_last'][:, -K:], obs['is_terminal'][:, -K:],
                obs['reward'][:, -K:], extout['ret'][:, 0].reshape(B, K),
                self.val(replay_input, 2), self.slowval(replay_input, 2),
                self.valnorm, update=training, horizon=self.config.horizon,
                **self.config.repl_loss)
            losses.update(replay_losses)
        model_loss = sum(self.scales[k] * v.mean() for k, v in losses.items())
        total = (model_loss + density_loss.mean() + ensemble_loss + error_loss + ext['value'].mean()
                 + intr['value'].mean() + ext['policy'].mean()
                 + self.config.explore.actor_weight * intr['policy'].mean())
        metrics = dict(metrics)
        metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
        metrics.update(prefix(extmets, 'survival'))
        metrics.update(prefix(imets, 'intrinsic'))
        metrics.update({'loss/density': density_loss.mean(), 'loss/ensemble': ensemble_loss,
            'loss/value': ext['value'].mean(), 'loss/ivalue': intr['value'].mean(),
            'explore/density': nov.mean(), 'explore/disagreement': chosen.mean()})
        metrics.update({'loss/error_head': error_loss, 'explore/observed_prediction_error': observed_error.mean(),
            'explore/predicted_error': predicted_error.mean(),
            'explore/weighted_disagreement': self.config.explore.disagreement * disagreement_reward.mean(),
            'explore/weighted_prediction_error': self.config.explore.get('prediction_error', 0.) * predicted_error.mean()})
        terminal = obs['is_terminal'].astype(jnp.float32)
        metrics['survival/replay_terminal_count'] = terminal.sum()
        metrics['survival/terminal_reward_prediction_sum'] = (
            self.rew(sg(real), 2).pred() * terminal).sum()
        outs = dict(tokens=tokens, repfeat=repfeat, losses=losses)
        return total, ((enc_carry, dyn_carry, dec_carry),
                       (enc_entries, dyn_entries, dec_entries), outs, metrics)
