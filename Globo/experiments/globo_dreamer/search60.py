"""60k Dreamer novelty search. Config24 model-free explorer is immutable."""
import argparse
import math
from pathlib import Path
import numpy as np
from .search import Search
from .search_score import score_run

FIXED = dict(mf_gamma=0.9036665304295952, mf_epsilon_end=0.015366900171141888,
    mf_online_coef=0.8728610896464566, mf_frontier_scale=4.574998390699497,
    dreamer_action_probability=0.13372618381960508, mf_lr=0.0001899492997635358,
    mf_offline_coef=1., density=1.)
BASE = dict(FIXED, disagreement=1., prediction_error=0., disagreement_power=1.,
            actor_weight=.2, intrinsic_horizon=100.)
BOUNDS = dict(disagreement=(.1,100.), prediction_error=(.1,30.),
              disagreement_power=(.5,2.), actor_weight=(.05,1.), intrinsic_horizon=(20.,300.))


def propose(rng, parent=None, sigma=.18):
    params = dict(parent or BASE)
    keys = list(BOUNDS)
    chosen = keys if parent is None else list(rng.choice(keys, size=int(rng.integers(1,3)), replace=False))
    for key in chosen:
        if key in ('disagreement','prediction_error') and rng.random()<.2:
            params[key]=0.
            continue
        lo,hi=map(math.log,BOUNDS[key])
        current=max(BOUNDS[key][0],params[key])
        z=rng.random() if parent is None else (math.log(current)-lo)/(hi-lo)+rng.normal(0,sigma)
        z=z%2; z=2-z if z>1 else z
        params[key]=math.exp(lo+z*(hi-lo))
    params.update(FIXED)
    return params


def score60(root,steps=60000):
    result=score_run(root,steps,tail_windows=20)
    late=result['windows'][-20:]
    result.update(objective_version=2, evaluation_start=40000,
        post40k_new_spatial_cells=sum(w['new_4m_cells'] for w in late),
        post40k_new_visual_views=sum(w['new_visual_views'] for w in late),
        final10k_new_spatial_cells=sum(w['new_4m_cells'] for w in late[-10:]),
        final10k_new_visual_views=sum(w['new_visual_views'] for w in late[-10:]))
    return result


class DreamerSearch(Search):
    prefix='s60'
    objective_version=2
    propose=staticmethod(propose)
    score=staticmethod(score60)

    def initial_configs(self):
        return [BASE,
            dict(BASE,disagreement=0.),  # density-only imagined reward control
            dict(BASE,disagreement=10.),  # stronger ensemble standard deviation
            dict(BASE,disagreement=30.,disagreement_power=2.),  # variance
            dict(BASE,disagreement=0.,prediction_error=10.),  # expected residual
            dict(BASE,disagreement=10.,prediction_error=10.),  # combined
            dict(BASE,disagreement=10.,actor_weight=.5),
            dict(BASE,disagreement=10.,intrinsic_horizon=200.)]

    def add_config(self,params,parent,sigma,reason):
        if any(params.get(k)!=v for k,v in FIXED.items()):
            raise ValueError('Config24 explorer and action mixture must stay fixed')
        super().add_config(params,parent,sigma,reason)

    def __init__(self,args):
        if args.steps!=60000: raise ValueError('This search evaluates 60k trials')
        fresh=not (Path(args.state)/'state.json').exists()
        super().__init__(args)
        if fresh:
            # First wave tests all eight hypotheses on the same seed. Subsequent
            # waves fill the other seeds; promotion still requires all three.
            order={s['seed']:i for i,s in enumerate(self.db['seeds'])}
            self.db['trials'].sort(key=lambda t:(order[t['seed']],t['config']))
        for c in self.db['configs']:
            assert all(c['params'].get(k)==v for k,v in FIXED.items())
        self.db.update(stage='dreamer_novelty_60k', fixed=FIXED,
            evaluation='Fresh matched-seed lifetimes; rank discoveries during observations 40001–60000',
            reference='40k configuration 24, rerun from scratch for 60k as config0',
            reward_note='Prediction-error reward is a learned expected residual, not access to future observations')
        # Include old kernel-blocked attempts in the headroom guard, without adopting them.
        import json
        old=self.state_root/'archives/search40k_closed/state.json'
        if old.exists():
            archive=json.loads(old.read_text())
            assert all(FIXED[k]==v for k,v in archive['configs'][24]['params'].items())
            self.db['prior_quarantined']=archive.get('quarantined',[])
            if self.db['prior_quarantined']:
                self.db.setdefault('quarantined',self.db['prior_quarantined'])
        self.save()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--state',default='/home/dev/globo_dreamer_state/search60k')
    p.add_argument('--seeds',default='/home/dev/globo_dreamer_state/seed_screen/selected.json')
    p.add_argument('--steps',type=int,default=60000)
    p.add_argument('--gpus',type=int,default=8)
    p.add_argument('--search-seed',type=int,default=20261008)
    p.add_argument('--poll-seconds',type=float,default=20)
    DreamerSearch(p.parse_args()).run()


if __name__=='__main__': main()
