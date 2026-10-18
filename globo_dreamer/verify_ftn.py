"""Pod-only integration check: real scaled Dreamer + all FTN tensor banks."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import sys
STATE=Path('/home/dev/globo_dreamer_state')
sys.path.insert(0,str(STATE/'vendor/dreamerv3'))
import elements
import ruamel.yaml
import numpy as np
import jax
from .agent import Agent
from .ftn import BayesWorldModel,InputRouter,H,K,hard_mask


def main():
    router=InputRouter();support=np.zeros((4,72));support[:,0]=[1,2,3,4];support[:,1]=[4,3,2,1]
    for _ in range(8):router.observe(support)
    novel=np.roll(support,8,axis=1)
    assert router.novel(novel) and not router.novel(support)
    router.archive(np.arange(H,dtype=np.float32))
    np.testing.assert_array_equal(router.match(support),np.arange(H,dtype=np.float32))
    assert router.match(novel) is None
    config=elements.Config(ruamel.yaml.YAML(typ='safe').load((STATE/'vendor/dreamerv3/dreamerv3/configs.yaml').read_text())['defaults'])
    configs=ruamel.yaml.YAML(typ='safe').load((STATE/'vendor/dreamerv3/dreamerv3/configs.yaml').read_text())
    config=config.update(configs['size12m']).update({'agent.dyn.rssm.hidden':1024,'agent.dyn.rssm.stoch':128,
        'agent.dyn.rssm.classes':32,'agent.enc.simple.depth':32,'agent.dec.simple.depth':64,
        'jax.prealloc':False,'jax.policy_devices':[0],'jax.train_devices':[0],
        'agent.imag_length':15,'agent.imag_last':8,'agent.repval_loss':False,'agent.report_gradnorms':False})
    cfg=dict(config.agent);cfg['explore']=dict(model_only=True,novelty_only=False,ensemble=5,density=1.,disagreement=1.,
        actor_weight=.2,feedback=True,horizon=100.,prediction_error=0.,disagreement_power=1.,reward_feedback=True)
    root=STATE/'diagnostics/ftn_integration';root.mkdir(parents=True,exist_ok=True)
    ac=elements.Config(**cfg,logdir=str(root),seed=19,jax=dict(config.jax,precompile=False),batch_size=8,batch_length=32,replay_context=0,
        report_length=256,replica=0,replicas=1)
    obs=dict(image=elements.Space(np.uint8,(64,64,3)),body=elements.Space(np.float32,(17,)),density=elements.Space(np.float32,(1,)),
        reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
    act=dict(action=elements.Space(np.int32,(),0,25))
    agent=BayesWorldModel(Agent(obs,act,ac),root,19)
    rng=np.random.default_rng(19)
    data={k:np.zeros((8,32,*v.shape),v.dtype) for k,v in {**obs,**act}.items()}
    data['image']=rng.integers(0,256,data['image'].shape,dtype=np.uint8)
    data['body']=rng.normal(0,.1,data['body'].shape).astype(np.float32)
    data['action']=rng.integers(0,25,(8,32),dtype=np.int32)
    data['is_first'][:,0]=True
    ids=np.flatnonzero(np.asarray(agent.action_mask)).astype(np.uint16)
    data['ftn_mask']=np.broadcast_to(ids,(8,32,K)).copy()
    before={k:np.array(v) for k,v in agent.base.items() if k in agent.keys}
    probe=next(k for k in agent.keys if k.startswith('enc/') and 'a' in agent.banks[k])
    atom_before=np.array(agent.banks[probe]['b'])
    print('BEGIN TRAIN',flush=True)
    _,_,metrics=agent.train(None,data)
    assert all(np.isfinite(v).all() for v in metrics.values()),metrics
    assert np.asarray(agent.train_mask).sum()==K
    for k,v in before.items():np.testing.assert_array_equal(np.asarray(agent.base[k]),v)
    active=np.asarray(agent.train_mask)>0
    np.testing.assert_array_equal(np.asarray(agent.banks[probe]['b'])[~active],atom_before[~active])
    assert np.any(np.asarray(agent.banks[probe]['b'])[active]!=atom_before[active])
    print('TRAIN OK',metrics,flush=True)
    import time
    for i in range(3):
        batch=jax.device_put(data,agent.agent.train_sharded) if i%2 else data
        start=time.perf_counter();_,_,m=agent.train(None,batch)
        print('HOT UPDATE',i,time.perf_counter()-start,'cache',agent.search._cache_size(),agent.update._cache_size(),
            'search',m['ftn_search_seconds'],'weight',m['ftn_update_seconds'],flush=True)
    assert agent.search._cache_size()==1 and agent.update._cache_size()==1
    carry=agent.init_policy(1)
    one={k:v[:1,0] for k,v in data.items() if k in obs}
    carry,actions,outputs=agent.policy(carry,one)
    assert outputs['ftn_mask'].shape==(1,K)
    for i in range(9):
        row={k:v[0] for k,v in one.items()};row.update(action=np.int32(1),ftn_mask=ids)
        agent.observe(row);carry,actions,outputs=agent.policy(carry,one)
    assert agent.action_reconfigs>=2
    repaired=agent.recontext(agent.materialize(agent.banks,agent.action_mask),agent.support_batch(),agent.nextkey())
    assert np.asarray(repaired[-1]['action']).item()==1
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(repaired))
    # Exercise preservation replay explicitly, even before random masks overlap.
    replay={k:v[:1] for k,v in data.items() if k!='ftn_mask'}
    primary={k:v for k,v in data.items() if k!='ftn_mask'}
    agent.banks,agent.optstate,mets=agent.update(agent.banks,agent.optstate,agent.train_mask,primary,
        agent.action_mask,replay,np.float32(1),agent.nextkey())
    assert float(mets['ftn_replay_loss'])>0 and np.isfinite(float(mets['ftn_total_loss']))
    print('POLICY OK',agent.action_reconfigs,flush=True)
    agent.render_masks(force=True)
    checkpoint=elements.Checkpoint(elements.Path(root/'checkpoint'))
    checkpoint.agent=agent
    checkpoint.save()
    before_logits=np.array(agent.action_logits)
    checkpoint.load()
    np.testing.assert_array_equal(np.asarray(agent.action_logits),before_logits)
    print('CHECKPOINT RESTORE OK',flush=True)
    print('PASS scaled FTN integration',flush=True)

if __name__=='__main__':main()
