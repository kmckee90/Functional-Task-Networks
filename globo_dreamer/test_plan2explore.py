"""CPU-only on pod: objective, gradients, and small end-to-end JAX agent."""
import os
import sys
from pathlib import Path
import tempfile
import itertools
import unittest
os.environ.setdefault('JAX_PLATFORMS','cpu')
STATE=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
sys.path.insert(0,str(STATE/'vendor/dreamerv3'))
import elements
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml
from .plan2explore import Plan2Explore,disagreement,bootstrap_loss


class Plan2ExploreTests(unittest.TestCase):
    def setUp(self):
        # Dreamer's setup globally disables implicit transfers after its test.
        jax.config.update('jax_transfer_guard','allow')

    def test_variance_bootstrap_reset_and_target_gradient(self):
        p=jnp.array([[[[0.,0.],[1.,2.]]],[[[2.,4.],[3.,6.]]]])
        np.testing.assert_allclose(disagreement(p),[[2.5,2.5]])
        target=jnp.zeros((1,2,2));valid=jnp.array([[False,True]]);indices=jnp.zeros((2,1),jnp.int32)
        self.assertAlmostEqual(float(bootstrap_loss(p,target,valid,indices)),25.)
        grad=jax.grad(bootstrap_loss,argnums=(0,1))(p,target,valid,indices)
        np.testing.assert_array_equal(grad[0][:,:,0],0.)
        np.testing.assert_array_equal(grad[1],0.)

    def test_small_agent_training_policy_report_checkpoint(self):
        cfgs=ruamel.yaml.YAML(typ='safe').load((STATE/'vendor/dreamerv3/dreamerv3/configs.yaml').read_text())
        cfg=elements.Config(cfgs['defaults']).update(cfgs['debug']).update({
            'jax.platform':'cpu','jax.debug':False,'jax.prealloc':False,'jax.compute_dtype':'float32',
            'agent.imag_length':3,'agent.imag_last':2,'agent.report_gradnorms':False})
        opts=dict(cfg.agent)
        opts['explore']=dict(ensemble=5,model_only=False,novelty_only=True,feedback=False,
            reward_feedback=False,prediction_error=0.,horizon=100.,p2e_scale=10000.)
        from .env import Globo
        with tempfile.TemporaryDirectory(dir=STATE/'diagnostics') as directory:
            env=Globo(Path(directory)/'env',video_every=0)
            config=elements.Config(**opts,jax=dict(cfg.jax,precompile=False),logdir=directory,
                seed=19,batch_size=2,batch_length=4,report_length=4,replay_context=0,replica=0,replicas=1)
            agent=Plan2Explore(env.obs_space,{'action':env.act_space['action']},config)
            rng=np.random.default_rng(1)
            data={k:np.zeros((2,4,*v.shape),v.dtype) for k,v in agent.spaces.items()}
            data['image']=rng.integers(0,256,(2,4,64,64,3),dtype=np.uint8)
            data['is_first'][:,0]=True
            data['action']=rng.integers(0,25,(2,4),dtype=np.int32)
            stream=iter(agent.stream(itertools.repeat(data)))
            carry=agent.init_train(2)
            for _ in range(3):
                carry,_,metrics=agent.train(carry,next(stream))
                self.assertTrue(all(np.isfinite(x).all() for x in metrics.values()))
            policyobs={k:v[:,0] for k,v in data.items() if k in env.obs_space}
            _,actions,_=agent.policy(agent.init_policy(2),policyobs)
            self.assertTrue(np.all((actions['action']>=0)&(actions['action']<25)))
            _,report=agent.report(agent.init_report(2),next(stream))
            self.assertEqual(report['posterior'].shape,(2,4,64,64,3))
            self.assertTrue(np.isfinite(report['forecast']).all())
            snapshot=agent.save();agent.load(snapshot)
            self.assertTrue(any(k.startswith('p2e_ensemble/') for k in agent.params))


if __name__=='__main__':unittest.main()
