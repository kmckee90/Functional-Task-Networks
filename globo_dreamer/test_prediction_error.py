"""Pod-only prediction novelty checks: executed action, resets, detached targets."""
import os
import sys
from pathlib import Path
import unittest
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))/'vendor/dreamerv3'))
import jax
import jax.numpy as jnp
import numpy as np
from .agent import prediction_error_loss


class ErrorTests(unittest.TestCase):
    def test_action_and_reset_masks_and_detached_target(self):
        estimates=jnp.array([[[.1,.2,.3],[.4,.5,.6]]])
        selection=jax.nn.one_hot(jnp.array([[1,2]]),3)
        target=jnp.array([[.7,99.]])
        valid=jnp.array([[1.,0.]])
        loss=prediction_error_loss(estimates,selection,target,valid)
        self.assertAlmostEqual(float(loss),.25,places=6)
        grads=jax.grad(prediction_error_loss, argnums=(0,2))(estimates,selection,target,valid)
        np.testing.assert_allclose(np.asarray(grads[0]),[[[0.,-1.,0.],[0.,0.,0.]]])
        np.testing.assert_array_equal(np.asarray(grads[1]),np.zeros((1,2)))
        self.assertEqual(float(prediction_error_loss(estimates,selection,target,jnp.zeros_like(valid))),0.)


if __name__=='__main__':unittest.main()
