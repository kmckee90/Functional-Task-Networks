"""API-name compatibility for the unmodified 2021 authors' implementation."""
import collections
import collections.abc
import importlib
import sys

import numpy as np
import networkx as nx

from storage import WORKSPACE


class LegacyArray(np.ndarray):
    def __bool__(self):
        if self.size==0:
            return False
        return super().__bool__()


class LegacyNumpy:
    """Restore NumPy 1.18's implicit object arrays for ragged shape metadata."""
    def __getattr__(self,name): return getattr(np,name)
    def array(self,value,*args,**kwargs):
        try:
            return np.array(value,*args,**kwargs).view(LegacyArray)
        except ValueError:
            if args or 'dtype' in kwargs:
                raise
            return np.array(value,dtype=object,**kwargs).view(LegacyArray)


def load_mntdp():
    for repo in ['MNTDP','SuperNetworks','CTrLBenchmark']:
        sys.path.insert(0,str(WORKSPACE/'third_party'/repo))
    for name in ['Mapping','MutableMapping','Sequence','Iterable','Callable']:
        if not hasattr(collections,name):
            setattr(collections,name,getattr(collections.abc,name))
    for name,value in [('int',int),('float',float),('object',object)]:
        if name not in np.__dict__:
            setattr(np,name,value)
    if not hasattr(nx.DiGraph,'node'):
        nx.DiGraph.node=property(lambda self:self._node)
    import pydot
    import pydot.core
    if not hasattr(pydot,'quote_if_necessary'):
        pydot.quote_if_necessary=pydot.core.quote_id_if_necessary
    import torchvision.datasets.utils as dataset_utils
    if not hasattr(dataset_utils,'gen_bar_updater'):
        def gen_bar_updater():
            from tqdm import tqdm
            progress=tqdm()
            def update(count,block_size,total_size):
                progress.total=total_size
                progress.update(count*block_size-progress.n)
            return update
        dataset_utils.gen_bar_updater=gen_bar_updater
    import ray.tune
    import ray.tune.search
    import ray.tune.logger
    # These logger classes are imported by the old source but are unused in
    # the serial native Ignite execution path used for these experiments.
    if not hasattr(ray.tune.logger,'JsonLogger'):
        ray.tune.logger.JsonLogger=ray.tune.logger.JsonLoggerCallback
    if not hasattr(ray.tune.logger,'CSVLogger'):
        ray.tune.logger.CSVLogger=ray.tune.logger.CSVLoggerCallback
    sys.modules['ray.tune.suggest']=ray.tune.search
    sys.modules['ray.tune.suggest.variant_generator']=importlib.import_module('ray.tune.search.variant_generator')
    from src.models.PSSN_llmodel import MNTDP
    from src.train.training import train
    for module in ['src.models.modular_model','src.models.PSSN_llmodel']:
        importlib.import_module(module).np=LegacyNumpy()
    return MNTDP, train


if __name__=='__main__':
    print(load_mntdp())
