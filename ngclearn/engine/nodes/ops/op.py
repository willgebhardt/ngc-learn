from ngclearn.engine.nodes.node import Node
from jax import numpy as jnp
from abc import ABC

class Op(Node, ABC):
    """
    Base operator (class from which other operator types inherit basic properties from)

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant

        key: PRNG Key to control determinism of any underlying synapses
            associated with this operator
    """
    def __init__(self, name, n_units, dt, key=None, debugging=False):
        super().__init__(name=name, dt=dt, key=key, debugging=debugging)
        self.n_units = n_units

    def set_to_rest(self, batch_size=1, hard=True):
        if hard:
            self.t = 0
            for key in self.comp:
                self.comp[key] = jnp.zeros([batch_size, self.n_units])

class_name = Op.__name__
