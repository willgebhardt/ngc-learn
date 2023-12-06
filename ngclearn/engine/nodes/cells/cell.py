from ngclearn.engine.nodes.node import Node
from jax import numpy as jnp
from abc import ABC

class Cell(Node, ABC):
    """
    Base cell element (class from which other cell types inherit basic properties from)

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        dt: integration time constant

        key: PRNG key to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, dt, key=None, debugging=False):
        super().__init__(name=name, dt=dt, key=key, debugging=debugging)
        self.n_units = n_units

    def set_to_rest(self, batch_size=1, hard=True):
        """
        Sets compartments of this cell to their resting state values or initial conditions

        Args:
            batch_size: how many samples are in parallel (number rows in batch)

            hard:
        """
        if hard:
            self.t = 0
            for key in self.comp:
                self.comp[key] = jnp.zeros([batch_size, self.n_units])

class_name = Cell.__name__
