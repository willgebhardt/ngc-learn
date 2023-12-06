from ngclearn.engine.nodes.ops.op import Op


class ScaleNode(Op):  # inherits from Node class
    """
    A scaling node

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant

        scale: scaling factor to apply to this node's output (compartment)

        key: PRNG Key to control determinism of any underlying synapses
            associated with this operator
    """
    def __init__(self, name, n_units, dt, scale, key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)
        self.scale = scale

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        self.comp['out'] = self.scale * (1 - self.comp['in']) ## hacky hard-coded inversion

class_name = ScaleNode.__name__
