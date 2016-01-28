from hiveplotter import HivePlot
from networkx import nx
import random
from unittest import TestCase


SEED = 1

NTYPES = ['A', 'B', 'C']


class SimpleCase(TestCase):
    def make_graph(self):
        G = nx.fast_gnp_random_graph(30, 0.2, seed=SEED)

        for node, data in G.nodes_iter(data=True):
            data['ntype'] = random.choice(NTYPES)

        for src, tgt, data in G.edges_iter(data=True):
            data['weight'] = random.random()

        return G

    def test_simple(self):
        G = self.make_graph()
        H = HivePlot(G, node_class_attribute='ntype')
        H.draw()
        H.save_plot('./output/main.pdf')

    def test_dump_cfg(self):
        G = self.make_graph()
        H = HivePlot(G, node_class_attribute='ntype')
        H.draw()
        print(H.dump_config())


if __name__ == '__main__':
    tests = SimpleCase()
    tests.test_simple()
