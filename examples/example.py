from hiveplotter import HivePlot
import networkx as nx
import random as rand


def make_graph():
    edge_types = ['a-edge', 'b-edge', 'c-edge']
    node_classes = ['x-node', 'y-node', 'z-node']
    G = nx.Graph()

    for node_ID in ("node{}".format(n) for n in range(250)):
        G.add_node(node_ID, type=rand.choice(node_classes))

    for _ in range(500):
        G.add_edge(*rand.sample(G.nodes(), 2), type=rand.choice(edge_types), weight=rand.triangular(0, 1, 0))

    return G


"""
Generate a graph with 250 nodes in 3 classes, and 500 edges in 3 classes between random pairs of nodes, randomly weighted.
"""

G = make_graph()

"""
The default hive plot. Note:
- The edge thickness and colour, proportional to the edge's weight.
- Nodes arrayed up the axes in order of their degree.
- The node size, representing the number of nodes at that locus on the axis.
"""

hive_plot = HivePlot(G)
hive_plot.draw(show=True)

"""
Colour changes, using the same graph. Note:
- The change to node colour.
- The random edge colouration and selection of gradient.
"""

hive_plot2 = HivePlot(G,
                      edge_colour_attribute='random',
                      edge_colour_gradient='Hue',
                      default_node_colour='Orange',
                      background_colour="Brown",
                      axis_colour="White",
                      label_colour="Yellow"
                      )
hive_plot2.draw(show=True)

"""
Some changes to th the representation of nodes and edges, using the same graph. Note:
- The explicit order of the 3 node classes.
- The edges coloured by their type, and the legend.
- The superimposed nodes represented by colour rather than size.
"""

hive_plot3 = HivePlot(G,
                      node_class_values=['z-node', 'y-node', 'x-node'],
                      edge_colour_attribute='type',
                      edge_colour_gradient='Rainbow',
                      edge_category_legend=True,
                      node_superimpose_representation='colour'
    )
hive_plot3.draw(show=True)

"""
Demonstrating the splitting of axes to allow visualisation of intra-class edges.
- The angle between split axes can be controlled with the split_angle kwarg.
"""

hive_plot4 = HivePlot(G,
                      split_axes=['y-node', 'x-node']
    )
hive_plot4.draw(show=True)

"""
While the plot is saved as a vector-drawn PDF, a bitmap can be returned for use with (for example) matplotlib.
"""

try:
    from matplotlib import pyplot as plt
    bitmap = hive_plot.as_bitmap(resolution=150)    # resolution in dpi
    plt.imshow(bitmap)
    plt.show()
except ImportError:
    raise ImportError('matplotlib not in requirements, but may be useful for example.py')