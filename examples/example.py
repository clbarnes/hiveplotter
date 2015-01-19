# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# hiveplotter usage

# <codecell>

from hiveplotter import HivePlot
import networkx as nx
import random as rand
from matplotlib import pyplot as plt

# Set up a graph with 250 nodes in 3 classes, and 500 edges between random pairs of nodes, in 3 random classes and with a random weight

edge_types = ['a-edge', 'b-edge', 'c-edge']
node_classes = ['x-node', 'y-node', 'z-node']
G = nx.Graph()

for node_ID in ("node{}".format(n) for n in range(250)):
    G.add_node(node_ID, type=rand.choice(node_classes), other_attr=rand.random())
    
for _ in range(500):
    G.add_edge(*rand.sample(G.nodes(), 2), type=rand.choice(edge_types), weight=rand.triangular(0, 1, 0))

# The hive plot for graph G with default arguments. Note:
# - The edge thickness and colour, proportional to the edge's weight.
# - Nodes arrayed up the axes in order of their degree (controllable with the order_nodes_by kwarg).
# - The node size, representing the number of nodes at that locus on the axis.

hive_plot = HivePlot(G, node_class_attribute='type', node_class_values=None)
hive_plot.draw()
plt.imshow(hive_plot.as_bitmap(resolution=150))

# Colour changes, using the same graph. Note:
# - The change to node colour.
# - The change to node locations, due to ordering them by other_attr
# - The random edge colouration and selection of gradient.

hive_plot2 = HivePlot(G,
                      edge_colour_attribute='random',
                      edge_colour_gradient='Hue',
                      default_node_colour='Orange',
                      order_nodes_by='other_attr',
                      background_colour='Brown',
                      axis_colour='White',
                      label_colour='Yellow'
                      )
hive_plot2.draw()
plt.imshow(hive_plot2.as_bitmap(resolution=150))

# Some changes to the representation of nodes and edges, using the same graph. Note:
# - The explicit order of the 3 node classes.
# - The edges coloured by their type, and the legend.
# - The superimposed nodes represented by colour rather than size.

hive_plot3 = HivePlot(G,
                      node_class_values=['z-node', 'y-node', 'x-node'],
                      edge_colour_attribute='type',
                      edge_colour_gradient='Rainbow',
                      edge_category_legend=True,
                      node_superimpose_representation='colour'
    )
hive_plot3.draw()
plt.imshow(hive_plot3.as_bitmap(resolution=150))

# Demonstrating the splitting of axes to allow visualisation of intra-class edges.
# - The angle between split axes can be controlled with the split_angle kwarg.

hive_plot4 = HivePlot(G,
                      split_axes=['y-node', 'x-node']
    )
hive_plot4.draw()
plt.imshow(hive_plot4.as_bitmap(resolution=150))