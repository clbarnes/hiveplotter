background_proportion = 1.2
"""How much larger the background is than the bounding box of all other elements of the plot."""

background_colour = "Black"
"""The colour of the background.
Can be string corresponding to one of PyX's named RGB or CMYK colours_, a 3- or 4-length tuple of RGB or CMYK values, or a greyscale value between 0 and 1.
.. _colours: http://pyx.sourceforge.net/manual/colorname.html"""

# axis parameters

axis_length = 10
"""Length, in cm, of the axes"""

proportional_offset_from_intersection = 0.2
"""The distance between the centre of the plot and the start of each axis, as a proportion of the axis length"""

split_axes = []
"""A list of node classes whose axes are to be split into 2 (allowing visualisation of edges between nodes of the same class"""

split_angle = 30
"""Angle, in degrees, between split axes"""

normalise_axis_length = False
"""Whether the set of nodes on an axis should be spread to take the whole length of the axis, rather than being compared to nodes on other axes as well"""

axis_colour = "Gray"
"""Colour of the axes.
Can be string corresponding to one of PyX's named RGB or CMYK colours_, a 3- or 4-length tuple of RGB or CMYK values, or a greyscale value between 0 and 1.
.. _colours: http://pyx.sourceforge.net/manual/colorname.html"""

axis_thickness = 0.15
"""Thickness of the axes"""

order_nodes_by = "degree"
"""How to distribute nodes on the axes. Use either an attribute name, or 'degree' for their degree"""

# axis label parameters

label_colour = "White"
"""Colour of the axis labels.
Can be string corresponding to one of PyX's named RGB or CMYK colours_, a 3- or 4-length tuple of RGB or CMYK values, or a greyscale value between 0 and 1.
.. _colours: http://pyx.sourceforge.net/manual/colorname.html"""

label_size = 15
"""Point size of the axis labels"""

label_spacing = 0.1
"""How far the label should be from the end of the axis, as a proportion of the length of the axis."""

# node parameters

normalise_node_distribution = False
"""Whether to spread the nodes evenly along the axis"""

node_superimpose_representation = "colour"
"""String of 'colour' or 'size'"""

node_size_range = (0.08, 0.3)
"""Range of radiuses of nodes. If node_superimpose_representation is 'colour', the minimum will be used"""

node_colour_gradient = "GreenRed"
"""Colour gradient from which to select node colours.
Should be a string corresponding to one of PyX's named colour gradients_.
.. _gradients: http://pyx.sourceforge.net/manual/gradientname.html"""

default_node_colour = "Green"
"""Default colour of nodes.
Can be string corresponding to one of PyX's named RGB or CMYK colours_, a 3- or 4-length tuple of RGB or CMYK values, or a greyscale value between 0 and 1.
.. _colours: http://pyx.sourceforge.net/manual/colorname.html"""

# edge parameters

edge_thickness_range = (0.002, 0.16)
"""Range of edge thicknesses to be used, as s tuple"""

edge_colour_attribute = "weight"
"""What attribute of the edge should be used to colour it. Should be a string referring to the attribute, or 'random'"""

edge_colour_gradient = "Jet"
"""Gradient from which to select edge colours.
Should be a string corresponding to one of PyX's named colour gradients_.
.. _gradients: http://pyx.sourceforge.net/manual/gradientname.html"""

default_edge_colour = "White"
"""Default colour of edges.
Can be string corresponding to one of PyX's named RGB or CMYK colours_, a 3- or 4-length tuple of RGB or CMYK values, or a greyscale value between 0 and 1.
.. _colours: http://pyx.sourceforge.net/manual/colorname.html"""

edge_category_colours = None
"""Dict of edge categories and the colours which should represent them"""

curved_edges = True
"""Whether the edges should be curved"""

normalise_edge_colours = False
"""Whether to spread edge colours evenly along the gradient"""