import pyx
import networkx as nx
import numpy as np
from collections import OrderedDict
from geom_utils import get_projection, place_point_proportion_along_line, mid_line
import copy
from collections import Counter
from colour_utils import convert_colour
import random as rand
import warnings
from component_classes import Axis, Edge


class HivePlot():
    """
    A class wrapping a networkx graph which can be used to generate highly customisable hive plots.
    """

    def __init__(self, network, node_class_attribute="type", node_class_values=None, **kwargs):
        """
        :param network: network to be plotted
        :type network: nx.Graph
        :param node_class_attribute: The attribute of nodes on which to split them onto 1-3 axes (default: "type")
        :type node_class_attribute: str
        :param node_class_values: Values of node_class_attribute which will be included in the plot (in clockwise order)
        :type node_class_values: list
        :param kwargs: Dictionary of attributes which can override default behaviour of the plot
        :type kwargs: dict
        """
        self.default_colour = "White"
        self.network = network
        self.node_class_attribute = node_class_attribute
        self.node_classes = self._split_nodes(node_class_values)

        # define default behaviour, which can be overridden with kwargs

        # setup parameters
        self.parent_hiveplot = None

        # background parameters
        self.background_proportion = 1.2
        self.background_colour = "Black"

        # axis parameters
        self.axis_length = 10
        self.proportional_offset_from_intersection = 0.2  # as a proportion of the longest axis
        self.split_axes = []
        self.split_angle = 30  # degrees
        self.normalise_axis_length = False
        self.axis_colour = "Gray"
        self.axis_thickness = 0.15
        self.order_nodes_by = "degree"

        # axis label parameters
        self.label_colour = "White"
        self.label_size = 15
        self.label_spacing = 0.1

        # node parameters
        self.normalise_node_distribution = False
        self.node_superimpose_representation = "colour"  # or "size"
        self.node_size_range = (0.08, 0.3)
        self.node_colour_gradient = "GreenRed"

        # edge parameters
        self.edge_thickness_range = (0.002, 0.16)
        self.edge_colour_attribute = "weight"
        self.edge_colour_gradient = "Jet"
        self.edge_category_colours = None
        self.edge_curvature = 1.7
        self.normalise_edge_colours = False

        self.__dict__.update(kwargs)

        # fields to be filled by the object
        self.canvas = None
        self.legend_categories = None
        self._background_layer = None
        self._foreground_layer = None
        self._legend_layer = None
        self.colour_definitions = self._convert_colours()
        self._axes = None

    def _split_nodes(self, node_class_names):
        """
        Split nodes based on the attribute specified in self.node_class_attribute
        :param node_class_names: Values of node_class_attribute which will be included in the plot (in clockwise order)
        :type node_class_names: list
        :return: A dictionary whose keys are the node class attribute values, and values are lists of nodes belonging to that class
        :rtype: dict
        """
        node_attribute_dict = nx.get_node_attributes(self.network, self.node_class_attribute)

        if node_class_names is None:
            node_class_names = list(node_attribute_dict.values())
            if len(node_class_names) > 3:
                raise ValueError("Nodes should be in 3 or fewer classes based on their {} attribute.".format(
                    self.node_class_attribute))
            node_class_names = sorted(node_class_names)
        else:
            for_deletion = []
            for node in node_attribute_dict:
                if node_attribute_dict[node] not in node_class_names:
                    for_deletion.append(node)
            for fd in for_deletion:
                node_attribute_dict.pop(fd)

        split_nodes = OrderedDict([(node_class, []) for node_class in node_class_names])
        node_list = list(node_attribute_dict)
        for i, node_class in enumerate(node_attribute_dict.values()):
            split_nodes[node_class].append(node_list[i])

        return split_nodes

    def _get_working_nodes(self):
        """
        :return: The set of all nodes to be included in the plot
        :rtype: set
        """
        working_nodes = set()

        for node_class_values in self.node_classes.values():
            working_nodes |= set(node_class_values)

        return working_nodes

    def _draw_background(self):
        """
        Draw the background of the plot slightly larger than everything in the foreground.
        :return: None
        :rtype: None
        """
        min_x, min_y, max_x, max_y = self._get_bbox()
        colour = convert_colour(self.background_colour)
        self._background_layer.fill(pyx.path.rect(min_x, min_y, max_x - min_x, max_y - min_y), [colour])

    def _get_bbox(self, extend=1.1):
        """
        Get the extents of the canvas.
        :param extend: the proportion of the extent which should be added to the size of the bounding box
        :type extend: float or int
        :return: extents of bounding box- (minX, minY, maxX, maxY)
        :rtype: tuple
        """
        bbox = self.canvas.bbox()
        min_x = pyx.unit.tocm(bbox.left())
        min_y = pyx.unit.tocm(bbox.bottom())
        max_x = pyx.unit.tocm(bbox.right())
        max_y = pyx.unit.tocm(bbox.top())

        return min_x * extend, min_y * extend, max_x * extend, max_y * extend

    def _draw_axes(self):
        for axis in self._axes.values():
            axis.draw(self._foreground_layer)

    def _draw_labels(self):
        text_alignment = [pyx.text.halign.boxcenter, pyx.text.valign.middle]

        for axis in self._axes.values():
            label_position = place_point_proportion_along_line(axis, 1 + self.label_spacing)
            txt_str = self._colour_text(self._size_text(axis.label, self.label_size), self.label_colour)

            self._foreground_layer.text(label_position[0], label_position[1], txt_str, text_alignment)

    def _size_text(self, text, size):
        return r"{\fontsize{" + str(size) + r"}{" + str(round(size * 1.2)) + r"}\selectfont " + text + r"}"

    def _colour_text(self, text, colour):
        return r"\textcolor{" + str(colour) + "}{" + text + "}"

    def _draw_edges(self, edges):
        for edge in edges:
            edge.draw(self._foreground_layer)

    def _draw_nodes(self):
        gradient = eval("pyx.color.gradient." + self.node_colour_gradient)
        node_positions = []
        for axis in self._axes.values():
            node_positions.extend(axis.nodes.values())
        node_position_weights = list(self._weight_by_colocation(node_positions).items())
        sorted_weights = sorted(node_position_weights, key=lambda x: x[1],
                                reverse=self.node_superimpose_representation != "colour")
        if self.node_superimpose_representation is "colour":
            node_size = min(self.node_size_range)
        else:
            node_size = None
        for coords, weight in sorted_weights:
            self._foreground_layer.stroke(pyx.path.circle(coords[0], coords[1],
                                                          node_size if node_size else map_to_interval(
                                                              self.node_size_range, weight)))
            self._foreground_layer.fill(pyx.path.circle(coords[0], coords[1],
                                                        node_size if node_size else map_to_interval(
                                                            self.node_size_range, weight)),
                                        [gradient.getcolor(weight)])

    def draw(self, save_path=None):
        """
        Draw the graph using the current settings
        :param save_path: path of PDF file to save graph into
        :type save_path: str
        :return: True for successful completion
        :rtype: bool
        """

        self._setup_latex()

        self.canvas = pyx.canvas.canvas()
        self._background_layer = self.canvas.layer("background")
        self._foreground_layer = self.canvas.layer("foreground", above="background")
        self._legend_layer = self.canvas.layer("legend", above="foreground")
        self._axes = self._create_axes()
        self._place_nodes()

        if len(self.node_classes) == 3:
            edge_lines = self._create_edge_info()
        else:
            raise NotImplementedError("2-axis plots not yet implemented")

        self._draw_axes()
        self._draw_edges(edge_lines)
        self._draw_nodes()
        self._draw_legend()
        self._draw_labels()
        self._draw_background()

        if save_path is not None:
            self.save_canvas(save_path)

        return True

    def save_canvas(self, path):
        """
        :param path: Save path
        :type path: str
        :return: True if complete
        :rtype: bool
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.canvas.writePDFfile(path)
        return True

    def _create_axes(self):
        """
        Generate axes on which to plot nodes. Copies parent hiveplot's axes if they exist.
        :return: A dictionary whose keys are the node class attribute values, and values are Axis objects
        :rtype: dict
        """

        if len(self.node_classes) != 3:
            raise NotImplementedError("Plots with number of axes other than 3 are not implemented yet (Issue #3)")

        if self.parent_hiveplot:
            return self.parent_hiveplot._axes

        axes = OrderedDict()

        angle = 0
        angle_spacing = 360 / len(self.node_classes)
        for node_class in self.node_classes:
            if node_class in self.split_axes:
                ccw_angle = angle - self.split_angle / 2
                axes[node_class + "_ccw"] = self._create_axis(ccw_angle, label=node_class)
                cw_angle = angle + self.split_angle / 2
                axes[node_class + "_cw"] = self._create_axis(cw_angle, label=node_class)
            else:
                new_axis = self._create_axis(angle, label=node_class)
                axes[node_class] = new_axis
            angle += angle_spacing

        return axes

    def _create_axis(self, angle, label=""):
        ax_start = get_projection((0, 0), angle, self.proportional_offset_from_intersection * self.axis_length)
        ax_stop = get_projection(ax_start, angle, self.axis_length)
        new_axis = Axis(ax_start, ax_stop)
        new_axis.set_visual_properties(convert_colour(self.axis_colour), self.axis_thickness, label=label)
        return new_axis

    def _place_nodes(self):
        """
        Places nodes on Axis objects. Does nothing if axes already have nodes.
        :param axes: A dictionary whose keys are the node class attribute values, and values are component_classes.Axis objects of the axis those nodes will be plotted on
        :type axes: dict
        """

        ordered_nodes = self._order_nodes()

        for node_class in self.node_classes:
            nodes_to_add = {key: value for key, value in ordered_nodes.items() if
                            key in self.node_classes[node_class]}
            if node_class in self.split_axes:
                self._add_node_to_axis_if_empty(node_class + "_ccw", nodes_to_add)
                self._add_node_to_axis_if_empty(node_class + "_cw", nodes_to_add)
            else:
                self._add_node_to_axis_if_empty(node_class, nodes_to_add)

    def _add_node_to_axis_if_empty(self, axis_key, nodes):
        if len(self._axes[axis_key].nodes) is 0:
            self._axes[axis_key].add_nodes(nodes)

    def _order_nodes(self):
        """
        Order nodes by their degree.
        :return: A dictionary whose keys are nodes, and values are proportions up their respective axes at which the nodes should be placed
        """
        working_nodes = self._get_working_nodes()
        if self.order_nodes_by is "degree":
            node_attrs = nx.degree(self.network, nbunch=working_nodes)
        else:
            node_attrs = {node[0]: node[1][self.order_nodes_by] for node in self.network.nodes(data=True)}

        if self.normalise_axis_length:
            ret_dict = dict()
            for nodes in self.node_classes:
                ret_dict.update(fit_attr_to_interval(
                    {key: value for key, value in node_attrs.items() if key in self.node_classes[nodes]},
                    distribute_evenly=self.normalise_node_distribution))
            return ret_dict
        else:
            return fit_attr_to_interval(node_attrs, distribute_evenly=self.normalise_node_distribution)

    def deepcopy(self):
        return copy.deepcopy(self)

    def _get_edge_colour_thickness(self):
        working_nodes = self._get_working_nodes()

        edge_thickness_data = get_attr_dict(self.network.edges_iter(data=True), "weight", 1)
        edge_thickness_values = fit_attr_to_interval(edge_thickness_data, interval=self.edge_thickness_range)
        edge_thickness_values = {key: value for key, value in edge_thickness_values.items()
                                 if {key[0], key[1]}.issubset(working_nodes)}

        gradient = eval("pyx.color.gradient." + self.edge_colour_gradient)
        if self.edge_colour_attribute is "random":
            edge_colour_values = {key: gradient.getcolor(rand.random()) for key in edge_thickness_data}
            return edge_colour_values, edge_thickness_values

        edge_colour_data = get_attr_dict(self.network.edges_iter(data=True), self.edge_colour_attribute, "unknown")
        edge_colour_data = {key: value for key, value in edge_colour_data.items()
                            if {key[0], key[1]}.issubset(working_nodes)}

        if self.edge_category_colours:
            actual_colours = {key: convert_colour(value) for key, value in self.edge_category_colours.items()}
            default_colour = convert_colour(self.default_colour)
            edge_colour_values = {key: actual_colours.get(value, default_colour) for key, value in
                                  edge_colour_data.items()}
        else:
            edge_colour_floats = fit_attr_to_interval(edge_colour_data)
            edge_colour_values = {key: gradient.getcolor(value) for key, value in edge_colour_floats.items()}

        return edge_colour_values, edge_thickness_values

    def _create_edge_info(self):
        """
        :param node_positions: dictionary whose keys are nodes and values are (x,y) coordinate tuples
        :type node_positions: dict
        :return: list of dictionaries with attributes about the edge (start coord, end coord, thickness, colour, mid point)
        :rtype: list
        """

        axes_this = list(self._axes)
        axes_ccw = [axes_this[-1]] + axes_this[0:-1]
        axes_cw = axes_this[1:] + [axes_this[0]]

        edge_colour_values, edge_thickness_values = self._get_edge_colour_thickness()

        plot_edges = list()
        for i, this_axis_name in enumerate(axes_this):
            this_axis = self._axes[this_axis_name]
            ccw_axis, cw_axis = self._axes[axes_ccw[i]], self._axes[axes_cw[i]]
            ccw_mid_ax_line = mid_line(ccw_axis.line, this_axis.line)
            cw_mid_ax_line = mid_line(cw_axis.line, this_axis.line)

            for edge in self.network.edges_iter(data=True):
                if edge[0] not in this_axis:
                    continue

                if edge[1] in ccw_axis:
                    new_edge = Edge(edge[0], this_axis, edge[1], ccw_axis, curvature=self.edge_curvature,
                                    mid_ax_line=ccw_mid_ax_line)
                elif edge[1] in cw_axis:
                    new_edge = Edge(edge[0], this_axis, edge[1], cw_axis, curvature=self.edge_curvature,
                                    mid_ax_line=cw_mid_ax_line)
                else:
                    continue

                new_edge.set_visual_properties(edge_colour_values[edge[:2]], edge_thickness_values[edge[:2]])
                plot_edges.append(new_edge)

        return plot_edges

    @staticmethod
    def _weight_by_colocation(node_positions):
        tally = Counter(node_positions)
        tally_arr = np.array(list(tally.items()), dtype="object")
        tally_arr[:, 1] = (tally_arr[:, 1] - 1) / (np.max(tally_arr[:, 1]) - 1)
        return dict(tally_arr)

    def _setup_latex(self):
        pyx.text.set(pyx.text.LatexRunner)
        pyx.text.preamble(r"\usepackage{color}")
        pyx.text.preamble(r"\usepackage[T1]{fontenc}")
        pyx.text.preamble(r"\usepackage{lmodern}")
        self._define_colours(self.colour_definitions)

    def _define_colours(self, colour_dict):
        for colour_name, colour_obj in colour_dict.items():
            pyx.text.preamble(r"\definecolor{%s}{cmyk}{%g,%g,%g,%g}" % (colour_name,
                                                                        colour_obj.c, colour_obj.m,
                                                                        colour_obj.y, colour_obj.k))

    def _draw_legend(self):
        if not self.edge_category_colours:
            return

        legend_items = []
        for category in self.edge_category_colours:
            legend_items.append(self._size_text(
                "%s %s" % (self._colour_text("--- ", self.edge_category_colours[category]),
                           self._colour_text(category, self.label_colour)), self.label_size)
            )

        legend_str = r"\linebreak".join(legend_items)
        max_x = self._get_bbox()[2]

        self._legend_layer.text(max_x, 0, legend_str,
                                [pyx.text.parbox(4), pyx.text.halign.left, pyx.text.valign.middle])

    def _convert_colours(self):
        d = {self.label_colour: convert_colour(self.label_colour),
             self.background_colour: convert_colour(self.background_colour),
             self.axis_colour: convert_colour(self.axis_colour)}

        if self.edge_category_colours:
            for value in self.edge_category_colours.values():
                d[value] = convert_colour(value)

        return d


def map_to_interval(num_range, proportion):
    """
    Return the number a certain proportion of the way along a number line between given values
    :param num_range: range in which to place value
    :type num_range: tuple
    :param proportion: proportion along range at which to place value
    :type proportion: float
    :return: value *proportion* of the way along *num_range*
    :rtype: float
    """
    mini = min(num_range)
    rng = max(num_range) - mini

    return mini + proportion * rng


def get_attr_dict(data_sequence, attr, default):
    """
    :param data_sequence: sequence of edges or nodes and their attributes
    :type data_sequence: list
    :param attr: name of attribute to select
    :type attr: str
    :param default: what to return when the attribute does not exist for an edge or node
    :type default: object
    :return: dictionary whose keys are lists identifying an edge or node, and values are the desired attribute
    :rtype: dict
    """
    return {datum[:-1]: datum[-1].get(attr, default) for datum in data_sequence}


def fit_attr_to_interval(attr_dict, random=False, distribute_evenly=False, interval=(0, 1)):
    """
    Convert an arbitrary set of attributes into a set of numerical attributes within a given range
    :param attr_dict: dictionary whose values are the attribute to convert
    :type attr_dict: dict
    :param random: whether to randomly assign attribute values
    :type random: bool
    :param distribute_evenly: whether to spread the values evenly within the range (keys sharing attributes will still share in the output)
    :type distribute_evenly: bool
    :param interval: range within which to fit the returned attributes
    :type interval: tuple
    :return: dictionary whose keys are the input keys, and values are the fitted numerical attribute values
    :rtype: dict
    """
    if random:
        return {key: map_to_interval(interval, rand.random()) for key in attr_dict}

    attr_values = np.array(list(attr_dict.values()))

    try:
        if distribute_evenly:
            raise AssertionError("Exception required for even distribution")
        attr_values = (attr_values - np.min(attr_values)) / np.ptp(attr_values)  # normalise to (0,1)
        attr_values = attr_values * (max(interval) - min(interval)) + min(interval)
        return dict(zip(list(attr_dict), attr_values))
    except (TypeError, AssertionError):
        uniques = sorted(set(attr_values))
        category_to_float = dict(zip(uniques, np.linspace(interval[0], interval[1], num=len(uniques))))
        return {key: category_to_float[value] for key, value in attr_dict.items()}