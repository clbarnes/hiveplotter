import pyx
import networkx as nx
import numpy as np
from collections import OrderedDict
from shapely import geometry as geom
import cmath
import math
import copy

class HivePlot():

    defaults = {
        "split_axes": None,
        "normalise_axis_length": False,
        "normalise_node_distribution": False
    }

    def __init__(self, network, node_class_attribute="type", node_class_values=None, **kwargs):
        """
        :param network: NetworkX representation of a network to be plotted
        :param node_class_attribute: The attribute of nodes on which to split them onto 1-3 axes (default: "type")
        :param node_class_values: Values of node_class_attribute which will be included in the plot (in clockwise order)
        :param kwargs: Dictionary of attributes which can override default behaviour of the plot
        :return:
        """
        self.network = network
        self.node_class_attribute = node_class_attribute
        self.node_classes = self._split_nodes(node_class_values)

        # define default behaviour, which can be overridden with kwargs
        self.axis_length = 100
        self.proportional_offset_from_intersection = 0.2  # as a proportion of the longest axis
        self.split_axes = []
        self.normalise_axis_length = False
        self.normalise_node_distribution = False
        #self.weight_edge_thickness = True
        self.edge_colour_data = "weight"    # should be numerical
        #self.background_color = "black"
        #self.axis_colour = "gray"
        self.normalise_link_colours = False
        self.node_size = 0.8
        self.canvas = None

        self.__dict__.update(kwargs)

        # define internal data stuctures used to

    def _split_nodes(self, node_class_names):
        """
        Split nodes based on the attribute specified in self.node_class_attribute
        :param node_class_names: Values of node_class_attribute which will be included in the plot (in clockwise order)
        :return: A dictionary whose keys are the node class attribute values, and values are lists of nodes belonging to that class
        """
        node_attribute_dict = nx.get_node_attributes(self.network, self.node_class_attribute)

        if node_class_names is None:
            node_class_names = set(node_attribute_dict.values())
            if len(node_class_names) > 3:
                raise ValueError("Nodes should be in 3 or fewer classes based on their {} attribute.".format(self.node_class_attribute))
            node_class_names = sorted(list(node_class_names))
        else:
            for node in node_attribute_dict:
                if node_attribute_dict[node] not in node_class_names:
                    node_attribute_dict.pop(node)

        split_nodes = OrderedDict({node_class: [] for node_class in node_class_names})
        node_list = list(node_attribute_dict)
        for i, node_class in enumerate(node_attribute_dict.values()):
            split_nodes[node_class].append(node_list[i])

        return split_nodes

    def _get_working_nodes(self):
        """
        :return: The set of all nodes to be included in the plot
        """
        working_nodes = set()

        for node_class_values in self.node_classes.values():
            working_nodes |= set(node_class_values)

        return working_nodes

    def draw(self, save_path):
        c = pyx.canvas.canvas()
        axes = self._create_axes()
        node_positions = self._place_nodes(axes)
        edge_lines = self._create_edge_info(node_positions)

        # draw axes
        for axis in axes.values():
            c.stroke(pyx.path.line(*HivePlot.linestring_to_coords(axis)), [pyx.style.linewidth(1), pyx.color.gray(0.7)])    #todo: customisable

        # draw edges
        for start, end, colour in edge_lines:
            c.stroke(pyx.path.line(start[0], start[1], end[0], end[1]), [pyx.style.linewidth(0.05), colour])

        # draw nodes
        for node, coords in node_positions.items():
            c.fill(pyx.path.circle(coords[0], coords[1], self.node_size), [pyx.color.rgb.red])   #todo: make colour interesting

        self.canvas = c

        self.save_canvas(save_path)

        return True

    def save_canvas(self, path):
        self.canvas.writePDFfile(path)
        return True

    @staticmethod
    def linestring_to_coords(linestring):
        ret = []
        for point in linestring.coords:
            for coord in point:
                ret.append(coord)

        return tuple(ret)

    def _create_axes(self):
        """
        Generate axes on which to plot nodes
        :return: A dictionary whose keys are the node class attribute values, and values are LineStrings of the axis those nodes will be plotted on
        """
        classes = list(self.node_classes)
        num_classes = len(classes)
        axes = OrderedDict()

        offset = self.proportional_offset_from_intersection * self.axis_length

        if num_classes == 1:
            axes[classes[0]] = geom.LineString([(0, 0), (0, self.axis_length)])
        elif num_classes == 2:
            axes[classes[0]] = geom.LineString([(0, offset), (0, self.axis_length + offset)])
            axes[classes[1]] = geom.LineString([(0, -offset), (0, - self.axis_length - offset)])
        elif num_classes == 3:
            axes[classes[0]] = geom.LineString([(0, offset), (0, self.axis_length + offset)])
            ax2_start = HivePlot._get_projection((0, 0), 120, offset)
            axes[classes[1]] = HivePlot._get_projecting_line(ax2_start, 120, self.axis_length)
            ax3_start = HivePlot._get_projection((0, 0), 240, offset)
            axes[classes[2]] = HivePlot._get_projecting_line(ax3_start, 240, self.axis_length)

        return axes

    def _place_nodes(self, axes):
        """
        Generate positions of nodes to be plotted
        :param axes: A dictionary whose keys are the node class attribute values, and values are LineStrings of the axis those nodes will be plotted on
        :return: A dictionary whose keys are the nodes to be plotted and values are coordinates of that node on the plot
        """
        node_positions = dict()

        ordered_nodes = self._order_nodes()

        for node_class in self.node_classes:
            axis = axes[node_class]
            for node in self.node_classes[node_class]:
                node_positions[node] = self._place_point_on_line(axis, ordered_nodes[node_class][node])

        return node_positions

    def _place_point_on_line(self, line, proportion):
        """
        Generate position of a point on a single line
        :param line: A LineString on which the point is to placed
        :param proportion: The proportion of the way along the line which the point should be placed
        :return: A tuple (x, y) of coordinates on the line
        """
        start, end = np.array(line.coords[0]), np.array(line.coords[1])
        vector = end - start
        vector_from_start = vector * proportion
        return tuple(start + vector_from_start)

    def _order_nodes(self):
        """
        Order nodes by their degree.
        :return: A dictionary whose keys are the node class attribute values, and values are a dictionaries whose keys are nodes belonging to that class and values are a proportion along the axis which that node should be placed
        """
        working_nodes = self._get_working_nodes()
        max_degree = max(nx.degree(self.network, nbunch=working_nodes).values())
        node_positions = OrderedDict({node_class: dict() for node_class in self.node_classes})
        for node_class in self.node_classes:
            degrees = nx.degree(self.network, nbunch=self.node_classes[node_class]).items()
            sorted_degrees = sorted(degrees, key=lambda degree: degree[1])
            degrees_arr = np.array(sorted_degrees, dtype="float64")
            if not self.normalise_axis_length:
                degrees_arr[:, 1] = degrees_arr[:, 1]/max_degree
            else:
                degrees_arr[:, 1] = degrees_arr[:, 1]/np.max(degrees_arr[:, 1])

            if self.normalise_node_distribution:
                degrees_arr[:, 1] = np.linspace(0, np.max(degrees_arr[:, 1]), num=len(degrees_arr[:, 1]))

            node_positions[node_class].update({deg[0]: deg[1] for deg in degrees_arr})

        return node_positions

    @staticmethod
    def _get_projection(startpoint, angle, distance):
        """
        :param startpoint: (x, y) tuple of starting coordinates
        :param angle: Bearing, clockwise from 0 degrees, in which to project
        :param distance: Length of projection
        :return: (x, y) tuple of ending coordinates
        """
        angle = 90-angle
        if angle < -180:
            angle += 360

        angle_r = math.radians(angle)

        start = complex(startpoint[0], startpoint[1])
        movement = cmath.rect(distance, angle_r)
        end = start + movement
        return end.real, end.imag

    @staticmethod
    def _get_projecting_line(startpoint, angle, distance):
        """
        :param startpoint: (x, y) tuple of starting coordinates
        :param angle: Bearing, clockwise from 0 degrees, in which to project
        :param distance: Length of projection
        :return: LineString of projection
        """
        return geom.LineString([startpoint, HivePlot._get_projection(startpoint, angle, distance)])

    def deepcopy(self):
        return copy.deepcopy(self)

    def _create_edge_info(self, node_positions):
        working_nodes = self._get_working_nodes()
        edges = dict()
        for edge in self.network.edges_iter(data=True):
            if self.network.node[edge[0]][self.node_class_attribute] == self.network.node[edge[1]][self.node_class_attribute]:
                continue
            undirected = set(edge[:2])
            if undirected.issubset(edges):
                edges[tuple(sorted(undirected))] += edge[2][self.edge_colour_data]
            elif undirected.issubset(working_nodes):
                edges[tuple(sorted(undirected))] = edge[2][self.edge_colour_data]

        maximum = max([weight for weight in edges.values()])
        for edge in edges:
            edges[edge] /= maximum

        if self.normalise_link_colours:
            old_new_weights = dict(zip(sorted(edges.values()), range(len(edges))))

            for edge in edges:
                edges[edge] = old_new_weights[edges[edge]]

        retlist = []
        for edge in edges.items():
            start, end = tuple(edge[0])
            retlist.append((node_positions[start], node_positions[end], pyx.color.rgb.blue))    #todo: make colour do something interesting

        return retlist