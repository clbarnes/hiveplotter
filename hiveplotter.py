import pyx
from pyx.metapost.path import beginknot, endknot, smoothknot, tensioncurve
import networkx as nx
import numpy as np
from collections import OrderedDict
from shapely import geometry as geom
from geom_utils import linestring_to_coords, get_projecting_line, get_projection, place_point_on_line
import copy
from collections import Counter
from colour_utils import convert_colour, categories_to_float
import random


class HivePlot():
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
        self.axis_length = 10
        self.proportional_offset_from_intersection = 0.2  # as a proportion of the longest axis
        self.split_axes = []
        self.normalise_axis_length = False
        self.normalise_node_distribution = False
        self.edge_thickness = 0.005
        # self.weight_edge_thickness = False
        self.edge_colour_attribute = "weight"
        self.edge_colour_gradient = "Jet"
        self.edge_category_colours = None
        self.edge_curvature = 1.7
        self.background_colour = "Black"
        self.label_colour = "White"
        self.label_size = 15
        self.axis_colour = "Gray"
        self.axis_thickness = 0.15
        self.normalise_link_colours = False
        self.node_size = 0.08
        self.canvas = None
        self.node_colour_gradient = "GreenRed"

        self.__dict__.update(kwargs)

    def _split_nodes(self, node_class_names):
        """
        Split nodes based on the attribute specified in self.node_class_attribute
        :param node_class_names: Values of node_class_attribute which will be included in the plot (in clockwise order)
        :return: A dictionary whose keys are the node class attribute values, and values are lists of nodes belonging to that class
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
        """
        working_nodes = set()

        for node_class_values in self.node_classes.values():
            working_nodes |= set(node_class_values)

        return working_nodes

    def _draw_background(self):
        max_length = (self.axis_length * (1 + self.proportional_offset_from_intersection)) * self.edge_curvature
        colour = convert_colour(self.background_colour)
        self.canvas.fill(pyx.path.rect(-max_length, -max_length, 2 * max_length, 2 * max_length), [colour])

    def _draw_axes(self, axes):
        axis_colour = convert_colour(self.axis_colour)
        for axis in axes.values():
            self.canvas.stroke(pyx.path.line(*linestring_to_coords(axis)),
                               [pyx.style.linewidth(self.axis_thickness), axis_colour])

    def _draw_labels(self, axes):
        text_alignment_list = [
            [pyx.text.halign.boxcenter, pyx.text.valign.bottom],
            [pyx.text.halign.boxleft, pyx.text.valign.top],
            [pyx.text.halign.boxright, pyx.text.valign.top]
        ]
        text_alignment = dict(zip(list(axes), text_alignment_list))

        for axis_name in axes:
            line_end = axes[axis_name].coords[1]
            # txt_str = axis_name
            txt_str = self._colour_text(self._size_text(axis_name, self.label_size))
            self.canvas.text(line_end[0], line_end[1],
                             txt_str,
                             text_alignment[axis_name])

    def _size_text(self, text, size):
        return r"{\fontsize{" + str(size) + r"}{" + str(round(size*1.2)) + r"}\selectfont " + text + r"}"

    def _colour_text(self, text):
        return r"\textcolor{COL}{" + text + "}"

    def _draw_edges(self, edge_lines):
        for edge_dict in edge_lines:
            start = edge_dict["start"]
            end = edge_dict["end"]
            colour = edge_dict["colour"]

            if "crossing_point" in edge_dict:
                midpoint = edge_dict["crossing_point"]

                edgepath = pyx.metapost.path.path([
                    beginknot(*start), tensioncurve(),
                    smoothknot(*midpoint), tensioncurve(),
                    endknot(*end)
                ])
            else:
                edgepath = pyx.path.line(pyx.path.line(start[0], start[1], end[0], end[1]))

            self.canvas.stroke(edgepath, [pyx.style.linewidth(self.edge_thickness), colour])

    def _draw_nodes(self, node_positions):
        gradient = eval("pyx.color.gradient." + self.node_colour_gradient)
        node_position_weights = self._weight_by_colocation(node_positions)
        for coords, weight in node_position_weights.items():
            self.canvas.stroke(pyx.path.circle(coords[0], coords[1], self.node_size))
            self.canvas.fill(pyx.path.circle(coords[0], coords[1], self.node_size),
                             [gradient.getcolor(weight)])

    def draw(self, save_path=None):
        """
        Draw the graph using the current settings
        :param save_path: path of PDF file to save graph into
        :return: True for successful completion
        """

        self._setup_latex()

        self.canvas = pyx.canvas.canvas()
        axes = self._create_axes()
        node_positions = self._place_nodes(axes)

        if len(axes) == 3:
            mid_ax_lines = self._create_mid_ax_lines(axes)

            # edge_lines = self._create_straight_edge_info(node_positions, curved=False)
            edge_lines = self._create_edge_info(node_positions, curved=True, mid_ax_lines=mid_ax_lines)
        else:
            raise NotImplementedError("2-axis plots not yet implemented")

        self._draw_background()
        self._draw_axes(axes)
        self._draw_labels(axes)
        self._draw_edges(edge_lines)
        self._draw_nodes(node_positions)

        if not save_path is None:
            self.save_canvas(save_path)

        return True

    def _create_mid_ax_lines(self, axes):
        """
        For every pair of axes, create a line segment which bisects the angle between them and extends from where a line connecting their base would intersect with the bisecting line, and where a line connecting their tips would intersect with the bisecting line.
        :param axes: A dictionary whose keys are the names of the axes in the plot, and values are the LineString objects describing those axes
        :return: A dictionary whose keys are tuples of the names of the two axes the line is between, and the values are LineString objects describing that line
        """
        ax_names = list(axes.keys())
        mid_ids = []
        for i, ax_name in enumerate(ax_names[:-1]):
            mid_ids.append(tuple(sorted([ax_name, ax_names[i + 1]])))
        mid_ids.append(tuple(sorted([ax_names[-1], ax_names[0]])))

        mid_ax_lines = dict()
        for mid_id in mid_ids:
            start_to_start = geom.LineString([axes[mid_id[0]].coords[0], axes[mid_id[1]].coords[0]])
            end_to_end = geom.LineString([axes[mid_id[0]].coords[1], axes[mid_id[1]].coords[1]])

            mid_ax_lines[mid_id] = geom.LineString([place_point_on_line(start_to_start, 0.5), place_point_on_line(end_to_end, 0.5)])

        return mid_ax_lines

    def save_canvas(self, path):
        self.canvas.writePDFfile(path)
        return True

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
            ax2_start = get_projection((0, 0), 120, offset)
            axes[classes[1]] = get_projecting_line(ax2_start, 120, self.axis_length)
            ax3_start = get_projection((0, 0), 240, offset)
            axes[classes[2]] = get_projecting_line(ax3_start, 240, self.axis_length)

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
                node_positions[node] = place_point_on_line(axis, ordered_nodes[node_class][node])

        return node_positions

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
            degrees_arr = np.array(sorted_degrees, dtype="object")
            if not self.normalise_axis_length:
                degrees_arr[:, 1] = degrees_arr[:, 1] / max_degree
            else:
                degrees_arr[:, 1] = degrees_arr[:, 1] / np.max(degrees_arr[:, 1])

            if self.normalise_node_distribution:
                degrees_arr[:, 1] = np.linspace(0, np.max(degrees_arr[:, 1]), num=len(degrees_arr[:, 1]))

            node_positions[node_class].update({deg[0]: deg[1] for deg in degrees_arr})

        return node_positions

    def deepcopy(self):
        return copy.deepcopy(self)

    def _create_edge_info(self, node_positions, curved=False, mid_ax_lines=None):
        working_nodes = self._get_working_nodes()
        edges = dict()
        is_colour_attr_numerical = False if self.edge_category_colours or self.edge_colour_attribute is "random" else None
        for edge in self.network.edges_iter(data=True):
            if is_colour_attr_numerical is None:
                try:
                    float(edge[2][self.edge_colour_attribute])
                    is_colour_attr_numerical = True
                except ValueError:
                    is_colour_attr_numerical = False
                    if self.edge_category_colours is None:
                        self.edge_category_colours = categories_to_float(self._get_edges_colour_attr())

            # prevent intra-axis edge
            if self.network.node[edge[0]][self.node_class_attribute] == self.network.node[edge[1]][
                    self.node_class_attribute]:
                continue

            key = tuple(sorted(edge[:2]))

            if key in edges:    # if edge already exists in data set
                if self.edge_colour_attribute is not "random" and is_colour_attr_numerical:
                    edges[key] += float(edge[2][self.edge_colour_attribute])
            elif set(key).issubset(working_nodes):
                if is_colour_attr_numerical:
                    edges[key] = float(edge[2][self.edge_colour_attribute])
                elif self.edge_colour_attribute is "random":
                    edges[key] = random.random()
                else:
                    edges[key] = self.edge_category_colours[edge[2][self.edge_colour_attribute]]

        if is_colour_attr_numerical:
            maximum = max([weight for weight in edges.values()])
            for edge in edges:
                edges[edge] /= maximum

            if self.normalise_link_colours:
                old_new_weights = dict(zip(sorted(edges.values()), range(len(edges))))

                for edge in edges:
                    edges[edge] = old_new_weights[edges[edge]]

        if curved:
            crossing_points = dict()
            intersection_points = {mid_ax_line: [] for mid_ax_line in mid_ax_lines}
            for edge in edges:
                mid_ax_line, intersection = self._get_name_and_point_of_intersection(edge, node_positions, mid_ax_lines)
                intersection_proportion = np.linalg.norm(np.array(intersection))/mid_ax_lines[mid_ax_line].length
                intersection_points[mid_ax_line].append((edge, intersection_proportion))

            for mid_ax_line in mid_ax_lines:
                sorted_edges = np.array(sorted(intersection_points[mid_ax_line], key=lambda point: point[1]))
                #maximum = sorted_edges[-1, 1]
                #sorted_edges[:, 1] = np.linspace(0, maximum, np.shape(sorted_edges)[0])

                for edge, crossing_proportion in sorted_edges:
                    crossing_points[edge] = place_point_on_line(mid_ax_lines[mid_ax_line], self.edge_curvature*crossing_proportion)

        gradient = eval("pyx.color.gradient." + self.edge_colour_gradient)
        retlist = []
        for edge in edges:
            start, end = edge
            colour = gradient.getcolor(edges[edge]) if isinstance(edges[edge], (float, int)) else convert_colour(edges[edge])
            entry = {
                "start": node_positions[start],
                "end": node_positions[end],
                "colour": colour
            }
            if curved:
                entry["crossing_point"] = crossing_points[edge]
            retlist.append(entry)
        return retlist

    @staticmethod
    def _get_name_and_point_of_intersection(edge, node_positions, mid_ax_lines):
        line = geom.LineString([node_positions[edge[0]], node_positions[edge[1]]])
        for mid_ax_line in mid_ax_lines:
            if line.intersects(mid_ax_lines[mid_ax_line]):
                return mid_ax_line, line.intersection(mid_ax_lines[mid_ax_line])

    @staticmethod
    def _weight_by_colocation(node_positions):
        tally = Counter(list(node_positions.values()))
        tally_arr = np.array(list(tally.items()), dtype="object")
        tally_arr[:, 1] = tally_arr[:, 1]/np.max(tally_arr[:, 1])
        return dict(tally_arr)

    def _setup_latex(self):
        pyx.text.set(pyx.text.LatexRunner)
        pyx.text.preamble(r"\usepackage{color}")
        pyx.text.preamble(r"\usepackage[T1]{fontenc}")
        pyx.text.preamble(r"\usepackage{lmodern}")
        col = convert_colour(self.label_colour)
        pyx.text.preamble(r"\definecolor{COL}{cmyk}{%g,%g,%g,%g}" % (col.c, col.m, col.y, col.k))

    def _get_edges_colour_attr(self):
        return set(nx.get_edge_attributes(self.network, self.edge_colour_attribute).values())