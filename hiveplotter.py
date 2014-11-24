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
import warnings


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
        self.network = network
        self.node_class_attribute = node_class_attribute
        self.node_classes = self._split_nodes(node_class_values)

        # define default behaviour, which can be overridden with kwargs

        # background parameters
        self.background_proportion = 1.2
        self.background_colour = "Black"

        # axis parameters
        self.axis_length = 10
        self.proportional_offset_from_intersection = 0.2  # as a proportion of the longest axis
        self.split_axes = []
        if self.split_axes:
            raise NotImplementedError("Axis splitting is not yet implemented.")
        self.normalise_axis_length = False
        self.axis_colour = "Gray"
        self.axis_thickness = 0.15

        # edge parameters
        self.edge_thickness_range = (0.005, 0.16)
        self.edge_colour_attribute = "weight"
        self.edge_colour_gradient = "Jet"
        self.edge_category_colours = None
        self.edge_curvature = 1.7
        self.normalise_edge_colours = False

        # axis label parameters
        self.label_colour = "White"
        self.label_size = 15

        # node parameters
        self.normalise_node_distribution = False
        self.node_size = 0.08
        self.node_colour_gradient = "GreenRed"

        # legend parameters
        self.legend_position = "top_right"

        self.__dict__.update(kwargs)

        # fields to be filled by the object
        self.canvas = None
        self.legend_categories = None
        self._background_layer = None
        self._foreground_layer = None
        self._legend_layer = None
        self.colour_definitions = self._convert_colours()


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
        # max_length = self.axis_length * (self.background_proportion + self.proportional_offset_from_intersection)
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

    def _draw_axes(self, axes):
        """
        :param axes: Dictionary of axis names and their geometries
        :type axes: dict
        :return: None
        :rtype: None
        """
        axis_colour = convert_colour(self.axis_colour)
        for axis in axes.values():
            self._foreground_layer.stroke(pyx.path.line(*linestring_to_coords(axis)),
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
            txt_str = self._colour_text(self._size_text(axis_name, self.label_size), self.label_colour)

            self._foreground_layer.text(line_end[0], line_end[1],
                                        txt_str,
                                        text_alignment[axis_name])

    def _size_text(self, text, size):
        return r"{\fontsize{" + str(size) + r"}{" + str(round(size * 1.2)) + r"}\selectfont " + text + r"}"

    def _colour_text(self, text, colour):
        return r"\textcolor{" + str(colour) + "}{" + text + "}"

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

            self._foreground_layer.stroke(edgepath, [
                pyx.style.linewidth(edge_dict["thickness"]),
                colour
            ])

    def _draw_nodes(self, node_positions):
        gradient = eval("pyx.color.gradient." + self.node_colour_gradient)
        node_position_weights = self._weight_by_colocation(node_positions)
        for coords, weight in node_position_weights.items():
            self._foreground_layer.stroke(pyx.path.circle(coords[0], coords[1], self.node_size))
            self._foreground_layer.fill(pyx.path.circle(coords[0], coords[1], self.node_size),
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
        axes = self._create_axes()
        node_positions = self._place_nodes(axes)

        if len(axes) == 3:
            mid_ax_lines = self._create_mid_ax_lines(axes)

            # edge_lines = self._create_straight_edge_info(node_positions, curved=False)
            edge_lines = self._create_edge_info(node_positions, curved=True, mid_ax_lines=mid_ax_lines)
        else:
            raise NotImplementedError("2-axis plots not yet implemented")

        self._draw_axes(axes)
        self._draw_labels(axes)
        self._draw_edges(edge_lines)
        self._draw_nodes(node_positions)
        self._draw_legend(axes)
        self._draw_background()

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

            mid_ax_lines[mid_id] = geom.LineString(
                [place_point_on_line(start_to_start, 0.5), place_point_on_line(end_to_end, 0.5)])

        return mid_ax_lines

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
        Generate axes on which to plot nodes
        :return: A dictionary whose keys are the node class attribute values, and values are LineStrings of the axis those nodes will be plotted on
        :rtype: bool
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
        edge_weights = set()
        for edge in self.network.edges_iter(data=True):
            if "weight" in edge[2]:
                edge_weights.add(edge[2]["weight"])
            else:
                edge_weights.add(1)

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

            if key in edges:  # if edge already exists in data set
                if self.edge_colour_attribute is not "random" and is_colour_attr_numerical:
                    edges[key] += float(edge[2][self.edge_colour_attribute])
            elif set(key).issubset(working_nodes):
                if is_colour_attr_numerical:
                    edges[key] = float(edge[2][self.edge_colour_attribute])
                elif self.edge_colour_attribute is "random":
                    edges[key] = random.random()
                else:
                    edges[key] = self.edge_category_colours[edge[2][self.edge_colour_attribute]]

        min_weight = min(edge_weights)
        weight_range = max(edge_weights) - min_weight
        min_thickness = min(self.edge_thickness_range)
        thickness_range = max(self.edge_thickness_range) - min_thickness
        edge_thickness_dict = dict()
        for edge in self.network.edges_iter(data=True):
            key = tuple(sorted(edge[:2]))
            if "weight" in edge[2] and weight_range:
                edge_thickness_dict[key] = ((edge[2][
                                                 "weight"] - min_weight) / weight_range * thickness_range) + min_thickness
            else:
                edge_thickness_dict[key] = min_thickness

        if is_colour_attr_numerical:
            maximum = max([colour_attr_val for colour_attr_val in edges.values()])
            for edge in edges:
                edges[edge] /= maximum

            if self.normalise_edge_colours:
                old_new_weights = dict(zip(sorted(edges.values()), range(len(edges))))

                for edge in edges:
                    edges[edge] = old_new_weights[edges[edge]]

        if curved:
            crossing_points = dict()
            intersection_points = {mid_ax_line: [] for mid_ax_line in mid_ax_lines}
            for edge in edges:
                mid_ax_line, intersection = self._get_name_and_point_of_intersection(edge, node_positions, mid_ax_lines)
                intersection_proportion = np.linalg.norm(np.array(intersection)) / mid_ax_lines[mid_ax_line].length
                intersection_points[mid_ax_line].append((edge, intersection_proportion))

            for mid_ax_line in mid_ax_lines:
                sorted_edges = np.array(sorted(intersection_points[mid_ax_line], key=lambda point: point[1]))
                # maximum = sorted_edges[-1, 1]
                #sorted_edges[:, 1] = np.linspace(0, maximum, np.shape(sorted_edges)[0])

                for edge, crossing_proportion in sorted_edges:
                    crossing_points[edge] = place_point_on_line(mid_ax_lines[mid_ax_line],
                                                                self.edge_curvature * crossing_proportion)

        gradient = eval("pyx.color.gradient." + self.edge_colour_gradient)
        retlist = []
        for edge in edges:
            start, end = edge
            colour = gradient.getcolor(edges[edge]) if isinstance(edges[edge], (float, int)) else convert_colour(
                edges[edge])
            entry = {
                "start": node_positions[start],
                "end": node_positions[end],
                "colour": colour,
                "thickness": edge_thickness_dict[edge]
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
        tally_arr[:, 1] = tally_arr[:, 1] / np.max(tally_arr[:, 1])
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

    def _get_edges_colour_attr(self):
        return set(nx.get_edge_attributes(self.network, self.edge_colour_attribute).values())

    def _draw_legend(self, axes):
        if not self.edge_category_colours:
            return

        legend_items = []
        for category in self.edge_category_colours:
            legend_items.append(self._size_text(
                "%s %s" % (self._colour_text("--- ", self.edge_category_colours[category]),
                           self._colour_text(category, self.label_colour)), self.label_size)
            )

        legend_str =  r"\linebreak".join(legend_items)
        max_x = max([axes[key].coords[1][0] for key in axes])

        self._legend_layer.text(max_x, 0, legend_str, [pyx.text.parbox(4), pyx.text.halign.left, pyx.text.valign.middle])

    def _convert_colours(self):
        d = {self.label_colour: convert_colour(self.label_colour),
             self.background_colour: convert_colour(self.background_colour),
             self.axis_colour: convert_colour(self.axis_colour)}

        if self.edge_category_colours:
            for value in self.edge_category_colours.values():
                d[value] = convert_colour(value)

        return d