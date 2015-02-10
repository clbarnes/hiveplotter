import pyx
from pyx.metapost.path import beginknot, endknot, smoothknot, tensioncurve

from shapely import geometry as geom
import numpy as np
from utils import geom_utils


class Axis():
    def __init__(self, start, end):
        self.line = geom.LineString(coordinates=[start, end])
        self.nodes = dict()
        self.colour = None
        self.thickness = None
        self.label = None

    def add_nodes(self, nodes_proportions):
        for node in nodes_proportions:
            self.nodes[node] = geom_utils.place_point_proportion_along_line(self.line, nodes_proportions[node])

    def __contains__(self, item):
        return item in self.nodes

    @property
    def coords(self):
        return self.line.coords

    @property
    def start(self):
        return self.line.coords[0]

    @property
    def end(self):
        return self.line.coords[1]

    def set_visual_properties(self, colour, thickness, label=""):
        self.colour = colour
        self.thickness = thickness
        self.label = label

    def draw(self, canvas):
        canvas.stroke(pyx.path.line(*geom_utils.linestring_to_coords(self.line)),
                      [pyx.style.linewidth(self.thickness), self.colour])


class Edge():
    def __init__(self, start_node, start_axis, end_node, end_axis, curved=True, mid_ax_line=None):
        self.start_node = start_node
        self.end_node = end_node
        self.start_point = start_axis.nodes[start_node]
        self.end_point = end_axis.nodes[end_node]
        self.mid_point = self._make_midpoint(start_axis, end_axis, curved, mid_ax_line)
        self.colour = None
        self.thickness = None

    def _make_midpoint(self, start_axis, end_axis, curved, mid_ax_line):
        if mid_ax_line is None:
            mid_ax_line = geom_utils.mid_line(start_axis.line, end_axis.line)

        intersection = mid_ax_line.intersection(geom.LineString(coordinates=[self.start_point, self.end_point]))
        if not curved:
            return intersection
        intersection_proportion = np.linalg.norm(np.array(intersection)-np.array(mid_ax_line.coords[0])) / mid_ax_line.length
        return geom_utils.place_point_proportion_along_line(mid_ax_line, intersection_proportion * start_axis.line.length/mid_ax_line.length)

    @property
    def coords(self):
        return [self.start_point, self.mid_point, self.end_point]

    @property
    def nodes(self):
        return self.start_node, self.end_node

    def set_visual_properties(self, colour, thickness):
        self.colour = colour
        self.thickness = thickness

    def draw(self, canvas):
        edgepath = pyx.metapost.path.path([
            beginknot(*self.start_point), tensioncurve(),
            smoothknot(*self.mid_point), tensioncurve(),
            endknot(*self.end_point)
        ])

        canvas.stroke(edgepath, [pyx.style.linewidth(self.thickness), self.colour])