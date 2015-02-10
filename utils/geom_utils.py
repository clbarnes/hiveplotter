import math
import cmath
from shapely import geometry as geom
import numpy as np


def linestring_to_coords(linestring):
    ret = []
    for point in linestring.coords:
        for coord in point:
            ret.append(coord)

    return tuple(ret)


def get_projection(startpoint, angle, distance):
    """
    :param startpoint: (x, y) tuple of starting coordinates
    :param angle: Bearing, clockwise from 0 degrees, in which to project
    :param distance: Length of projection
    :return: (x, y) tuple of ending coordinates
    """
    angle = 90 - angle
    if angle < -180:
        angle += 360

    angle_r = math.radians(angle)

    start = complex(startpoint[0], startpoint[1])
    movement = cmath.rect(distance, angle_r)
    end = start + movement
    return end.real, end.imag


def get_projecting_line(startpoint, angle, distance):
    """
    :param startpoint: (x, y) tuple of starting coordinates
    :param angle: Bearing, clockwise from 0 degrees, in which to project
    :param distance: Length of projection
    :return: LineString of projection
    """
    return geom.LineString([startpoint, get_projection(startpoint, angle, distance)])


def place_point_proportion_along_line(line, proportion):
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


def multipoint_centroid(*args):
    arr = np.array(args)
    return tuple(np.mean(arr, axis=0))


def mid_line(line1, line2):
    start_point = multipoint_centroid(line1.coords[0], line2.coords[0])
    end_point = multipoint_centroid(line1.coords[-1], line2.coords[-1])
    return geom.LineString(coordinates=[start_point, end_point])