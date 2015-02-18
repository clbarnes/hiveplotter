[![Stories in Ready](https://badge.waffle.io/clbarnes/hiveplotter.png?label=ready&title=Ready)](https://waffle.io/clbarnes/hiveplotter)
hiveplotter
===========

A python library for creating highly customisable [hive plots](http://www.hiveplot.net/) from [NetworkX](https://networkx.github.io/) graphs.

This project is released under the [BSD License](https://raw.githubusercontent.com/clbarnes/hiveplotter/master/LICENSE).

WIP

Requires:
 - LaTeX (for labels)
 - Ghostscript (for displaying plot as a bitmap)
 - Dependencies as listed in setup.py: shapely, networkx, pyx, numpy, PIL
 
See hiveplotter/examples for usage.

Default behaviour can be found in hiveplotter_defaults.ini
Default behaviour can be overridden with keyword arguments in the HivePlot constructor, or by pointing to an alternative config file (kwarg config_path).
Most behaviour can be can be changed between instantiation of the hive plot and drawing.
