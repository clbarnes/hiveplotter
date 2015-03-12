from distutils.core import setup
import shutil
import site
import os

CONF_PATH = os.path.join(site.getsitepackages()[0], 'hiveplotter_defaults.ini')
shutil.copy('hiveplotter_defaults.ini', CONF_PATH)

setup(
    name='hiveplotter',
    version='0.1',
    packages=['hiveplotter', 'hiveplotter.util'],
    url='https://github.com/clbarnes/hiveplotter',
    license='BSD',
    author='clbarnes',
    author_email='cbarnes@mrc-lmb.cam.ac.uk',
    description='A python library for creating hive plots from networkx graphs',
    requires=['shapely', 'networkx', 'pyx', 'numpy', 'PIL']
)
