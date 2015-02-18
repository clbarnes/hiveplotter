from distutils.core import setup

setup(
    name='hiveplotter',
    version='0.1',
    packages=[],
    py_modules=['hiveplotter'],
    url='https://github.com/clbarnes/hiveplotter',
    license='BSD',
    author='clbarnes',
    author_email='cbarnes@mrc-lmb.cam.ac.uk',
    description='A python library for creating hive plots from networkx graphs',
    requires=['shapely', 'networkx', 'pyx', 'numpy', 'PIL']
)
