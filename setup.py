from setuptools import setup, find_packages

setup(
    name='hiveplotter',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={'hiveplotter': ['hiveplotter/*.ini']},
    url='https://github.com/clbarnes/hiveplotter',
    license='BSD',
    author='Chris L. Barnes',
    author_email='cbarnes@mrc-lmb.cam.ac.uk',
    description='A python library for creating hive plots from networkx graphs',
    requires=['shapely', 'networkx', 'pyx', 'numpy', 'pillow']
)
