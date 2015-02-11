from distutils.core import setup
import sys

with open("LICENSE", 'r') as license_file:
    print(license_file.read())

while True:
    user_input = input("Do you agree? (yes/no): ")
    if user_input.lower() == "yes":
        break
    elif user_input.lower() == "no":
        print('Aborting installation...')
        sys.exit()
    else:
        print('Input not recognised (yes/no)')

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
