from setuptools import find_packages, setup

setup(
    name='matchain',
    version='1.0.0',
    author='Andreas Eibeck',
    license='BSD-3-Clause license',
    description='Record linkage - simple, flexible, efficient.',
    long_description=open('README.md').read(),
    packages=find_packages(exclude=("tests")),
    python_requires='>=3.8, <4',
)
