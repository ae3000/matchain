from setuptools import find_packages, setup

setup(
    name='matchain',
    version='1.0.0',
    author='Andreas Eibeck',
    license='MIT',
    description='a flexible tool for matching data sets',
    long_description=open('README.md').read(),
    packages=find_packages(exclude=("tests")),
    python_requires='>=3.8, <4',
)
