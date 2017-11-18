from setuptools import setup, find_packages

setup(
    name='shapelets-lts',
    version='0.2.2',
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib',
    ],
    packages=find_packages()
)
