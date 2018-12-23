from setuptools import setup, find_packages

setup(
    name='shapelets-lts',
    version='0.3.0.dev1',
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib',
    ],
    extras_require={'dev': ['ipython', 'nose'], },
    packages=find_packages()
)
