from setuptools import setup, find_packages

setup(
    name='shapelets-lts',
    version='0.3.0',
    install_requires=[
        'numpy>=1.15.4,<2.0.0',
        'pandas>=0.23.4,<2.0.0'
        'scipy>=1.2.0,<2.0.0',
        'scikit-learn>=0.20.2,<2.0.0',
        'matplotlib>=2.2.3,<3.0.0',
        'seaborn>=0.9.0,<2.0.0'
    ],
    extras_require={'dev': ['nose>=1.3.7,<2.0.0', 'ipython>=5.8.0,<6.0.0']},
    packages=find_packages()
)
