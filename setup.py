from setuptools import setup


setup(name='LASSOBench',
      packages=['LASSOBench'],
      install_requires=['sparse-ho @ https://github.com/QB3/sparse-ho/archive/master.zip', 'celer', 'pyDOE', 'libsvmdata', 'matplotlib>=2.0.0',
            'numpy>=1.12', 'scipy>=0.18.0',
            'scikit-learn>=0.21', 'seaborn>=0.7', 'GPy>=1.9.2', 'pyDOE>=0.3.8'],
      )
