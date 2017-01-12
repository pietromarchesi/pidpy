
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import numpy as np
from distutils.extension import Extension


def readme():
    with open('README.rst') as f:
        return f.read()


# ext_1 = Extension(SRC_DIR + ".wrapped",
#                   [SRC_DIR + "/lib/cfunc.c", SRC_DIR + "/wrapped.pyx"],
#                   libraries=[],
#                   include_dirs=[np.get_include()])
#
#
# EXTENSIONS = [ext_1]

setup(name='pidpy',
      version='0.1',
      description='Partial Information Decomposition in Python',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Information Theory',
      ],
      keywords='partial information decomposition synergy '
               'redundancy unique',
      #url='http://github.com/',
      author='Pietro Marchesi',
      author_email='pietromarchesi92@gmail.com',
      license='new BSD',
      packages=['pidpy'],
      install_requires=[
          'numpy',
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite = 'nose.collector',
      tests_require = ['nose'],
      ext_modules=[Extension('pidpy.utilsc', ['pidpy/utilsc.c'])],
      include_dirs=[np.get_include()]
      )