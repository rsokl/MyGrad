import versioneer
from setuptools import find_packages, setup

DISTNAME = 'mygrad'
LICENSE = 'MIT'
AUTHOR = 'Ryan Soklaski'
AUTHOR_EMAIL = 'rsoklaski<at>gmail<dot>com'
URL = 'https://github.com/rsokl/MyGrad'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['numpy >= 1.12', "numba>=0.34.0"]
TESTS_REQUIRE = ['pytest >= 3.8', 'hypothesis >= 4.0', 'scipy']

DESCRIPTION = "A sleek auto-differentiation library that wraps numpy."
LONG_DESCRIPTION = """
mygrad is a simple, NumPy-centric autograd library. An autograd library enables 
you to automatically compute derivatives of mathematical functions. This library is 
designed to serve primarily as an education tool for learning about gradient-based 
machine learning; it is easy to install, has a readable and easily customizable code base, 
and provides a sleek interface that mimics NumPy. Furthermore, it leverages NumPy's 
vectorization to achieve good performance despite the library's simplicity.

This is not meant to be a competitor to libraries like PyTorch (which mygrad most 
closely resembles) or TensorFlow. Rather, it is meant to serve as a useful tool for 
students who are learning about training neural networks using back propagation.
"""


setup(name=DISTNAME,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      extras_require={
          'rnn': ["numba>=0.34.0"],  # GRU and vanilla RNN require numba-acceleration
      },
      url=URL,
      python_requires='!=2.*,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
      packages=find_packages(exclude="tests/*"),
      )
