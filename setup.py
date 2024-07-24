import versioneer
from setuptools import find_packages, setup

DISTNAME = "mygrad"
LICENSE = "MIT"
AUTHOR = "Ryan Soklaski"
AUTHOR_EMAIL = "rsoklaski@gmail.com"
URL = "https://github.com/rsokl/MyGrad"
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
]

# tests/tensor_ops/test_getitem.py::test_getitem_advindex_bool_bkwdprop segfaults
INSTALL_REQUIRES = ["numpy >= 1.20, !=1.25.0", "typing-extensions >= 4.1.0, !=4.6.0"]
TESTS_REQUIRE = ["pytest >= 3.8", "hypothesis >= 6.17.1", "scipy"]

DESCRIPTION = "Brings drop-in automatic differentiation to NumPy"
LONG_DESCRIPTION = """
MyGrad is a lightweight library that adds automatic differentiation to NumPy – its only dependency is NumPy!
It's primary goal is to make automatic differentiation an accessible and easy to use across the Python/NumPy ecosystem.

MyGrad introduces a tensor object, which behaves like NumPy's ndarray object, but that builds a computational
graph, which enables MyGrad to perform reverse-mode differentiation (i.e. "backpropagation"). By exploiting
NumPy's mechanisms for ufunc/function overrides, MyGrad's tensor works "natively" with NumPy's suite of mathematical
functions so that they can be chained together into a differentiable computational graph.

NumPy's systems for broadcasting operations, producing views of arrays, performing in-place operations, and permitting
both "basic" and "advanced" indexing of arrays are all supported by MyGrad to a high-fidelity.
"""


setup(
    name=DISTNAME,
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
        "rnn": ["numba>=0.34.0"]  # GRU and vanilla RNN require numba-acceleration
    },
    url=URL,
    download_url="https://github.com/rsokl/mygrad/tarball/v" + versioneer.get_version(),
    python_requires=">=3.7",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
)
