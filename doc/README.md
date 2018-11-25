# Building MyGrad's Docs

Install the the following:

```shell
pip install sphinx
pip install sphinx_rtd_theme
pip install nbsphinx
pip install numpydoc
```

To build the docs for the first time, navigate to `mygrad/doc/source/` 
and generate the proper `.rst` files via:

```shell
sphinx-autogen -o generated/ *.rst
```

Then navigate to `mygrad/doc/` and run:

```shell
make html
```

on Windows, the command is:

```shell
make.bat html
```

This will create the `mygrad/doc/build/` directory, which contains the html for 
the documentation. You can delete this directory and rerun the above command to 
force a clean build of the html. This is typically needed if a new page has been 
introduced, but the quick-navigation index does not change.