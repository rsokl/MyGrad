# Building MyGrad's Docs

Install the following:

```shell
pip install sphinx
pip install sphinx_rtd_theme
pip install nbsphinx
pip install numpydoc
pip install matplotlib
pip install pydata_sphinx_theme
```

To build the docs for the first time, navigate to `mygrad/docs/source/` 
and generate the proper `.rst` files via:

```shell
sphinx-autogen -o generated/ *.rst
```

In the case of Windows, Powershell might have trouble with the wildcard, so use:

```shell
Get-ChildItem -Filter "*.rst" | ForEach-Object { sphinx-autogen -o generated/ $_.FullName }
```

Then navigate to `mygrad/docs/` and run:

```shell
make html
```

on Windows, the command is:

```shell
make.bat html
```

This will create the `mygrad/docs/build/` directory, which contains the html for 
the documentation. You can delete this directory and rerun the above command to 
force a clean build of the html. This is typically needed if a new page has been 
introduced, but the quick-navigation index does not change.
