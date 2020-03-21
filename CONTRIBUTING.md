# Contributing
Thanks for getting involved! PRs and github issues are more than welcome.

You will need to install `tox` to run tests and auto-formatting:

```shell script
pip install tox
```

# Adhering to MyGrad's style / formatting
## Auto-formatting
MyGrad uses: 
 - [flake8](https://flake8.pycqa.org/en/latest/) to enforce the [pep8 style](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Writing_Good_Code.html#The-PEP8-Style-Guide-to-Python-Code)
 - [isort](https://github.com/timothycrosley/isort) to organize import statements
 - [black](https://black.readthedocs.io/en/stable/)
 
Once you have made your changes/additions to MyGrad's code, you can run all of these auto-formatting tools via `tox`:

```shell
tox -e format
```

This will (temporarily) install and run all of these formatting tools on MyGrad's code base for you.


## Docstring format
MyGrad adheres to the [NumPy docstring](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Writing_Good_Code.html#The-NumPy-Documentation-Style) style, 
and strives to adhere to formally-correct [type hints](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Writing_Good_Code.html#Type-Hinting) in the `Parameters`
section.

 
# Running tests
PRs will automatically execute a full test suite on Travis.

To test locally

```shell
tox -e py
```

will run the tests in a hermetic environment (without even having to install `mygrad`, and

```shell
pytest tests
```

will run in your local python environment.

