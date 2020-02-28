# Contributing
Thanks for getting involved! PRs and github issues are more than welcome.

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
