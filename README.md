# MyGrad

A pure-python autograd/tensor library (CogWorks 2017)

`mygrad` was created as a prototype for the CogWorks 2017 summer program, in the [Beaver Works Summer Institute at MIT](https://beaverworks.ll.mit.edu/CMS/bw/bwsi). It was developed by [Ryan Soklaski](https://github.com/LLrsokl), the lead instructor of CogWorks 2017, and by [Nick Stanisha](https://github.com/nickstanisha). 

## Installation Instructions
Clone MyGrad, navigate to the resulting directory, and run

```shell
python setup.py develop
```

If you want to run unit tests, install `pytest` and `hypothesis`:

```shell
conda install pytest
pip install hypothesis
```

And, in the MyGrad directory, run:
```shell
pytest tests
```
