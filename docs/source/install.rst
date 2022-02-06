.. MyGrad documentation master file, created by
   sphinx-quickstart on Sun Oct 21 09:57:03 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installing MyGrad
=================
MyGrad requires numpy. It is highly recommended that you utilize numpy built with MKL for access to optimized math
routines (e.g. install numpy via anaconda). You can install MyGrad using pip:

.. code-block:: shell

  pip install mygrad


You can instead install MyGrad from its source code. Clone `this repository <https://github.com/rsokl/MyGrad>`_ and
navigate to the MyGrad directory, then run:

.. code-block:: shell

  pip install .


Support for Python and NumPy
----------------------------
MyGrad abides by the `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_ recommendation, and adopts
a common “time window-based” policy for support of NumPy versions. Accordingly, MyGrad's drop schedule for NumPy versions can be found `here <https://numpy.org/neps/nep-0029-deprecation_policy.html#drop-schedule>`_. 

Note, however, that MyGrad will maintain a wider window of support for minor Python 
versions than is specified by NEP 29. Because our only dependency is NumPy, and because
we strive to remain an exceptionally lightweight and flexible dependency to our users, 
we will support minor versions of Python until their end of life, *or* until our lowest
supported version of NumPy drops support for that version of Python -- whichever occurs
first.