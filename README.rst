.. image:: https://img.shields.io/pypi/v/dwave-gate.svg
    :target: https://pypi.org/project/dwave-gate

.. image:: https://img.shields.io/pypi/pyversions/dwave-gate.svg
    :target: https://pypi.org/project/dwave-gate

.. image:: https://circleci.com/gh/dwavesystems/dwave-gate.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-gate

.. image:: https://codecov.io/gh/dwavesystems/dwave-gate/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-gate

dwave-gate
==========

.. index-start-marker

``dwave-gate`` is a software package for constructing, modifying and running quantum circuits on the
included simulator. It provides a set of tools that enables you to:

* Construct quantum circuits using an intuitive context-manager interface.

* Utilize a comprehensive library of quantum gates with simple access to matrix representations,
  various decompositions, and more.

* Simulate circuits on a performant (C++) state-vector simulator.

* Easily create your own quantum gates and templates. Any circuit can be either directly applied in
  another circuit or converted into a quantum operation.

.. index-end-marker

Example usage
-------------

.. example-start-marker

This example uses the ``dwave.gate.Circuit`` object's  context manager to append operations to
a two-qubit circuit.

.. code-block:: python

    import dwave.gate.operations as ops
    from dwave.gate import Circuit

    circuit = Circuit(2)

    with circuit.context as (q, c):
        ops.X(q[0])
        ops.Hadamard(q[1])
        ops.CZ(q[0], q[1])
        ops.Hadamard(q[1])

You can run the ``dwave.gate.simulator`` simulator on such circuits,

>>> from dwave.gate.simulator import simulate
>>> simulate(circuit)

and then access the resulting state via the state attribute.

>>> circuit.state
array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])

.. example-end-marker

Installation
------------

.. installation-start-marker

The simplest way to install ``dwave-gate`` is from `PyPI <https://pypi.org/project/dwave-gate>`_:

.. code-block:: bash

    pip install dwave-gate

It can also be installed from source by cloning this GitHub repository and running:

.. code-block:: bash

    make install

The makefile will also simplify running tests (``make test``), coverage (``make coverage``),
documentation (``make docs``), as well as formatting (``make format``) the code using the `Black
<https://black.readthedocs.io/>`_ formatter (set to a line-length of 100) and `isort
<https://pycqa.github.io/isort/>`_. It's available on both Unix as well as Windows systems, via the
`make.bat` batch file.

Alternatively, the package can be built and installed in development mode using Python and pip. The
simulator operations would need to be generated first by executing `operation_generation.py`, found
in `dwave/gate/simulator`.

.. code-block:: bash

    python setup.py build_ext --inplace
    pip install -e .

Tests and coverage can be run using Pytest.

.. code-block:: bash

    python -m pytest tests/ --cov=dwave.gate

.. note::

    For the QIR compiler and loader to work the PyQIR (v0.9.0) is required. It can be
    installed manually with ``pip install pyqir==0.9.0`` or as an optional dependency:

    .. code-block:: bash

        pip install dwave-gate[qir]

.. installation-end-marker

License
-------

Released under the Apache License 2.0. See LICENSE file.

Contributing
------------

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.

Release Notes
~~~~~~~~~~~~~

``dwave-gate`` uses `reno <https://docs.openstack.org/reno/>`_ to manage its release notes.

When making a contribution to ``dwave-gate`` that will affect users, create a new release note file
by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``. Remove any sections not relevant
to your changes. Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_ for details.
