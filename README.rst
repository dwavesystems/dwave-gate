D-Wave Gate
===========

.. index-start-marker

``dwave-gate`` is a software package and set of tools to construct, modify and run quantum circuits.

Features
--------

* Construct quantum circuits using an intuitive context manager interface.

* Utilize our comprehensive library of quantum gates, and have quick access to their matrix representations, various decompositions and more.

* Simulate circuits on our performant state-vector simulator, written in C++.

* Easily create your own quantum gates and templates. Any circuit can either be directly applied in another circuit or converted into a quantum operation.


Example Usage
-------------

Construct a two-qubit circuit using the ``dwave.gate.Circuit`` object, to which you can easily append
operations using the circuits context-manager.

.. code-block:: python

    import dwave.gate.operations as ops
    from dwave.gate import Circuit

    circuit = Circuit(2)

    with circuit.context as q:
        ops.Hadamard(q[1])
        ops.CZ(q[0], q[1])
        ops.Hadamard(q[1])

The above circuit can then be simulated using ``dwave.gate.simulator.simulate()``.

>>> from dwave.gate.simulator import simulate
>>> simulate(circuit)
array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])

.. index-end-marker

Installation
------------

.. installation-start-marker

The simplest way to install ``dwave-gate`` is from PyPI:

.. code-block:: bash

    pip install dwave-gate

It can also be installed from source by cloning the GitHub repository and running:

.. code-block:: bash

    make install

The makefile will also simplify running tests (``make test``), coverage (``make coverage``) and
formatting (``make format``) the code using the `Black <https://black.readthedocs.io/>`_ formatter,
set to a line-lenth of 100, and `isort <https://pycqa.github.io/isort/>`_.

.. installation-end-marker


Release Notes
~~~~~~~~~~~~~

``dwave-gate`` uses `reno <https://docs.openstack.org/reno/>`_ to manage its release notes.

When making a contribution to ``dwave-gate`` that will affect users, create a new release note file by
running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``. Remove any sections not relevant
to your changes. Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_ for details.
