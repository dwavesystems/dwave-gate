D-Wave Gate Model Software
==========================

.. index-start-marker

``dwave-gate`` is a software package and set of tools to construct, modify and run quantum circuits.

Features
--------

* Construct quantum circuits by appending operations to a ``Circuit`` object.

* Simulate circuits using a performant state-vector simulator built in C++.

* Have easy access to quantum operations, their matrix representations, decompositions and easily
  create custom operations and circuit templates.


Example Usage
-------------

Construct a two-qubit circuit using the ``dwgms.Circuit`` object, to which you can easily append
operations using the circuits context-manager.

.. code-block:: python

    from dwgms import Circuit
    import dwgms.operations as ops

    circuit = Circuit(2)
    with circuit.context as q:
        ops.X(q[0])
        ops.Hadamard(q[1])
        ops.CZ(q[0], q[1])
        ops.Hadamard(q[1])

The above circuit can then be simulated using ``dwgms.simulator.simulate()``.

>>> from dwgms.simulator import simulate
>>> simulate(circuit)
array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])

.. index-end-marker

Installation
------------

.. installation-start-marker

The simplest way to install ``dwave-gate`` is from source, by cloning the repository, navigating to
the root folder and running

.. code-block:: bash

    make install

The makefile will also simplify running tests (``make test``), coverage (``make coverage``) and
formatting (``make format``) the code with the help of the `Black <https://black.readthedocs.io/>`_
formatter, set to a line-lenth of 100, and `isort <https://pycqa.github.io/isort/>`_.

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
