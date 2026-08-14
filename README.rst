.. image:: https://img.shields.io/pypi/v/dwave-gate.svg
    :target: https://pypi.org/project/dwave-gate

.. image:: https://codecov.io/gh/dwavesystems/dwave-gate/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-gate

.. image:: https://circleci.com/gh/dwavesystems/dwave-gate.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-gate

==========
dwave-gate
==========

.. start_gate_about

``dwave-gate`` provides the functionality for generating quantum circuit
description language (QCDL) programs. QCDL is an embedded domain-specific
language (DSL) that uses Python as its host language, allowing you to interleave
classical Python preprocessing with quantum gate operations. Programs are
defined with the ``@qcdl`` decorator, which converts an ordinary Python function
into a QCDL program generator that accepts qubit arguments and returns a
dictionary suitable for submission to compiler or simulator services.

The following example creates a Bell state circuit, producing a dictionary
that can be passed to a compiler or simulator:

.. code-block:: python

    from dwave.gate.qcdl import qcdl
    from dwave.gate.qcdl.operations import cx, h, measure

    @qcdl(num_qubits=2)
    def main(q0, q1):
        h(q0)        # Hadamard gate on q0
        cx(q0, q1)   # CNOT with q0 as control, q1 as target
        measure(q0)
        measure(q1)

    qcdl_program = main()

To run the program on a solver, use the Ocean SDK's cloud client to locate a solver
that supports QCDL and submit the dictionary via ``sample_qcdl``:

.. code-block:: python

    import orjson
    from dwave.cloud import Client
    from dwave.gate.results import Result

    client = Client.from_config()
    solver = client.get_solver(supported_problem_types__contains="qcdl")

    response = solver.sample_qcdl(qcdl_program, shots=3)
    answer = orjson.loads(response.answer_data.read())
    result = Result(**answer)

Measurements and sample counts are now accessible by calling 
``result.measurements`` or ``result.get_counts()`` respectively.

.. end_gate_about

Installation
============

**Installation from PyPI:**

.. code-block:: bash

    pip install dwave-gate

**Installation from source:**

.. code-block:: bash

    pip install .

**Development setup**

Install development requirements and the package in editable mode:

.. code-block:: bash

    pip install --group dev
    pip install --editable .

Tests require the ``test`` dependency group:

.. code-block:: bash

    pip install --group test
    python -m pytest

License
=======

Released under the Apache License 2.0. See LICENSE file.

Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.

Release Notes
-------------

We use `reno <https://docs.openstack.org/reno/>`_ to manage release notes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
