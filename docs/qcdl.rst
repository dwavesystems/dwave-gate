.. _gate_qcdl:

====
QCDL
====

The :ref:`gate_workflow` section provides an introduction to using QCDL to
program quantum circuits.

Classes
=======

.. currentmodule:: dwave.gate.qcdl

.. autoclass:: Scope
    :members:
    :show-inheritance:
    :inherited-members:

Utilities
=========

.. currentmodule:: dwave.gate.qcdl

.. autofunction:: print_qcdl

.. autofunction:: display_qcdl

.. autoclass:: LogicalOutcomeToInteger
    :members:
    :show-inheritance:

Mirroring Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: dwave.gate.qcdl.implementations
   :members:

Decorators
==========

.. currentmodule:: dwave.gate.qcdl

.. autofunction:: dwave.gate.qcdl.registers.arbitrary_function

.. autofunction:: procedure

.. autofunction:: qcdl


QCDL Development
================

These classes are of interest mostly to developers of QCDL.

.. automodule:: dwave.gate.qcdl
    :show-inheritance:
    :members: QCDLProgram, QCDLModule, QCDLModuleContainer, QCDLProcedureDef,
        QCDLStatement

.. automodule:: dwave.gate.qcdl.qcdl_circuit
    :show-inheritance:
    :members: QCDLCircuit

.. automodule:: dwave.gate.qcdl.components
    :show-inheritance:
    :members: Procedure