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

.. automodule:: dwave.gate.qcdl
    :show-inheritance:
    :members: Qcdl, QcdlModule, QcdlModuleContainer, QcdlProcedureDef,
        QcdlStatement
