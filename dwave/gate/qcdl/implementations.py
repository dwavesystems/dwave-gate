# Copyright 2026 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Implementations of various simple algorithms in QCDL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .components import QcdlModule, Scope, procedure
from .constants import LogicalOutcomeToInteger

if TYPE_CHECKING:
    from .registers import Register


def _get_receivers(
    sender: QcdlModule,
    register: Register,
    receivers: list[QcdlModule] | None = None,
) -> list[QcdlModule]:
    if receivers is None:
        receivers = [q for q in register.qcdl_modules if repr(q) != repr(sender)]
    else:
        if sender in receivers:
            raise ValueError(f"{sender=} can't also be a receiver")
    return receivers


@procedure
def mirror_bool_register(
    sender: QcdlModule,
    register: Register,
    receivers: list[QcdlModule] | None = None,
) -> None:
    """Broadcast a bool register from one qubit to others.

    This may be useful for, for example, a register used with an mced.

    This function assumes that:

    * The register was already allocated on all the qubits.
    * That the register on the sender has only a 0 or a 1. This
      assumption is not verified at runtime.

    Args:
        sender (QcdlModule): The DR qubit which has already assigned
           the measurement outcome to the register.
        register (Register): The register to mirror.
        receivers (list[QcdlModule] | None, optional): The DR qubits which need to have
           their copy of the register updated. If not provided, it will
           chose the other modules in the Register.
    """
    receivers = _get_receivers(sender=sender, register=register, receivers=receivers)
    if not receivers:
        return
    receivers_scope = Scope(*receivers, use_scope_id=False)
    dst_reg = receivers_scope.Register(name=register.name, alias=True)

    sender.comment()
    sender.comment("broadcasting bool: register == 1")
    sender.one_to_all(receivers_scope, register == 1)
    with receivers_scope.If(None) as _Else:
        dst_reg <<= 1
    with _Else():
        dst_reg <<= 0


@procedure
def mirror_measurement_register(
    sender: QcdlModule,
    register: Register,
    receivers: list[QcdlModule] | None = None,
) -> None:
    """Broadcast a 2-bit DR measurement from one qubit to others.

    A Register is actually a collection of registers corresponding to the scope
    for which the Register was created. If a register is passed to a `measure`
    operation, only the copy of the register on the measured qubit is updated;
    an extra step involving digital communication is necessary to ensure that
    registers on all other qubits in the Register are also updated.

    The outcome from a qubit measurement can be immediately written to a
    register on that qubit. This function can mirror that outcome to the same
    register stored on each of a set of other qubits (often all of them).

    This function assumes that:

    * The register was already allocated on all the qubits.
    * The measurement outcome is already written to the register on the sender
      qubit. This assumption is not verified at runtime.
    * That it's a 2 bit DR measurement that we need to broadcast. This
      assumption is not verified at runtime.

    Args:
        sender (QcdlModule): The DR qubit which has already assigned
           the measurement outcome to the register.
        register (Register): The register to mirror.
        receivers (list[QcdlModule] | None, optional): The DR qubits which need to have
           their copy of the register updated. If not provided, it will
           chose the other modules in the Register.
    """
    receivers = _get_receivers(sender=sender, register=register, receivers=receivers)
    if not receivers:
        return
    receivers_scope = Scope(*receivers, use_scope_id=False)
    dst_reg = receivers_scope.Register(name=register.name, alias=True)
    dst_reg <<= LogicalOutcomeToInteger.ZERO.value

    for val in [LogicalOutcomeToInteger.ONE, LogicalOutcomeToInteger.SPLAT]:
        sender.comment()
        sender.comment(f"broadcasting bit: register == {val=}")
        sender.one_to_all(receivers_scope, register == val.value)
        with receivers_scope.If(None):
            dst_reg <<= val.value
