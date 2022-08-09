import copy
import functools
from typing import Hashable, Sequence

from dwgms import Circuit
from dwgms.registers import QuantumRegister
from dwgms.utils import TemplateError, generate_id


allowed_template_methods = [
    "circuit",
]


class template:
    """Decorator class to create an operation template.

    Decorate a custom class with the ``template`` decorator to turn it into a
    usuable operation. The class must contain:

    - A ``circuit`` method, only with ``self`` in the signature, which applies
      all the operations contained in the template. Any qubits and parameters
      can be accessed via ``self.qubits`` and ``self.params`` respectively.
    - A ``num_qubits`` class attribute with the number of required qubits.
    - A ``num_params`` class attribute with the number of required parameters.

    Args:
        cls: Decorated class, i.e., the custom operator class.

    Example:

        Creating a custom Rotation gate can be done using ``RZ`` and ``RY``
        operations along with the template decorator, which can later be used
        the same way as any other built-in operation.

        .. code-block:: python

            @template
            class Rotation:
                num_qubits = 2
                num_params = 3

                def circuit(self):
                    RZ(self.params[0], self.qubits[0])
                    RY(self.params[1], self.qubits[1])
                    RZ(self.params[2], self.qubits[0])
    """

    def __init__(self, cls):
        for m in dir(cls):
            if not callable(getattr(cls, m)) or m.startswith("__"):
                continue
            if m not in allowed_template_methods:
                raise TemplateError(f"Template contains invalid method '{m}'.")

        functools.update_wrapper(self, cls)
        self._cls = cls

        self._cls.num_qubits = getattr(cls, "num_qubits", 1)
        self._cls.num_params = getattr(cls, "num_params", 0)

        label = generate_id(prefix=f"{self._cls.__name__}Register")
        self._qubits = QuantumRegister(label=label)

        self._params = None
        self._matrix = None
        self._instancelike = False

        self._cls.params = property(lambda _: self.params)
        self._cls.qubits = property(lambda _: self.qubits)

    def __call__(self, params=None, qubits=None) -> "template":
        """Calls the class and applied the custom circuit.

        Args:
            params: Optional parameter(s) required by the circuit.
            qubits: Qubits on which the operation should be applied. Only
                required when applying an operation within a circuit context.

        Returns:
            template: Copy of the decorated class, with qubits and/or parameters
            stored as attributes.
        """
        template_copy = self.copy()
        template_copy._params = params
        if qubits is not None:
            qubits = self._check_qubits(qubits)
            template_copy._qubits.add(qubits)
        template_copy._instancelike = True

        template_copy._cls().circuit()
        return template_copy

    def _check_qubits(self, qubits) -> Sequence[Hashable]:
        """Asserts size and type of the qubit(s) and returns the correct type.

        Args:
            qubits: Qubits to check.

        Returns:
            tuple: Sequence of qubits as a tuple.
        """
        # TODO: update to check for single qubit instead of str
        if isinstance(qubits, str) or not isinstance(qubits, Sequence):
            qubits = [qubits]

        if len(qubits) != self._cls.num_qubits:
            raise ValueError(
                f"Operation '{self._cls.__name__} requires "
                f"{self._cls.num_qubits} qubits, got {len(qubits)}."
            )

        # cast to tuple for convention
        return tuple(qubits)

    @property
    def params(self):
        """Parameters of the template."""
        return self._params

    @property
    def qubits(self):
        """Qubits that the template is applied to."""
        return self._qubits

    @property
    def label(self):
        """Template operation label."""
        if self.params is not None:
            params = f"({', '.join(str(p) for p in self.params)})"
            return self._cls.__name__ + params
        return self._cls.__name__

    @property
    def matrix(self):
        """The matrix representation of the template operator.

        Note that this property call constructs, and caches, the matrix lazily
        by building the unitary based on the operations in the ``circuit``
        methods.
        """
        if self._matrix is None:
            tmp_circuit = Circuit(self._cls.num_qubits)
            with tmp_circuit.context as q:
                self(self.params, q)

            self._matrix = tmp_circuit.build_unitary()
        return self._matrix

    def copy(self) -> "template":
        """Copies the templated operation.

        Returns:
            template: Copy of the templated operation.
        """
        new_copy = self.__class__(self._cls)
        new_copy._cls._params = copy.deepcopy(self.params)
        new_copy._cls._qubits = copy.deepcopy(self.qubits)
        new_copy._matrix = copy.deepcopy(self._matrix)
        return new_copy

    def __repr__(self) -> str:
        "Representation of the template operation."
        if self._instancelike:
            return f"<Operation: {self.label}, qubits={self.qubits.data}>"
        return f"{self._cls.__module__}.{self._cls.__name__}"
