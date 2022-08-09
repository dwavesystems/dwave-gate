import inspect
import uuid
from typing import Any, Callable, Optional


class classproperty:
    """Decorator class for creating a property for a class method.

    Note that, apart from being much simplified, the ``classproperty`` decorator
    differs from the regular ``property`` decorator by supporting access to both
    the class and the instance (the latter is only available when calling the
    method on an instance of the class).

    Args:
        func: Method decorated by the ``classproperty`` decorator.
    """

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __get__(self, instance: Optional[object], cls: type) -> Any:
        """Return the internal function call.

        Depending on whether the internal function signature contains 2
        parameters (class and instance object; normally `cls` and `self`) or
        only 1 parameter (only class; normally `cls`), different function calls
        are returned.

        Args:
            instance: Instance object if called on such.
            cls: Class or class of instance.

        Returns:
            Any: Output of the decorated method, if any.
        """
        num_parameters = len(inspect.signature(self._func).parameters)
        if num_parameters == 1:
            return self._func(cls)
        return self._func(cls, instance)


def generate_id(prefix: str = None, k: int = 6) -> str:
    """Generate a random ID label.

    A random string of integers with an optional prefix.

    Args:
        prefix: Prefix for the ID label. Can be any string except
            ``"qubit"``/``"bit"``, in which case the prefix defaults to
            ``'q'``/``'c'``.
        k: Number of integers in the ID, excluding the prefix.

    Returns:
        str: Random ID string label.
    """
    reg_id = str(uuid.uuid4().int)[:k]

    if prefix == "qubit":
        return "q" + reg_id
    elif prefix == "bit":
        return "c" + reg_id

    if prefix is None:
        return reg_id
    return prefix + reg_id


class IntegerCounter:
    """Integer counter to generate and keep track of labels.

    Args:
        start: Integer index at which to start the counting.
        prefix: Prefix for the ID label (defaults to none).
    """

    def __init__(self, start: int = 0, prefix: str = None) -> None:
        self._start = start
        self._counter = start - 1
        self._prefix = prefix or ""

    def reset(self, start: int = None) -> None:
        """Resets the counter to restart at the initialized starting index."""
        if start:
            self._start = start
        self._counter = self._start - 1

    @property
    def counter(self) -> int:
        """Current counter ID label."""
        return self._counter

    def increment(self, n=1) -> None:
        """Increment the counter without returning a label.

        Args:
            n: The number of steps to increment the counter with.
        """
        self._counter += n

    def next(self) -> str:
        """Increment the counter with 1, returning the ID label.

        Returns:
            str: The next ID label.
        """
        self._counter += 1
        return self._prefix + str(self._counter)


#####################
# Custom exceptions #
#####################


class CircuitError(Exception):
    """Exception to be raised when there is an error with a Circuit."""


class TemplateError(Exception):
    """Exception to be raised when there is an error with a Template."""
