# Copyright 2022-2023 D-Wave Systems Inc.
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

""":class:`mixedproperty` decorator.

Contains a decorator for creating mixed properties, which differ from regular properties by
allowing access to both the class and the instance.
"""

from __future__ import annotations

__all__ = [
    "mixedproperty",
]

import functools
import inspect
from typing import Any, Callable, Optional


class mixedproperty:
    """Decorator class for creating a property for a class/instance method.

    Note that the ``mixedproperty`` decorator differs from the regular
    ``property`` decorator by supporting access to both the class (``cls``) and
    the instance (``self``). The latter is only available when calling the
    method on an instance of the class. The signature parameters are positional
    and assume that the first parameter is the class and the second parameter
    (optional) is the instance.

    This property can optionally also accept parameters (either positional or
    keyword) when initialized. Allowed parameters are listed under 'Args' below.

    Args:
        self_required: Whether ``self`` is required for the property to be able
            to return the requested property, i.e., the function will only work
            when called on an instance and not on the class.

    Example:

        .. code-block:: python

            class MixedpropertyExample:
                _a = "a_class"
                _b = "b_class"

                def __init__(self):
                    self._b = "b_instance"
                    self._c = "c_instance"

                # can be called on both instance and class and will
                # return the same result regardless
                @mixedproperty
                def a(cls):
                    return cls._a

                # can be called on both instance and class and will
                # return different results (e.g., different defaults)
                @mixedproperty
                def b(cls, self):
                    if self:
                        return self._b
                    return cls._b

                # can be called on both instance and class and will
                # return 'None' if called on class
                @mixedproperty(self_required=True)
                def c(cls, self):
                    return self._c
    """

    def __init__(self, *args, **kwargs) -> None:
        self._self_required: bool = kwargs.pop("self_required", False)
        if args and isinstance(args[0], Callable):
            self.__call__(args[0])

    def __call__(self, callable_: Callable) -> mixedproperty:
        """Initialization of the decorated function.

        Args:
            callable_: Method decorated by the ``mixedproperty`` decorator."""

        # don't patch '__qualname__' due to Sphinx calling the mixedproperty
        # at docsbuild, raising exceptions and thus not rendering all entries
        for attr in ("__module__", "__name__", "__doc__", "__annotations__"):
            self.attr = getattr(callable_, attr)
        getattr(self, "__dict__").update(getattr(callable_, "__dict__", {}))

        self.__wrapped__ = callable_
        self._callable = callable_
        return self

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
        num_parameters = len(inspect.signature(self._callable).parameters)

        # if called on class while requiring access to 'self', return 'None'
        if self._self_required and num_parameters >= 2 and instance is None:
            return None

        if num_parameters == 1:
            return self._callable(cls)

        return self._callable(cls, instance)

    def __set__(self, instance: Optional[object], value: Any) -> None:
        raise AttributeError("can't set attribute")
