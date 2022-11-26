# Copyright 2022 D-Wave Systems Inc.
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

import itertools
from numbers import Number
from typing import Container, Hashable, Iterator, Mapping

from cpython.dict cimport PyDict_Contains, PyDict_Size
from cpython.long cimport PyLong_Check
from cpython.ref cimport PyObject


cdef extern from "Python.h":
    # not yet available as of cython 0.29.22
    PyObject* PyDict_GetItemWithError(object p, object key) except? NULL


cdef class cyRegister:
    def __init__(self, object iterable=None):
        self._index_to_label = dict()
        self._label_to_index = dict()
        self._stop = 0

        if iterable is not None:
            if isinstance(iterable, cyRegister):
                self.__init_cyregister__(iterable)
            elif isinstance(iterable, range) and iterable.start == 0 and iterable.step == 1:
                # Unlike range, ._stop is a private property so we don't allow
                # it to be negative
                self._stop = max(iterable.stop, 0)
            else:
                self._extend(iterable, permissive=True)

    def __init_cyregister__(self, cyRegister iterable):
        # everything is hashable, and this is not a deep copy
        self._index_to_label.update(iterable._index_to_label)
        self._label_to_index.update(iterable._label_to_index)
        self._stop = iterable._stop

    def __contains__(self, v):
        return bool(self.count(v))

    def __copy__(self):
        return self.copy()

    def __getitem__(self, idx):
        try:
            return self.at(idx)
        except TypeError:
            pass

        if not isinstance(idx, slice):
            raise TypeError(f"indices must be integers or slices, not {type(idx)}")

        cdef Py_ssize_t start = 0 if idx.start is None else idx.start
        cdef Py_ssize_t stop = self.size() if idx.stop is None else idx.stop
        cdef Py_ssize_t step = 1 if idx.step is None else idx.step

        cdef Py_ssize_t i
        cdef cyRegister new = type(self)()
        for i in range(start, stop, step):
            new._append(self.at(i), permissive=False)
        return new

    def __iter__(self):
        cdef Py_ssize_t i

        if self._is_range():
            yield from range(self._stop)
        else:
            for i in range(self._stop):
                yield self.at(i)

    def __len__(self):
        return self.size()

    cpdef object _append(self, object v=None, bint permissive=False):
        """Append a new register item.

        Args:
            v (hashable, optional): Add a new item. If ``None``, a new label will be generated. The
                generated label will be the index of the new variable if that index is available,
                otherwise it will be the lowest available non-negative integer.

            permissive (bool, optional, default=False): If ``False``, appending a variable that
                already exists will raise a :exc:`ValueError`. If ``True``, appending a variable
                that already exists will not change the container.

        Returns:
            hashable: The label of the appended variable.

        Raises:
            ValueError: If the variable is present and ``permissive`` is ``False``.
        """
        if v is None:
            v = self._stop

            if not self._is_range() and self.count(v):
                v = 0
                while self.count(v):
                    v += 1

        elif self.count(v):
            if permissive:
                return v
            else:
                raise ValueError('{!r} is already in register'.format(v))

        idx = self._stop

        if idx != v:
            self._label_to_index[v] = idx
            self._index_to_label[idx] = v

        self._stop += 1
        return v

    cpdef void _clear(self):
        """Remove all items in a register."""
        self._label_to_index.clear()
        self._index_to_label.clear()
        self._stop = 0

    cpdef bint _is_range(self):
        """Return whether the register items are currently labelled [0, n)."""
        return not PyDict_Size(self._label_to_index)

    cpdef object _extend(self, object iterable, bint permissive=False):
        """Add new items to the register.

        Args:
            iterable (iterable[hashable]): An iterable of hashable objects.

            permissive (bool, optional, default=False): If ``False``, appending an item that
                already exists will raise a :exc:`ValueError`. If ``True``, appending an item that
                already exists will not change the container.

        Raises:
            ValueError: If the item is present in the register and ``permissive`` is ``False``.
        """
        # todo: performance improvements for range etc. Unlike in the __init__
        # we cannot make assumptions about our current state, so we would need
        # to check all of that
        for v in iterable:
            self._append(v, permissive=permissive)

    cpdef object _pop(self):
        """Pop the last item in the register.

        Returns:
            hashable: The removed item.

        Raises:
            :exc:`IndexError`: If the container is empty.
        """
        if not self.size():
            raise IndexError("Cannot pop when register is empty")

        self._stop = idx = self._stop - 1

        label = self._index_to_label.pop(idx, idx)
        self._label_to_index.pop(label, None)
        return label

    def _relabel(self, mapping):
        """Relabel the register items in-place.

        Args:
            mapping (dict[hashable, hashable]): Mapping from current item labels to new, as a dict.
                If an incomplete mapping is specified, unmapped variables keep their current labels.
        """
        for submap in _iter_safe_relabels(mapping, self):
            for old, new in submap.items():
                if old == new:
                    continue

                idx = self._label_to_index.pop(old, old)

                if new != idx:
                    self._label_to_index[new] = idx
                    self._index_to_label[idx] = new  # overwrites old idx
                else:
                    self._index_to_label.pop(idx, None)

    def _relabel_as_integers(self):
        """Relabel the register items as integers in-place.

        Returns:
            dict[int, hashable]: A mapping that will restore the original labels.

        Examples:

            >>> reg = dwave.gate.registers.Register(['a', 'b', 'c', 'd'])
            >>> print(reg)
            Register(['a', 'b', 'c', 'd'])
            >>> mapping = reg._relabel_as_integers()
            >>> print(reg)
            Register([0, 1, 2, 3])
            >>> reg._relabel(mapping)  # restore the original labels
            >>> print(reg)
            Register(['a', 'b', 'c', 'd'])
        """
        mapping = self._index_to_label.copy()
        self._index_to_label.clear()
        self._label_to_index.clear()
        return mapping

    cpdef _remove(self, v):
        """Remove the given register item.

        Args:
            v: A register item label.
        """
        cdef Py_ssize_t vi = self.index(v)  # raises error if not present

        # we can do better in terms of performance, but this is easy so until
        # we have reason to believe it's a bottleneck, let's just keep
        # the easy approach
        mapping = dict()
        cdef Py_ssize_t i
        for i in range(vi, self.size() - 1):
            mapping[self.at(i)] = self.at(i+1)
        self._pop()
        self._relabel(mapping)

    cdef object at(self, Py_ssize_t idx):
        """Get item ``idx``.

        This method is useful for accessing from cython since __getitem__ goes
        through python.
        """
        if idx < 0:
            idx = self._stop + idx

        if not 0 <= idx < self._stop:
            raise IndexError('index out of range')

        cdef object v
        cdef object pyidx = idx
        cdef PyObject* obj
        if self._is_range():
            v = pyidx
        else:
            # faster than self._index_to_label.get
            obj = PyDict_GetItemWithError(self._index_to_label, pyidx)
            if obj == NULL:
                v = pyidx
            else:
                v = <object>obj  # correctly handles the ref count

        return v

    cpdef cyRegister copy(self):
        """Return a copy of the Register object."""
        cdef cyRegister new = self.__new__(type(self))
        new._index_to_label = dict(self._index_to_label)
        new._label_to_index = dict(self._label_to_index)
        new._stop = self._stop
        return new

    cdef Py_ssize_t _count_int(self, object v) except -1:
        # only works when v is an int
        cdef Py_ssize_t vi = v

        if self._is_range():
            return 0 <= vi < self._stop

        # need to make sure that we're not using the integer elsewhere
        return (0 <= vi < self._stop
                and not PyDict_Contains(self._index_to_label, v)
                or PyDict_Contains(self._label_to_index, v))

    cpdef Py_ssize_t count(self, object v) except -1:
        """Return the number of times ``v`` appears in the register items.

        Because the variables are always unique, this will always return 1 or 0.
        """
        if PyLong_Check(v):
            return self._count_int(v)

        # handle other numeric types
        if isinstance(v, Number):
            v_int = int(v)  # assume this is safe because it's a number
            if v_int == v:
                return self._count_int(v_int)  # it's an integer afterall!

        try:
            return v in self._label_to_index
        except TypeError:
            # unhashable
            return False

    cpdef Py_ssize_t index(self, object v, bint permissive=False) except -1:
        """Return the index of ``v``.

        Args:
            v (hashable): A register item.

            permissive (bool, optional, default=False): If ``True``, the variable will be inserted,
                guaranteeing an index can be returned.

        Returns:
            int: The index of the given register item.

        Raises:
            :exc:`ValueError`: If the register item is present and ``permissive`` is ``False``.

        """
        if permissive:
            self._append(v, permissive=True)
        if not self.count(v):
            raise ValueError('unknown register item {!r}'.format(v))

        if self._is_range():
            return v if PyLong_Check(v) else int(v)

        # faster than self._label_to_index.get
        cdef PyObject* obj = PyDict_GetItemWithError(self._label_to_index, v)
        if obj == NULL:
            pyobj = v
        else:
            pyobj = <object>obj  # correctly updates ref count

        return pyobj if PyLong_Check(pyobj) else int(pyobj)

    cdef Py_ssize_t size(self):
        """The number of register items.

        This method is useful for accessing from cython since __len__ goes
        through python.
        """
        return self._stop


def _iter_safe_relabels(mapping, existing):
    """Iterator over "safe" intermediate relabelings.

    Args:
        mapping (dict[hashable, hashable]): A map from old labels to new. New labels must be hashable and unique.
        existing (container[hashable]): A container of existing labels.

    Yields:
        iterator[dict[hashable, hashable]]: A "safe" relabelling.
    """
    # put the new labels into a set for fast lookup, also ensures that the
    # values are valid labels
    # We could use a set, but using a dict makes for nicer error messages later
    try:
        new_labels = {new: old for old, new in mapping.items()}
    except TypeError:
        raise ValueError("mapping values must be hashable objects")

    if len(new_labels) < len(mapping):
        for old, new in mapping.items():
            if new_labels[new] != old:
                raise ValueError("cannot map two items to the same label: "
                                f"{old!r} and {new_labels[new]!r} are both mapped to {new!r}")
        raise RuntimeError  # should never get here, but just in case...

    old_labels = mapping.keys()

    for v in new_labels:
        if v in existing and v not in old_labels:
            raise ValueError(f"an item cannot be relabelled {v!r} without also "
                            "relabeling the existing item of the same name")

    if any(v in new_labels for v in old_labels):
        yield from _resolve_label_conflict(mapping, existing, old_labels, new_labels)
    else:
        yield mapping

def _resolve_label_conflict(mapping, existing, old_labels=None, new_labels=None):
    """Resolve a self-labeling conflict by creating an intermediate labeling.

    Args:
        mapping (dict): A dict mapping the current variable labels to new ones.
        existing (set-like): The existing labels.
        old_labels (set, optional, default=None): The keys of mapping. Can be passed in for
            performance reasons. These are not checked.
        new_labels (set, optional, default=None): The values of mapping. Can be passed in
            for performance reasons. These are not checked.

    Returns:
        tuple[dict, dict]: A 2-tuple containing a map from the keys of mapping to an
        intermediate labeling and a map from the intermediate labeling to the values of
        mapping.
    """
    if old_labels is None:
        old_labels = set(mapping)
    if new_labels is None:
        new_labels = set(mapping.values())

    # counter will be used to generate the intermediate labels, as an easy optimization
    # we start the counter with a high number because often register items are labeled by
    # integers starting from 0
    counter = itertools.count(2 * len(mapping))

    old_to_intermediate = {}
    intermediate_to_new = {}

    for old, new in mapping.items():
        if old == new:
            # we can remove self-labels
            continue

        if old in new_labels or new in old_labels:

            # try to get a new unique label
            lbl = next(counter)
            while lbl in new_labels or lbl in old_labels or lbl in existing:
                lbl = next(counter)

            # add it to the mapping
            old_to_intermediate[old] = lbl
            intermediate_to_new[lbl] = new

        else:
            old_to_intermediate[old] = new
            # don't need to add it to intermediate_to_new because it is a self-label

    return old_to_intermediate, intermediate_to_new
