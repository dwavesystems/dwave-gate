# distutils: language = c++
# cython: language_level=3

# Confidential & Proprietary Information: D-Wave Systems Inc.

# The following code is a derivative work of the code from the dwavesystems/dimod package,
# specifically dimod/dimod/cyvariables.pxd, which is licensed under the Apache License 2.0.

__all__ = ['cyRegister']

cdef class cyRegister:
    cdef object _index_to_label
    cdef object _label_to_index
    cdef Py_ssize_t _stop

    cdef object at(self, Py_ssize_t)
    cdef Py_ssize_t size(self)

    cpdef object _append(self, object v=*, bint permissive=*)
    cpdef void _clear(self)
    cpdef object _extend(self, object iterable, bint permissive=*)
    cpdef bint _is_range(self)
    cpdef object _pop(self)
    cpdef cyRegister copy(self)
    cdef Py_ssize_t _count_int(self, object) except -1
    cpdef Py_ssize_t count(self, object) except -1
    cpdef Py_ssize_t index(self, object, bint permissive=*) except -1
    cpdef _remove(self, object)