#![cfg(feature = "macros")]

use pyo3::prelude::*;
use pyo3::py_run;
use pyo3::types::{PyAny, PyDict, PyString, PyTuple, PyType};

mod test_utils;

// A simple metaclass with no extra data fields or custom __new__.
// It relies on the inherited `type.__new__` for class creation.
#[pyclass(metaclass)]
struct SimpleMeta;

#[pymethods]
impl SimpleMeta {
    // Overrides isinstance(x, C) where C has metaclass SimpleMeta.
    // Always returns True for testing purposes.
    fn __instancecheck__(&self, _instance: &Bound<'_, PyAny>) -> bool {
        true
    }

    // Overrides issubclass(X, C) where C has metaclass SimpleMeta.
    // Always returns True for testing purposes.
    fn __subclasscheck__(&self, _subclass: &Bound<'_, PyAny>) -> bool {
        true
    }

    // Overrides C[item] where C has metaclass SimpleMeta.
    fn __getitem__(&self, item: Py<PyAny>) -> Py<PyAny> {
        item
    }
}

// A metaclass that overrides __call__ so that calling C() returns a tuple
// (cls, args, kwargs) for testing.
#[pyclass(metaclass)]
struct CallMeta;

#[pymethods]
impl CallMeta {
    #[pyo3(signature = (*args, **_kwargs))]
    fn __call__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        // Return (type, args) to show we intercept the call
        let result = PyTuple::new(py, [slf.as_any().clone(), args.as_any().clone()])?;
        Ok(result.into())
    }
}

// A metaclass with a custom __new__ that returns Py<Self>.
// Users who want custom metaclass __new__ must:
// 1. Use both #[new] and #[classmethod].
// 2. Take `cls: &Bound<'_, PyType>` as first argument.
// 3. Take the standard metaclass creation args (name, bases, namespace).
// 4. Call type_new directly via FFI (the Python-level type.__new__ performs a
//    safety check that rejects PyO3 metaclasses, so use the C function directly).
// 5. Return Py<Self> (not just Self).
#[pyclass(metaclass)]
struct CustomNewMeta;

#[pymethods]
impl CustomNewMeta {
    #[new]
    #[classmethod]
    fn new(
        cls: &Bound<'_, PyType>,
        name: &Bound<'_, PyString>,
        bases: &Bound<'_, PyTuple>,
        namespace: &Bound<'_, PyDict>,
    ) -> PyResult<Py<Self>> {
        let py = cls.py();
        // Build the 3-element args tuple that type_new expects.
        let args =
            PyTuple::new(py, [name.as_any(), bases.as_any(), namespace.as_any()])?;
        // Call type_new (CPython's C-level slot) directly.
        // The Python-level `type.__new__` rejects PyO3 metaclasses via a safety
        // check (`metatype->tp_new != type_new`).  Using the slot directly lets
        // us bypass that check while still creating a well-formed type object.
        let obj_ptr = unsafe {
            let tp_new = pyo3::ffi::PyType_Type
                .tp_new
                .expect("type_new must be set on PyType_Type");
            tp_new(cls.as_type_ptr(), args.as_ptr(), std::ptr::null_mut())
        };
        if obj_ptr.is_null() {
            return Err(pyo3::PyErr::fetch(py));
        }
        Ok(unsafe {
            Bound::<PyAny>::from_owned_ptr(py, obj_ptr)
                .cast_into_unchecked::<Self>()
                .unbind()
        })
    }
}

#[test]
fn test_simple_metaclass_type_hierarchy() {
    Python::attach(|py| {
        let meta = py.get_type::<SimpleMeta>();
        py_run!(
            py,
            meta,
            r#"
# SimpleMeta must be a subclass of type
assert issubclass(meta, type), f"Expected issubclass(meta, type) but got False"
# A class created with metaclass=meta must be an instance of meta
class C(metaclass=meta): pass
assert isinstance(C, meta), f"Expected isinstance(C, meta) but got False"
assert type(C) is meta, f"Expected type(C) is meta but got {type(C)}"
"#
        );
    });
}

#[test]
fn test_metaclass_instancecheck() {
    Python::attach(|py| {
        let meta = py.get_type::<SimpleMeta>();
        py_run!(
            py,
            meta,
            r#"
class C(metaclass=meta): pass
# __instancecheck__ on SimpleMeta always returns True
assert isinstance(42, C)
assert isinstance("hello", C)
assert isinstance(None, C)
"#
        );
    });
}

#[test]
fn test_metaclass_subclasscheck() {
    Python::attach(|py| {
        let meta = py.get_type::<SimpleMeta>();
        py_run!(
            py,
            meta,
            r#"
class C(metaclass=meta): pass
# __subclasscheck__ on SimpleMeta always returns True
assert issubclass(int, C)
assert issubclass(str, C)
assert issubclass(list, C)
"#
        );
    });
}

#[test]
fn test_metaclass_getitem() {
    Python::attach(|py| {
        let meta = py.get_type::<SimpleMeta>();
        py_run!(
            py,
            meta,
            r#"
class C(metaclass=meta): pass
# __getitem__ on SimpleMeta returns the item unchanged
assert C[int] is int
assert C[str] is str
# Tuple subscript creates a tuple
tup = C[int, str]
assert tup == (int, str)
"#
        );
    });
}

#[test]
fn test_metaclass_call() {
    Python::attach(|py| {
        let meta = py.get_type::<CallMeta>();
        py_run!(
            py,
            meta,
            r#"
class D(metaclass=meta): pass
# Calling D() invokes CallMeta.__call__(D, (), {})
result = D()
assert isinstance(result, tuple)
assert result[0] is D
"#
        );
    });
}

#[test]
fn test_metaclass_custom_new() {
    Python::attach(|py| {
        let meta = py.get_type::<CustomNewMeta>();
        py_run!(
            py,
            meta,
            r#"
class E(metaclass=meta): pass
assert isinstance(E, meta), f"Expected isinstance(E, meta) but got False"
assert issubclass(meta, type), f"Expected issubclass(meta, type) but got False"
"#
        );
    });
}

#[test]
fn test_metaclass_is_subclassable() {
    Python::attach(|py| {
        let meta = py.get_type::<SimpleMeta>();
        // SimpleMeta should be subclassable (i.e., can be used as a metaclass)
        py_run!(
            py,
            meta,
            r#"
# Metaclasses can be subclassed in Python
class SubMeta(meta): pass
class F(metaclass=SubMeta): pass
assert isinstance(F, SubMeta)
assert isinstance(F, meta)
"#
        );
    });
}
