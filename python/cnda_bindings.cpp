#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cnda/contiguous_nd.hpp>
#include <cstdint>
#include <string>
#include <cstring>

namespace py = pybind11;

// Helper function to implement from_numpy logic (used by both dtype-specific and generic functions)
template<typename T>
cnda::ContiguousND<T> from_numpy_impl(const py::array_t<T>& arr, bool copy) {
    
    // Extract shape
    std::vector<std::size_t> shape;
    shape.reserve(static_cast<std::size_t>(arr.ndim()));
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(static_cast<std::size_t>(arr.shape(i)));
    }

    // deep copy
    if (copy) {
        py::array_t<T, py::array::c_style | py::array::forcecast> arr_c(arr);
        cnda::ContiguousND<T> result(shape);

        const T* src = arr_c.data();
        T* dst = result.data();

        std::memcpy(dst, src, result.size() * sizeof(T));
        return result;
    }

    // zero-copy
    // Check if array is C-contiguous
    if (!(arr.flags() & py::array::c_style)) {
        // ValueError for shape/layout mismatch
        throw py::value_error(
            "from_numpy with copy=False requires C-contiguous (row-major) array. "
            "Use copy=True to force a copy, or ensure the input array is C-contiguous."
        );
    }

    // Additional stride validation for safety
    cnda::ContiguousND<T> tmp(shape);
    const auto& expected = tmp.strides();
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        auto stride_elems = arr.strides(i) / static_cast<py::ssize_t>(sizeof(T));
        if (stride_elems != static_cast<py::ssize_t>(expected[static_cast<std::size_t>(i)])) {
            // ValueError for shape/layout mismatch
            throw py::value_error(
                "from_numpy(copy=False) requires standard row-major strides. "
                "The input has non-standard strides; use copy=True."
            );
        }
    }

    T* data_ptr = const_cast<T*>(arr.data());  // Need mutable access for zero-copy view

    struct NumpyOwner {
        py::array arr;
        explicit NumpyOwner(py::array a) : arr(std::move(a)) {}
    };

    // Use shared_ptr constructor with custom deleter for better exception safety
    std::shared_ptr<void> owner(
        new NumpyOwner(py::array(arr)),  // Explicitly copy py::array to extend lifetime
        [](void* p) {
            delete static_cast<NumpyOwner*>(p);
        }
    );

    return cnda::ContiguousND<T>(std::move(shape), data_ptr, std::move(owner));
}

// Helper function to bind a specific dtype variant
template<typename T>
void bind_contiguous_nd(py::module_ &m, const std::string &dtype_suffix) {
    using ContiguousND_T = cnda::ContiguousND<T>;
    
    std::string class_name = "ContiguousND_" + dtype_suffix;
    
    py::class_<ContiguousND_T>(m, class_name.c_str())
        .def(py::init<std::vector<std::size_t>>(),
             py::arg("shape"),
             "Construct a ContiguousND array with the given shape")
        
        // Metadata accessors
        .def_property_readonly("shape", 
            [](const ContiguousND_T &self) -> py::tuple {
                auto s = self.shape();
                py::tuple result(s.size());
                for (size_t i = 0; i < s.size(); ++i) {
                    result[i] = s[i];
                }
                return result;
            },
            "Return the shape as a tuple")
        
        .def_property_readonly("strides",
            [](const ContiguousND_T &self) -> py::tuple {
                auto s = self.strides();
                py::tuple result(s.size());
                for (size_t i = 0; i < s.size(); ++i) {
                    result[i] = s[i];
                }
                return result;
            },
            "Return the strides (in elements) as a tuple")
        
        .def_property_readonly("ndim",
            &ContiguousND_T::ndim,
            "Return the number of dimensions")
        
        .def_property_readonly("size",
            &ContiguousND_T::size,
            "Return the total number of elements")
        
        // Indexing support - we need to handle variable number of indices
        // For Python, we'll use __call__ with *args
        .def("__call__",
            [](ContiguousND_T &self, py::args indices) -> T& {
                std::vector<std::size_t> idx_vec;
                for (auto idx : indices) {
                    idx_vec.push_back(idx.cast<std::size_t>());
                }
                
                if (idx_vec.size() != self.ndim()) {
                    throw std::out_of_range("Number of indices does not match ndim");
                }
                
                // Compute offset using strides
                std::size_t offset = 0;
                const auto& shape = self.shape();
                const auto& strides = self.strides();
                
                for (size_t i = 0; i < idx_vec.size(); ++i) {
                    if (idx_vec[i] >= shape[i]) {
                        throw std::out_of_range("Index out of bounds");
                    }
                    offset += idx_vec[i] * strides[i];
                }
                
                return self.data()[offset];
            },
            py::return_value_policy::reference_internal,
            "Access element at given indices")
        
        // Also support __getitem__ with tuple for more Pythonic access
        .def("__getitem__",
            [](const ContiguousND_T &self, std::size_t index) -> T {
                // Single index access for 1D arrays
                if (self.ndim() != 1) {
                    throw std::out_of_range("Single index only valid for 1D arrays");
                }
                
                const auto& shape = self.shape();
                if (index >= shape[0]) {
                    throw std::out_of_range("Index out of bounds");
                }
                
                return self.data()[index];
            },
            "Get element at given index (1D arrays)")
        
        .def("__getitem__",
            [](const ContiguousND_T &self, py::tuple indices) -> T {
                std::vector<std::size_t> idx_vec;
                for (auto idx : indices) {
                    idx_vec.push_back(idx.cast<std::size_t>());
                }
                
                if (idx_vec.size() != self.ndim()) {
                    throw std::out_of_range("Number of indices does not match ndim");
                }
                
                // Compute offset using strides
                std::size_t offset = 0;
                const auto& shape = self.shape();
                const auto& strides = self.strides();
                
                for (size_t i = 0; i < idx_vec.size(); ++i) {
                    if (idx_vec[i] >= shape[i]) {
                        throw std::out_of_range("Index out of bounds");
                    }
                    offset += idx_vec[i] * strides[i];
                }
                
                return self.data()[offset];
            },
            "Get element at given indices")
        
        .def("__setitem__",
            [](ContiguousND_T &self, std::size_t index, T value) {
                // Single index access for 1D arrays
                if (self.ndim() != 1) {
                    throw std::out_of_range("Single index only valid for 1D arrays");
                }
                
                const auto& shape = self.shape();
                if (index >= shape[0]) {
                    throw std::out_of_range("Index out of bounds");
                }
                
                self.data()[index] = value;
            },
            "Set element at given index (1D arrays)")
        
        .def("__setitem__",
            [](ContiguousND_T &self, py::tuple indices, T value) {
                std::vector<std::size_t> idx_vec;
                for (auto idx : indices) {
                    idx_vec.push_back(idx.cast<std::size_t>());
                }
                
                if (idx_vec.size() != self.ndim()) {
                    throw std::out_of_range("Number of indices does not match ndim");
                }
                
                // Compute offset using strides
                std::size_t offset = 0;
                const auto& shape = self.shape();
                const auto& strides = self.strides();
                
                for (size_t i = 0; i < idx_vec.size(); ++i) {
                    if (idx_vec[i] >= shape[i]) {
                        throw std::out_of_range("Index out of bounds");
                    }
                    offset += idx_vec[i] * strides[i];
                }
                
                self.data()[offset] = value;
            },
            "Set element at given indices")

        .def("at",
            [](const ContiguousND_T &self, py::tuple indices) -> T {
                std::vector<std::size_t> idx_vec;
                for (auto idx : indices) {
                    idx_vec.push_back(idx.cast<std::size_t>());
                }
                // pybind11 can't automatically convert py::tuple to std::initializer_list,
                // so we convert to std::vector first and then call a helper.
                // To avoid creating a new C++ helper, we can just call the existing `at`
                // by creating an initializer list, but that's complicated.
                // The simplest is to just re-implement the logic here, which is what is done for other methods.
                if (idx_vec.size() != self.ndim()) {
                    throw std::out_of_range("at(): rank mismatch");
                }
                std::size_t offset = 0;
                const auto& shape = self.shape();
                const auto& strides = self.strides();
                for (size_t i = 0; i < idx_vec.size(); ++i) {
                    if (idx_vec[i] >= shape[i]) {
                        throw std::out_of_range("at(): index out of bounds");
                    }
                    offset += idx_vec[i] * strides[i];
                }
                return self.data()[offset];
            }, "Safe access with bounds checking")
        
        // String representation
        .def("__repr__",
            [dtype_suffix](const ContiguousND_T &self) {
                std::string repr = "<ContiguousND_" + dtype_suffix + " shape=(";
                auto s = self.shape();
                for (size_t i = 0; i < s.size(); ++i) {
                    if (i > 0) repr += ", ";
                    repr += std::to_string(s[i]);
                }
                repr += "), size=" + std::to_string(self.size()) + ">";
                return repr;
            })
        
        // Raw data pointer access (for testing zero-copy)
        .def("data_ptr",
            [](const ContiguousND_T &self) -> std::uintptr_t {
                return reinterpret_cast<std::uintptr_t>(self.data());
            },
            "Return the raw pointer address as an integer (for zero-copy verification)")
        
        // NumPy interop: to_numpy
        .def("to_numpy",
            [](py::object self_obj, bool copy) -> py::array_t<T> {
                // Get the C++ object from the Python object
                auto& self = self_obj.cast<ContiguousND_T&>();
                
                if (copy) {
                    // Deep copy: create new NumPy array and copy data
                    std::vector<py::ssize_t> shape_ssize;
                    for (auto s : self.shape()) {
                        shape_ssize.push_back(static_cast<py::ssize_t>(s));
                    }
                    
                    // Create NumPy array with C-contiguous layout
                    py::array_t<T> result(shape_ssize);
                    
                    // Copy data
                    T* dst = result.mutable_data();
                    const T* src = self.data();
                    std::memcpy(dst, src, self.size() * sizeof(T));
                    
                    return result;
                } else {
                    // Zero-copy: create view that keeps Python object alive
                    std::vector<py::ssize_t> shape_ssize;
                    std::vector<py::ssize_t> strides_bytes;
                    
                    for (auto s : self.shape()) {
                        shape_ssize.push_back(static_cast<py::ssize_t>(s));
                    }
                    
                    // Convert element strides to byte strides
                    for (auto s : self.strides()) {
                        strides_bytes.push_back(static_cast<py::ssize_t>(s * sizeof(T)));
                    }
                    
                    T* data_ptr = self.data();
                    
                    // Create a capsule that keeps the Python object alive
                    // Store a copy of the Python object handle
                    auto* py_obj_ptr = new py::object(self_obj);
                    py::capsule capsule(py_obj_ptr, [](void *p) {
                        // When the NumPy array is destroyed, delete the Python object handle
                        auto* obj = static_cast<py::object*>(p);
                        delete obj;
                    });
                    
                    return py::array_t<T>(
                        shape_ssize,
                        strides_bytes,
                        data_ptr,
                        capsule
                    );
                }
            },
            py::arg("copy") = false,
            R"pbdoc(
                Export to NumPy array.
                
                Parameters
                ----------
                copy : bool, optional (default=False)
                    If False, returns a zero-copy view with lifetime managed by capsule deleter.
                    If True, returns a deep copy with independent lifetime.
                
                Returns
                -------
                numpy.ndarray
                    NumPy array sharing memory (copy=False) or independent copy (copy=True).
                
                Notes
                -----
                When copy=False, the returned NumPy array shares memory with the ContiguousND
                object. The lifetime is managed by a capsule deleter that keeps the underlying
                C++ object alive as long as the NumPy array exists.
                
                Examples
                --------
                >>> import cnda
                >>> arr = cnda.ContiguousND_f32([3, 4])
                >>> arr[0, 0] = 1.0
                >>> np_arr = arr.to_numpy(copy=False)  # Zero-copy view
                >>> np_arr[0, 0] == 1.0
                True
            )pbdoc");
    
    // Add static from_numpy function
    m.def(("from_numpy_" + dtype_suffix).c_str(),
        [](py::array_t<T> arr, bool copy) -> ContiguousND_T {
            return from_numpy_impl<T>(arr, copy);
        },
        py::arg("arr"),
        py::arg("copy") = false,
        R"pbdoc(
            Create ContiguousND from NumPy array.
            
            Parameters
            ----------
            arr : numpy.ndarray
                Input NumPy array with matching dtype.
            copy : bool, optional (default=False)
                If False, requires C-contiguous layout and raises error on mismatch.
                If True, always performs a deep copy regardless of layout.
            
            Returns
            -------
            ContiguousND
                New ContiguousND object with data from the NumPy array.
            
            Raises
            ------
            TypeError
                If dtype does not match (for generic from_numpy).
            ValueError
                If copy=False and array is not C-contiguous (layout/shape mismatch).
            
            Notes
            -----
            Zero-copy requirements (copy=False):
            - Array must be C-contiguous (row-major)
            - Dtype must match exactly
            - Proper memory alignment
            
            When copy=True, a deep copy is always performed regardless of layout.
            
            Examples
            --------
            >>> import numpy as np
            >>> import cnda
            >>> x = np.array([[1, 2], [3, 4]], dtype=np.float32)
            >>> arr = cnda.from_numpy_f32(x, copy=False)
            >>> arr[0, 0]
            1.0
        )pbdoc"
    );
}

PYBIND11_MODULE(cnda, m) {
    m.doc() = "CNDA: Contiguous N-Dimensional Array library with zero-copy NumPy interoperability";
    
    // Register exception translators for proper Python error types
    // std::out_of_range -> IndexError (out-of-bounds access)
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const std::out_of_range &e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        }
    });
    
    // std::invalid_argument -> ValueError (for shape/layout mismatches not caught by py::value_error)
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const std::invalid_argument &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });
    
    // std::runtime_error -> RuntimeError (for lifetime/ownership issues)
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const std::runtime_error &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // Export version
#ifdef CNDA_VERSION
    m.attr("__version__") = CNDA_VERSION;
#else
    m.attr("__version__") = "0.1.0";
#endif
    
    // Bind all dtype variants with distinct names
    bind_contiguous_nd<float>(m, "f32");
    bind_contiguous_nd<double>(m, "f64");
    bind_contiguous_nd<std::int32_t>(m, "i32");
    bind_contiguous_nd<std::int64_t>(m, "i64");
    
    // Add convenience type aliases for easier documentation
    m.attr("ContiguousND_float") = m.attr("ContiguousND_f32");
    m.attr("ContiguousND_double") = m.attr("ContiguousND_f64");
    m.attr("ContiguousND_int32") = m.attr("ContiguousND_i32");
    m.attr("ContiguousND_int64") = m.attr("ContiguousND_i64");
    
    // Add generic from_numpy that auto-detects dtype
    m.def("from_numpy",
        [](py::array arr, bool copy) -> py::object {
            auto dtype = arr.dtype();
            
            // Call the appropriate dtype-specific from_numpy function
            if (dtype.is(py::dtype::of<float>())) {
                auto typed_arr = py::array_t<float>(arr);
                return py::cast(from_numpy_impl<float>(typed_arr, copy));
            } else if (dtype.is(py::dtype::of<double>())) {
                auto typed_arr = py::array_t<double>(arr);
                return py::cast(from_numpy_impl<double>(typed_arr, copy));
            } else if (dtype.is(py::dtype::of<std::int32_t>())) {
                auto typed_arr = py::array_t<std::int32_t>(arr);
                return py::cast(from_numpy_impl<std::int32_t>(typed_arr, copy));
            } else if (dtype.is(py::dtype::of<std::int64_t>())) {
                auto typed_arr = py::array_t<std::int64_t>(arr);
                return py::cast(from_numpy_impl<std::int64_t>(typed_arr, copy));
            } else {
                throw py::type_error(
                    "Unsupported dtype: " + py::str(dtype).cast<std::string>() +
                    ". Supported types: float32, float64, int32, int64"
                );
            }
        },
        py::arg("arr"),
        py::arg("copy") = false,
        R"pbdoc(
            Create ContiguousND from NumPy array with automatic dtype detection.
            
            This is a convenience function that automatically selects the appropriate
            ContiguousND type based on the NumPy array's dtype.
            
            Parameters
            ----------
            arr : numpy.ndarray
                Input NumPy array.
            copy : bool, optional (default=False)
                If False, requires C-contiguous layout and raises error on mismatch.
                If True, always performs a deep copy regardless of layout.
            
            Returns
            -------
            ContiguousND_*
                ContiguousND object of appropriate dtype (f32, f64, i32, or i64).
            
            Raises
            ------
            TypeError
                If dtype is not supported (must be float32, float64, int32, or int64).
            ValueError
                If copy=False and array is not C-contiguous.
            
            Examples
            --------
            >>> import numpy as np
            >>> import cnda
            >>> x = np.array([[1, 2], [3, 4]], dtype=np.float32)
            >>> arr = cnda.from_numpy(x, copy=False)  # Returns ContiguousND_f32
            >>> type(arr).__name__
            'ContiguousND_f32'
        )pbdoc"
    );
}
