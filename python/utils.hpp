#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cnda/contiguous_nd.hpp>
#include <vector>
#include <cstring>

namespace py = pybind11;

// Helper function to implement from_numpy logic (used by both dtype-specific and generic functions)
template<typename T>
inline cnda::ContiguousND<T> from_numpy_impl(const py::array_t<T>& arr, bool copy) {
    
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
        PyErr_SetString(PyExc_ValueError,
            "from_numpy with copy=False requires C-contiguous (row-major) array. "
            "Use copy=True to force a copy, or ensure the input array is C-contiguous.");
        throw py::error_already_set();
    }

    // Additional stride validation for safety
    cnda::ContiguousND<T> tmp(shape);
    const auto& expected = tmp.strides();
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        auto stride_bytes = arr.strides(i);
        auto elem_size    = static_cast<py::ssize_t>(sizeof(T));

        if (stride_bytes % elem_size != 0) {
            PyErr_SetString(PyExc_ValueError,
                "from_numpy(copy=False) requires strides that are a multiple of sizeof(T). "
                "The input has incompatible strides; use copy=True.");
            throw py::error_already_set();
        }

        auto stride_elems = stride_bytes / elem_size;
        if (stride_elems != static_cast<py::ssize_t>(
                                expected[static_cast<std::size_t>(i)])) {
            PyErr_SetString(PyExc_ValueError,
                "from_numpy(copy=False) requires standard row-major strides. "
                "The input has non-standard strides; use copy=True.");
            throw py::error_already_set();
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

// Helper function to convert py::tuple to vector of indices
inline std::vector<std::size_t> tuple_to_indices(py::tuple indices) {
    std::vector<std::size_t> idx_vec;
    idx_vec.reserve(indices.size());
    for (auto idx : indices) {
        idx_vec.push_back(idx.cast<std::size_t>());
    }
    return idx_vec;
}

// Helper function to compute flat index from indices vector
template<typename T>
inline std::size_t compute_offset(const cnda::ContiguousND<T>& self, const std::vector<std::size_t>& idx_vec) {
    const auto& shape = self.shape();
    const auto& strides = self.strides();
    
    // Always perform bounds checking in Python bindings
    if (idx_vec.size() != self.ndim()) {
        throw std::out_of_range("Number of indices does not match ndim");
    }
    for (size_t i = 0; i < idx_vec.size(); ++i) {
        if (idx_vec[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    
    std::size_t offset = 0;
    for (size_t i = 0; i < idx_vec.size(); ++i) {
        offset += idx_vec[i] * strides[i];
    }
    return offset;
}

// Forward declaration - bind_contiguous_nd is defined in module.cpp
template<typename T>
void bind_contiguous_nd(py::module_& m, const std::string& dtype_suffix);
