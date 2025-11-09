#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cnda/contiguous_nd.hpp>
#include <cstdint>
#include <string>

namespace py = pybind11;

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
            });
}

PYBIND11_MODULE(cnda, m) {
    m.doc() = "CNDA: Contiguous N-Dimensional Array library with zero-copy NumPy interoperability";
    
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
}
