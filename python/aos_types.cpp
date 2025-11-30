/**
 * @file aos_types.cpp
 * @brief Python bindings for Array-of-Structures (AoS) types
 * 
 * This file contains all AoS-specific Python bindings, including:
 * - Type registration for Vec2f, Vec3f, Cell2D, Cell3D, Particle, MaterialPoint
 * - Validated from_numpy_* functions with strict dtype checking
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cnda/contiguous_nd.hpp>
#include <cnda/aos_types.hpp>
#include "utils.hpp"
#include "aos_types.hpp"

namespace py = pybind11;

// Forward declarations from main bindings
template<typename T>
cnda::ContiguousND<T> from_numpy_impl(const py::array_t<T>& arr, bool copy);

template<typename T>
void bind_contiguous_nd(py::module_ &m, const std::string &dtype_suffix);

namespace cnda {
namespace aos_bindings {

/**
 * @brief Register all AoS type bindings to Python module
 * 
 * This function is called from PYBIND11_MODULE to set up all
 * AoS types and their validated from_numpy functions.
 * 
 * @param m The pybind11 module to bind to
 */
void register_aos_bindings(py::module_ &m) {
    // ========================================================================
    // Bind AoS Types
    // ========================================================================
    // These calls create ContiguousND_TypeName classes with standard interface
    // Field accessors are automatically bound via bind_aos_fields<T>()
    
    bind_contiguous_nd<cnda::aos::Vec2f>(m, "Vec2f");
    bind_contiguous_nd<cnda::aos::Vec3f>(m, "Vec3f");
    bind_contiguous_nd<cnda::aos::Cell2D>(m, "Cell2D");
    bind_contiguous_nd<cnda::aos::Cell3D>(m, "Cell3D");
    bind_contiguous_nd<cnda::aos::Particle>(m, "Particle");
    bind_contiguous_nd<cnda::aos::MaterialPoint>(m, "MaterialPoint");
    
    // ========================================================================
    // Override from_numpy for AoS types with strict dtype validation
    // ========================================================================
    // These functions validate structured dtype before conversion to prevent
    // silent data corruption. Generic from_numpy_* are already created by
    // bind_contiguous_nd, but we override them with validated versions.
    
    // Vec2f: [('x', float32), ('y', float32)]
    m.def("from_numpy_Vec2f",
        [](py::array arr, bool copy) -> cnda::ContiguousND<cnda::aos::Vec2f> {
            validate_structured_dtype<cnda::aos::Vec2f>(arr.dtype(), "Vec2f");
            auto typed_arr = py::array_t<cnda::aos::Vec2f>::ensure(arr);
            if (!typed_arr) throw py::type_error("Failed to convert to typed array");
            return from_numpy_impl<cnda::aos::Vec2f>(typed_arr, copy);
        },
        py::arg("arr"), py::arg("copy") = false,
        R"pbdoc(Create ContiguousND_Vec2f from NumPy structured array.
        
        Requires structured dtype: [('x', np.float32), ('y', np.float32)]
        
        Raises
        ------
        TypeError : If dtype does not match expected structured dtype.
        ValueError : If copy=False and array is not C-contiguous.
        )pbdoc");
    
    // Vec3f: [('x', float32), ('y', float32), ('z', float32)]
    m.def("from_numpy_Vec3f",
        [](py::array arr, bool copy) -> cnda::ContiguousND<cnda::aos::Vec3f> {
            validate_structured_dtype<cnda::aos::Vec3f>(arr.dtype(), "Vec3f");
            auto typed_arr = py::array_t<cnda::aos::Vec3f>::ensure(arr);
            if (!typed_arr) throw py::type_error("Failed to convert to typed array");
            return from_numpy_impl<cnda::aos::Vec3f>(typed_arr, copy);
        },
        py::arg("arr"), py::arg("copy") = false,
        R"pbdoc(Create ContiguousND_Vec3f from NumPy structured array.
        
        Requires structured dtype: [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        
        Raises
        ------
        TypeError : If dtype does not match expected structured dtype.
        ValueError : If copy=False and array is not C-contiguous.
        )pbdoc");
    
    // Cell2D: [('u', float32), ('v', float32), ('flag', int32)]
    m.def("from_numpy_Cell2D",
        [](py::array arr, bool copy) -> cnda::ContiguousND<cnda::aos::Cell2D> {
            validate_structured_dtype<cnda::aos::Cell2D>(arr.dtype(), "Cell2D");
            auto typed_arr = py::array_t<cnda::aos::Cell2D>::ensure(arr);
            if (!typed_arr) throw py::type_error("Failed to convert to typed array");
            return from_numpy_impl<cnda::aos::Cell2D>(typed_arr, copy);
        },
        py::arg("arr"), py::arg("copy") = false,
        R"pbdoc(Create ContiguousND_Cell2D from NumPy structured array.
        
        Requires structured dtype: [('u', np.float32), ('v', np.float32), ('flag', np.int32)]
        
        Raises
        ------
        TypeError : If dtype does not match expected structured dtype.
        ValueError : If copy=False and array is not C-contiguous.
        )pbdoc");
    
    // Cell3D: [('u', float32), ('v', float32), ('w', float32), ('flag', int32)]
    m.def("from_numpy_Cell3D",
        [](py::array arr, bool copy) -> cnda::ContiguousND<cnda::aos::Cell3D> {
            validate_structured_dtype<cnda::aos::Cell3D>(arr.dtype(), "Cell3D");
            auto typed_arr = py::array_t<cnda::aos::Cell3D>::ensure(arr);
            if (!typed_arr) throw py::type_error("Failed to convert to typed array");
            return from_numpy_impl<cnda::aos::Cell3D>(typed_arr, copy);
        },
        py::arg("arr"), py::arg("copy") = false,
        R"pbdoc(Create ContiguousND_Cell3D from NumPy structured array.
        
        Requires structured dtype: [('u', np.float32), ('v', np.float32), ('w', np.float32), ('flag', np.int32)]
        
        Raises
        ------
        TypeError : If dtype does not match expected structured dtype.
        ValueError : If copy=False and array is not C-contiguous.
        )pbdoc");
    
    // Particle: [('x', float64), ..., ('mass', float64)]
    m.def("from_numpy_Particle",
        [](py::array arr, bool copy) -> cnda::ContiguousND<cnda::aos::Particle> {
            validate_structured_dtype<cnda::aos::Particle>(arr.dtype(), "Particle");
            auto typed_arr = py::array_t<cnda::aos::Particle>::ensure(arr);
            if (!typed_arr) throw py::type_error("Failed to convert to typed array");
            return from_numpy_impl<cnda::aos::Particle>(typed_arr, copy);
        },
        py::arg("arr"), py::arg("copy") = false,
        R"pbdoc(Create ContiguousND_Particle from NumPy structured array.
        
        Requires structured dtype: [('x', np.float64), ('y', np.float64), ('z', np.float64),
                                     ('vx', np.float64), ('vy', np.float64), ('vz', np.float64),
                                     ('mass', np.float64)]
        
        Raises
        ------
        TypeError : If dtype does not match expected structured dtype.
        ValueError : If copy=False and array is not C-contiguous.
        )pbdoc");
    
    // MaterialPoint: [('density', float32), ('temperature', float32), ('pressure', float32), ('id', int32)]
    m.def("from_numpy_MaterialPoint",
        [](py::array arr, bool copy) -> cnda::ContiguousND<cnda::aos::MaterialPoint> {
            validate_structured_dtype<cnda::aos::MaterialPoint>(arr.dtype(), "MaterialPoint");
            auto typed_arr = py::array_t<cnda::aos::MaterialPoint>::ensure(arr);
            if (!typed_arr) throw py::type_error("Failed to convert to typed array");
            return from_numpy_impl<cnda::aos::MaterialPoint>(typed_arr, copy);
        },
        py::arg("arr"), py::arg("copy") = false,
        R"pbdoc(Create ContiguousND_MaterialPoint from NumPy structured array.
        
        Requires structured dtype: [('density', np.float32), ('temperature', np.float32),
                                     ('pressure', np.float32), ('id', np.int32)]
        
        Raises
        ------
        TypeError : If dtype does not match expected structured dtype.
        ValueError : If copy=False and array is not C-contiguous.
        )pbdoc");
}

} // namespace aos_bindings
} // namespace cnda
