#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cnda/contiguous_nd.hpp>
#include <cnda/aos_types.hpp>
#include "utils.hpp"
#include <vector>
#include <map>
#include <string>
#include <cstddef>

namespace py = pybind11;

namespace cnda {
namespace aos_bindings {

// ============================================================================
// AoS Type Registration System
// ============================================================================

/**
 * @brief Specification for a single field in an AoS struct
 */
struct FieldSpec {
    std::string name;
    py::dtype dtype;
    
    FieldSpec(const std::string& n, const py::dtype& d) : name(n), dtype(d) {}
};

/**
 * @brief Registry for AoS type specifications
 * 
 * This allows users to register custom AoS types with their expected fields.
 * Future enhancement: expose this to Python for user-defined types.
 */
class AoSTypeRegistry {
public:
    /**
     * @brief Register an AoS type with its expected field specification
     * 
     * @param type_name Human-readable name (e.g., "Vec2f")
     * @param fields Expected fields in order
     */
    static void register_type(const std::string& type_name, 
                             const std::vector<FieldSpec>& fields) {
        get_registry()[type_name] = fields;
    }
    
    /**
     * @brief Get field specification for a registered type
     * 
     * @param type_name Type name to lookup
     * @return Field specification, or empty vector if not registered
     */
    static std::vector<FieldSpec> get_fields(const std::string& type_name) {
        auto& registry = get_registry();
        auto it = registry.find(type_name);
        if (it != registry.end()) {
            return it->second;
        }
        return {};
    }
    
private:
    static std::map<std::string, std::vector<FieldSpec>>& get_registry() {
        static std::map<std::string, std::vector<FieldSpec>> registry;
        return registry;
    }
};

// ============================================================================
// Generic Dtype Validation
// ============================================================================

/**
 * @brief Generic dtype validator that works for any registered AoS type
 * 
 * This replaces all the type-specific validation functions.
 * 
 * @param dtype NumPy dtype to validate
 * @param type_name Type name for error messages
 * @param expected_fields Expected field specification
 * 
 * @throws TypeError if validation fails
 */
inline void validate_aos_dtype(const py::dtype& dtype, 
                               const std::string& type_name,
                               const std::vector<FieldSpec>& expected_fields) {
    // Check if structured
    if (dtype.kind() != 'V') {  // 'V' = void/structured
        std::string field_list = "[";
        for (size_t i = 0; i < expected_fields.size(); ++i) {
            if (i > 0) field_list += ", ";
            field_list += "'" + expected_fields[i].name + "'";
        }
        field_list += "]";
        
        std::string msg = type_name + " requires structured dtype with fields " + field_list +
                         ", got plain array with dtype kind '" + std::string(1, dtype.kind()) + "'";
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        throw py::error_already_set();
    }
    
    // Check field count
    py::tuple names = dtype.attr("names");
    if (names.size() != expected_fields.size()) {
        std::string field_list = "[";
        for (size_t i = 0; i < expected_fields.size(); ++i) {
            if (i > 0) field_list += ", ";
            field_list += "'" + expected_fields[i].name + "'";
        }
        field_list += "]";
        
        std::string msg = type_name + " requires exactly " + 
                         std::to_string(expected_fields.size()) + " fields " + field_list +
                         ", got " + std::to_string(names.size()) + " fields";
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        throw py::error_already_set();
    }
    
    // Check field names and types
    auto fields = dtype.attr("fields");
    for (size_t i = 0; i < expected_fields.size(); ++i) {
        std::string actual_name = names[i].cast<std::string>();
        const std::string& expected_name = expected_fields[i].name;
        
        // Check field name
        if (actual_name != expected_name) {
            std::string msg = type_name + " field " + std::to_string(i) + 
                            " must be '" + expected_name + "', got '" + actual_name + "'";
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            throw py::error_already_set();
        }
        
        // Check field type
        auto field_info = fields[py::str(actual_name.c_str())];
        auto field_dtype = field_info[py::int_(0)].cast<py::dtype>();
        
        if (!field_dtype.is(expected_fields[i].dtype)) {
            std::string expected_type_str = py::str(expected_fields[i].dtype).cast<std::string>();
            std::string actual_type_str = py::str(field_dtype).cast<std::string>();
            
            std::string msg = type_name + "." + actual_name + " requires " + expected_type_str +
                            ", got " + actual_type_str;
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            throw py::error_already_set();
        }
    }
}

/**
 * @brief Validate NumPy structured dtype matches expected AoS type
 * 
 * Template function that uses the registry system.
 * Throws TypeError with descriptive message if validation fails.
 */
template<typename T>
void validate_structured_dtype(const py::dtype& dtype, const std::string& type_name);

// ============================================================================
// Type-Specific Specializations (Use Registry)
// ============================================================================

/**
 * @brief Helper macro to reduce boilerplate for validate_structured_dtype specializations
 * 
 * This macro generates a template specialization that:
 * 1. Retrieves field specification from the registry
 * 2. Checks if the type is registered (throws RuntimeError if not)
 * 3. Delegates to validate_aos_dtype for actual validation
 * 
 * @param CppType The C++ struct type (e.g., cnda::aos::Vec2f)
 * @param TypeName The registry name string (e.g., "Vec2f")
 */
#define CNDA_VALIDATE_AOS_TYPE(CppType, TypeName) \
    template<> \
    inline void validate_structured_dtype<CppType>( \
        const py::dtype& dtype, const std::string& type_name) { \
        auto fields = AoSTypeRegistry::get_fields(TypeName); \
        if (fields.empty()) { \
            PyErr_SetString(PyExc_RuntimeError, \
                           TypeName " not registered in AoS type registry"); \
            throw py::error_already_set(); \
        } \
        validate_aos_dtype(dtype, type_name, fields); \
    }

// Generate specializations for all built-in AoS types
CNDA_VALIDATE_AOS_TYPE(cnda::aos::Vec2f,        "Vec2f")
CNDA_VALIDATE_AOS_TYPE(cnda::aos::Vec3f,        "Vec3f")
CNDA_VALIDATE_AOS_TYPE(cnda::aos::Cell2D,       "Cell2D")
CNDA_VALIDATE_AOS_TYPE(cnda::aos::Cell3D,       "Cell3D")
CNDA_VALIDATE_AOS_TYPE(cnda::aos::Particle,     "Particle")
CNDA_VALIDATE_AOS_TYPE(cnda::aos::MaterialPoint, "MaterialPoint")

#undef CNDA_VALIDATE_AOS_TYPE

// ============================================================================
// Registration Function
// ============================================================================

/**
 * @brief Register all built-in AoS types
 * 
 * Call this during module initialization to set up the registry.
 */
inline void register_builtin_aos_types() {
    // Vec2f: [('x', float32), ('y', float32)]
    AoSTypeRegistry::register_type("Vec2f", {
        FieldSpec("x", py::dtype::of<float>()),
        FieldSpec("y", py::dtype::of<float>())
    });
    
    // Vec3f: [('x', float32), ('y', float32), ('z', float32)]
    AoSTypeRegistry::register_type("Vec3f", {
        FieldSpec("x", py::dtype::of<float>()),
        FieldSpec("y", py::dtype::of<float>()),
        FieldSpec("z", py::dtype::of<float>())
    });
    
    // Cell2D: [('u', float32), ('v', float32), ('flag', int32)]
    AoSTypeRegistry::register_type("Cell2D", {
        FieldSpec("u", py::dtype::of<float>()),
        FieldSpec("v", py::dtype::of<float>()),
        FieldSpec("flag", py::dtype::of<std::int32_t>())
    });
    
    // Cell3D: [('u', float32), ('v', float32), ('w', float32), ('flag', int32)]
    AoSTypeRegistry::register_type("Cell3D", {
        FieldSpec("u", py::dtype::of<float>()),
        FieldSpec("v", py::dtype::of<float>()),
        FieldSpec("w", py::dtype::of<float>()),
        FieldSpec("flag", py::dtype::of<std::int32_t>())
    });
    
    // Particle: [('x', float64), ('y', float64), ('z', float64), 
    //            ('vx', float64), ('vy', float64), ('vz', float64), ('mass', float64)]
    AoSTypeRegistry::register_type("Particle", {
        FieldSpec("x", py::dtype::of<double>()),
        FieldSpec("y", py::dtype::of<double>()),
        FieldSpec("z", py::dtype::of<double>()),
        FieldSpec("vx", py::dtype::of<double>()),
        FieldSpec("vy", py::dtype::of<double>()),
        FieldSpec("vz", py::dtype::of<double>()),
        FieldSpec("mass", py::dtype::of<double>())
    });
    
    // MaterialPoint: [('density', float32), ('temperature', float32), 
    //                 ('pressure', float32), ('id', int32)]
    AoSTypeRegistry::register_type("MaterialPoint", {
        FieldSpec("density", py::dtype::of<float>()),
        FieldSpec("temperature", py::dtype::of<float>()),
        FieldSpec("pressure", py::dtype::of<float>()),
        FieldSpec("id", py::dtype::of<std::int32_t>())
    });
}

// ============================================================================
// Field Access Bindings
// ============================================================================

// Helper to bind struct field access for AoS types (default: do nothing)
template<typename T>
inline void bind_aos_fields(py::class_<cnda::ContiguousND<T>>& cls) {
    // No-op for non-struct types
}

// Specialization for Vec2f
template<>
inline void bind_aos_fields<cnda::aos::Vec2f>(py::class_<cnda::ContiguousND<cnda::aos::Vec2f>>& cls) {
    cls.def("get_x", [](const cnda::ContiguousND<cnda::aos::Vec2f>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].x;
    }, "Get x component at indices");
    cls.def("get_y", [](const cnda::ContiguousND<cnda::aos::Vec2f>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].y;
    }, "Get y component at indices");
    cls.def("set_x", [](cnda::ContiguousND<cnda::aos::Vec2f>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].x = val;
    }, "Set x component at indices");
    cls.def("set_y", [](cnda::ContiguousND<cnda::aos::Vec2f>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].y = val;
    }, "Set y component at indices");
}

// Specialization for Vec3f
template<>
inline void bind_aos_fields<cnda::aos::Vec3f>(py::class_<cnda::ContiguousND<cnda::aos::Vec3f>>& cls) {
    cls.def("get_x", [](const cnda::ContiguousND<cnda::aos::Vec3f>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].x;
    }, "Get x component at indices");
    cls.def("get_y", [](const cnda::ContiguousND<cnda::aos::Vec3f>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].y;
    }, "Get y component at indices");
    cls.def("get_z", [](const cnda::ContiguousND<cnda::aos::Vec3f>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].z;
    }, "Get z component at indices");
    cls.def("set_x", [](cnda::ContiguousND<cnda::aos::Vec3f>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].x = val;
    }, "Set x component at indices");
    cls.def("set_y", [](cnda::ContiguousND<cnda::aos::Vec3f>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].y = val;
    }, "Set y component at indices");
    cls.def("set_z", [](cnda::ContiguousND<cnda::aos::Vec3f>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].z = val;
    }, "Set z component at indices");
}

// Specialization for Cell2D
template<>
inline void bind_aos_fields<cnda::aos::Cell2D>(py::class_<cnda::ContiguousND<cnda::aos::Cell2D>>& cls) {
    cls.def("get_u", [](const cnda::ContiguousND<cnda::aos::Cell2D>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].u;
    }, "Get u velocity at indices");
    cls.def("get_v", [](const cnda::ContiguousND<cnda::aos::Cell2D>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].v;
    }, "Get v velocity at indices");
    cls.def("get_flag", [](const cnda::ContiguousND<cnda::aos::Cell2D>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].flag;
    }, "Get flag at indices");
    cls.def("set_u", [](cnda::ContiguousND<cnda::aos::Cell2D>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].u = val;
    }, "Set u velocity at indices");
    cls.def("set_v", [](cnda::ContiguousND<cnda::aos::Cell2D>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].v = val;
    }, "Set v velocity at indices");
    cls.def("set_flag", [](cnda::ContiguousND<cnda::aos::Cell2D>& self, py::tuple indices, std::int32_t val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].flag = val;
    }, "Set flag at indices");
}

// Specialization for Cell3D
template<>
inline void bind_aos_fields<cnda::aos::Cell3D>(py::class_<cnda::ContiguousND<cnda::aos::Cell3D>>& cls) {
    cls.def("get_u", [](const cnda::ContiguousND<cnda::aos::Cell3D>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].u;
    }, "Get u velocity at indices");
    cls.def("get_v", [](const cnda::ContiguousND<cnda::aos::Cell3D>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].v;
    }, "Get v velocity at indices");
    cls.def("get_w", [](const cnda::ContiguousND<cnda::aos::Cell3D>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].w;
    }, "Get w velocity at indices");
    cls.def("get_flag", [](const cnda::ContiguousND<cnda::aos::Cell3D>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].flag;
    }, "Get flag at indices");
    cls.def("set_u", [](cnda::ContiguousND<cnda::aos::Cell3D>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].u = val;
    }, "Set u velocity at indices");
    cls.def("set_v", [](cnda::ContiguousND<cnda::aos::Cell3D>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].v = val;
    }, "Set v velocity at indices");
    cls.def("set_w", [](cnda::ContiguousND<cnda::aos::Cell3D>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].w = val;
    }, "Set w velocity at indices");
    cls.def("set_flag", [](cnda::ContiguousND<cnda::aos::Cell3D>& self, py::tuple indices, std::int32_t val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].flag = val;
    }, "Set flag at indices");
}

// Specialization for Particle
template<>
inline void bind_aos_fields<cnda::aos::Particle>(py::class_<cnda::ContiguousND<cnda::aos::Particle>>& cls) {
    cls.def("get_x", [](const cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].x;
    }, "Get x position at indices");
    cls.def("get_y", [](const cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].y;
    }, "Get y position at indices");
    cls.def("get_z", [](const cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].z;
    }, "Get z position at indices");
    cls.def("get_mass", [](const cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].mass;
    }, "Get mass at indices");
    cls.def("set_x", [](cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices, double val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].x = val;
    }, "Set x position at indices");
    cls.def("set_y", [](cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices, double val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].y = val;
    }, "Set y position at indices");
    cls.def("set_z", [](cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices, double val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].z = val;
    }, "Set z position at indices");
    cls.def("set_mass", [](cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices, double val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].mass = val;
    }, "Set mass at indices");
    // Velocity field accessors
    cls.def("get_vx", [](const cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].vx;
    }, "Get vx velocity at indices");
    cls.def("get_vy", [](const cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].vy;
    }, "Get vy velocity at indices");
    cls.def("get_vz", [](const cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].vz;
    }, "Get vz velocity at indices");
    cls.def("set_vx", [](cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices, double val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].vx = val;
    }, "Set vx velocity at indices");
    cls.def("set_vy", [](cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices, double val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].vy = val;
    }, "Set vy velocity at indices");
    cls.def("set_vz", [](cnda::ContiguousND<cnda::aos::Particle>& self, py::tuple indices, double val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].vz = val;
    }, "Set vz velocity at indices");
}

// Specialization for MaterialPoint
template<>
inline void bind_aos_fields<cnda::aos::MaterialPoint>(py::class_<cnda::ContiguousND<cnda::aos::MaterialPoint>>& cls) {
    cls.def("get_density", [](const cnda::ContiguousND<cnda::aos::MaterialPoint>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].density;
    }, "Get density at indices");
    cls.def("get_temperature", [](const cnda::ContiguousND<cnda::aos::MaterialPoint>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].temperature;
    }, "Get temperature at indices");
    cls.def("get_pressure", [](const cnda::ContiguousND<cnda::aos::MaterialPoint>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].pressure;
    }, "Get pressure at indices");
    cls.def("get_id", [](const cnda::ContiguousND<cnda::aos::MaterialPoint>& self, py::tuple indices) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        return self.data()[offset].id;
    }, "Get id at indices");
    cls.def("set_density", [](cnda::ContiguousND<cnda::aos::MaterialPoint>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].density = val;
    }, "Set density at indices");
    cls.def("set_temperature", [](cnda::ContiguousND<cnda::aos::MaterialPoint>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].temperature = val;
    }, "Set temperature at indices");
    cls.def("set_pressure", [](cnda::ContiguousND<cnda::aos::MaterialPoint>& self, py::tuple indices, float val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].pressure = val;
    }, "Set pressure at indices");
    cls.def("set_id", [](cnda::ContiguousND<cnda::aos::MaterialPoint>& self, py::tuple indices, std::int32_t val) {
        auto idx_vec = tuple_to_indices(indices);
        std::size_t offset = compute_offset(self, idx_vec);
        self.data()[offset].id = val;
    }, "Set id at indices");
}

// Function to register NumPy dtypes for all AoS types
inline void register_numpy_dtypes() {
    PYBIND11_NUMPY_DTYPE(cnda::aos::Vec2f, x, y);
    PYBIND11_NUMPY_DTYPE(cnda::aos::Vec3f, x, y, z);
    PYBIND11_NUMPY_DTYPE(cnda::aos::Cell2D, u, v, flag);
    PYBIND11_NUMPY_DTYPE(cnda::aos::Cell3D, u, v, w, flag);
    PYBIND11_NUMPY_DTYPE(cnda::aos::Particle, x, y, z, vx, vy, vz, mass);
    PYBIND11_NUMPY_DTYPE(cnda::aos::MaterialPoint, density, temperature, pressure, id);
}

/**
 * @brief Register all AoS type bindings to Python module
 * 
 * This function binds all AoS types and their validated from_numpy functions.
 * Implementation is in aos_types.cpp
 * 
 * @param m The pybind11 module to bind to
 */
void register_aos_bindings(pybind11::module_ &m);

} // namespace aos_bindings
} // namespace cnda
