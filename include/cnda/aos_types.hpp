#pragma once
#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace cnda {
namespace aos {

// Helper trait: AoS-compatible type
// Requirement:
//   - standard-layout: well-defined memory layout
//   - trivially-copyable: safe for memcpy / contiguous storage
template <typename T>
struct is_aos_compatible
    : std::integral_constant<bool,
        std::is_standard_layout<T>::value &&
        std::is_trivially_copyable<T>::value> {};

// Example composite types for Array-of-Structures (AoS) support
// These are POD-like types suitable for contiguous storage
// and NumPy structured dtype interop.

/**
 * @brief Simple 2D vector with float components
 *
 * Compatible with NumPy dtype: [('x', '<f4'), ('y', '<f4')]
 */
struct Vec2f {
    float x;
    float y;
};

/**
 * @brief Simple 3D vector with float components
 *
 * Compatible with NumPy dtype: [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
 */
struct Vec3f {
    float x;
    float y;
    float z;
};

/**
 * @brief Simulation cell with velocity components and a flag
 *
 * Typical use case: fluid dynamics grid cell with u, v velocities and status flag
 * Compatible with NumPy dtype: [('u', '<f4'), ('v', '<f4'), ('flag', '<i4')]
 */
struct Cell2D {
    float       u;    // velocity in x-direction
    float       v;    // velocity in y-direction
    std::int32_t flag; // status flag (boundary, fluid, etc.)
};

/**
 * @brief 3D simulation cell with velocity components and a flag
 *
 * Compatible with NumPy dtype: [('u', '<f4'), ('v', '<f4'), ('w', '<f4'), ('flag', '<i4')]
 */
struct Cell3D {
    float       u;    // velocity in x-direction
    float       v;    // velocity in y-direction
    float       w;    // velocity in z-direction
    std::int32_t flag; // status flag
};

/**
 * @brief Particle with position, velocity, and mass
 *
 * Compatible with NumPy dtype:
 *   [('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
 *    ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
 *    ('mass', '<f8')]
 */
struct Particle {
    double x, y, z;        // position
    double vx, vy, vz;     // velocity
    double mass;           // mass
};

/**
 * @brief Material point with scalar properties
 *
 * Compatible with NumPy dtype:
 *   [('density', '<f4'), ('temperature', '<f4'),
 *    ('pressure', '<f4'), ('id', '<i4')]
 */
struct MaterialPoint {
    float       density;
    float       temperature;
    float       pressure;
    std::int32_t id;
};

// Compile-time checks to ensure AoS compatibility
static_assert(is_aos_compatible<Vec2f>::value,        "Vec2f must be AoS-compatible");
static_assert(is_aos_compatible<Vec3f>::value,        "Vec3f must be AoS-compatible");
static_assert(is_aos_compatible<Cell2D>::value,       "Cell2D must be AoS-compatible");
static_assert(is_aos_compatible<Cell3D>::value,       "Cell3D must be AoS-compatible");
static_assert(is_aos_compatible<Particle>::value,     "Particle must be AoS-compatible");
static_assert(is_aos_compatible<MaterialPoint>::value,"MaterialPoint must be AoS-compatible");

} // namespace aos
} // namespace cnda
