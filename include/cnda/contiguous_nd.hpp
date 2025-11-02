#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <initializer_list>

namespace cnda {

/**
 * Row-major contiguous N-D container (header-only, minimal v0.2).
 * - Strides are measured in ELEMENTS (not bytes).
 * - O(1) accessors: shape(), strides(), ndim(), size(), data().
 */
template <class T>
class ContiguousND {
public:
  // Construct with shape; allocate contiguous buffer of product(shape).
  explicit ContiguousND(std::vector<std::size_t> shape)
    : m_shape(std::move(shape))
  {
    // Compute ndim & size (product of dimensions).
    m_ndim = m_shape.size();
    m_size = 1;
    for (std::size_t d : m_shape) {
      // Allow zero-sized dimensions; size=0 then buffer stays empty.
      // If you prefer to forbid zeros, throw here.
      m_size *= d;
    }

    // Compute row-major strides in ELEMENTS.
    // Example: shape [3,4] => strides [4,1]
    m_strides.assign(m_ndim, 0);
    if (m_ndim > 0) {
      m_strides[m_ndim - 1] = 1;
      for (std::size_t k = m_ndim; k-- > 1; ) {
        m_strides[k - 1] = m_strides[k] * m_shape[k];
      }
    }

    // Allocate contiguous storage.
    m_buffer.resize(m_size);
  }

  // -------- O(1) accessors --------
  const std::vector<std::size_t>& shape()   const { return m_shape;   }
  const std::vector<std::size_t>& strides() const { return m_strides; }
  std::size_t                      ndim()    const { return m_ndim;    }
  std::size_t                      size()    const { return m_size;    }
  T*                               data()          { return m_buffer.data(); }
  const T*                         data()    const { return m_buffer.data(); }
  // -------- Index helpers --------
  // Compute flat row-major index from an initializer_list of indices.
  // If CNDA_BOUNDS_CHECK is defined, validates ndim and bounds.
  std::size_t index(std::initializer_list<std::size_t> idxs) const {
    std::size_t off = 0;
#ifdef CNDA_BOUNDS_CHECK
    if (idxs.size() != m_ndim) {
      throw std::out_of_range("index: rank mismatch");
    }
#endif
    std::size_t axis = 0;
    for (auto v : idxs) {
#ifdef CNDA_BOUNDS_CHECK
      if (axis >= m_ndim) {
        throw std::out_of_range("index: rank mismatch");
      }
      if (v >= m_shape[axis]) {
        throw std::out_of_range("index: out of bounds");
      }
#endif
      off += v * m_strides[axis];
      ++axis;
    }
    return off;
  }
  // -------- operator() overloads (1D/2D/3D) --------
  // 1D
  T& operator()(std::size_t i) {
#ifdef CNDA_BOUNDS_CHECK
    if (m_ndim != 1 || i >= m_shape[0]) {
      throw std::out_of_range("operator(): out of bounds (1D)");
    }
#endif
    return m_buffer[i];
  }
  const T& operator()(std::size_t i) const {
#ifdef CNDA_BOUNDS_CHECK
    if (m_ndim != 1 || i >= m_shape[0]) {
      throw std::out_of_range("operator() const: out of bounds (1D)");
    }
#endif
    return m_buffer[i];
  }
  // 2D
  T& operator()(std::size_t i, std::size_t j) {
#ifdef CNDA_BOUNDS_CHECK
    if (m_ndim != 2 || i >= m_shape[0] || j >= m_shape[1]) {
      throw std::out_of_range("operator(): out of bounds (2D)");
    }
#endif
    return m_buffer[i * m_strides[0] + j * m_strides[1]];
  }
  const T& operator()(std::size_t i, std::size_t j) const {
#ifdef CNDA_BOUNDS_CHECK
    if (m_ndim != 2 || i >= m_shape[0] || j >= m_shape[1]) {
      throw std::out_of_range("operator() const: out of bounds (2D)");
    }
#endif
    return m_buffer[i * m_strides[0] + j * m_strides[1]];
  }
  // 3D
  T& operator()(std::size_t i, std::size_t j, std::size_t k) {
#ifdef CNDA_BOUNDS_CHECK
    if (m_ndim != 3 || i >= m_shape[0] || j >= m_shape[1] || k >= m_shape[2]) {
      throw std::out_of_range("operator(): out of bounds (3D)");
    }
#endif
    return m_buffer[i * m_strides[0] + j * m_strides[1] + k * m_strides[2]];
  }
  const T& operator()(std::size_t i, std::size_t j, std::size_t k) const {
#ifdef CNDA_BOUNDS_CHECK
    if (m_ndim != 3 || i >= m_shape[0] || j >= m_shape[1] || k >= m_shape[2]) {
      throw std::out_of_range("operator() const: out of bounds (3D)");
    }
#endif
    return m_buffer[i * m_strides[0] + j * m_strides[1] + k * m_strides[2]];
  }
  
private:
  std::vector<std::size_t> m_shape;
  std::vector<std::size_t> m_strides; // in ELEMENTS (not bytes)
  std::size_t              m_ndim = 0;
  std::size_t              m_size = 0;
  std::vector<T>           m_buffer;  // contiguous storage
};

} // namespace cnda