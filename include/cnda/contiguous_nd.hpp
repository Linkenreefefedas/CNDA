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
  // -------- operator() overload (N-dimensional) --------
  // Variadic template for N-dimensional access
  template<typename... Indices>
  T& operator()(Indices... indices) {
    constexpr std::size_t n = sizeof...(Indices);
    std::size_t idx_array[n] = {static_cast<std::size_t>(indices)...};
    
#ifdef CNDA_BOUNDS_CHECK
    if (n != m_ndim) {
      throw std::out_of_range("operator(): rank mismatch");
    }
#endif
    
    std::size_t offset = 0;
    for (std::size_t i = 0; i < n; ++i) {
#ifdef CNDA_BOUNDS_CHECK
      if (idx_array[i] >= m_shape[i]) {
        throw std::out_of_range("operator(): index out of bounds");
      }
#endif
      offset += idx_array[i] * m_strides[i];
    }
    return m_buffer[offset];
  }
  
  template<typename... Indices>
  const T& operator()(Indices... indices) const {
    constexpr std::size_t n = sizeof...(Indices);
    std::size_t idx_array[n] = {static_cast<std::size_t>(indices)...};
    
#ifdef CNDA_BOUNDS_CHECK
    if (n != m_ndim) {
      throw std::out_of_range("operator() const: rank mismatch");
    }
#endif
    
    std::size_t offset = 0;
    for (std::size_t i = 0; i < n; ++i) {
#ifdef CNDA_BOUNDS_CHECK
      if (idx_array[i] >= m_shape[i]) {
        throw std::out_of_range("operator() const: index out of bounds");
      }
#endif
      offset += idx_array[i] * m_strides[i];
    }
    return m_buffer[offset];
  }
  
private:
  std::vector<std::size_t> m_shape;
  std::vector<std::size_t> m_strides; // in ELEMENTS (not bytes)
  std::size_t              m_ndim = 0;
  std::size_t              m_size = 0;
  std::vector<T>           m_buffer;  // contiguous storage
};

} // namespace cnda