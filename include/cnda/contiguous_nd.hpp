#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>

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

private:
  std::vector<std::size_t> m_shape;
  std::vector<std::size_t> m_strides; // in ELEMENTS (not bytes)
  std::size_t              m_ndim = 0;
  std::size_t              m_size = 0;
  std::vector<T>           m_buffer;  // contiguous storage
};

} // namespace cnda
