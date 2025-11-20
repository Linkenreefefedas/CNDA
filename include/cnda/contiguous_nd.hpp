#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <initializer_list>
#include <memory>

namespace cnda {

template <class T>
class ContiguousND {
public:
  // Construct with shape; allocate contiguous buffer of product(shape).
  explicit ContiguousND(std::vector<std::size_t> shape)
    : m_shape(std::move(shape))
  {
    compute_metadata();

    // owning storage
    m_buffer.resize(m_size);
    m_data = m_buffer.data();
    // m_external_owner stays empty (null)
  }

  // non-owning view constructor（for from_numpy(copy=False)）
  ContiguousND(std::vector<std::size_t> shape,
                 T* external_data,
                 std::shared_ptr<void> external_owner)
      : m_shape(std::move(shape)),
        m_data(external_data),
        m_external_owner(std::move(external_owner))
    {
      compute_metadata();
    }

  // -------- O(1) accessors --------
  const std::vector<std::size_t>& shape()   const noexcept { return m_shape;   }
  const std::vector<std::size_t>& strides() const noexcept { return m_strides; }
  std::size_t ndim() const noexcept { return m_ndim; }
  std::size_t size() const noexcept { return m_size; }
  
  T* data() noexcept { return m_data; }
  const T* data() const noexcept { return m_data; }
  
  // Check if this is a non-owning view
  bool is_view() const noexcept { return m_external_owner != nullptr; }

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
  // Safe access with bounds checking, mirrors std::vector::at()
  T& at(std::initializer_list<std::size_t> idxs) {
      if (idxs.size() != m_ndim) {
        throw std::out_of_range("at(): rank mismatch");
      }
      
      std::size_t offset = 0;
      std::size_t axis = 0;
      for (auto v : idxs) {
        if (v >= m_shape[axis]) {
          throw std::out_of_range("at(): index out of bounds");
        }
        offset += v * m_strides[axis];
        ++axis;
      }
      return m_data[offset];
  }

  const T& at(std::initializer_list<std::size_t> idxs) const {
      if (idxs.size() != m_ndim) {
        throw std::out_of_range("at() const: rank mismatch");
      }
      
      std::size_t offset = 0;
      std::size_t axis = 0;
      for (auto v : idxs) {
        if (v >= m_shape[axis]) {
          throw std::out_of_range("at() const: index out of bounds");
        }
        offset += v * m_strides[axis];
        ++axis;
      }
      return m_data[offset];
  }

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
        return m_data[offset];
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
        return m_data[offset];
  }
  
private:
  std::vector<std::size_t> m_shape;
  std::vector<std::size_t> m_strides; // in ELEMENTS (not bytes)
  std::size_t m_ndim = 0;
  std::size_t m_size = 0;
  std::vector<T> m_buffer;  // contiguous storage
  T* m_data = nullptr;
  std::shared_ptr<void> m_external_owner;  // non-null if non-owning view
  
  // Helper to compute ndim, size, and strides from shape
  void compute_metadata() {
    m_ndim = m_shape.size();
    m_size = 1;
    for (std::size_t d : m_shape) {
      m_size *= d;
    }

    // row-major strides
    m_strides.assign(m_ndim, 0);
    if (m_ndim > 0) {
      m_strides[m_ndim - 1] = 1;
      for (std::size_t k = m_ndim; k-- > 1; ) {
        m_strides[k - 1] = m_strides[k] * m_shape[k];
      }
    }
  }
};

} // namespace cnda