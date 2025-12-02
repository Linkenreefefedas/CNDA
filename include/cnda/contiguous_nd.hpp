#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <array>
#include <algorithm>

namespace cnda {

template <class T>
class ContiguousND {
  // AoS safety checks: T must be POD-like
  static_assert(std::is_standard_layout<T>::value,
                "ContiguousND requires T to be standard-layout type");
  static_assert(std::is_trivially_copyable<T>::value,
                "ContiguousND requires T to be trivially copyable");
  
public:
  // -------- Constructors --------
  explicit ContiguousND(std::vector<std::size_t> shape)
      : m_shape(std::move(shape))
  {
      compute_metadata();
      m_buffer.resize(m_size);
      m_data = m_buffer.data();
  }

  ContiguousND(std::vector<std::size_t> shape,
               T* external_data,
               std::shared_ptr<void> external_owner)
      : m_shape(std::move(shape)),
        m_data(external_data),
        m_external_owner(std::move(external_owner))
  {
      compute_metadata();
  }

  // -------- Move Semantics --------
  ContiguousND(ContiguousND&& other) noexcept
      : m_shape(std::move(other.m_shape)),
        m_strides(std::move(other.m_strides)),
        m_ndim(other.m_ndim),
        m_size(other.m_size),
        m_buffer(std::move(other.m_buffer)),
        m_external_owner(std::move(other.m_external_owner))
  {
      // Repoint m_data to the moved buffer if it was self-owned
      if (other.m_data && !other.m_buffer.empty() && other.m_data == other.m_buffer.data()) {
          m_data = m_buffer.data();
      } else {
          m_data = other.m_data;
      }
      other.m_data = nullptr;
      other.m_ndim = 0;
      other.m_size = 0;
  }

  ContiguousND& operator=(ContiguousND&& other) noexcept {
      if (this != &other) {
          m_shape = std::move(other.m_shape);
          m_strides = std::move(other.m_strides);
          m_ndim = other.m_ndim;
          m_size = other.m_size;
          m_buffer = std::move(other.m_buffer);
          m_external_owner = std::move(other.m_external_owner);
          
          // Repoint m_data to the moved buffer if it was self-owned
          if (other.m_data && !other.m_buffer.empty() && other.m_data == other.m_buffer.data()) {
              m_data = m_buffer.data();
          } else {
              m_data = other.m_data;
          }
          other.m_data = nullptr;
          other.m_ndim = 0;
          other.m_size = 0;
      }
      return *this;
  }

  // Delete copy operations to prevent accidental expensive copies
  // Users should explicitly use clone() or similar if deep copy is needed
  ContiguousND(const ContiguousND&) = delete;
  ContiguousND& operator=(const ContiguousND&) = delete;

  // -------- Basic Accessors --------
  const std::vector<std::size_t>& shape()   const noexcept { return m_shape;   }
  const std::vector<std::size_t>& strides() const noexcept { return m_strides; }
  std::size_t ndim() const noexcept { return m_ndim; }
  std::size_t size() const noexcept { return m_size; }

  T* data() noexcept { return m_data; }
  const T* data() const noexcept { return m_data; }

  bool is_view() const noexcept { return m_external_owner != nullptr; }

  // -------- Core offset computation (shared by all accessors) --------
  std::size_t compute_offset(const std::size_t* idx_array, std::size_t n, bool check_bounds) const {
      bool enforce_bounds = check_bounds;
#ifdef CNDA_BOUNDS_CHECK
      enforce_bounds = true;
#endif

      if (enforce_bounds && n != m_ndim) {
          throw std::out_of_range("at(): rank mismatch");
      }
#ifndef CNDA_BOUNDS_CHECK
      // When bounds check is disabled and not forced, clamp to safe range
      if (!enforce_bounds && n != m_ndim) {
          n = std::min(n, m_ndim);
      }
#endif

      std::size_t off = 0;
      for (std::size_t i = 0; i < n; ++i) {
          if (enforce_bounds && idx_array[i] >= m_shape[i]) {
              throw std::out_of_range("at(): index out of bounds");
          }
          off += idx_array[i] * m_strides[i];  // stride in ELEMENTS
      }
      return off;
  }

  // -------- initializer_list version --------
  std::size_t index(std::initializer_list<std::size_t> idxs, bool check_bounds = false) const {
      std::vector<std::size_t> tmp(idxs.begin(), idxs.end());
      return compute_offset(tmp.data(), tmp.size(), check_bounds);
  }

  // -------- at() with bounds guaranteed --------
  T& at(std::initializer_list<std::size_t> idxs) {
      return m_data[index(idxs, true)];
  }
  const T& at(std::initializer_list<std::size_t> idxs) const {
      return m_data[index(idxs, true)];
  }

  // -------- Variadic operator() --------
  // Optimized 1D access
  template <typename Index>
  inline T& operator()(Index i0) {
#ifdef CNDA_BOUNDS_CHECK
      if (m_ndim != 1) throw std::out_of_range("operator(): rank mismatch");
      std::size_t idx = static_cast<std::size_t>(i0);
      if (idx >= m_shape[0]) throw std::out_of_range("operator(): index out of bounds");
      return m_data[idx * m_strides[0]];
#else
      return m_data[static_cast<std::size_t>(i0) * m_strides[0]];
#endif
  }
  
  template <typename Index>
  inline const T& operator()(Index i0) const {
#ifdef CNDA_BOUNDS_CHECK
      if (m_ndim != 1) throw std::out_of_range("operator(): rank mismatch");
      std::size_t idx = static_cast<std::size_t>(i0);
      if (idx >= m_shape[0]) throw std::out_of_range("operator(): index out of bounds");
      return m_data[idx * m_strides[0]];
#else
      return m_data[static_cast<std::size_t>(i0) * m_strides[0]];
#endif
  }
  
  // Optimized 2D access
  template <typename Index1, typename Index2>
  inline T& operator()(Index1 i0, Index2 i1) {
#ifdef CNDA_BOUNDS_CHECK
      if (m_ndim != 2) throw std::out_of_range("operator(): rank mismatch");
      std::size_t idx0 = static_cast<std::size_t>(i0);
      std::size_t idx1 = static_cast<std::size_t>(i1);
      if (idx0 >= m_shape[0] || idx1 >= m_shape[1]) 
          throw std::out_of_range("operator(): index out of bounds");
      return m_data[idx0 * m_strides[0] + idx1 * m_strides[1]];
#else
      return m_data[static_cast<std::size_t>(i0) * m_strides[0] + 
                    static_cast<std::size_t>(i1) * m_strides[1]];
#endif
  }
  
  template <typename Index1, typename Index2>
  inline const T& operator()(Index1 i0, Index2 i1) const {
#ifdef CNDA_BOUNDS_CHECK
      if (m_ndim != 2) throw std::out_of_range("operator(): rank mismatch");
      std::size_t idx0 = static_cast<std::size_t>(i0);
      std::size_t idx1 = static_cast<std::size_t>(i1);
      if (idx0 >= m_shape[0] || idx1 >= m_shape[1]) 
          throw std::out_of_range("operator(): index out of bounds");
      return m_data[idx0 * m_strides[0] + idx1 * m_strides[1]];
#else
      return m_data[static_cast<std::size_t>(i0) * m_strides[0] + 
                    static_cast<std::size_t>(i1) * m_strides[1]];
#endif
  }
  
  // Optimized 3D access
  template <typename Index1, typename Index2, typename Index3>
  inline T& operator()(Index1 i0, Index2 i1, Index3 i2) {
#ifdef CNDA_BOUNDS_CHECK
      if (m_ndim != 3) throw std::out_of_range("operator(): rank mismatch");
      std::size_t idx0 = static_cast<std::size_t>(i0);
      std::size_t idx1 = static_cast<std::size_t>(i1);
      std::size_t idx2 = static_cast<std::size_t>(i2);
      if (idx0 >= m_shape[0] || idx1 >= m_shape[1] || idx2 >= m_shape[2]) 
          throw std::out_of_range("operator(): index out of bounds");
      return m_data[idx0 * m_strides[0] + idx1 * m_strides[1] + idx2 * m_strides[2]];
#else
      return m_data[static_cast<std::size_t>(i0) * m_strides[0] + 
                    static_cast<std::size_t>(i1) * m_strides[1] +
                    static_cast<std::size_t>(i2) * m_strides[2]];
#endif
  }
  
  template <typename Index1, typename Index2, typename Index3>
  inline const T& operator()(Index1 i0, Index2 i1, Index3 i2) const {
#ifdef CNDA_BOUNDS_CHECK
      if (m_ndim != 3) throw std::out_of_range("operator(): rank mismatch");
      std::size_t idx0 = static_cast<std::size_t>(i0);
      std::size_t idx1 = static_cast<std::size_t>(i1);
      std::size_t idx2 = static_cast<std::size_t>(i2);
      if (idx0 >= m_shape[0] || idx1 >= m_shape[1] || idx2 >= m_shape[2]) 
          throw std::out_of_range("operator(): index out of bounds");
      return m_data[idx0 * m_strides[0] + idx1 * m_strides[1] + idx2 * m_strides[2]];
#else
      return m_data[static_cast<std::size_t>(i0) * m_strides[0] + 
                    static_cast<std::size_t>(i1) * m_strides[1] +
                    static_cast<std::size_t>(i2) * m_strides[2]];
#endif
  }
  
  // Optimized 4D access
  template <typename Index1, typename Index2, typename Index3, typename Index4>
  inline T& operator()(Index1 i0, Index2 i1, Index3 i2, Index4 i3) {
#ifdef CNDA_BOUNDS_CHECK
      if (m_ndim != 4) throw std::out_of_range("operator(): rank mismatch");
      std::size_t idx0 = static_cast<std::size_t>(i0);
      std::size_t idx1 = static_cast<std::size_t>(i1);
      std::size_t idx2 = static_cast<std::size_t>(i2);
      std::size_t idx3 = static_cast<std::size_t>(i3);
      if (idx0 >= m_shape[0] || idx1 >= m_shape[1] || idx2 >= m_shape[2] || idx3 >= m_shape[3]) 
          throw std::out_of_range("operator(): index out of bounds");
      return m_data[idx0 * m_strides[0] + idx1 * m_strides[1] + idx2 * m_strides[2] + idx3 * m_strides[3]];
#else
      return m_data[static_cast<std::size_t>(i0) * m_strides[0] + 
                    static_cast<std::size_t>(i1) * m_strides[1] +
                    static_cast<std::size_t>(i2) * m_strides[2] +
                    static_cast<std::size_t>(i3) * m_strides[3]];
#endif
  }
  
  template <typename Index1, typename Index2, typename Index3, typename Index4>
  inline const T& operator()(Index1 i0, Index2 i1, Index3 i2, Index4 i3) const {
#ifdef CNDA_BOUNDS_CHECK
      if (m_ndim != 4) throw std::out_of_range("operator(): rank mismatch");
      std::size_t idx0 = static_cast<std::size_t>(i0);
      std::size_t idx1 = static_cast<std::size_t>(i1);
      std::size_t idx2 = static_cast<std::size_t>(i2);
      std::size_t idx3 = static_cast<std::size_t>(i3);
      if (idx0 >= m_shape[0] || idx1 >= m_shape[1] || idx2 >= m_shape[2] || idx3 >= m_shape[3]) 
          throw std::out_of_range("operator(): index out of bounds");
      return m_data[idx0 * m_strides[0] + idx1 * m_strides[1] + idx2 * m_strides[2] + idx3 * m_strides[3]];
#else
      return m_data[static_cast<std::size_t>(i0) * m_strides[0] + 
                    static_cast<std::size_t>(i1) * m_strides[1] +
                    static_cast<std::size_t>(i2) * m_strides[2] +
                    static_cast<std::size_t>(i3) * m_strides[3]];
#endif
  }

  // General N-D fallback for 5+ dimensions
  template <typename... Indices, 
            typename = typename std::enable_if<(sizeof...(Indices) >= 5)>::type>
  T& operator()(Indices... indices) {
      constexpr std::size_t N = sizeof...(Indices);
      std::array<std::size_t, N> idx_array
          {{ static_cast<std::size_t>(indices)... }};
#ifdef CNDA_BOUNDS_CHECK
      return m_data[ compute_offset(idx_array.data(), N, true) ];
#else
      return m_data[ compute_offset(idx_array.data(), N, false) ];
#endif
  }

  template <typename... Indices,
            typename = typename std::enable_if<(sizeof...(Indices) >= 5)>::type>
  const T& operator()(Indices... indices) const {
      constexpr std::size_t N = sizeof...(Indices);
      std::array<std::size_t, N> idx_array
          {{ static_cast<std::size_t>(indices)... }};
#ifdef CNDA_BOUNDS_CHECK
      return m_data[ compute_offset(idx_array.data(), N, true) ];
#else
      return m_data[ compute_offset(idx_array.data(), N, false) ];
#endif
  }

private:
  std::vector<std::size_t> m_shape;
  std::vector<std::size_t> m_strides;  // stride in ELEMENTS
  std::size_t m_ndim = 0;
  std::size_t m_size = 0;

  std::vector<T> m_buffer;
  T* m_data = nullptr;
  std::shared_ptr<void> m_external_owner;

  void compute_metadata() noexcept {
      m_ndim = m_shape.size();
      m_size = 1;
      for (std::size_t d : m_shape) {
          m_size *= d;
      }

      // Row-major, ELEMENT-BASED strides
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
