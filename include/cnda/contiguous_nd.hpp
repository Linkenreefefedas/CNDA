#pragma once
#include <vector>
#include <cstddef>
#include <initializer_list>

namespace cnda {

// v0.1：僅提供最小介面雛形，方便之後增量開發與測試。
template <class T>
class ContiguousND {
public:
  explicit ContiguousND(std::vector<std::size_t> /*shape*/) {}
  // 後續週次再補：shape()/strides()/size()/data()/operator()...
};

} // namespace cnda
