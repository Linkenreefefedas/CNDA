#include "cnda/contiguous_nd.hpp"
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace cnda;

TEST_CASE("Move constructor with owned buffer", "[move_semantics]") {
    ContiguousND<int> arr1({2, 3});
    arr1(0, 0) = 42;
    arr1(1, 2) = 99;
    
    int* old_data = arr1.data();
    ContiguousND<int> arr2(std::move(arr1));
    
    REQUIRE(arr2(0, 0) == 42);
    REQUIRE(arr2(1, 2) == 99);
    REQUIRE(arr2.shape()[0] == 2);
    REQUIRE(arr2.shape()[1] == 3);
    REQUIRE(arr1.data() == nullptr);
}

TEST_CASE("Move assignment with owned buffer", "[move_semantics]") {
    ContiguousND<int> arr1({3, 2});
    arr1(0, 0) = 11;
    arr1(2, 1) = 22;
    
    ContiguousND<int> arr2({1, 1});
    arr2 = std::move(arr1);
    
    REQUIRE(arr2(0, 0) == 11);
    REQUIRE(arr2(2, 1) == 22);
    REQUIRE(arr2.shape()[0] == 3);
    REQUIRE(arr2.shape()[1] == 2);
    REQUIRE(arr1.data() == nullptr);
}

TEST_CASE("Move with external data (view)", "[move_semantics]") {
    std::vector<double> external_buffer(10, 3.14);
    auto owner = std::make_shared<std::vector<double>>(external_buffer);
    
    ContiguousND<double> arr1({2, 5}, owner->data(), owner);
    double* external_ptr = arr1.data();
    
    ContiguousND<double> arr2(std::move(arr1));
    
    REQUIRE(arr2.data() == external_ptr);
    REQUIRE(arr2.is_view());
    REQUIRE(arr1.data() == nullptr);
}

TEST_CASE("Move in std::vector", "[move_semantics]") {
    std::vector<ContiguousND<int>> vec;
    
    ContiguousND<int> arr({3, 3});
    arr(1, 1) = 55;
    
    vec.push_back(std::move(arr));
    
    REQUIRE(vec[0](1, 1) == 55);
    REQUIRE(arr.data() == nullptr);
}

TEST_CASE("Move preserves metadata", "[move_semantics]") {
    ContiguousND<float> arr1({4, 5, 6});
    arr1(3, 4, 5) = 123.45f;
    
    auto original_shape = arr1.shape();
    auto original_strides = arr1.strides();
    auto original_ndim = arr1.ndim();
    auto original_size = arr1.size();
    
    ContiguousND<float> arr2(std::move(arr1));
    
    REQUIRE(arr2.shape() == original_shape);
    REQUIRE(arr2.strides() == original_strides);
    REQUIRE(arr2.ndim() == original_ndim);
    REQUIRE(arr2.size() == original_size);
    REQUIRE(arr2(3, 4, 5) == 123.45f);
}

TEST_CASE("Self-move assignment is safe", "[move_semantics]") {
    ContiguousND<int> arr({2, 2});
    arr(0, 0) = 10;
    arr(1, 1) = 20;
    
    // This should be safe (though not useful)
    arr = std::move(arr);
    
    REQUIRE(arr.data() != nullptr);
    REQUIRE(arr(0, 0) == 10);
    REQUIRE(arr(1, 1) == 20);
}
