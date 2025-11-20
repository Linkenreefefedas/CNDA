"""
Test suite for error semantics in CNDA Python bindings.

This test suite validates that errors are raised with the correct Python exception types
according to the proposal:

- TypeError → dtype mismatch
- ValueError → shape/layout mismatch
- RuntimeError → invalid lifetime (if applicable)
- IndexError → out-of-bounds when bounds-check enabled

The tests cover both positive (error is raised) and negative (no error) cases.
"""

import sys
import os
import pytest


@pytest.fixture(scope="module")
def np():
    """Import numpy or skip tests if not available."""
    pytest.importorskip('numpy')
    import numpy
    return numpy


@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


# ==============================================================================
# TypeError Tests (Dtype Mismatch)
# ==============================================================================

class TestTypeErrorDtypeMismatch:
    """Test that TypeError is raised for dtype mismatches."""

    _UNSUPPORTED_ARRAYS = [
        pytest.param(lambda np: np.array([[1, 2], [3, 4]], dtype=dtype), id=name)
        for name, dtype in (
            ("uint8", "uint8"),
            ("uint16", "uint16"),
            ("uint32", "uint32"),
            ("uint64", "uint64"),
            ("int8", "int8"),
            ("int16", "int16"),
        )
    ] + [
        pytest.param(lambda np: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16), id="float16"),
        pytest.param(lambda np: np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64), id="complex64"),
        pytest.param(lambda np: np.array([[1 + 2j, 3 + 4j]], dtype=np.complex128), id="complex128"),
        pytest.param(lambda np: np.array([[True, False], [False, True]], dtype=np.bool_), id="bool"),
        pytest.param(lambda np: np.array([["a", "b"], ["c", "d"]], dtype=object), id="object"),
    ]

    @pytest.mark.parametrize("array_factory", _UNSUPPORTED_ARRAYS)
    def test_unsupported_dtypes_raise_typeerror(self, np, cnda, array_factory):
        """Unsupported numpy dtypes must raise TypeError."""
        x = array_factory(np)

        with pytest.raises(TypeError, match="Unsupported dtype"):
            cnda.from_numpy(x, copy=False)

    @pytest.mark.parametrize(
        "dtype_factory",
        [
            pytest.param(lambda mod: mod.float32, id="f32"),
            pytest.param(lambda mod: mod.float64, id="f64"),
            pytest.param(lambda mod: mod.int32, id="i32"),
            pytest.param(lambda mod: mod.int64, id="i64"),
        ],
    )
    def test_supported_dtypes_do_not_raise(self, np, cnda, dtype_factory):
        """Supported numpy dtypes should be accepted without errors."""
        dtype = dtype_factory(np)
        x = np.array([[1, 2], [3, 4]], dtype=dtype, order='C')

        arr = cnda.from_numpy(x, copy=False)
        assert arr is not None
    
    def test_dtype_specific_function_with_wrong_dtype(self, np, cnda):
        """Test that dtype-specific functions can handle dtype conversion with pybind11."""
        x_f64 = np.array([[1.0, 2.0]], dtype=np.float64)
        
        # pybind11 may perform implicit conversion or raise TypeError
        # This test documents the actual behavior
        try:
            arr = cnda.from_numpy_f32(x_f64, copy=False)
            # If it succeeds, pybind11 performed conversion
            assert arr is not None
        except TypeError:
            # If it raises TypeError, that's also acceptable behavior
            pass


# ==============================================================================
# ValueError Tests (Shape/Layout Mismatch)
# ==============================================================================

class TestValueErrorLayoutMismatch:
    """Test that ValueError is raised for layout/shape mismatches."""

    _NON_C_CONTIG_FACTORIES = [
        pytest.param(
            lambda np: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F'),
            id="fortran",
        ),
        pytest.param(
            lambda np: np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32, order='C'
            ).T,
            id="transposed",
        ),
        pytest.param(
            lambda np: np.arange(20, dtype=np.float32).reshape(5, 4)[::2, ::2],
            id="sliced",
        ),
        pytest.param(
            lambda np: np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
            id="non_standard_strides",
        ),
    ]

    @pytest.mark.parametrize("array_factory", _NON_C_CONTIG_FACTORIES)
    def test_non_c_contiguous_arrays_raise_valueerror(self, np, cnda, array_factory):
        """Non C-contiguous arrays with copy=False must raise ValueError."""
        x = array_factory(np)
        assert not x.flags['C_CONTIGUOUS']

        with pytest.raises(ValueError, match="C-contiguous|strides"):
            cnda.from_numpy_f32(x, copy=False)

    def test_c_contiguous_no_valueerror(self, np, cnda):
        """Test that C-contiguous array does NOT raise ValueError (negative test)."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='C')
        assert x.flags['C_CONTIGUOUS']

        # Should not raise ValueError
        arr = cnda.from_numpy_f32(x, copy=False)
        assert arr is not None

    @pytest.mark.parametrize(
        "array_factory",
        [
            pytest.param(
                lambda np: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F'),
                id="fortran",
            ),
            pytest.param(
                lambda np: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).T,
                id="transposed",
            ),
        ],
    )
    def test_copy_true_allows_layout_fix(self, np, cnda, array_factory):
        """copy=True should accept arrays even if layout is wrong."""
        x = array_factory(np)
        assert not x.flags['C_CONTIGUOUS']

        arr = cnda.from_numpy_f32(x, copy=True)
        assert arr is not None

    def test_generic_from_numpy_fortran_raises_valueerror(self, np, cnda):
        """Test that generic from_numpy raises ValueError for Fortran-order."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F')

        with pytest.raises(ValueError, match="C-contiguous"):
            cnda.from_numpy(x, copy=False)


# ==============================================================================
# IndexError Tests (Out-of-Bounds Access)
# ==============================================================================

class TestIndexErrorBoundsCheck:
    """Test that IndexError is raised for out-of-bounds access."""
    
    _OUT_OF_BOUNDS_CASES = [
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([3, 4]),
            lambda arr: arr[3, 0],
            id="getitem_first_dim",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([3, 4]),
            lambda arr: arr[0, 4],
            id="getitem_second_dim",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([3, 4]),
            lambda arr: arr[5, 10],
            id="getitem_both_dims",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([3, 4]),
            lambda arr: arr.__setitem__((3, 0), 1.0),
            id="setitem_first_dim",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([3, 4]),
            lambda arr: arr.__setitem__((0, 4), 1.0),
            id="setitem_second_dim",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([3, 4]),
            lambda arr: arr(3, 0),
            id="call_operator",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([5]),
            lambda arr: arr[5],
            id="1d_out_of_bounds",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([2, 3, 4]),
            lambda arr: arr[2, 0, 0],
            id="3d_first_dim",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([2, 3, 4]),
            lambda arr: arr[0, 3, 0],
            id="3d_second_dim",
        ),
        pytest.param(
            lambda cnda: cnda.ContiguousND_f32([2, 3, 4]),
            lambda arr: arr[0, 0, 4],
            id="3d_third_dim",
        ),
    ]

    @pytest.mark.bounds_check_required
    @pytest.mark.parametrize("arr_factory, operation", _OUT_OF_BOUNDS_CASES)
    def test_out_of_bounds_ops_raise_indexerror(self, cnda, arr_factory, operation):
        """All invalid coordinate accesses must raise IndexError."""
        arr = arr_factory(cnda)

        with pytest.raises(IndexError, match="out of bounds|Index out of bounds"):
            operation(arr)

    @pytest.mark.parametrize(
        "indices",
        [
            pytest.param((3, 0), id="first_dim"),
            pytest.param((0, 4), id="second_dim"),
        ],
    )
    def test_at_method_out_of_bounds(self, cnda, indices):
        """arr.at() must raise IndexError for out-of-bounds indices."""
        arr = cnda.ContiguousND_f32([3, 4])

        with pytest.raises(IndexError, match=r"at\(\): index out of bounds"):
            arr.at(indices)
    
    def test_in_bounds_access_no_indexerror(self, cnda):
        """Test that in-bounds access does NOT raise IndexError (negative test)."""
        arr = cnda.ContiguousND_f32([3, 4])
        arr[0, 0] = 1.0
        arr[2, 3] = 2.0
        
        # Should not raise IndexError
        val1 = arr[0, 0]
        val2 = arr[2, 3]
        assert val1 == 1.0
        assert val2 == 2.0
    
# ==============================================================================
# IndexError Tests (Rank Mismatch)
# ==============================================================================

class TestIndexErrorRankMismatch:
    """Test that IndexError is raised for rank (dimension) mismatches."""
    
    _TOO_FEW_CASES = [
        pytest.param(lambda arr: arr[0], id="getitem"),
        pytest.param(lambda arr: arr.__setitem__(0, 1.0), id="setitem"),
        pytest.param(lambda arr: arr(0), id="call"),
    ]

    _TOO_MANY_CASES = [
        pytest.param(lambda arr: arr[0, 0, 0], id="getitem"),
        pytest.param(lambda arr: arr.__setitem__((0, 0, 0), 1.0), id="setitem"),
        pytest.param(lambda arr: arr(0, 0, 0), id="call"),
    ]

    @pytest.mark.bounds_check_required
    @pytest.mark.parametrize("operation", _TOO_FEW_CASES)
    def test_rank_mismatch_too_few_indices(self, cnda, operation):
        """Too few indices must raise IndexError."""
        arr = cnda.ContiguousND_f32([3, 4])

        with pytest.raises(IndexError, match="rank mismatch|Number of indices|Single index only valid"):
            operation(arr)

    @pytest.mark.bounds_check_required
    @pytest.mark.parametrize("operation", _TOO_MANY_CASES)
    def test_rank_mismatch_too_many_indices(self, cnda, operation):
        """Too many indices must raise IndexError."""
        arr = cnda.ContiguousND_f32([3, 4])

        with pytest.raises(IndexError, match="rank mismatch|Number of indices"):
            operation(arr)

    @pytest.mark.parametrize(
        "indices",
        [
            pytest.param((1,), id="too_few"),
            pytest.param((1, 2, 3), id="too_many"),
        ],
    )
    def test_at_method_wrong_ndim(self, cnda, indices):
        """Test that .at() raises IndexError for rank mismatch."""
        arr = cnda.ContiguousND_f32([3, 4])

        with pytest.raises(IndexError, match=r"at\(\): rank mismatch"):
            arr.at(indices)
    
    def test_correct_number_of_indices_no_indexerror(self, cnda):
        """Test that correct number of indices does NOT raise IndexError (negative test)."""
        arr = cnda.ContiguousND_f32([3, 4])
        arr[1, 2] = 5.0
        
        # Should not raise IndexError
        val = arr[1, 2]
        assert val == 5.0
    
    def test_1d_array_single_index_no_indexerror(self, cnda):
        """Test that 1D array with single index does NOT raise IndexError (negative test)."""
        arr = cnda.ContiguousND_f32([5])
        arr[3] = 7.0
        
        # Should not raise IndexError
        val = arr[3]
        assert val == 7.0


# ==============================================================================
# RuntimeError Tests (Lifetime/Ownership Issues)
# ==============================================================================

class TestRuntimeErrorLifetime:
    """Test RuntimeError for lifetime/ownership issues (if applicable)."""
    
    def test_capsule_keeps_data_alive(self, np, cnda):
        """Test that capsule mechanism prevents use-after-free (negative test for RuntimeError)."""
        # This is a negative test - we should NOT get RuntimeError
        # because the capsule keeps the data alive
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Delete NumPy reference
        del x
        
        import gc
        gc.collect()
        
        # Should NOT raise RuntimeError - capsule keeps data alive
        val = arr[1, 1]
        assert val == 4.0
    
    def test_to_numpy_zero_copy_lifetime(self, np, cnda):
        """Test that zero-copy to_numpy keeps C++ data alive (negative test for RuntimeError)."""
        # This is a negative test - we should NOT get RuntimeError
        arr = cnda.ContiguousND_f32([2, 2])
        arr[0, 0] = 1.5
        arr[1, 1] = 9.9
        
        # Export to NumPy with zero-copy
        np_arr = arr.to_numpy(copy=False)
        
        # Delete C++ object reference
        del arr
        
        import gc
        gc.collect()
        
        # Should NOT raise RuntimeError - capsule keeps C++ data alive
        assert np_arr[0, 0] == 1.5
        assert np_arr[1, 1] == 9.9


# ==============================================================================
# Mixed Error Type Tests
# ==============================================================================

class TestMixedErrorTypes:
    """Test scenarios that could trigger multiple error types."""
    
    def test_dtype_error_takes_precedence_over_layout(self, np, cnda):
        """Test that dtype error (TypeError) is raised before layout error (ValueError)."""
        # Create array with unsupported dtype AND wrong layout
        x = np.array([[1, 2], [3, 4]], dtype=np.uint8, order='F')
        
        # Should raise TypeError for unsupported dtype
        with pytest.raises(TypeError, match="Unsupported dtype"):
            cnda.from_numpy(x, copy=False)
    
    def test_all_error_types_independent(self, np, cnda):
        """Test that different error types are independent and correctly raised."""
        # TypeError: unsupported dtype
        with pytest.raises(TypeError):
            x_bad_dtype = np.array([[1, 2]], dtype=np.uint8)
            cnda.from_numpy(x_bad_dtype, copy=False)
        
        # ValueError: wrong layout
        with pytest.raises(ValueError):
            x_bad_layout = np.array([[1.0, 2.0]], dtype=np.float32, order='F')
            cnda.from_numpy_f32(x_bad_layout, copy=False)
    
    @pytest.mark.bounds_check_required
    def test_indexerror_independent(self, cnda):
        """Test that IndexError is raised for out-of-bounds access (requires CNDA_BOUNDS_CHECK)."""
        # IndexError: out of bounds
        with pytest.raises(IndexError):
            arr = cnda.ContiguousND_f32([2, 2])
            _ = arr[2, 0]


# ==============================================================================
# All Dtypes Error Tests
# ==============================================================================

class TestAllDtypesErrors:
    """Test error handling for all supported dtypes."""
    
    def test_f32_layout_error(self, np, cnda):
        """Test ValueError for f32 with wrong layout."""
        x = np.array([[1.0, 2.0]], dtype=np.float32, order='F')
        with pytest.raises(ValueError, match="C-contiguous|strides"):
            cnda.from_numpy_f32(x, copy=False)
    
    def test_f64_layout_error(self, np, cnda):
        """Test ValueError for f64 with wrong layout."""
        x = np.array([[1.0, 2.0]], dtype=np.float64, order='F')
        with pytest.raises(ValueError, match="C-contiguous|strides"):
            cnda.from_numpy_f64(x, copy=False)
    
    def test_i32_layout_error(self, np, cnda):
        """Test ValueError for i32 with wrong layout."""
        x = np.array([[1, 2]], dtype=np.int32, order='F')
        with pytest.raises(ValueError, match="C-contiguous|strides"):
            cnda.from_numpy_i32(x, copy=False)
    
    def test_i64_layout_error(self, np, cnda):
        """Test ValueError for i64 with wrong layout."""
        x = np.array([[1, 2]], dtype=np.int64, order='F')
        with pytest.raises(ValueError, match="C-contiguous|strides"):
            cnda.from_numpy_i64(x, copy=False)
    
    @pytest.mark.bounds_check_required
    def test_all_dtypes_bounds_error(self, cnda):
        """Test IndexError for out-of-bounds on all dtypes."""
        test_cases = [
            cnda.ContiguousND_f32,
            cnda.ContiguousND_f64,
            cnda.ContiguousND_i32,
            cnda.ContiguousND_i64,
        ]
        
        for cnda_class in test_cases:
            arr = cnda_class([2, 2])
            with pytest.raises(IndexError):
                _ = arr[2, 0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
