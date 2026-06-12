use nalgebra::{DMatrix, RealField};
use num_traits::ToPrimitive;
use numpy::ndarray::{ArrayBase, Dim, ViewRepr};
use numpy::{Element, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Compute the span of a matrix given a `DMatrix``. Returns the indices of the columns (from left-to-right) which are linearly independent, or returns `None` when the matrix is zero rank.
fn _span<T>(mat: &DMatrix<T>, eps: Option<T::RealField>) -> Option<Vec<usize>>
where
    T: RealField,
{
    let eps = eps.unwrap_or(T::RealField::from_f64(1E-12).unwrap());
    if mat.ncols() == 0 {
        return None;
    }
    if mat.ncols() == 1 {
        if mat.rank(eps) == 0 {
            return None;
        }
        Some(vec![0])
    } else {
        let mut basis: Vec<usize> = Vec::new();
        for (j, _col) in mat.column_iter().enumerate() {
            let mut cand = basis.clone();
            cand.push(j);
            let sub_cand = mat.select_columns(&cand);
            let sub_basis = sub_cand.columns(0, sub_cand.ncols() - 1);
            let sub_cand_rank = sub_cand.rank(eps.clone());
            let mut sub_basis_rank = 0;
            if !sub_basis.is_empty() {
                sub_basis_rank = sub_basis.rank(eps.clone());
            }
            if sub_cand_rank == 0 {
                continue;
            }
            if sub_cand_rank > sub_basis_rank {
                basis = cand;
            }
            if j == mat.ncols() - 1 {
                if sub_cand.rank(eps.clone()) == 0 {
                    return None;
                }
            }
        }
        Some(basis)
    }
}

#[pyfunction]
fn span<'py>(arr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    /// Convert a view of a 2-D NumPy array into a nalgebra DMatrix<f64>, a matrix with dtype float64
    fn arr_to_dmatrix<T>(view: &ArrayBase<ViewRepr<&T>, Dim<[usize; 2]>>) -> PyResult<DMatrix<f64>>
    where
        T: ToPrimitive + Copy,
    {
        let (nrows, ncols) = view.dim();

        let slice = view
            .as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("Array 'arr' must be C-contiguous"))?;

        let data_f64: Vec<f64> = slice
            .iter()
            .map(|v| {
                v.to_f64().ok_or_else(|| {
                    PyRuntimeError::new_err("Could not cast one or more array elements to f64")
                })
            })
            .collect::<PyResult<Vec<f64>>>()?;

        Ok(DMatrix::from_row_slice(nrows, ncols, &data_f64))
    }

    /// Take a non-empty basis and return a NumPy array with these columns
    fn basis_to_pyarray2<'py, T>(
        py: Python<'py>,
        basis: Vec<usize>,
        view: &ArrayBase<ViewRepr<&T>, Dim<[usize; 2]>>,
    ) -> PyResult<Bound<'py, PyArray2<T>>>
    where
        T: Element + ToPrimitive + Copy,
    {
        let (nrows, _ncols) = view.dim();

        let mut out: Vec<Vec<T>> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let mut row: Vec<T> = Vec::with_capacity(basis.len());
            for &j in &basis {
                row.push(view[(i, j)]);
            }
            out.push(row);
        }

        PyArray2::from_vec2(py, &out)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build output array: {e}")))
    }

    let numpy = arr.py().import("numpy")?;
    let ndarray_type = numpy.getattr("ndarray")?;

    // Validate the inputs
    if !arr.is_instance(&ndarray_type)? {
        let received_type: String = arr.getattr("__class__")?.getattr("__name__")?.extract()?;
        return Err(PyTypeError::new_err(format!(
            "'arr' must be a NumPy array but received type '{received_type}'"
        )));
    }

    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim != 2 {
        return Err(PyValueError::new_err(format!(
            "'arr' must be 2-dimensional but received arr.ndim={ndim}"
        )));
    }

    let dtype_kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    if dtype_kind != "f" && dtype_kind != "i" && dtype_kind != "u" {
        let dtype_kind_mapped = match dtype_kind.as_str() {
            "c" => "complex",
            "b" => "bool",
            _ => dtype_kind.as_str(),
        };
        return Err(PyTypeError::new_err(format!(
            "Datatype of 'arr' must be float or int but received '{dtype_kind_mapped}'",
        )));
    }

    let is_finite = numpy.call_method1("isfinite", (arr,))?;
    if !is_finite.call_method0("all")?.extract()?  {
        return Err(PyValueError::new_err(
            "'arr' must not contain nan or inf value"
        ));
    }

    // Cast to f64, compute spanning indices, and return a new view of the original matrix
    macro_rules! try_print_for_dtype {
        ($t:ty) => {
            if let Ok(a) = arr.downcast::<PyArray2<$t>>() {
                let readonly = a.readonly();
                let view = readonly.as_array();
                let (nrows, _ncols) = view.dim();

                let mat = arr_to_dmatrix(&view)?;

                let basis = match _span(&mat, None) {
                    Some(b) => b,
                    None => return Ok(PyArray2::<$t>::zeros(arr.py(), [nrows, 0], false).into_any()),
                };

                return Ok(basis_to_pyarray2(arr.py(), basis, &view)?.into_any());
            }
        };
    }

    try_print_for_dtype!(f64);
    try_print_for_dtype!(f32);
    try_print_for_dtype!(i64);
    try_print_for_dtype!(i32);
    try_print_for_dtype!(i16);
    try_print_for_dtype!(i8);
    try_print_for_dtype!(u64);
    try_print_for_dtype!(u32);
    try_print_for_dtype!(u16);
    try_print_for_dtype!(u8);

    Err(PyRuntimeError::new_err(
        "Unsupported numeric dtype for 'arr'",
    ))
}

#[pymodule]
fn rlinalg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(span, m)?)?;

    Ok(())
}
