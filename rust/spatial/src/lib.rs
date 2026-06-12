use std::fmt::format;

use cddlib_rs::{Matrix, Polyhedron, CddNumber, CddResult, Inequality};
use nalgebra::{DMatrix};
use num_traits::ToPrimitive;
use numpy::ndarray::{Array2, ArrayBase, Dim, ViewRepr, ShapeBuilder};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[allow(non_snake_case)]
fn _enum_gens(Ab: &DMatrix<f64>, Ab_eq: &DMatrix<f64>) -> CddResult<(DMatrix<f64>, DMatrix<f64>)> {    
    let nrows = Ab.nrows();
    let ncols = Ab.ncols();

    let mut matrix: Matrix<f64, Inequality> = Matrix::new(
        nrows,
        ncols,
        f64::DEFAULT_NUMBER_TYPE,
    )?;

    // Convert the format Ab = [A, b] to the [b, -A] format expected by cddlib
    for row in 0..nrows {
        // First column is b
        matrix.set(row, 0, &Ab[(row, ncols - 1)]);

        // Remaining columns are -A
        for col in 0..(ncols - 1) {
            matrix.set(row, col + 1, &(-Ab[(row, col)]));
        }
    }

    let poly = Polyhedron::from_matrix(&matrix)?;

    let v_rep = poly.generators()?;

    let nrows = v_rep.rows();
    let ncols = v_rep.cols();
    let gens = DMatrix::from_fn(nrows, ncols, |r, c| v_rep.get(r, c));
    let gens_ref = &gens;

    // Based on whether the first entry is a 1, stack it to vertices
    let verts_data: Vec<f64> = (0..gens.nrows())
        .filter(|&row| (gens_ref[(row, 0)] - 1.0).abs() < f64::EPSILON)
        .flat_map(|row| {
            (1..gens.ncols()).map(move |col| gens_ref[(row, col)])
        }).collect();

    let verts = DMatrix::from_row_slice(verts_data.len() / (ncols - 1), ncols - 1, &verts_data);

    // Based on whether the first entry is a 0, stack it to rays
    let rays_data: Vec<f64> = (0..gens.nrows())
        .filter(|&row| gens_ref[(row, 0)].abs() < f64::EPSILON)
        .flat_map(|row| {
            (1..gens.ncols()).map(move |col| gens_ref[(row, col)])
        }).collect();

    let rays = DMatrix::from_row_slice(rays_data.len() / (ncols - 1), ncols - 1, &rays_data);

    Ok((verts, rays))
}

#[pyfunction]
fn enum_gens<'py>(arr: &Bound<'py, PyAny>, arr_eq: &Bound<'py, PyAny>) -> PyResult<PyObject> {
    /// Cast a NumPy array (of type float/int) to a DMatrix float64
    fn cast_numeric_array<'py>(array: &Bound<'py, PyAny>) -> PyResult<DMatrix<f64>> {
        if let Ok(view) = array.downcast::<PyArray2<f64>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<f32>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<i64>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<i32>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<i16>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<i8>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<u64>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<u32>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<u16>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }
        if let Ok(view) = array.downcast::<PyArray2<u8>>() {
            let readonly = view.readonly();
            return arr_to_dmatrix(&readonly.as_array());
        }

        Err(PyRuntimeError::new_err(
            "Unsupported numeric dtype for numeric array",
        ))
    }
    
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

    let numpy = arr.py().import("numpy")?;
    let ndarray_type = numpy.getattr("ndarray")?;

    // Validate the inputs
    if !arr.is_instance(&ndarray_type)? {
        let received_type: String = arr.getattr("__class__")?.getattr("__name__")?.extract()?;
        return Err(PyTypeError::new_err(format!(
            "'arr' must be a NumPy array but received type '{received_type}'"
        )));
    }
    if !arr_eq.is_instance(&ndarray_type)? {
        let received_type: String = arr_eq.getattr("__class__")?.getattr("__name__")?.extract()?;
        return Err(PyTypeError::new_err(format!(
            "'arr_eq' must be a NumPy array but received type '{received_type}'"
        )));
    }

    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim != 2 {
        return Err(PyValueError::new_err(format!(
            "'arr' must be 2-dimensional but received arr.ndim={ndim}"
        )));
    }
    let ndim_eq: usize = arr_eq.getattr("ndim")?.extract()?;
    if ndim_eq != 2 {
        return Err(PyValueError::new_err(format!(
            "'arr_eq' must be 2-dimensional but received arr_eq.ndim{ndim_eq}"
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
    let dtype_kind_eq: String = arr_eq.getattr("dtype")?.getattr("kind")?.extract()?;
    if dtype_kind_eq != "f" && dtype_kind_eq != "i" && dtype_kind_eq != "u" {
        let dtype_kind_mapped = match dtype_kind_eq.as_str() {
            "c" => "complex",
            "b" => "bool",
            _ => dtype_kind.as_str(),
        };
        return Err(PyTypeError::new_err(format!(
            "Datatype of 'arr_eq' must be float or int but received '{dtype_kind_mapped}'",
        )));
    }

    let is_finite = numpy.call_method1("isfinite", (arr,))?;
    if !is_finite.call_method0("all")?.extract()?  {
        return Err(PyValueError::new_err(
            "'arr' must not contain nan or inf value"
        ));
    }
    let is_finite_eq = numpy.call_method1("isfinite", (arr_eq,))?;
    if !is_finite_eq.call_method0("all")?.extract()?  {
        return Err(PyValueError::new_err(
            "'arr_eq' must not contain nan or inf value"
        ));
    }

    // Convert the received from NumPy array to DMatrix f64
    let mat = cast_numeric_array(arr)?;
    let mat_eq = cast_numeric_array(arr_eq)?;

    let (mat_verts, mat_rays) = _enum_gens(&mat, &mat_eq)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let verts_array = Array2::from_shape_vec(
        (mat_verts.nrows(), mat_verts.ncols()).f(),
        mat_verts.as_slice().to_vec(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to build output array: {e}")))?;
    
    let verts = PyArray2::from_array(arr.py(), &verts_array);

    let rays_array = Array2::from_shape_vec(
        (mat_rays.nrows(), mat_rays.ncols()).f(),
        mat_rays.as_slice().to_vec(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to build output array: {e}")))?;

    let rays = PyArray2::from_array(arr_eq.py(), &rays_array);

    Ok(PyTuple::new(arr.py(), &[verts, rays])?.to_object(arr.py()))
}

#[pymodule]
fn rspatial(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enum_gens, m)?)?;

    Ok(())
}