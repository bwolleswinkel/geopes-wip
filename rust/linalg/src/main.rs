use std::ops::{Add, AddAssign, Mul, MulAssign};

use nalgebra::{DMatrix, DVector, RealField, Scalar};
use num_traits::{One, Zero};

fn _outer_prod<T>(vec_1: &DVector<T>, vec_2: &DVector<T>) -> DMatrix<T>
where
    // This is needed such that it works for both integers and floats
    T: Scalar + Copy + Mul<Output = T> + Add<Output = T> + AddAssign + MulAssign + Zero + One,
{
    vec_1 * vec_2.transpose()
}

/// Returns a matrix containing a linearly independent subset of columns from `mat`.
///
/// If `eps` is `None`, a default tolerance of `1e-12` is used.
/// The result preserves the left-to-right order of selected columns.
fn span<T>(mat: &DMatrix<T>, eps: Option<T::RealField>) -> Vec<usize>
where
    T: RealField,
{
    if mat.ncols() <= 1 {
        vec![0]
    } else {
        let eps = eps.unwrap_or(T::RealField::from_f64(1E-12).unwrap());
        let mut basis: Vec<usize> = vec![0];
        for (j, _col) in mat.column_iter().enumerate() {
            if j == 0 {
                continue;
            }
            let mut cand = basis.clone();
            cand.push(j);
            let sub_cand = mat.select_columns(&cand);
            let sub_basis = sub_cand.columns(0, sub_cand.ncols() - 1);
            if sub_cand.rank(eps.clone()) > sub_basis.rank(eps.clone()) {
                basis = cand;
            }
        }
        basis
    }
}

fn main() {
    let nrows: usize = 4;
    let ncols: usize = 4;

    let data: Vec<f32> = vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];

    // let data: Vec<i32> = vec![
    //     1, 1, 0, 2,
    //     0, 1, 0, 1,
    //     0, 0, 1, 0,
    //     1, 0, 0, 1
    // ];

    let matrix = DMatrix::from_row_slice(nrows, ncols, &data);
    println!("Org={matrix}");

    let m_f64 = matrix.clone().cast::<f64>();
    let basis = span(&m_f64, None);
    let mat_lin_indp = matrix.select_columns(&basis);
    let rank: usize = m_f64.rank(1E-12);
    println!("Linearly independent cols={mat_lin_indp} (rank={rank})");
}
