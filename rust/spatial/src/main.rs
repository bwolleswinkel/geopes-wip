use cddlib_rs::{Matrix, Polyhedron, CddNumber, CddResult, Inequality};
use nalgebra::{DMatrix};

fn enum_gens(mat: &DMatrix<f64>) -> CddResult<(DMatrix<f64>, DMatrix<f64>)> {
    let mut matrix: Matrix<f64, Inequality> = Matrix::new(
        mat.nrows(),
        mat.ncols(),
        f64::DEFAULT_NUMBER_TYPE,
    )?;

    for row in 0..mat.nrows() {
        for col in 0..mat.ncols() {
            matrix.set(row, col, &mat[(row, col)]);
        }
    };

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

fn main() {
    #[allow(non_snake_case)]
    let A = vec![
        vec![ 1,  0],
        vec![ 0,  1],
        vec![-1,  0],
        vec![ 0, -1],
    ];

    let b = vec![1, 1, 0, 0];

    // H-representation: Each row is [b, -A] representing Ax <= b
    let nrows = b.len();
    let ncols = A[0].len() + 1;
    let mut data = Vec::with_capacity(nrows * ncols);
    for (&b_i, a_row) in b.iter().zip(A.iter()) {
        data.push(b_i as f64);
        data.extend(a_row.iter().map(|&a| -a as f64));
    }
    let mat = &DMatrix::from_row_slice(nrows, ncols, &data);

    let (verts, rays) = enum_gens(mat).expect("Conversion from H-representation to V-representation failed");

    println!("verts={verts}");

    println!("rays={rays}");
}
