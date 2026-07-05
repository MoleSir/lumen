use lumen_core::{FloatDType, Tensor};
use crate::{LinalgError, LinalgResult};
use crate::view::prelude::*;

/// Result of eigenvalue decomposition of a square matrix
///
/// Given a square matrix `A` of size `n x n`, the eigen decomposition factorizes it as:
/// ```text
/// A = V * Λ * V⁻¹
/// ```
/// where:
/// - `Λ` is a diagonal matrix containing the eigenvalues
/// - `V` is a matrix whose columns are the eigenvectors
///
/// This struct stores the eigenvalues and eigenvectors in a convenient format.
pub struct EigResult<T: FloatDType> {
    /// Eigenvalues of the matrix (length n)
    pub eig_values: Vec<T>,
    /// Eigenvectors of the matrix (n x n), each column corresponds to an eigenvector
    pub eig_vectors: Tensor<T>,
}

impl<T: FloatDType> EigResult<T> {
    fn new(eig_values: Vec<T>, eig_vectors:  Tensor<T>,) -> Self {
        Self { eig_values, eig_vectors }
    }

    /// Reconstruct the original matrix from its eigen decomposition
    ///
    /// Computes:
    /// ```text
    /// A ≈ V * diag(eig_values) * V⁻¹
    /// ```
    /// # Notes
    /// - For symmetric or Hermitian matrices, `V⁻¹` can be replaced by `Vᵀ`.
    /// - This method returns an approximation due to floating point errors.
    pub fn reconstruct(&self) -> LinalgResult<Tensor<T>> {
        let d = Tensor::diag(&self.eig_values).unwrap();
        let v = self.eig_vectors.matmul(&d)?;
        let v = v.matmul(&self.eig_vectors.transpose_last()?)?;
        Ok(v)
    }
}

pub fn qr_eig<T: FloatDType>(a: &Tensor<T>, max_iters: usize, tol: T) -> LinalgResult<EigResult<T>> {
    if !crate::is_symmetric(a)? {
        return Err(LinalgError::ExpectSymmetricMatrix { op: "jacobi_eig" })?;
    }

    let (n, _) = a.dims2()?;
    let mut h = a.copy()?;
    let mut v = Tensor::<T>::eye(n)?;

    for _ in 0..max_iters {
        let result = crate::qr(&h)?;
        let (q, r) = (result.q, result.r);

        h = r.matmul(&q)?;
        v = v.matmul(&q)?;

        // off-diagonal norm
        let mut off_diag_norm = T::zero();
        matrix_view!(hv = h);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off_diag_norm += hv.g(i,j).abs();
                }
            }
        }


        if off_diag_norm < tol {
            break;
        }
    }

    // let hv = h.matrix_view_unsafe().unwrap();
    matrix_view!(hv = h);
    let eig_values = (0..n).map(|i| { hv.g(i,i) }).collect::<Vec<_>>();
    Ok(EigResult { eig_values, eig_vectors: v })
}

/// Eig by jacobi
/// 
/// # References
/// - https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
/// - https://www.quantstart.com/articles/Jacobi-Method-in-Python-and-NumPy/
/// - https://oldsite.pup.ac.in/e-content/science/physics/mscphy58.pdf
/// 
pub fn jacobi_eig<T: FloatDType>(mat: &Tensor<T>, tol: T) -> LinalgResult<EigResult<T>> {

    fn max_elem<T: FloatDType, S: AsRef<[T]>>(mat: &MatrixView<'_, T, S>) -> (T, (usize, usize)) {
        let (n, _) = mat.shape();
        let mut max_value = mat.g(0, 1).abs();
        let mut max_position = (0, 1);
        for i in 0..(n - 1) {
            for j in (i+1)..n {
                if mat.g(i, j).abs() > max_value {
                    max_value = mat.g(i, j).abs();
                    max_position = (i, j);
                }
            }
        }
        (max_value, max_position)
    }

    if !crate::is_symmetric(mat)? {
        return Err(LinalgError::ExpectSymmetricMatrix { op: "jacobi_eig" })?;
    }

    matrix_view!(mat_view = mat);
    let (n, m) = mat_view.shape();
    if m != n {
        Err(LinalgError::ExpectMatrixSquare { shape: mat_view.shape(), op: "jacobi_eig" })?;
    }

    {
        let max_rot = 5 * n.pow(2);
        let mut a = mat_view.copy()?;
        let mut r = Tensor::<T>::eye(n)?;
    
        for _ in 0..max_rot {
            let (ai, ri) = {
                matrix_view!(av = a);
                let (max_value, (p, q)) = max_elem(&av);
                assert!(p != q);
                if max_value < tol {
                    let eig_vals: Vec<T> = (0..n).map(|i| av.g(i,i)).collect();
        
                    // sort by eig_vals
                    let mut idx: Vec<usize> = (0..n).collect();
                    idx.sort_by(|&i, &j| eig_vals[j].abs().partial_cmp(&eig_vals[i].abs()).unwrap());
                    let eig_vals_sorted: Vec<T> = idx.iter().map(|&i| eig_vals[i]).collect();
                    let eig_vecs_sorted = r.copy()?;
                    {
                        matrix_view_mut!(mv = eig_vecs_sorted);
                        matrix_view!(rv = r);
        
                        for (new_j, &old_j) in idx.iter().enumerate() {
                            for i in 0..n {
                                mv.s(i, new_j, rv.g(i, old_j));
                            }
                        }
                    }
                
                    return Ok(EigResult::new(eig_vals_sorted, eig_vecs_sorted));
                }

                jacobi_rotate(&a, p, q)?
            };

            a = ai;
            r = r.matmul(&ri)?;
        }
    
        Err(LinalgError::JacobiMethodDidNotConverge)?
    }
}

fn jacobi_rotate<T: FloatDType>(mat: &Tensor<T>, p: usize, q: usize) -> LinalgResult<(Tensor<T>, Tensor<T>)> {
    let (m, _) = mat.dims2()?;
    
    {
        matrix_view!(matv = mat);

        if matv.g(p, q) == T::zero() {
            return Ok((mat.clone(), Tensor::<T>::eye(m)?));
        }
    
        let (a, b, ab) = (matv.g(p, p), matv.g(q, q), matv.g(p, q));
        let (c, s) = jacobi_calculate_cs(a, b, ab);
    
        let r = Tensor::<T>::eye(m)?;
        {
            matrix_view_mut!(r);
            r.s(p, p, c); 
            r.s(q, q, c); 
            r.s(p, q, s); 
            r.s(q, p, -s); 
        }
    
        let a = r.transpose_last()?.matmul(mat)?;
        let a = a.matmul(&r)?;
        
        Ok((a, r))
    }
}

fn jacobi_calculate_cs<T: FloatDType>(a: T, b: T, ab: T) -> (T, T) {
    let tau = (b - a) / (T::from_f64(2.) * ab);
    let t = if tau > T::zero() {
        T::one() / (tau + (T::one() + tau.powi(2)).sqrt())
    } else {
        -T::one() / (tau.abs() + (T::one() + tau.powi(2)).sqrt())
    };
    let c = T::one() / (T::one() + t.powi(2)).sqrt();
    let s = c * t;
    (c, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_eig_identity() {
        let a = Tensor::<f64>::eye(3).unwrap();
        let result = jacobi_eig(&a, 1e-12).unwrap();

        for v in result.eig_values.iter() {
            assert!((v - 1.0f64).abs() < 1e-10);
        }

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8).unwrap());
    }

    #[test]
    fn test_jacobi_eig_diagonal_matrix() {
        let a = Tensor::new(&[
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, -2.0],
        ]).unwrap();

        let result = jacobi_eig(&a, 1e-12).unwrap();

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8).unwrap());
    }

    #[test]
    fn test_jacobi_eig_symmetric_offdiag() {
        let a = Tensor::new(&[
            [2.0, 1.0],
            [1.0, 2.0],
        ]).unwrap();
        let result = jacobi_eig(&a, 1e-12).unwrap();

        let mut sorted = result.eig_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0f64).abs() < 1e-6);
        assert!((sorted[1] - 3.0f64).abs() < 1e-6);

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8).unwrap());
    }

    #[test]
    fn test_jacobi_eig_zero_matrix() {
        let a = Tensor::zeros((3, 3)).unwrap();
        let result = jacobi_eig(&a, 1e-12).unwrap();

        for v in result.eig_values.iter() {
            assert!(f64::abs(*v) < 1e-12);
        }

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8).unwrap());
    }

    #[test]
    fn test_jacobi_eig_random_symmetric() {
        let a = Tensor::<f64>::randn(0.0, 1.0, (4, 4)).unwrap();
        let a_sym = (&a + &a.transpose_last().unwrap()).mul(0.5).unwrap();

        let result = jacobi_eig(&a_sym, 1e-10).unwrap();
        let rec = result.reconstruct().unwrap();

        assert!(rec.allclose(&a_sym, 1e-6, 1e-6).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_jacobi_eig_non_square() {
        let a = Tensor::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]).unwrap();
    
        let _ = jacobi_eig(&a, 1e-12).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_jacobi_eig_non_symmetric() {
        let a = Tensor::new(&[
            [1.0, 2.0],
            [0.0, 3.0],
        ]).unwrap();
    
        let result = jacobi_eig(&a, 1e-12).unwrap();
        println!("{}", result.reconstruct().unwrap());
    }

    #[test]
    fn test_jacobi_eig_zeros() {
        let a = Tensor::<f64>::zeros((4, 4)).unwrap();
        let _ = jacobi_eig(&a, 1e-12).unwrap();
    }
    
    #[test]
    fn test_qr_eig_identity() {
        let a = Tensor::<f64>::eye(3).unwrap();
        let result = qr_eig(&a, 100, 1e-12).unwrap();

        for v in result.eig_values.iter() {
            assert!((v - 1.0f64).abs() < 1e-10);
        }

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8).unwrap());
    }

    #[test]
    fn test_qr_eig_diagonal_matrix() {
        let a = Tensor::new(&[
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, -2.0],
        ]).unwrap();

        let result = qr_eig(&a, 100, 1e-12).unwrap();

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8).unwrap());
    }

    #[test]
    fn test_qr_eig_symmetric_offdiag() {
        let a = Tensor::new(&[
            [2.0, 1.0],
            [1.0, 2.0],
        ]).unwrap();
        let result = qr_eig(&a, 100, 1e-12).unwrap();

        let mut sorted = result.eig_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0f64).abs() < 1e-6);
        assert!((sorted[1] - 3.0f64).abs() < 1e-6);

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8).unwrap());
    }

    #[test]
    fn test_qr_eig_random_symmetric() {
        let a = Tensor::<f64>::randn(0.0, 1.0, (4, 4)).unwrap();
        let a_sym = (&a + &a.transpose_last().unwrap()).mul(0.5).unwrap();

        let result = qr_eig(&a_sym, 200, 1e-10).unwrap();
        let rec = result.reconstruct().unwrap();

        assert!(rec.allclose(&a_sym, 1e-3, 1e-3).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_qr_eig_non_square() {
        let a = Tensor::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]).unwrap();

        let _ = qr_eig(&a, 100, 1e-12).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_qr_eig_non_symmetric() {
        let a = Tensor::new(&[
            [1.0, 2.0],
            [0.0, 3.0],
        ]).unwrap();

        let result = qr_eig(&a, 100, 1e-12).unwrap();
        println!("{}", result.reconstruct().unwrap());
    }
}
