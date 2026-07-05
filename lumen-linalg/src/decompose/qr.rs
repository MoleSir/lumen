use lumen_core::{FloatDType, Tensor, Result};
use crate::LinalgResult;
use crate::view::prelude::*;

/// Result of QR decomposition of a matrix
///
/// Given a matrix `A` of size `n x m`, the QR decomposition factorizes it as:
/// ```text
/// A = Q * R
/// ```
/// where:
/// - `Q` is an orthogonal matrix (n x n)  
/// - `R` is an upper triangular matrix (n x m)
pub struct QrResult<T: FloatDType> {
    /// Orthogonal matrix Q
    pub q: Tensor<T>,
    /// Upper triangular matrix R
    pub r: Tensor<T>,
}

impl<T: FloatDType> QrResult<T> {
    /// Reconstruct the original matrix from its QR decomposition
    ///
    /// Computes:
    /// ```text
    /// A ≈ Q * R
    /// ```
    /// # Notes
    /// - Reconstruction may be approximate due to floating point arithmetic.
    pub fn reconstruct(&self) -> Result<Tensor<T>> {
        self.q.matmul(&self.r)
    }
}

/// Computes the QR decomposition of a matrix `A` using Householder reflections.
///
/// # Description
/// QR decomposition factorizes a matrix `A` into an orthogonal matrix `Q`
/// and an upper triangular matrix `R` such that `A = Q * R`.
///
/// This implementation uses **Householder reflections**, which are numerically
/// stable and suitable for both square and rectangular matrices.
///
/// # Parameters
/// - `a`: The input matrix `A` to decompose. Can be rectangular (`m x n`).
///
/// # Returns
/// `(Q, R)` where:
///   - `Q` is an orthogonal matrix of size `m x m` (`Q^T * Q = I`).
///   - `R` is an upper triangular matrix of size `m x n`.
///
/// # Notes
/// - This implementation produces a full `Q` of size `m x m`. The upper-left
///   `m x n` block can be used for a reduced QR decomposition if desired.
/// - The algorithm iteratively constructs Householder vectors to zero out
///   sub-diagonal elements column by column.
/// - `Q` is built as the product of Householder transformations applied to the identity matrix.
///
/// # Example
/// ```rust
/// # use lumen_core::Tensor;
/// let a = Tensor::new(&[
///     [12.0, -51.0, 4.0],
///     [6.0, 167.0, -68.0],
///     [-4.0, 24.0, -41.0],
/// ]).unwrap();
/// let result = lumen_linalg::qr(&a).unwrap();
/// // Now a ≈ Q * R
/// // Q is orthogonal, R is upper triangular
/// ```
pub fn qr<T: FloatDType>(mat: &Tensor<T>) -> LinalgResult<QrResult<T>> {
    let (m, n) = mat.dims2()?;
    let r = mat.copy()?; // (m, n)
    let q = Tensor::<T>::eye(m)?; // (m, m)
    
    {
        matrix_view_mut!(r);
        matrix_view_mut!(q);

        for k in 0..n {
            let x_arr = Tensor::<T>::zeros(m - k)?;
            vector_view_mut!(x = x_arr);
            
            for i in 0..(m - k) {
                x.s(i, r.g(k + i, k));
            }
    
            // v
            let v_arr = x.copy()?;
            vector_view_mut!(v = v_arr);

            let sign = if x.g(0) >= T::zero() { T::one() } else { -T::one() };
            let norm_x = x.norm();
            v.s(0, v.g(0) + sign * norm_x);
    
            // H = I - 2 vv^T / (v^T v)
            let beta = (T::one() + T::one()) / v.dot(&v)?;
            for j in k..n {
                // r[k.., j] -= beta * v * (v^T r[k.., j])
                let mut proj = T::zero();
                for i in 0..v.len() {
                    proj += v.g(i) * r.g(k + i, j);
                }
                proj *= beta;
                for i in 0..v.len() {
                    r.s(k + i, j, r.g(k + i, j) - proj * v.g(i));
                }
            }
    
            // q[:, k..] -= q[:, k..] * beta * v * v^T
            for i in 0..m {
                let mut proj = T::zero();
                for j in 0..v.len() {
                    proj += q.g(i, k + j) * v.g(j);
                }
                proj *= beta;
                for j in 0..v.len() {
                    q.s(i, k + j, q.g(i, k + j) - proj * v.g(j));
                }
            }
        }        
    }

    Ok( QrResult { q, r })
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;

    #[test]
    fn test_qr_simple() {
        let a = Tensor::new(&[
            [12., -51., 4.],
            [6., 167., -68.],
            [-4., 24., -41.],
        ]).unwrap();

        let result = crate::qr(&a).unwrap();
        let (q, r) = (result.q, result.r);

        // 检查 A ≈ Q * R
        let a_rec = q.matmul(&r).unwrap();
        println!("{}", a_rec);
        assert!(a_rec.allclose(&a, 1e-6, 1e-6).unwrap());

        // 检查 Q 是否正交：Q^T Q ≈ I
        let qtq = q.transpose_last().unwrap().matmul(&q).unwrap();
        let i = Tensor::<f64>::eye(q.dims()[1]).unwrap();
        assert!(qtq.allclose(&i, 1e-6, 1e-6).unwrap());
    }

    #[test]
    fn test_qr_identity() {
        let a = Tensor::<f64>::eye(4).unwrap();
        let _ = crate::qr(&a).unwrap();
    }

    #[test]
    fn test_qr_rectangular() {
        // 4x3 矩阵
        let a = Tensor::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 10.],
            [1., 0., 0.],
        ]).unwrap();

        let result = crate::qr(&a).unwrap();
        let (q, r) = (result.q, result.r);

        // A ≈ Q * R
        let a_rec = q.matmul(&r).unwrap();
        assert!(a_rec.allclose(&a, 1e-6, 1e-6).unwrap());

        // Q^T Q ≈ I
        let qtq = q.transpose_last().unwrap().matmul(&q).unwrap();
        let m = q.dims()[1];
        let i = Tensor::<f64>::eye(m).unwrap();
        assert!(qtq.allclose(&i, 1e-6, 1e-6).unwrap());
    }
}
