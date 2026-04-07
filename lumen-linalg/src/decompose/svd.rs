use rand_distr::{Distribution, StandardNormal};
use crate::{linalg::{self, Norm}, view::MatrixViewUsf, FloatDType, Tensor, Result};

/// Result of Singular Value Decomposition (SVD) of a matrix
///
/// Given a matrix `A` of size `n x m`, the SVD factorizes it as:
/// ```text
/// A = U * Σ * Vᵀ
/// ```
/// where:
/// - `U` is an orthogonal matrix (n x n)
/// - `Σ` (sigma) is a diagonal matrix containing the singular values
/// - `V` is an orthogonal matrix (m x m)
pub struct SvdResult<T: FloatDType> {
    /// Left orthogonal matrix `U` of size (n x n)
    /// Columns of `U` are the left singular vectors of the original matrix.
    pub u: Tensor<T>,

    /// Singular values of the matrix, stored as a vector
    /// Represents the diagonal entries of Σ in `A = U * Σ * Vᵀ`.
    pub sigmas: Vec<T>,

    /// Right orthogonal matrix `V` of size (m x m)
    /// Columns of `V` are the right singular vectors of the original matrix.
    pub v: Tensor<T>,
}

impl<T: FloatDType> SvdResult<T> {
    /// Reconstruct the original matrix from its SVD decomposition
    ///
    /// Computes:
    /// ```text
    /// A ≈ U * diag(sigmas) * Vᵀ
    /// ```
    /// # Notes
    /// - The reconstruction may be approximate due to floating point errors.
    pub fn reconstruct(&self) -> Result<Tensor<T>> {
        let sigma = Tensor::diag(&self.sigmas)?;
        let rec = self.u.matmul(&sigma)?.matmul(&self.v.transpose_last()?)?;
        Ok(rec)
    }
}

/// Computes the **Singular Value Decomposition (SVD)** of a matrix.
///
/// Given an input matrix `A` of shape `(n, m)`, this function factorizes it as:
///
/// ```text
/// A = U * Σ * Vᵀ
/// ```
///
/// where:
/// - `U` is an orthogonal matrix of shape `(n, n)`, whose columns are the **left singular vectors**.
/// - `Σ` is a diagonal matrix of shape `(n, m)` containing the **singular values** (non-negative, sorted in descending order).
/// - `V` is an orthogonal matrix of shape `(m, m)`, whose columns are the **right singular vectors**.
///
/// # Returns
/// A [`SvdResult<T>`] struct containing:
/// - `u`: the left singular vectors `U`
/// - `sigmas`: the singular values (as a vector, instead of full Σ matrix)
/// - `v`: the right singular vectors `V`
///
/// # Example
/// ```rust
/// # use numrst::{Tensor, linalg};
/// let a = Tensor::new(&[
///     [3.0, 1.0, 1.0],
///     [-1.0, 3.0, 1.0],
/// ]).unwrap();
///
/// let svd_result = linalg::svd(&a).unwrap();
///
/// // Singular values
/// println!("Σ = {:?}", svd_result.sigmas);
///
/// // Reconstruct A from U, Σ, V
/// let a_reconstructed = svd_result.reconstruct().unwrap();
/// println!("Reconstructed A = {:?}", a_reconstructed);
/// ```
///
/// # References
/// - [Singular Value Decomposition (Wikipedia)](https://en.wikipedia.org/wiki/Singular_value_decomposition)
/// - ["Implementing SVD in Rust"](https://medium.com/gaussian-machine/implementing-svd-algorithm-in-rust-ac1489eb7ca4)
pub fn svd<T: FloatDType>(arr: &Tensor<T>) -> Result<SvdResult<T>> {
    let a = arr.matrix_view_unsafe()?;
    let (n, m) = a.shape(); 
    
    unsafe {
        let (a1_arr, do_transpose) = if n < m {
            (a.transpose().copy(), true)
        } else {
            (a.copy(), false)
        };
        let mut a1 = a1_arr.matrix_view_unsafe()?;
        let (n, m) = a1.shape(); 
        assert!(n >= m);
        
        let hr = householder_reflection_bidiagonalization(&a1_arr)?;
        let r = n.min(m);

        let hr_0 = hr.0.matrix_view_unsafe()?;
        let hr_1 = hr.1.matrix_view_unsafe()?;
        let hr_2 = hr.2.matrix_view_unsafe()?;
        let mut u = hr_0.slice(0..r, 0..n)?;
        a1 = hr_1.slice(0..r, 0..r)?;
        let mut v = hr_2.slice(0..m, 0..r)?;

        let eps = T::from_f64(1.0e-7);

        loop {
            for i in 0..r-1 {
                if a1[(i, i+1)].abs() < eps*(a1[(i, i)].abs() + a1[(i+1, i+1)].abs()) {
                    a1[(i, i+1)] = T::zero();
                }
            }
    
            let mut q = 0;
            for i in (0..r-1).rev() {
                if a1[(i, i+1)].abs() > T::zero() {
                    q = i+1;
                    break;
                }
            }


            if q == 0 {
                break;
            }
    
            let mut p = 0;
            for i in (0..q).rev() {
                if a1[(i, i+1)].abs() == T::zero() {
                    p = i+1;
                    break;
                }
            }
    
            let mut flag: bool = false;


            for i in p..q {
                if a1[(i, i)].abs() == T::zero() {
                    flag = true;
                    for j in i+1..r {
                        let b = givens_left_rotation(&a1, i+1, j, true)?;
                        
                        givens_left_rotation_multiply(&mut a1, b.0, b.1, i+1, 0, r-1)?;
                        givens_left_rotation_multiply(&mut u, b.0, b.1, i+1, 0, n-1)?;
                    }
                }
            }
    
            if !flag && p < q {
                golub_kahan(&mut a1, &mut u, &mut v, n, m, p, q)?;
            }
        }
        
        if do_transpose {
            Ok(SvdResult { 
                u: v.copy(), 
                sigmas: a1.diag().collect(),
                v: u.transpose().copy(),
            })
            // Ok((v.copy(), a1.transpose().copy(), u.copy()))
        } else {
            Ok(SvdResult { 
                u: u.transpose().copy(),
                sigmas: a1.diag().collect(),
                v: v.copy(),
            })
        }            
    }
}

/// `SvdLowrankResult` represents the result of a low-rank SVD decomposition of a matrix.
pub struct SvdLowrankResult<T: FloatDType> {
    /// A `Vec` of left singular vectors (`u_i`), each of shape `(m,)`.
    pub us: Vec<Tensor<T>>,

    /// A `Vec` of singular values (`σ_i`), of length `r`.
    pub sigmas: Vec<T>,

    /// A `Vec` of right singular vectors (`v_i`), each of shape `(n,)`.
    pub vs: Vec<Tensor<T>>,
}

impl<T: FloatDType> SvdLowrankResult<T> {
    /// Reconstructs the approximate original matrix as `U * Σ * V^T`.
    pub fn reconstruct(&self) -> Result<Tensor<T>> {
        let u = self.u()?;
        let sigma = self.sigma()?;
        let v = self.v()?;
        let rec = u.matmul(&sigma)?.matmul(&v.transpose_last()?)?;
        Ok(rec)
    }

    /// Stacks the left singular vectors into a matrix of shape `(m, r)`.
    pub fn u(&self) -> Result<Tensor<T>> {
        Tensor::stack(&self.us, 1)
    }

    /// Constructs a diagonal matrix of shape `(r, r)` from the singular values.
    pub fn sigma(&self) -> Result<Tensor<T>> {
        Tensor::diag(&self.sigmas)
    }

    /// Stacks the right singular vectors into a matrix of shape `(n, r)`.
    pub fn v(&self) -> Result<Tensor<T>> {
        Tensor::stack(&self.vs, 1)
    }
}

pub fn svd_lowrank<T: FloatDType>(a: &Tensor<T>, rank: usize, num_iters: usize) -> Result<SvdLowrankResult<T>> 
where 
    StandardNormal: Distribution<T>
{
    let (m, n) = a.dims2()?;
    let k = m.min(n).min(rank);

    let mut vs = vec![];
    let mut ss = vec![];
    let mut us = vec![];

    let a_copy = a.copy(); // (m, n)

    for _ in 0..k {
        // 1. right vector 
        // RS = A.T @ A === (n, m) @ (m, n) = (n, n)
        let rs = a_copy.transpose_last()?.matmul(&a_copy)?;
        
        // power_iteration get eig value and vector
        let (_, eig_v) = power_iteration(&rs, num_iters)?; // (n,)
        eig_v.div_assign(linalg::norm(&eig_v, Norm::Frobenius))?;
        vs.push(eig_v.clone());

        // 2. sigma and left vector
        let av = linalg::mat_mul_vec(a, &eig_v)?; // (m. n) @ (n) = (m,)
        let sigma = linalg::norm(&av, Norm::Frobenius);
        // let eig_u = av.div(sigma)?; // (m,)
        av.div_assign(sigma)?;
        let eig_u = av;

        us.push(eig_u.clone());  
        ss.push(sigma);

        // 3. update a copy
        // (m, 1) @ (1, n) => (m, n)
        let uv = linalg::outer(&eig_u, &eig_v)?;
        uv.mul_assign(sigma)?; // (m, n)
        a_copy.sub_assign(&uv)?; // (m, n)
    }

    return Ok( SvdLowrankResult { us, sigmas: ss, vs} )
}

fn power_iteration<T>(mat: &Tensor<T>, num_iters: usize) -> Result<(T, Tensor<T>)> 
where 
    T: FloatDType,
    StandardNormal: Distribution<T>
{
    let (m, n) = mat.dims2()?; // (n, n)
    assert_eq!(m, n);
    let mut b = Tensor::<T>::randn(T::from_f64(0.), T::from_f64(1.), (n,))?; // (n,)

    for _ in 0..num_iters {
        // (n, n) @ (n,) = (n,)
        b = linalg::mat_mul_vec(mat, &b)?;
        let norm_b = linalg::norm(&b, Norm::Frobenius);
        b.div_assign(norm_b)?;
    }

    // (n) @ (n, n) @ (n)
    let eig_value = linalg::vec_mul_mat(&b, mat)?; // (n,)
    let eig_value = linalg::dot(&eig_value, &b)?;
    let eig_vector = b;

    Ok((eig_value, eig_vector)) // ( (), (n,) )
}

fn householder_reflection_bidiagonalization<T: FloatDType>(arr: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    let av = arr.matrix_view_unsafe()?;
    let (n, m) = av.shape();

    let q_lt_arr = Tensor::<T>::eye(n)?; // (n, n)
    let mut q_lt = q_lt_arr.matrix_view_unsafe().unwrap();
    let q_rt_arr = Tensor::<T>::eye(m)?; // (m, m)
    let mut q_rt = q_rt_arr.matrix_view_unsafe().unwrap();

    let r_arr = arr.copy();
    let mut r = r_arr.matrix_view_unsafe().unwrap();

    for i in 0..n.min(m) {
        let n1 = n - i;
        let mut nm = T::zero();

        let u_arr = Tensor::<T>::zeros(n1)?;
        let mut u = u_arr.vector_view_unsafe().unwrap();

        for i1 in i..n {
            nm += r[(i1, i)].powi(2);
            u[i1 - i] = r[(i1, i)];
        }
        
        u[0] -= r[(i, i)].signum() * nm.sqrt();
        let z = (nm - r[(i, i)] * r[(i, i)] + u[0] * u[0]).sqrt();
        

        if z > T::zero() {
            for i1 in 0..n1 {
                u[i1] = u[i1] / z;
            }

            let r1_arr = Tensor::<T>::zeros(m-i)?;
            let mut r1 = r1_arr.vector_view_unsafe().unwrap();

            for i1 in i..n {
                for j1 in i..m {
                    r1[j1-i] += u[i1-i] * r[(i1, j1)];
                }
            }

            for i1 in i..n {
                for j1 in i..m {
                    r[(i1, j1)] -= T::from_f64(2.0) * u[i1-i] * r1[j1-i];
                }
            }

            let q1_arr = Tensor::<T>::zeros(n)?;
            let mut q1 = q1_arr.vector_view_unsafe()?;
            
            for i1 in i..n {
                for j1 in 0..n {
                    q1[j1] += u[i1-i]*q_lt[(i1, j1)];
                }
            }

            for i1 in i..n {
                for j1 in 0..n {
                    q_lt[(i1, j1)] -= T::from_f64(2.0) * u[i1-i] * q1[j1];
                }
            }
        }

        if m-i-1 > 0 {
            let n1 = m-i-1;

            let mut nm = T::zero();
            let u_arr = Tensor::<T>::zeros(n1)?;
            let mut u = u_arr.vector_view_unsafe().unwrap();

            for j1 in i+1..m {
                nm += r[(i, j1)]*r[(i, j1)];
                u[j1-i-1] = r[(i, j1)];
            }

            u[0] -= r[(i, i+1)].signum() *nm.sqrt();
            let z = (nm-r[(i, i+1)]*r[(i, i+1)]+u[0]*u[0]).sqrt();

            if z > T::zero() {
                for i1 in 0..n1 {
                    u[i1] = u[i1]/z;
                }

                let r1_arr = Tensor::<T>::zeros(n-i)?;
                let mut r1 = r1_arr.vector_view_unsafe().unwrap();

                for i1 in i..n {
                    for j1 in i+1..m {
                        r1[i1-i] += u[j1-i-1] * r[(i1, j1)];
                    }
                }

                for i1 in i..n {
                    for j1 in i+1..m {
                        r[(i1, j1)] -= T::from_f64(2.0) * u[j1-i-1] * r1[i1-i];
                    }
                }
                
                let q1_arr = Tensor::<T>::zeros(m)?;
                let mut q1 = q1_arr.vector_view_unsafe()?;

                for i1 in 0..m {
                    for j1 in i+1..m {
                        q1[i1] += u[j1-i-1]*q_rt[(i1, j1)];
                    }
                }

                for i1 in 0..m {
                    for j1 in i+1..m {
                        q_rt[(i1, j1)] -= T::from_f64(2.0) * u[j1-i-1] * q1[i1];
                    }
                }
            }
        }

    }
    
    Ok((q_lt_arr, r_arr, q_rt_arr))
}

fn eigenvalue_bidiagonal<T: FloatDType>(a: &MatrixViewUsf<T>, i2: usize, j2:usize) -> Result<T> {
    let h = i2.min(j2) + 1;

    let mut d1 = T::zero();
    let mut d2 = T::zero();
    let mut d3 = T::zero();
    let mut d4 = T::zero();

    if h >= 3 {
        d1 = a[((h-3), h-2)];
    }

    if h >= 2 {
        d2 = a[((h-2), h-2)];
        d3 = a[((h-2), h-1)];
    }

    if h >= 1 {
        d4 = a[((h-1), h-1)];
    }

      
    let a1 = d2*d2 + d1*d1;
    let a2 = d2*d3;
    let a3 = d2*d3;
    let a4 = d4*d4 + d3*d3;

    let u = T::one();
    let b = -(a1 + a4);
    let c = a1 * a4 - a2 * a3;

    let v1 = (-b + (b*b- T::from_f64(4.0) *u*c).sqrt())/(T::from_f64(2.0)*u);
    let v2 = (-b - (b*b- T::from_f64(4.0) *u*c).sqrt())/(T::from_f64(2.0)*u);

    if (v1-a4).abs() < (v2-a4).abs() {
        Ok(v1)
    } else {
        Ok(v2)
    }        
}

fn givens_right_rotation<T: FloatDType>(a: &MatrixViewUsf<T>, i:usize, j:usize, flip:bool) -> Result<(T, T)> {
    let x = a[(i, j-1)];
    let y = a[(i, j)];
    let r = (x*x + y*y).sqrt();

    if flip {
        Ok((y / r, -x / r))
    } else {
        Ok((x/r, -y/r))
    }
}

fn givens_right_rotation_multiply<T: FloatDType>(a: &mut MatrixViewUsf<T>, c: T, s: T, j: usize, r1: usize, r2: usize) -> Result<()> {    
    for i1 in r1..r2+1 {
        let p = a[(i1, j-1)];
        let q = a[(i1, j)];
        a[(i1, j-1)] = c*p - s*q;
        a[(i1, j)] = s*p + c*q;
    }

    Ok(())
}

// Get (c, s) value for Givens left rotation matrix
fn givens_left_rotation<T: FloatDType>(
    a: &MatrixViewUsf<T>, i: usize, j: usize, flip: bool
) -> Result<(T, T)> {
    let x = a[(i - 1, j)];
    let y = a[(i, j)];
    let r = (x * x + y * y).sqrt();

    if r == T::zero() {
        // 避免除零，可以定义成单位旋转
        return Ok((T::one(), T::zero()));
    }

    if flip {
        Ok((y / r, -x / r))
    } else {
        Ok((x / r, -y / r))
    }
}

// Multiply matrix by Givens left rotation matrix
fn givens_left_rotation_multiply<T: FloatDType>(a: &mut MatrixViewUsf<T>, c: T, s: T, i: usize, c1: usize, c2: usize) -> Result<()> {
    for j1 in c1..=c2 {
        let p = a[(i - 1, j1)];
        let q = a[(i, j1)];
        a[(i - 1, j1)] = c * p - s * q;
        a[(i, j1)] = s * p + c * q;
    }

    Ok(())
}

fn golub_kahan<T: FloatDType>(a: &mut MatrixViewUsf<T>, l: &mut MatrixViewUsf<T>, r: &mut MatrixViewUsf<T>, n:usize, m:usize, i:usize, j:usize) -> Result<()> {
    let mu = eigenvalue_bidiagonal(a, j, j)?;
    
    let u = a[(i, i)];
    let v = a[(i, i+1)];

    a[(i, i)] = u*u-mu;
    a[(i, i+1)] = u*v;

    for k in i..j {
        let mut x;
        let mut y;

        if k > i {
            x = k-1;
            y = k+1; 
        }
        else {
            x = i;
            y = i+1; 
        }

        let b = givens_right_rotation(a, x, y, false)?;

        if k == i {
            a[(i, i)] = u;
            a[(i, i+1)] = v;
        }

        givens_right_rotation_multiply(a, b.0, b.1, y, i, j)?;
        givens_right_rotation_multiply(r, b.0, b.1, y, 0, m-1)?;

        if k > i {
            x = k+1;
            y = k;
        }
        else {
            x = i+1;
            y = i;
        }
            
        let b = givens_left_rotation(&a, x, y, false)?;

        givens_left_rotation_multiply(a, b.0, b.1, x, i, j)?;
        givens_left_rotation_multiply(l, b.0, b.1, x,0, n-1)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_svd_identity() {
        let a = Tensor::<f64>::eye(3).unwrap();
        let result = svd(&a).unwrap();

        // 检查奇异值
        for &sigma in result.sigmas.iter() {
            assert!((sigma - 1.0).abs() < 1e-10);
        }

        // 检查重构
        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-10, 1e-10));

        // 检查正交性
        assert!(check_orthogonal(&result.u, 1e-10));
        assert!(check_orthogonal(&result.v, 1e-10));
    }

    #[test]
    fn test_svd_diagonal_matrix() {
        let a = Tensor::new(&[
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 2.0],
        ]).unwrap();

        let result = svd(&a).unwrap();
        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-10, 1e-10));

        assert!(check_orthogonal(&result.u, 1e-10));
        assert!(check_orthogonal(&result.v, 1e-10));
    }

    #[test]
    fn test_svd_zeros() {
        let a: Tensor<f64> = Tensor::<f64>::zeros((4, 4)).unwrap();
        let _ = svd(&a).unwrap();
    }

    #[test]
    fn test_svd_rectangular_matrix_m_gt_n() {
        let a = Tensor::<f64>::randn(0.0, 1.0, (5, 3)).unwrap();
        let result = svd(&a).unwrap();

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));

        assert!(check_orthogonal(&result.u, 1e-6));
        assert!(check_orthogonal(&result.v, 1e-6));
    }

    #[test]
    fn test_svd_rectangular_matrix_m_lt_n() {
        let a = Tensor::<f64>::randn(0.0, 1.0, (3, 5)).unwrap();
        let result = svd(&a).unwrap();

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));

        assert!(check_orthogonal(&result.u, 1e-6));
        assert!(check_orthogonal(&result.v, 1e-6));
    }

    #[test]
    fn test_svd_rank_deficient() {
        let a = Tensor::new(&[
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]).unwrap();
        let result = svd(&a).unwrap();

        // 奇异值应有一个接近 0
        let zero_sigma = result.sigmas.iter().filter(|&&x| f64::abs(x) < 1e-5).count();
        assert!(zero_sigma >= 1);
    
        // 重构矩阵近似原矩阵
        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-7, 1e-7));

        // U、V 正交
        assert!(check_orthogonal(&result.u, 1e-6));
        assert!(check_orthogonal(&result.v, 1e-6));
    }

    #[test]
    fn test_svd_lowrank() {
        let a = Tensor::<f64>::randn(0.0, 1.0, (6, 4)).unwrap();

        for rank in 1..=4 {
            let result = svd_lowrank(&a, rank, 100).unwrap();
            let rec = result.reconstruct().unwrap();
            println!("{}", linalg::norm(&(&rec - &a), Norm::Frobenius));
        }
    }

    #[test]
    fn test_svd_lowrank_symmetric() {
        let a = Tensor::<f64>::randn(0.0, 1.0, (5, 5)).unwrap();
        let a = &a + &a.transpose_last().unwrap(); // make symmetric
    
        let result = svd_lowrank(&a, 5, 100).unwrap();
        let rec = result.reconstruct().unwrap();
        let error = linalg::norm(&(&rec - &a), Norm::Frobenius);
        println!("symmetric matrix reconstruction error: {}", error);
        assert!(error < 1e-10);
    }

    #[test]
    fn test_svd_lowrank_rank1() {
        let u = Tensor::<f64>::randn(0.0, 1.0, (6,)).unwrap();
        let v = Tensor::<f64>::randn(0.0, 1.0, (4,)).unwrap();
        let a = linalg::outer(&u, &v).unwrap(); // rank-1 matrix
    
        let result = svd_lowrank(&a, 1, 100).unwrap();
        let rec = result.reconstruct().unwrap();
        let error = linalg::norm(&(&rec - &a), Norm::Frobenius);
        println!("rank-1 reconstruction error: {}", error);
        assert!(error < 1e-12);
    }

    #[test]
    fn test_svd_lowrank_tall() {
        let a = Tensor::<f64>::randn(0.0, 1.0, (10, 3)).unwrap();
    
        let result = svd_lowrank(&a, 3, 100).unwrap();
        let rec = result.reconstruct().unwrap();
        let error = linalg::norm(&(&rec - &a), Norm::Frobenius);
        println!("tall matrix reconstruction error: {}", error);
        assert!(error < 1e-10);
    }
    
    #[test]
    fn test_svd_lowrank_wide() {
        let a = Tensor::<f64>::randn(0.0, 1.0, (3, 8)).unwrap();
    
        let result = svd_lowrank(&a, 3, 100).unwrap();
        let rec = result.reconstruct().unwrap();
        let error = linalg::norm(&(&rec - &a), Norm::Frobenius);
        println!("wide matrix reconstruction error: {}", error);
        assert!(error < 1e-10);
    }    
    
    fn check_orthogonal<T: FloatDType>(mat: &Tensor<T>, tol: T) -> bool {
        let mat_t = mat.transpose_last().unwrap();
        let prod = mat_t.matmul(mat).unwrap();
        let n = prod.dims2().unwrap().0;
        let id = Tensor::<T>::eye(n).unwrap();
        prod.allclose(&id, tol.to_f64(), tol.to_f64())
    }
}
