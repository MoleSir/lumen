use lumen_core::{Tensor, NumDType};
use super::LinalgResult;
use crate::view::prelude::*;

// pub fn dot<T: NumDType>(a: &Tensor<T>, b: &Tensor<T>) -> LinalgResult<T> {
//     let a = a.vector_view_unsafe()?;
//     let b = b.vector_view_unsafe()?;
    

//     if a.len() != b.len() {
//         Err(LinalgError::VectorLenMismatch { len1: a.len(), len2: b.len(), op: "dot" })?;
//     }

//     unsafe {
//         let result = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<T>();
//         Ok(result)   
//     }
// }

// pub fn outer<T: NumDType>(a: &Tensor<T>, b: &Tensor<T>) -> LinalgResult<Tensor<T>> {
//     let a = a.vector_view_unsafe()?;
//     let b = b.vector_view_unsafe()?;

//     let m = a.len();
//     let n = b.len();

//     unsafe {
//         let res_arr = Tensor::zeros((m, n))?;
//         let mut res = res_arr.matrix_view_unsafe().unwrap();
//         for i in 0..m {
//             for j in 0..n {
//                 let v = a.g(i) * b.g(j);
//                 res.s(i, j, v);
//             }
//         }
//         Ok(res_arr)

//     }
// }

// pub fn matmul<T: NumDType>(a: &Tensor<T>, b: &Tensor<T>) -> LinalgResult<Tensor<T>> {
//     let a = a.matrix_view_unsafe()?;
//     let b = b.matrix_view_unsafe()?; 

//     let (m, k1) = a.shape();
//     let (k2, n) = b.shape();

//     if k1 != k2 {
//         Err(LinalgError::MatmulShapeMismatch { lhs: a.shape(), rhs: b.shape() })?;
//     }
//     let z = k1;

//     unsafe {
//         let res_arr = Tensor::<T>::zeros((m, n))?;
//         let mut res = res_arr.matrix_view_unsafe().unwrap();
//         for i in 0..m {
//             for j in 0..n {
//                 let mut sum = T::zero();
//                 for k in 0..z {
//                     sum += a.g(i, k) * b.g(k, j);
//                 }
//                 res[(i, j)] = sum;
//             }
//         }
//         Ok(res_arr)
//     }

// }

// pub fn mat_mul_vec<T: NumDType>(mat: &Tensor<T>, vec: &Tensor<T>) -> LinalgResult<Tensor<T>> {
//     let mat = mat.matrix_view_unsafe()?;
//     let vec = vec.vector_view_unsafe()?;

//     let (m, k1) = mat.shape();
//     let k2 = vec.len();

//     if k1 != k2 {
//         Err(LinalgError::MatMulVecShapeMismatch { shape: mat.shape(), len: vec.len() })?;
//     }
//     let k = k1; // (m, k) @ (k) = (m)

//     unsafe  {
//         let res_arr = Tensor::<T>::zeros(m)?;
//         let mut res = res_arr.vector_view_unsafe().unwrap();
//         for i in 0..m {
//             let mut sum = T::zero();
//             for j in 0..k {
//                 sum += mat.g(i, j) * vec.g(j);
//             }
//             res.s(i, sum);
//         }

//         Ok(res_arr)
//     }
// }

// pub fn vec_mul_mat<T: FloatDType>(vec: &Tensor<T>, mat: &Tensor<T>) -> LinalgResult<Tensor<T>> {
//     let vec = vec.vector_view_unsafe()?;
//     let mat = mat.matrix_view_unsafe()?;

//     let k1 = vec.len();
//     let (k2, n) = mat.shape();

//     if k1 != k2 {
//         Err(LinalgError::VecMulMatShapeMismatch { len: vec.len(), shape: mat.shape() })?;
//     }
//     let k = k1; // (1, k) @ (k, n) = (1, n)

//     unsafe {
//         let res_arr = Tensor::<T>::zeros(n)?;
//         let mut res = res_arr.vector_view_unsafe().unwrap();
//         for i in 0..n {
//             let mut sum = T::zero();
//             for j in 0..k {
//                 sum += vec.g(j) * mat.g(j, i);
//             }
//             res.s(i, sum);
//         }

//         Ok(res_arr)
//     }

// }

// pub fn trace<T: NumDType>(mat: &Tensor<T>) -> LinalgResult<T> {
//     let mat = mat.matrix_view_unsafe()?;
//     let (m, n) = mat.shape();
//     if m != n {
//         Err(LinalgError::ExpectMatrixSquare { shape: mat.shape(), op: "trace" })?;
//     }
    
//     unsafe {
//         let t = (0..m).into_iter()
//         .map(|i| mat.g(i, i))
//         .product::<T>();

//         Ok(t)
//     }
// }

// pub fn is_square<T: NumDType>(mat: &Tensor<T>) -> LinalgResult<bool> {
//     let mat = mat.matrix_view_unsafe()?;
//     let (m, n) = mat.shape();
//     Ok(m == n)
// }

pub fn is_symmetric<T: NumDType>(mat: &Tensor<T>) -> LinalgResult<bool> {
    matrix_view!(mat);
    let mat_trans = mat.transpose();
    Ok( mat.eqal(&mat_trans) )
}

// pub fn check_square<T: NumDType>(mat: &Tensor<T>, op: &'static str) -> LinalgResult<()> {
//     let mat = mat.matrix_view_unsafe()?;
//     let (m, n) = mat.shape();
//     if m != n {
//         Err(LinalgError::ExpectMatrixSquare { shape: mat.shape(), op })?
//     } else {
//         Ok(())
//     }
// }

// #[cfg(test)]
// mod test {
//     use crate::{Tensor, linalg};

//     #[test]
//     fn test_dot_basic() {
//         let a = Tensor::new(&[1., 2., 3.]).unwrap();
//         let b = Tensor::new(&[4., 5., 6.]).unwrap();
//         let res = linalg::dot(&a, &b).unwrap();
//         // 1*4 + 2*5 + 3*6 = 32
//         assert_eq!(res, 32.);
//     }

//     #[test]
//     fn test_dot_zero_vector() {
//         let a = Tensor::new(&[0., 0., 0.]).unwrap();
//         let b = Tensor::new(&[1., 2., 3.]).unwrap();
//         let res = linalg::dot(&a, &b).unwrap();
//         assert_eq!(res, 0.);
//     }

//     #[test]
//     fn test_dot_negative() {
//         let a = Tensor::new(&[1., -2., 3.]).unwrap();
//         let b = Tensor::new(&[-1., 4., -3.]).unwrap();
//         let res = linalg::dot(&a, &b).unwrap();
//         // 1*-1 + -2*4 + 3*-3 = -1 -8 -9 = -18
//         assert_eq!(res, -18.);
//     }

//     #[test]
//     fn test_dot_incompatible_size() {
//         let a = Tensor::new(&[1., 2., 3.]).unwrap();
//         let b = Tensor::new(&[4., 5.]).unwrap();
//         let res = linalg::dot(&a, &b);
//         assert!(res.is_err());
//     }


//     #[test]
//     fn test_matmul_basic() {
//         let a = Tensor::new(&[
//             [1., 2.],
//             [3., 4.],
//         ]).unwrap();
//         let b = Tensor::new(&[
//             [5., 6.],
//             [7., 8.],
//         ]).unwrap();

//         let c = linalg::matmul(&a, &b).unwrap();
//         let expected = Tensor::new(&[
//             [19., 22.],
//             [43., 50.],
//         ]).unwrap();

//         assert!(c.allclose(&expected, 1e-6, 1e-6));
//     }

//     #[test]
//     fn test_matmul_incompatible() {
//         let a = Tensor::new(&[[1., 2.]]).unwrap(); // 1x2
//         let b = Tensor::new(&[[3., 4.], [5., 6.], [7., 8.]]).unwrap(); // 3x2
//         let res = linalg::matmul(&a, &b);
//         assert!(res.is_err());
//     }

//     #[test]
//     fn test_trace_square() {
//         let a = Tensor::new(&[
//             [1., 2.],
//             [3., 4.],
//         ]).unwrap();
//         let t = linalg::trace(&a).unwrap();
//         assert_eq!(t, 1. * 4.); // 注意 trace 是 sum 对角线，如果你之前实现是 sum，这里要改
//     }

//     #[test]
//     fn test_trace_non_square() {
//         let a = Tensor::new(&[
//             [1., 2., 3.],
//             [4., 5., 6.],
//         ]).unwrap();
//         let t = linalg::trace(&a);
//         assert!(t.is_err());
//     }

//     #[test]
//     fn test_is_square_true() {
//         let a = Tensor::new(&[
//             [1., 2.],
//             [3., 4.],
//         ]).unwrap();
//         let res = linalg::is_square(&a).unwrap();
//         assert!(res);
//     }

//     #[test]
//     fn test_is_square_false() {
//         let a = Tensor::new(&[
//             [1., 2., 3.],
//             [4., 5., 6.],
//         ]).unwrap();
//         let res = linalg::is_square(&a).unwrap();
//         assert!(!res);
//     }
// }
