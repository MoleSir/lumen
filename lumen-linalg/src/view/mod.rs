pub mod prelude;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::sync::{RwLockReadGuard, RwLockWriteGuard};
use lumen_core::{FloatDType, Layout, NumDType, Storage, Tensor, WithDType};
use crate::{LinalgError, LinalgResult};

// ===================================================================================== //
//                                 Guards
// ===================================================================================== //

pub fn tensor_read_guard<T: WithDType>(t: &Tensor<T>) -> lumen_core::Result<TensorReadGuard<T>> {
    let guard = t.storage_read()?;
    Ok(TensorReadGuard { guard, layout: t.layout() })
}

pub fn tensor_write_guard<T: WithDType>(t: &Tensor<T>) -> lumen_core::Result<TensorWriteGuard<T>> {
    let guard = t.storage_write()?;
    Ok(TensorWriteGuard { guard, layout: t.layout() })
}

pub struct TensorReadGuard<'a, T: WithDType> {
    guard: RwLockReadGuard<'a, Storage<T>>,
    layout: &'a Layout,
}

pub struct TensorWriteGuard<'a, T: WithDType> {
    guard: RwLockWriteGuard<'a, Storage<T>>,
    layout: &'a Layout,
}

impl<'a, T: WithDType> TensorReadGuard<'a, T> {
    pub fn as_matrix(&self) -> lumen_core::Result<MatrixView<'_, T, &[T]>> {
        let (rows, cols) = self.layout.shape().dims2()?;
        Ok(MatrixView {
            data: self.guard.data(),
            offset: self.layout.start_offset(),
            stride_r: self.layout.stride()[0],
            stride_c: self.layout.stride()[1],
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    pub fn as_vector(&self) -> lumen_core::Result<VectorView<'_, T, &[T]>> {
        let len = self.layout.shape().dims1()?;
        Ok(VectorView {
            data: self.guard.data(),
            offset: self.layout.start_offset(),
            stride: self.layout.stride()[0],
            len,
            _marker: PhantomData,
        })
    }
}

impl<'a, T: WithDType> TensorWriteGuard<'a, T> {
    pub fn as_matrix_mut(&mut self) -> lumen_core::Result<MatrixView<'_, T, &mut [T]>> {
        let (rows, cols) = self.layout.shape().dims2()?;
        Ok(MatrixView {
            data: self.guard.data_mut(),
            offset: self.layout.start_offset(),
            stride_r: self.layout.stride()[0],
            stride_c: self.layout.stride()[1],
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    pub fn as_vector_mut(&mut self) -> lumen_core::Result<VectorView<'_, T, &mut [T]>> {
        let len = self.layout.shape().dims1()?;
        Ok(VectorView {
            data: self.guard.data_mut(),
            offset: self.layout.start_offset(),
            stride: self.layout.stride()[0],
            len,
            _marker: PhantomData,
        })
    }
}

// ===================================================================================== //
//                                 Unified Views
// ===================================================================================== //

pub struct VectorView<'a, T: WithDType, S> {
    pub(crate) data: S,
    pub(crate) offset: usize,
    pub(crate) stride: usize,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<&'a T>,
}

pub struct MatrixView<'a, T: WithDType, S> {
    pub(crate) data: S,
    pub(crate) offset: usize,
    pub(crate) stride_r: usize,
    pub(crate) stride_c: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _marker: PhantomData<&'a T>,
}

pub type Vector<'a, T> = VectorView<'a, T, &'a [T]>;
pub type VectorMut<'a, T> = VectorView<'a, T, &'a mut [T]>;
pub type Matrix<'a, T> = MatrixView<'a, T, &'a [T]>;
pub type MatrixMut<'a, T> = MatrixView<'a, T, &'a mut [T]>;

// ===================================================================================== //
//                                 Vector Implementations
// ===================================================================================== //

impl<'a, T: WithDType, S: AsRef<[T]>> VectorView<'a, T, S> {
    #[inline] 
    pub fn len(&self) -> usize { self.len }
    
    #[inline] 
    pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline]
    pub fn g(&self, index: usize) -> T {
        self.data.as_ref()[self.offset + index * self.stride]
    }

    pub fn copy(&self) -> lumen_core::Result<Tensor<T>> {
        let mut vec = Vec::with_capacity(self.len);
        for i in 0..self.len {
            vec.push(self.g(i));
        }
        Tensor::new(vec)
    }
}

impl<'a, T: FloatDType, S: AsRef<[T]>> VectorView<'a, T, S> {
    pub fn norm(&self) -> T {
        let mut s = T::ZERO;
        for i in 0..self.len {
            s += self.g(i).sqr();
        }
        s.sqrt()
    }
}

impl<'a, T: NumDType, S: AsRef<[T]>> VectorView<'a, T, S> {
    pub fn dot<S2: AsRef<[T]>>(&self, other: &VectorView<'_, T, S2>) -> LinalgResult<T> {
        check_vector_len(self, other, "dot")?;
        let mut sum = T::ZERO;
        for i in 0..self.len {
            sum = sum + self.g(i) * other.g(i);
        }
        Ok(sum)
    }
}

impl<'a, T: WithDType, S: AsMut<[T]> + AsRef<[T]>> VectorView<'a, T, S> {
    #[inline]
    pub fn s(&mut self, index: usize, val: T) {
        let idx = self.offset + index * self.stride;
        self.data.as_mut()[idx] = val;
    }

    pub fn copy_from<S2: AsRef<[T]>>(&mut self, other: &VectorView<'_, T, S2>) -> LinalgResult<()> {
        check_vector_len(self, other, "copy_from")?;
        for i in 0..self.len {
            self.s(i, other.g(i));
        }
        Ok(())
    }
}

impl<'a, T: NumDType, S: AsMut<[T]> + AsRef<[T]>> VectorView<'a, T, S> {
    pub fn scale(&mut self, alpha: T) {
        for i in 0..self.len {
            self.s(i, self.g(i) * alpha);
        }
    }

    pub fn axpy<S2: AsRef<[T]>>(&mut self, alpha: T, other: &VectorView<'_, T, S2>) -> LinalgResult<()> {
        check_vector_len(self, other, "axpy")?;
        for i in 0..self.len {
            self.s(i, self.g(i) + alpha * other.g(i));
        }
        Ok(())
    }
}

impl<'a, T: WithDType, S: AsRef<[T]>> Index<usize> for VectorView<'a, T, S> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data.as_ref()[self.offset + index * self.stride]
    }
}

impl<'a, T: WithDType, S: AsRef<[T]> + AsMut<[T]>> IndexMut<usize> for VectorView<'a, T, S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data.as_mut()[self.offset + index * self.stride]
    }
}

// ===================================================================================== //
//                                 Matrix Implementations
// ===================================================================================== //

impl<'a, T: WithDType, S: AsRef<[T]>> MatrixView<'a, T, S> {
    #[inline]
    pub fn g(&self, r: usize, c: usize) -> T {
        self.data.as_ref()[self.offset + r * self.stride_r + c * self.stride_c]
    }

    #[inline]
    pub fn row(&self, r: usize) -> VectorView<'_, T, &[T]> {
        VectorView {
            data: self.data.as_ref(),
            offset: self.offset + r * self.stride_r,
            stride: self.stride_c,
            len: self.cols,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn col(&self, c: usize) -> VectorView<'_, T, &[T]> {
        VectorView {
            data: self.data.as_ref(),
            offset: self.offset + c * self.stride_c,
            stride: self.stride_r,
            len: self.rows,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    pub fn eqal<S2: AsRef<[T]>>(&self, other: &MatrixView<'_, T, S2>) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.g(r, c) != other.g(r, c) {
                    return false;
                }
            }
        }
        true
    }

    pub fn copy(&self) -> lumen_core::Result<Tensor<T>> {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                data.push(self.g(r, c));
            }
        }   
        Tensor::from_vec(data, (self.rows, self.cols))
    }
}

impl<'a, T: WithDType, S: AsRef<[T]> + Copy> MatrixView<'a, T, S> {
    pub fn transpose(&self) -> Self {
        Self {
            data: self.data, // &[T] 自动 Copy
            offset: self.offset,
            stride_r: self.stride_c,
            stride_c: self.stride_r,
            rows: self.cols,
            cols: self.rows,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T: WithDType, S: AsMut<[T]> + AsRef<[T]>> MatrixView<'a, T, S> {
    #[inline]
    pub fn s(&mut self, r: usize, c: usize, val: T) {
        let idx = self.offset + r * self.stride_r + c * self.stride_c;
        self.data.as_mut()[idx] = val;
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> VectorView<'_, T, &mut [T]> {
        VectorView {
            data: self.data.as_mut(),
            offset: self.offset + r * self.stride_r,
            stride: self.stride_c,
            len: self.cols,
            _marker: PhantomData,
        }
    }

    pub fn swap_rows(&mut self, r1: usize, r2: usize) -> LinalgResult<()> {
        if r1 == r2 { return Ok(()); }
        for c in 0..self.cols {
            let t = self.g(r1, c);
            let val2 = self.g(r2, c);
            self.s(r1, c, val2);
            self.s(r2, c, t);
        }
        Ok(())
    }

    pub fn swap_rows_partial(&mut self, r1: usize, r2: usize, mxa_c: usize) -> LinalgResult<()> {
        if r1 == r2 { return Ok(()); }
        for c in 0..mxa_c {
            let t = self.g(r1, c);
            let val2 = self.g(r2, c);
            self.s(r1, c, val2);
            self.s(r2, c, t);
        }
        Ok(())
    }
}

impl<'a, T: NumDType, S: AsMut<[T]> + AsRef<[T]>> MatrixView<'a, T, S> {
    pub fn scale_row(&mut self, r: usize, alpha: T) -> LinalgResult<()> {
        for c in 0..self.cols {
            let val = self.g(r, c) * alpha;
            self.s(r, c, val);
        }
        Ok(())
    }

    pub fn axpy_row(&mut self, r1: usize, r2: usize, alpha: T) -> LinalgResult<()> {
        if r1 == r2 { return Ok(()); }
        for c in 0..self.cols {
            let val = self.g(r1, c) + alpha * self.g(r2, c);
            self.s(r1, c, val);
        }
        Ok(())
    }
}

impl<'a, T: WithDType, S: AsRef<[T]>> Index<(usize, usize)> for MatrixView<'a, T, S> {
    type Output = T;
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.data.as_ref()[self.offset + r * self.stride_r + c * self.stride_c]
    }
}

impl<'a, T: WithDType, S: AsRef<[T]> + AsMut<[T]>> IndexMut<(usize, usize)> for MatrixView<'a, T, S> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut Self::Output {
        &mut self.data.as_mut()[self.offset + r * self.stride_r + c * self.stride_c]
    }
}

fn check_vector_len<T: WithDType, S1: AsRef<[T]>, S2: AsRef<[T]>>(
    v1: &VectorView<T, S1>, 
    v2: &VectorView<T, S2>,
    op: &'static str,
) -> crate::LinalgResult<()> {
    if v1.len != v2.len {
        Err(LinalgError::VectorLenMismatch { len1: v1.len, len2: v2.len, op })?;
    }
    Ok(())
}
