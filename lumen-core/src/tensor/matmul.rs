use std::ops::Deref;
use num_traits::Zero;
use crate::{AutogradMetaT, Error, Layout, NumDType, Result, Shape, Storage, StorageIndices};
use super::Tensor;

impl<T: NumDType> Tensor<T> {
    /// Returns the matrix-multiplication of the input tensor with the other provided tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A tensor with dimensions `b1, b2, ..., bi, m, k`.
    /// * `rhs` - A tensor with dimensions `b1, b2, ..., bi, k, n`.
    ///
    /// The resulting tensor has dimensions `b1, b2, ..., bi, m, n`.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let a_dims = self.shape().dims();
        let b_dims = rhs.shape().dims();

        let dim = a_dims.len();

        if dim < 2 || b_dims.len() != dim {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            })?
        }

        let m = a_dims[dim - 2];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];
        let n = b_dims[dim - 1];

        let c_shape = Shape::from(&a_dims[..dim - 2]).extend(&[m, n]);
        if c_shape.element_count() == 0 || k == 0 {
            return Self::zeros(c_shape);
        }
        let batching: usize = a_dims[..dim - 2].iter().product();
        let batching_b: usize = b_dims[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            })?
        }

        // (..., m, k) @ (..., k, n)
        let c_storage = Self::do_matmul(
            self.storage_read()?.deref(),
            self.layout(),
            rhs.storage_read()?.deref(),
            rhs.layout(),
            (batching, m, n, k),
        );

        let meta = T::AutogradMeta::on_matmul_op(self, rhs);
        Ok(Self::from_storage(c_storage, c_shape, meta))
    }

    fn do_matmul(
        lhs: &Storage<T>, lhs_layout: &Layout, 
        rhs: &Storage<T>, rhs_layout: &Layout, 
        bmnk: (usize, usize, usize, usize)
    ) -> Storage<T> 
        where T: num_traits::Num + Copy + Zero
    {    
        let lhs_rank = lhs_layout.shape().rank();
        let rhs_rank = rhs_layout.shape().rank();
        let (bs, ms, ns, ks) = bmnk;
        let mns = ms * ns;
        let mut dst = vec![T::zero(); bs * mns];

        use std::any::TypeId;

        let l_stride_m = lhs_layout.stride()[lhs_rank - 2] as isize;
        let l_stride_k = lhs_layout.stride()[lhs_rank - 1] as isize;
        let r_stride_k = rhs_layout.stride()[rhs_rank - 2] as isize;
        let r_stride_n = rhs_layout.stride()[rhs_rank - 1] as isize;

        let type_id = TypeId::of::<T>();

        if type_id == TypeId::of::<f32>() {
            for b in 0..bs {
                let start_index = b*mns;
                let dst_slice = &mut dst[start_index..start_index+mns];
                let l_batch_offset = Self::compute_batch_offset(b, lhs_layout);
                let r_batch_offset = Self::compute_batch_offset(b, rhs_layout);

                unsafe {
                    let lhs_ptr = lhs.data().as_ptr().add(l_batch_offset) as *const f32;
                    let rhs_ptr = rhs.data().as_ptr().add(r_batch_offset) as *const f32;
                    let dst_ptr = dst_slice.as_mut_ptr() as *mut f32;

                    matrixmultiply::sgemm(
                        ms, ks, ns, 
                        1.0, 
                        lhs_ptr, l_stride_m, l_stride_k, 
                        rhs_ptr, r_stride_k, r_stride_n, 
                        0.0, 
                        dst_ptr, ns as isize, 1
                    );
                }
            }
        }
        else if type_id == TypeId::of::<f64>() {
            for b in 0..bs {
                let start_index = b*mns;
                let dst_slice = &mut dst[start_index..start_index+mns];
                let l_batch_offset = Self::compute_batch_offset(b, lhs_layout);
                let r_batch_offset = Self::compute_batch_offset(b, rhs_layout);

                unsafe {
                    let lhs_ptr = lhs.data().as_ptr().add(l_batch_offset) as *const f64;
                    let rhs_ptr = rhs.data().as_ptr().add(r_batch_offset) as *const f64;
                    let dst_ptr = dst_slice.as_mut_ptr() as *mut f64;
                    
                    matrixmultiply::dgemm(
                        ms, ks, ns, 
                        1.0, 
                        lhs_ptr, l_stride_m, l_stride_k, 
                        rhs_ptr, r_stride_k, r_stride_n, 
                        0.0, 
                        dst_ptr, ns as isize, 1
                    );
                }
            }
        }
        else {
            let lhs_data = lhs.data();
            let rhs_data = rhs.data();

            // dst.par_chunks_mut(ms * ns).enumerate().for_each(|(b, dst_slice)| {
            (0..bs).for_each(|b| {
                let l_batch_offset = Self::compute_batch_offset(b, lhs_layout);
                let r_batch_offset = Self::compute_batch_offset(b, rhs_layout);
                let dst_slice = &mut dst[b*mns..b*mns+mns];

                for m in 0..ms {
                    for k in 0..ks {
                        let l_idx = l_batch_offset + (m as isize * l_stride_m + k as isize * l_stride_k) as usize;
                        let l_val = unsafe { *lhs_data.get_unchecked(l_idx) };
                        
                        let r_base = r_batch_offset + (k as isize * r_stride_k) as usize;
                        let dst_base = m * ns;

                        for n in 0..ns {
                            unsafe {
                                let r_idx = r_base + (n as isize * r_stride_n) as usize;
                                let r_val = *rhs_data.get_unchecked(r_idx);
                                
                                let dst_ptr = dst_slice.get_unchecked_mut(dst_base + n);
                                *dst_ptr = *dst_ptr + l_val * r_val;
                            }
                        }
                    }
                }
            });
        }
        
        Storage::new(dst)
    }

    fn compute_batch_offset(b: usize, layout: &Layout) -> usize {
        let rank = layout.shape().rank();
        let batch_dims = &layout.dims()[..rank - 2];
        let batch_strides = &layout.stride()[..rank - 2];
        
        let mut offset = layout.start_offset();
        let mut temp_b = b;
        
        for i in (0..batch_dims.len()).rev() {
            let idx = temp_b % batch_dims[i];
            offset += idx * batch_strides[i];
            temp_b /= batch_dims[i];
        }
        offset
    }
}

impl<T: NumDType> Tensor<T> {
    /// In-place version of matrix multiplication: self += lhs @ rhs
    pub fn add_matmul_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.check_implace_op()?;

        let a_dims = lhs.shape().dims();
        let b_dims = rhs.shape().dims();
        let c_dims = self.shape().dims();

        let dim = a_dims.len();

        // 1. 基础维度检查
        if dim < 2 || b_dims.len() != dim || c_dims.len() != dim {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "add_matmul_ (rank mismatch)",
            });
        }

        let m = a_dims[dim - 2];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];
        let n = b_dims[dim - 1];
        
        let self_m = c_dims[dim - 2];
        let self_n = c_dims[dim - 1];

        // 2. 矩阵乘法规则检查 (M,K) @ (K,N) -> (M,N)
        if k != k2 || m != self_m || n != self_n {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "add_matmul_ (dimension mismatch)",
            });
        }

        // 3. Batch 检查
        let batching_lhs: usize = a_dims[..dim - 2].iter().product();
        let batching_rhs: usize = b_dims[..dim - 2].iter().product();
        let batching_self: usize = c_dims[..dim - 2].iter().product();

        if batching_lhs != batching_rhs || batching_lhs != batching_self {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "add_matmul_ (batch mismatch)",
            });
        }

        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        // 4. 执行计算
        self.do_add_matmul_(
            &mut *self.storage_write()?, self.layout(),
            lhs.storage_read()?.deref(), lhs.layout(),
            rhs.storage_read()?.deref(), rhs.layout(),
            (batching_lhs, m, n, k),
        );
        
        Ok(())
    }

    fn do_add_matmul_(
        &self,
        dst_storage: &mut Storage<T>, dst_layout: &Layout,
        lhs: &Storage<T>, lhs_layout: &Layout,
        rhs: &Storage<T>, rhs_layout: &Layout,
        bmnk: (usize, usize, usize, usize)
    ) {
        let (bs, ms, ns, ks) = bmnk;
        
        let lhs_rank = lhs_layout.shape().rank();
        let rhs_rank = rhs_layout.shape().rank();
        let dst_rank = dst_layout.shape().rank();


        use std::any::TypeId;

        let l_stride_m = lhs_layout.stride()[lhs_rank - 2] as isize;
        let l_stride_k = lhs_layout.stride()[lhs_rank - 1] as isize;
        let r_stride_k = rhs_layout.stride()[rhs_rank - 2] as isize;
        let r_stride_n = rhs_layout.stride()[rhs_rank - 1] as isize;
        let d_stride_m = dst_layout.stride()[dst_rank - 2] as isize;
        let d_stride_n = dst_layout.stride()[dst_rank - 1] as isize;

        let type_id = TypeId::of::<T>();

        if type_id == TypeId::of::<f32>() {
            for b in 0..bs {
                let l_offset = Self::compute_batch_offset(b, lhs_layout);
                let r_offset = Self::compute_batch_offset(b, rhs_layout);
                let d_offset = Self::compute_batch_offset(b, dst_layout);

                unsafe {
                    let lhs_ptr = lhs.data().as_ptr().add(l_offset) as *const f32;
                    let rhs_ptr = rhs.data().as_ptr().add(r_offset) as *const f32;
                    let dst_ptr = dst_storage.data_mut().as_mut_ptr().add(d_offset) as *mut f32;

                    matrixmultiply::sgemm(
                        ms, ks, ns,
                        1.0, // alpha
                        lhs_ptr, l_stride_m, l_stride_k,
                        rhs_ptr, r_stride_k, r_stride_n,
                        1.0, // beta = 1.0 表示 self = 1.0 * (lhs*rhs) + 1.0 * self
                        dst_ptr, d_stride_m, d_stride_n,
                    );
                }
            }
        } else if type_id == TypeId::of::<f64>() {
            for b in 0..bs {
                let l_offset = Self::compute_batch_offset(b, lhs_layout);
                let r_offset = Self::compute_batch_offset(b, rhs_layout);
                let d_offset = Self::compute_batch_offset(b, dst_layout);

                unsafe {
                    let lhs_ptr = lhs.data().as_ptr().add(l_offset) as *const f64;
                    let rhs_ptr = rhs.data().as_ptr().add(r_offset) as *const f64;
                    let dst_ptr = dst_storage.data_mut().as_mut_ptr().add(d_offset) as *mut f64;

                    matrixmultiply::dgemm(
                        ms, ks, ns,
                        1.0,
                        lhs_ptr, l_stride_m, l_stride_k,
                        rhs_ptr, r_stride_k, r_stride_n,
                        1.0,
                        dst_ptr, d_stride_m, d_stride_n,
                    );
                }
            }
        } else {
            let lhs_data = lhs.data();
            let rhs_data = rhs.data();
            
            match dst_layout.storage_indices() {
                StorageIndices::Contiguous(index) => {
                    let dst = &mut dst_storage.data_mut()[index.begin_index..index.end_index];
                    let mns = ms * ns;

                    (0..bs).for_each(|b| {
                        let l_batch_offset = Self::compute_batch_offset(b, lhs_layout);
                        let r_batch_offset = Self::compute_batch_offset(b, rhs_layout);
                        let dst_slice = &mut dst[b*mns..b*mns+mns];
        
                        for m in 0..ms {
                            for k in 0..ks {
                                let l_idx = l_batch_offset + (m as isize * l_stride_m + k as isize * l_stride_k) as usize;
                                let l_val = unsafe { *lhs_data.get_unchecked(l_idx) };
                                
                                let r_base = r_batch_offset + (k as isize * r_stride_k) as usize;
                                let dst_base = m * ns;
        
                                for n in 0..ns {
                                    unsafe {
                                        let r_idx = r_base + (n as isize * r_stride_n) as usize;
                                        let r_val = *rhs_data.get_unchecked(r_idx);
                                        
                                        let dst_ptr = dst_slice.get_unchecked_mut(dst_base + n);
                                        *dst_ptr = *dst_ptr + l_val * r_val;
                                    }
                                }
                            }
                        }
                    });
                },
                StorageIndices::Uncontiguous(_) => {
                    (0..bs).for_each(|b| {
                        let l_batch_offset = Self::compute_batch_offset(b, lhs_layout);
                        let r_batch_offset = Self::compute_batch_offset(b, rhs_layout);
                        let d_batch_offset = Self::compute_batch_offset(b, dst_layout);
        
                        for m in 0..ms {
                            let d_m_base = d_batch_offset + (m as isize * d_stride_m) as usize;
                            for k in 0..ks {
                                let l_idx = l_batch_offset + (m as isize * l_stride_m + k as isize * l_stride_k) as usize;
                                let l_val = unsafe { *lhs_data.get_unchecked(l_idx) };
                                
                                let r_k_base = r_batch_offset + (k as isize * r_stride_k) as usize;
        
                                for n in 0..ns {
                                    unsafe {
                                        let r_idx = r_k_base + (n as isize * r_stride_n) as usize;
                                        let r_val = *rhs_data.get_unchecked(r_idx);
                                        
                                        let d_idx = d_m_base + (n as isize * d_stride_n) as usize;
                                        let dst_ptr = dst_storage.data_mut();
                                        dst_ptr[d_idx] += l_val * r_val;
                                    }
                                }
                            }
                        }
                    });
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use crate::{s, DType, IndexOp, Slice};

    use super::*;

    #[test]
    fn test_matmul_2d() {
        // A: (2, 3), B: (3, 2)
        let a = Tensor::arange(0, 6).unwrap().reshape((2, 3)).unwrap(); // [[0,1,2],[3,4,5]]
        let b = Tensor::arange(0, 6).unwrap().reshape((3, 2)).unwrap(); // [[0,1],[2,3],[4,5]]
        let c = a.matmul(&b).unwrap();

        let expected = Tensor::new(&[
            [0*0 + 1*2 + 2*4, 0*1 + 1*3 + 2*5], // [10, 13]
            [3*0 + 4*2 + 5*4, 3*1 + 4*3 + 5*5], // [28, 40]
        ]).unwrap();

        assert!(c.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_add_matmul_2d() {
        // A: (2, 3), B: (3, 2)
        let a = Tensor::arange(0.0, 6.0).unwrap().reshape((2, 3)).unwrap(); // [[0,1,2],[3,4,5]]
        let b = Tensor::arange(0.0, 6.0).unwrap().reshape((3, 2)).unwrap(); // [[0,1],[2,3],[4,5]]
        // let c = a.matmul(&b).unwrap();
        let c = Tensor::zeros((2, 2)).unwrap(); // (2, 2)
        c.add_matmul_(&a, &b).unwrap();

        let expected = Tensor::new(&[
            [0*0 + 1*2 + 2*4, 0*1 + 1*3 + 2*5], // [10, 13]
            [3*0 + 4*2 + 5*4, 3*1 + 4*3 + 5*5], // [28, 40]
        ]).unwrap().cast::<f64>().unwrap();

        assert!(c.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_large_matmul() {
        let w = Tensor::randn(0.0, 1.0, (128, 1024, 1024)).unwrap();
        let x = Tensor::randn(0.0, 1.0, (128, 512, 1024)).unwrap();

        let start = std::time::Instant::now();
        let y = x.matmul(&w).unwrap();
        let end = std::time::Instant::now();
        
        eprintln!("{:?}", end - start);
    }

    #[test]
    fn test_small_matmul() {
        let w = Tensor::randn(0.0, 1.0, (32, 32)).unwrap();
        let x = Tensor::randn(0.0, 1.0, (32, 32)).unwrap();

        let start = std::time::Instant::now();
        let y = x.matmul(&w).unwrap();
        let end = std::time::Instant::now();
        
        eprintln!("{:?}", end - start);
    }

    #[test]
    fn test_matmul_batch() {
        // A: (2, 2, 3), B: (2, 3, 2)
        let a = Tensor::arange(0., 12.).unwrap().reshape((2, 2, 3)).unwrap();
        let b = Tensor::arange(0., 12.).unwrap().reshape((2, 3, 2)).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.dims(), &[2, 2, 2]);

        // batch 0
        let a0 = Tensor::new(&[[0.,1.,2.],[3.,4.,5.]]).unwrap();
        let b0 = Tensor::new(&[[0.,1.],[2.,3.],[4.,5.]]).unwrap();
        let c0 = a0.matmul(&b0).unwrap();

        // batch 1
        let a1 = Tensor::new(&[[6.,7.,8.],[9.,10.,11.]]).unwrap();
        let b1 = Tensor::new(&[[6.,7.],[8.,9.],[10.,11.]]).unwrap();
        let c1 = a1.matmul(&b1).unwrap();

        assert!(c0.allclose(&c.index(0).unwrap(), 1e-5, 1e-8).unwrap());
        assert!(c1.allclose(&c.index(1).unwrap(), 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_matmul_not_continues() {
        let a = Tensor::arange(0., 125.).unwrap().reshape((5, 5, 5)).unwrap();

        let sub_a = a.index((s!(1:3), s!(3:5), 2)).unwrap();
        let mut vals = Vec::new();
        for i in 1..3 {
            for j in 3..5 {
                vals.push((i * 25 + j * 5 + 2) as f64);
            }
        }
        let expected = Tensor::from_vec(vals, (2, 2)).unwrap();
        assert!(sub_a.allclose(&expected, 0.0, 0.0).unwrap());

        let b = Tensor::randn(0.0, 1.0, (2, 5)).unwrap();

        let res = sub_a.matmul(&b).unwrap();
        let res_expected = expected.matmul(&b).unwrap();
        assert!(res.allclose(&res_expected, 1e-5, 1e-8).unwrap());
    }
}