use num_traits::Zero;
use crate::{AutogradMetaT, Error, Layout, NumDType, Result, Shape, Storage};
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
            &self.storage(),
            self.layout(),
            &rhs.storage(),
            rhs.layout(),
            (batching, m, n, k),
        );

        let meta = T::AutogradMeta::on_matmul_op(self, rhs);
        Ok(Self::build(c_storage, c_shape, meta))
    }

    fn do_matmul(
        lhs: &Storage<T>, lhs_layout: &Layout, 
        rhs: &Storage<T>, rhs_layout: &Layout, 
        bmnk: (usize, usize, usize, usize)
    ) -> Storage<T> 
        where T: num_traits::Num + Copy + Zero
    {
        let lhs_data = lhs.data();
        let rhs_data = rhs.data();
    
        let lhs_rank = lhs_layout.shape().rank();
        let rhs_rank = rhs_layout.shape().rank();
        let (bs, ms, ns, ks) = bmnk;
        
        let mut dst = vec![T::zero(); bs * ms * ns];
    
        // 获取最后两个维度的 stride
        let l_stride_m = lhs_layout.stride()[lhs_rank - 2];
        let l_stride_k = lhs_layout.stride()[lhs_rank - 1];
        let r_stride_k = rhs_layout.stride()[rhs_rank - 2];
        let r_stride_n = rhs_layout.stride()[rhs_rank - 1];
    
        // 获取用于 Batch 迭代的维度和步幅（排除最后两个维度）
        let batch_dims = &lhs_layout.dims()[..lhs_rank - 2];
        let l_batch_strides = &lhs_layout.stride()[..lhs_rank - 2];
        let r_batch_strides = &rhs_layout.stride()[..rhs_rank - 2];
    
        for b in 0..bs {
            // 计算当前 batch 在 lhs 和 rhs 中的起始偏移量
            // 这里我们需要将平面的索引 b 还原为多维索引
            let mut l_batch_offset = lhs_layout.start_offset();
            let mut r_batch_offset = rhs_layout.start_offset();
            let mut temp_b = b;
            
            // 逆向计算每个 batch 维度的索引并乘以该维度的 stride
            for i in (0..batch_dims.len()).rev() {
                let idx = temp_b % batch_dims[i];
                l_batch_offset += idx * l_batch_strides[i];
                r_batch_offset += idx * r_batch_strides[i];
                temp_b /= batch_dims[i];
            }
    
            let dst_batch_offset = b * ms * ns;
    
            for m in 0..ms {
                for n in 0..ns {
                    let mut v = T::zero();
                    for k in 0..ks {
                        // 使用真实的 stride 进行寻址，无论是否转置都能正确工作
                        let l_idx = l_batch_offset + m * l_stride_m + k * l_stride_k;
                        let r_idx = r_batch_offset + k * r_stride_k + n * r_stride_n;
                        
                        v = v + lhs_data[l_idx] * rhs_data[r_idx];
                    }
                    dst[dst_batch_offset + m * ns + n] = v;
                }
            }
        }
        
        Storage::new(dst)
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use crate::{rng, DType, IndexOp, Range};

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

        assert!(c.allclose(&expected, 1e-5, 1e-8));
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

        assert!(c0.allclose(&c.index(0).unwrap(), 1e-5, 1e-8));
        assert!(c1.allclose(&c.index(1).unwrap(), 1e-5, 1e-8));
    }

    #[test]
    fn test_matmul_not_continues() {
        let a = Tensor::arange(0., 125.).unwrap().reshape((5, 5, 5)).unwrap();

        let sub_a = a.index((rng!(1:3), rng!(3:5), 2)).unwrap();
        let mut vals = Vec::new();
        for i in 1..3 {
            for j in 3..5 {
                vals.push((i * 25 + j * 5 + 2) as f64);
            }
        }
        let expected = Tensor::from_vec(vals, (2, 2)).unwrap();
        assert!(sub_a.allclose(&expected, 0.0, 0.0));

        let b = Tensor::randn(0.0, 1.0, (2, 5)).unwrap();

        let res = sub_a.matmul(&b).unwrap();
        let res_expected = expected.matmul(&b).unwrap();
        assert!(res.allclose(&res_expected, 1e-5, 1e-8));
    }
}