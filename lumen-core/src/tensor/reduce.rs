use std::cmp::Ordering;
use crate::{AutogradMetaT, Dim, NumDType, Result, Storage, Tensor, WithDType};
use paste::paste;

/*

        let (storage, shape) = self.compute_reduce_op(reduce_dim, op, keepdim)?;
        Ok(Tensor::<R>::from_storage(storage, shape, meta))
*/
macro_rules! reduce_impl {
    ($fn_name:ident, $reduce:ident, $op:ident) => {
        paste! {
            pub fn $fn_name<D: Dim>(&self, axis: D) -> Result<Self> {
                let (storage, dims) = self.compute_reduce_op(axis, $reduce, false)?;
                let meta = T::AutogradMeta::on_reduce_op(self, &dims, crate::ReduceOp::$op);
                Ok(Self::from_storage(storage, dims, meta))
            }
        
            pub fn [< $fn_name _keepdim >]<D: Dim>(&self, axis: D) -> Result<Self> {
                let (storage, dims) = self.compute_reduce_op(axis, $reduce, true)?;
                let meta = T::AutogradMeta::on_reduce_op(self, &dims, crate::ReduceOp::$op);
                Ok(Self::from_storage(storage, dims, meta))
            }

            pub fn [< $fn_name _all >](&self) -> Result<Self> {
                self.flatten_all()?.$fn_name(0)
            }
        }
    };
}

impl<T: NumDType> Tensor<T> {
    reduce_impl!(sum, ReduceSum, Sum);
    reduce_impl!(min, ReduceMin, Min);
    reduce_impl!(max, ReduceMax, Max);
    reduce_impl!(mean, ReduceMean, Mean);

    pub fn var_keepdim<D: Dim>(&self, axis: D) -> Result<Self> {
        let mean = self.mean_keepdim(axis)?; // (..., 1, ...)
        let delta = self.broadcast_sub(&mean)?; // (..., n, ...)
        let delta_pow = &delta * &delta; // (..., n, ...)

        delta_pow.mean_keepdim(axis)
    }

    pub fn var_unbiased_keepdim<D: Dim>(&self, axis: D) -> Result<Self> {
        let n = T::from_usize(self.dim(axis)?);
        let biased_var = self.var_keepdim(axis)?;
        
        // cor scale: N / (N - 1)
        let correction = n / (n - T::one());
        Ok(correction * biased_var)
    }
    
    pub fn var<D: Dim>(&self, axis: D) -> Result<Self> {
        let v = self.var_keepdim(axis)?;
        let v = v.squeeze(axis)?;
        Ok(v)
    }

    pub fn var_unbiased<D: Dim>(&self, axis: D) -> Result<Self> {
        let v = self.var_unbiased_keepdim(axis)?;
        let v = v.squeeze(axis)?;
        Ok(v)
    }

    pub fn var_all(&self) -> Result<Self> {
        self.flatten_all()?.var(0)
    }

    pub fn var_unbiased_all(&self) -> Result<Self> {
        self.flatten_all()?.var_unbiased(0)
    }

    pub fn argmin_keepdim<D: Dim>(&self, axis: D) -> Result<Tensor<u32>> {
        let (storage, dims) = self.compute_reduce_op(axis, ReduceArgMin, true)?;
        Ok(Tensor::from_storage(storage, dims, Default::default()))
    }

    pub fn argmin<D: Dim>(&self, axis: D) -> Result<Tensor<u32>> {
        let (storage, dims) = self.compute_reduce_op(axis, ReduceArgMin, false)?;
        Ok(Tensor::from_storage(storage, dims, Default::default()))
    }

    pub fn argmax_keepdim<D: Dim>(&self, axis: D) -> Result<Tensor<u32>> {
        let (storage, dims) = self.compute_reduce_op(axis, ReduceArgMax, true)?;
        Ok(Tensor::from_storage(storage, dims, Default::default()))
    }

    pub fn argmax<D: Dim>(&self, axis: D) -> Result<Tensor<u32>> {
        let (storage, dims) = self.compute_reduce_op(axis, ReduceArgMax, false)?;
        Ok(Tensor::from_storage(storage, dims, Default::default()))
    }
}

impl Tensor<bool> {
    pub fn all_all(&self) -> crate::Result<bool> {
        self.iter().map(|mut i| i.all(|a| a))
    }

    pub fn any_all(&self) -> crate::Result<bool> {
        self.iter().map(|mut i| i.any(|a| a))
    } 

    pub fn all_keepdim<D: Dim>(&self, axis: D) -> Result<Tensor<bool>> {
        self.reduce_op(axis, ReduceAll, true, Default::default())
    }

    pub fn any_keepdim<D: Dim>(&self, axis: D) -> Result<Tensor<bool>> {
        self.reduce_op(axis, ReduceAny, true, Default::default())
    }

    pub fn all<D: Dim>(&self, axis: D) -> Result<Tensor<bool>> {
        self.reduce_op(axis, ReduceAll, false, Default::default())
    }

    pub fn any<D: Dim>(&self, axis: D) -> Result<Tensor<bool>> {
        self.reduce_op(axis, ReduceAny, false, Default::default())
    }
}

impl<T: WithDType> Tensor<T> {
    fn reduce_op<'a, R, Op, D>(&'a self, reduce_dim: D, op: Op, keepdim: bool, meta: R::AutogradMeta) -> Result<Tensor<R>> 
    where 
        R: WithDType,
        D: Dim,
        Op: ReduceOp<T, Output = R>,
    {
        let (storage, shape) = self.compute_reduce_op(reduce_dim, op, keepdim)?;
        Ok(Tensor::<R>::from_storage(storage, shape, meta))
    }

    fn compute_reduce_op<'a, R, Op, D>(&'a self, reduce_dim: D, _: Op, keepdim: bool) -> Result<(Storage<R>, Vec<usize>)> 
    where 
        R: WithDType,
        D: Dim,
        Op: ReduceOp<T, Output = R>,
    {
        let reduce_dim = reduce_dim.to_index(self.shape(), Op::NAME)?;
        assert!(reduce_dim < self.layout().dims().len());
        let reduce_dim_stride = self.layout().stride()[reduce_dim];
        let reduce_dim_size = self.layout().dims()[reduce_dim];
   
        let dst_len = self.layout().element_count() / reduce_dim_size;
        let mut dst: Vec<R> = Vec::with_capacity(dst_len);
        let dst_to_set = dst.spare_capacity_mut();

        let storage = self.storage_read()?;
        let storage_slice = storage.data();

        let layout = self.layout().narrow(reduce_dim, 0, 1)?;
        for (dst_index, src_index) in layout.storage_indices().enumerate() {
            if reduce_dim_stride == 1 {
                let end_index = src_index + reduce_dim_size;
                let chunk = &storage_slice[src_index..end_index];
                let v = Op::op(chunk.iter().copied());
                dst_to_set[dst_index].write(v);
            } else {
                let arr: DimArray<'_, T> = DimArray {
                    src: &storage_slice[src_index..],
                    size: reduce_dim_size,
                    stride: reduce_dim_stride
                };
                let iter: DimArrayIter<'_, T> = arr.into_iter();
                let v = Op::op(iter);
                dst_to_set[dst_index].write(v);
            }  
        }
        unsafe { dst.set_len(dst_len) };

        let storage = Storage::new(dst);
        let mut shape = self.dims().to_vec();

        if keepdim {
            shape[reduce_dim] = 1;
        } else {
            shape.remove(reduce_dim);
        }

        Ok((storage, shape))
    }
}

impl<T: NumDType> Tensor<T> {
    pub fn topk<D: Dim>(&self, k: usize, dim: D) -> crate::Result<(Tensor<T>, Tensor<u32>)> {
        let reduce_dim = dim.to_index(self.shape(), "topk")?;
        assert!(reduce_dim < self.layout().dims().len());
        let reduce_dim_stride = self.layout().stride()[reduce_dim];
        let reduce_dim_size = self.layout().dims()[reduce_dim];
        if k > reduce_dim_size {
            return Err(crate::Error::IndexOutOfRange { max_size: reduce_dim_size, index: k, op: "topk"});
        }

        // 输出目标：值和数组的总大小
        let dst_len = self.layout().element_count() / reduce_dim_size * k;
        // 预先开辟足够的空间
        let mut indices: Vec<u32> = Vec::with_capacity(dst_len);
        let mut values: Vec<T> = Vec::with_capacity(dst_len);
        let mut indexed: Vec<(u32, T)> = Vec::with_capacity(reduce_dim_size);

        // 先去掉 dim 维度，遍历其他所有的可能的排列
        let storage = self.storage_read()?;
        let storage_slice = storage.data();
        let layout = self.layout().narrow(reduce_dim, 0, 1)?;
        for src_index in layout.storage_indices() {
            // 取出这个维度的一组数据
            let arr: DimArray<'_, T> = DimArray {
                src: &storage_slice[src_index..],
                size: reduce_dim_size,
                stride: reduce_dim_stride
            };
            // 索引 + 值进行排序
            indexed.clear();
            indexed.extend(arr.into_iter().enumerate().map(|(i, v)| (i as u32, v)));

            assert!(indexed.len() >= k);
            if k < indexed.len() {
                // 先在 O(N) 时间内找出前 k 大的划分
                indexed.select_nth_unstable_by(k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
                // 然后只对前 k 个进行排序 O(k log k)
                indexed[..k].sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
            } else {
                indexed.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
            }

            let topk_slice = &indexed[..k];

            values.extend(topk_slice.iter().map(|(_, v)| *v));
            indices.extend(topk_slice.iter().map(|(i, _)| *i));
        }

        assert_eq!(indices.len(), dst_len);
        assert_eq!(values.len(), dst_len);

        let mut physical_shape = self.dims().to_vec();
        physical_shape.remove(reduce_dim);
        physical_shape.push(k);
        
        let temp_values = Tensor::from_vec(values, physical_shape.clone())?;
        let temp_indices = Tensor::from_vec(indices, physical_shape.clone())?;

        // 如果 reduce_dim 本来就在最后，不需要转置
        if reduce_dim == self.dims().len() - 1 {
            return Ok((temp_values, temp_indices));
        }

        // 把最后一个维度移动回 reduce_dim 的位置
        let ndims = physical_shape.len();
        let mut permute_axes: Vec<usize> = (0..ndims - 1).collect(); // [0, 1, 2, ..., N-2]
        permute_axes.insert(reduce_dim, ndims - 1);

        let final_values = temp_values.permute(permute_axes.as_ref())?.contiguous()?;
        let final_indices = temp_indices.permute(permute_axes)?.contiguous()?;

        Ok((final_values, final_indices))
    }
}

pub trait ReduceOp<D: WithDType> {
    type Output: WithDType;
    const NAME: &'static str;
    fn op<I>(iter: I) -> Self::Output
    where
        I: Iterator<Item = D> + ExactSizeIterator + Clone;
}

pub struct ReduceAll;
impl ReduceOp<bool> for ReduceAll {
    type Output = bool;
    const NAME: &'static str = "all";
    fn op<I>(mut iter: I) -> Self::Output 
    where I: Iterator<Item = bool> + ExactSizeIterator + Clone {
        iter.all(|b| b)
    }
}

pub struct ReduceAny;
impl ReduceOp<bool> for ReduceAny {
    type Output = bool;
    const NAME: &'static str = "any";
    fn op<I>(mut iter: I) -> Self::Output 
    where I: Iterator<Item = bool> + ExactSizeIterator + Clone {
        iter.any(|b| b)
    }
}

pub struct ReduceSum;
impl<D: NumDType> ReduceOp<D> for ReduceSum {
    const NAME: &'static str = "sum";
    type Output = D;
    fn op<I>(iter: I) -> Self::Output 
    where I: Iterator<Item = D> + ExactSizeIterator + Clone {
        iter.sum::<D>()
    }
} 

pub struct ReduceMean;
impl<D: NumDType> ReduceOp<D> for ReduceMean {
    type Output = D;
    const NAME: &'static str = "mean";
    fn op<I>(iter: I) -> Self::Output 
    where I: Iterator<Item = D> + ExactSizeIterator + Clone {
        let len = iter.len();
        iter.sum::<D>() / D::from_usize(len)
    }
} 

pub struct ReduceVar;
impl<D: NumDType> ReduceOp<D> for ReduceVar {
    type Output = D;
    const NAME: &'static str = "var";
    fn op<I>(iter: I) -> Self::Output 
    where I: Iterator<Item = D> + ExactSizeIterator + Clone {
        let len = iter.len();
        if len == 0 { return D::zero(); }

        // 利用 Clone 给 Mean 使用，本迭代器保留用于第二次遍历
        let mean = ReduceMean::op(iter.clone());
        
        let mut sum_sq_diff = D::zero();
        for v in iter {
            let diff = v - mean;
            sum_sq_diff += diff * diff;
        }

        sum_sq_diff / D::from_usize(len)
    }
} 

pub struct ReduceProduct;
impl<D: NumDType> ReduceOp<D> for ReduceProduct {
    type Output = D;
    const NAME: &'static str = "product";
    fn op<I>(iter: I) -> Self::Output 
    where I: Iterator<Item = D> + ExactSizeIterator + Clone {
        iter.product::<D>()
    }
} 

pub struct ReduceMin;
impl<D: NumDType> ReduceOp<D> for ReduceMin {
    type Output = D;
    const NAME: &'static str = "min";
    fn op<I>(iter: I) -> Self::Output 
    where I: Iterator<Item = D> + ExactSizeIterator + Clone {
        iter.reduce(|a, b| D::minimum(a, b)).unwrap()
    }
} 

pub struct ReduceMax;
impl<D: NumDType> ReduceOp<D> for ReduceMax {
    type Output = D;
    const NAME: &'static str = "max";
    fn op<I>(iter: I) -> Self::Output 
    where I: Iterator<Item = D> + ExactSizeIterator + Clone {
        iter.reduce(|a, b| D::maximum(a, b)).unwrap()
    }
} 

pub struct ReduceArgMin;
impl<D: NumDType> ReduceOp<D> for ReduceArgMin {
    type Output = u32;
    const NAME: &'static str = "arg_min";
    fn op<I>(iter: I) -> Self::Output 
    where I: Iterator<Item = D> + ExactSizeIterator + Clone {
        iter.enumerate()
            .reduce(|(ia, a), (ib, b)| {
                if a.partial_cmp(&b) == Some(std::cmp::Ordering::Less) {
                    (ia, a)
                } else {
                    (ib, b)
                }
            }).unwrap().0 as u32
    }
} 

pub struct ReduceArgMax;
impl<D: NumDType> ReduceOp<D> for ReduceArgMax {
    type Output = u32;
    const NAME: &'static str = "arg_max";
    fn op<I>(iter: I) -> Self::Output 
    where I: Iterator<Item = D> + ExactSizeIterator + Clone {
        iter.enumerate()
            .reduce(|(ia, a), (ib, b)| {
                if a.partial_cmp(&b) == Some(std::cmp::Ordering::Greater) {
                    (ia, a)
                } else {
                    (ib, b)
                }
            }).unwrap().0 as u32
    }
}

#[derive(Clone)]
pub struct DimArray<'a, T> {
    src: &'a [T],
    size: usize,
    stride: usize
}

impl<'a, T: WithDType> DimArray<'a, T> {
    pub fn get(&self, index: usize) -> T {
        self.src[index * self.stride]
    }

    #[allow(unused)]
    pub fn to_vec(&self) -> Vec<T> {
        let mut v = vec![];
        for i in 0..self.size {
            v.push(self.get(i));
        }
        v
    }
}

impl<'a, T: WithDType> IntoIterator for DimArray<'a, T> {
    type IntoIter = DimArrayIter<'a, T>;
    type Item = T;
    fn into_iter(self) -> Self::IntoIter {
        DimArrayIter::<'a, T> {
            array: self,
            index: 0,
        }
    }
}

#[derive(Clone)]
pub struct DimArrayIter<'a, T> {
    array: DimArray<'a, T>,
    index: usize,
}

impl<'a, T: WithDType> Iterator for DimArrayIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.index >= self.array.size {
            None
        } else {
            let index = self.index;
            self.index += 1;
            Some(self.array.get(index))
        }
    }
}

impl<'a, T: WithDType> ExactSizeIterator for DimArrayIter<'a, T> {
    fn len(&self) -> usize {
        self.array.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // sum_all(axis=0) -> [4, 6, 8]
        let arr = Tensor::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let s = arr.sum(0).unwrap();
        let expected = Tensor::new(&[4, 6, 8]).unwrap();
        assert!(s.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_sum_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // sum_all(axis=1) -> [6, 12]
        let arr = Tensor::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let s = arr.sum(1).unwrap();
        let expected = Tensor::new(&[6, 12]).unwrap();
        assert!(s.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_sum_ones_axis() {
        // ones( (2,3), dtype=I32 )
        // [[1,1,1],
        //  [1,1,1]]
        let arr = Tensor::ones((2, 3)).unwrap();
        let s0 = arr.sum(0).unwrap(); // -> [2,2,2]
        let s1 = arr.sum(1).unwrap(); // -> [3,3]

        let expected0 = Tensor::new(&[2, 2, 2]).unwrap();
        let expected1 = Tensor::new(&[3, 3]).unwrap();

        assert!(s0.allclose(&expected0, 1e-5, 1e-8).unwrap());
        assert!(s1.allclose(&expected1, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_min_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // min_all(axis=0) -> [1, 1, 0]
        let arr = Tensor::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.min(0).unwrap();
        let expected = Tensor::new(&[1, 1, 0]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_max_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // max_all(axis=1) -> [3, 3]
        let arr = Tensor::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.max(1).unwrap();
        let expected = Tensor::new(&[3, 3]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_aragmin_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // min_all(axis=0) -> [1, 1, 0]
        let arr = Tensor::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.argmin(0).unwrap();
        let expected = Tensor::new(&[0, 1, 1]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_sum_all() {
        // [[1, 2], [3, 4]] -> 1+2+3+4 = 10
        let arr = Tensor::new(&[[1, 2], [3, 4]]).unwrap();
        let s = arr.sum_all().unwrap();        
        let expected = Tensor::new(10).unwrap(); 
        assert!(s.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_mean_all() {
        // [[1.0, 2.0], [3.0, 4.0]] -> Sum=10.0, Count=4 -> Mean=2.5
        let arr = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let m = arr.mean_all().unwrap();
        let expected = Tensor::new(2.5).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_min_max_all() {
        // [[10, 2, 5], [8, 1, 9]]
        // Global Min: 1
        // Global Max: 10
        let arr = Tensor::new(&[[10, 2, 5], [8, 1, 9]]).unwrap();
        
        let min_val = arr.min_all().unwrap();
        let max_val = arr.max_all().unwrap();
        
        let expected_min = Tensor::new(1).unwrap();
        let expected_max = Tensor::new(10).unwrap();
        
        assert!(min_val.allclose(&expected_min, 1e-5, 1e-8).unwrap());
        assert!(max_val.allclose(&expected_max, 1e-5, 1e-8).unwrap());
    }
    
    #[test]
    fn test_argmax_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // max_all(axis=1) -> [3, 3]
        let arr = Tensor::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.argmax(1).unwrap();
        let expected = Tensor::new(&[2, 0]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_reductions_with_negatives() {
        // [[-2.0, 0.0, 2.0]]
        // sum_all = 0.0
        // mean_all = 0.0
        // var(axis=1) -> 4.0 (unbiased: (4+0+4)/2)
        
        let arr = Tensor::new(&[[-2.0, 0.0, 2.0]]).unwrap();
        
        assert!(arr.sum_all().unwrap().allclose(&Tensor::new(0.0).unwrap(), 1e-5, 1e-8).unwrap());
        assert!(arr.mean_all().unwrap().allclose(&Tensor::new(0.0).unwrap(), 1e-5, 1e-8).unwrap());
        
        let expected_var = Tensor::new(2.66666666666666666).unwrap();
        assert!(arr.var_all().unwrap().allclose(&expected_var, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_topk_1d() {
        // [10, 50, 20, 40, 30]
        // topk(k=3, axis=0) -> 找最大的 3 个
        // 期望 Values:  [50, 40, 30]
        // 期望 Indices: [ 1,  3,  4]
        let arr = Tensor::new(&[10, 50, 20, 40, 30]).unwrap();
        let (val, idx) = arr.topk(3, 0).unwrap();

        let expected_val = Tensor::new(&[50, 40, 30]).unwrap();
        let expected_idx = Tensor::new(&[1u32, 3, 4]).unwrap(); // 注意索引是 u32

        assert!(val.allclose(&expected_val, 1e-5, 1e-8).unwrap());
        assert!(idx.allclose(&expected_idx, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_topk_matrix_axis1() {
        // 测试按行求 TopK (axis=1，也是连续内存)
        // [[1, 5, 2],
        //  [8, 4, 6]]
        // topk(k=2, axis=1)
        // 第一行 top2 -> [5, 2], 索引 -> [1, 2]
        // 第二行 top2 -> [8, 6], 索引 -> [0, 2]
        let arr = Tensor::new(&[[1, 5, 2], [8, 4, 6]]).unwrap();
        let (val, idx) = arr.topk(2, 1).unwrap();

        let expected_val = Tensor::new(&[[5, 2], [8, 6]]).unwrap();
        let expected_idx = Tensor::new(&[[1u32, 2], [0, 2]]).unwrap();

        assert!(val.allclose(&expected_val, 1e-5, 1e-8).unwrap());
        assert!(idx.allclose(&expected_idx, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_topk_matrix_axis0() {
        // 【关键测试】测试按列求 TopK (axis=0，跨步幅提取)
        // 用来验证内存 Layout 是否修复正确！
        // [[1, 5, 2],
        //  [8, 4, 6],
        //  [3, 7, 9]]
        // topk(k=2, axis=0) -> 找每列最大的 2 个
        // 第 0 列: [1, 8, 3] -> top2 是 8(idx:1), 3(idx:2)
        // 第 1 列: [5, 4, 7] -> top2 是 7(idx:2), 5(idx:0)
        // 第 2 列: [2, 6, 9] -> top2 是 9(idx:2), 6(idx:1)
        // 所以重组后的预期矩阵应该长这样：
        // Values:  [[8, 7, 9], 
        //           [3, 5, 6]]
        // Indices: [[1, 2, 2], 
        //           [2, 0, 1]]
        let arr = Tensor::new(&[
            [1, 5, 2], 
            [8, 4, 6], 
            [3, 7, 9]
        ]).unwrap();
        let (val, idx) = arr.topk(2, 0).unwrap();

        let expected_val = Tensor::new(&[[8, 7, 9], [3, 5, 6]]).unwrap();
        let expected_idx = Tensor::new(&[[1u32, 2, 2], [2, 0, 1]]).unwrap();

        assert!(val.allclose(&expected_val, 1e-5, 1e-8).unwrap());
        assert!(idx.allclose(&expected_idx, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_topk_full_sort() {
        // 测试当 k 等于该维度大小时，相当于进行沿轴的全量降序排序
        // [[3, 1, 2]] -> topk(3, axis=1)
        let arr = Tensor::new(&[[3, 1, 2]]).unwrap();
        let (val, idx) = arr.topk(3, 1).unwrap();

        let expected_val = Tensor::new(&[[3, 2, 1]]).unwrap();
        let expected_idx = Tensor::new(&[[0u32, 2, 1]]).unwrap();

        assert!(val.allclose(&expected_val, 1e-5, 1e-8).unwrap());
        assert!(idx.allclose(&expected_idx, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_large_ops() {
        let a = Tensor::randn(0.0, 1.0, (1000, 4096)).unwrap();
        let now = std::time::Instant::now();
        let s = a.sum_keepdim(1).unwrap();
        println!("{:?}", std::time::Instant::now() - now);
        println!("{:?}", s.dims());
    }
}
