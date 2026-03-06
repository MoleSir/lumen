use crate::{AutogradMetaT, Dim, NumDType, Result, Storage, StorageRef, Tensor, WithDType};
use paste::paste;

use super::ResettableIterator;

macro_rules! reduce_impl {
    ($fn_name:ident, $reduce:ident, $op:ident) => {
        paste! {
            pub fn $fn_name<D: Dim>(&self, axis: D) -> Result<Self> {
                let (storage, dims) = self.compute_reduec_axis_op(axis, $reduce::op, stringify!($fn_name))?;
                let meta = T::AutogradMeta::on_reduce_op(self, &dims, crate::ReduceOp::$op);
                let res = Self::from_storage(storage, dims, meta);
                res.squeeze(axis)
            }
        
            pub fn [< $fn_name _keepdim >]<D: Dim>(&self, axis: D) -> Result<Self> {
                let (storage, dims) = self.compute_reduec_axis_op(axis, $reduce::op, stringify!([< $fn_name _keepdim >]))?;
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
        let (storage, dims) = self.compute_reduec_axis_op(axis, ReduceArgMin::op, "argmin")?;
        Ok(Tensor::from_storage(storage, dims, Default::default()))
    }

    pub fn argmin<D: Dim>(&self, axis: D) -> Result<Tensor<u32>> {
        let (storage, dims) = self.compute_reduec_axis_op(axis, ReduceArgMin::op, "argmin_keepdim")?;
        let res = Tensor::from_storage(storage, dims, Default::default());
        res.squeeze(axis)
    }

    pub fn argmax_keepdim<D: Dim>(&self, axis: D) -> Result<Tensor<u32>> {
        let (storage, dims) = self.compute_reduec_axis_op(axis, ReduceArgMax::op, "argmax")?;
        Ok(Tensor::from_storage(storage, dims, Default::default()))
    }

    pub fn argmax<D: Dim>(&self, axis: D) -> Result<Tensor<u32>> {
        let (storage, dims) = self.compute_reduec_axis_op(axis, ReduceArgMax::op, "argmax_keepdim")?;
        let res = Tensor::from_storage(storage, dims, Default::default());
        res.squeeze(axis)
    }
}

impl Tensor<bool> {
    pub fn all(&self) -> crate::Result<bool> {
        self.iter().map(|mut i| i.all(|a| a))
    }

    pub fn any(&self) -> crate::Result<bool> {
        self.iter().map(|mut i| i.any(|a| a))
    } 

    pub fn all_axis<D: Dim>(&self, axis: D) -> Result<Tensor<bool>> {
        self.reduec_axis_op(axis, ReduceAll::op, Default::default(), "all")
    }

    pub fn any_axis<D: Dim>(&self, axis: D) -> Result<Tensor<bool>> {
        self.reduec_axis_op(axis, ReduceAny::op, Default::default(), "any")
    }
}

impl<T: WithDType> Tensor<T> {
    fn reduec_axis_op<'a, F, R: WithDType, D: Dim>(&'a self, reduce_dim: D, f: F, meta: R::AutogradMeta, op_name: &'static str) -> Result<Tensor<R>> 
    where 
        F: Fn(&mut DimArrayIter<'a, T>) -> R
    {
        let (storage, shape) = self.compute_reduec_axis_op(reduce_dim, f, op_name)?;
        Ok(Tensor::<R>::from_storage(storage, shape, meta))
    }

    fn compute_reduec_axis_op<'a, F, R: WithDType, D: Dim>(&'a self, reduce_dim: D, f: F, op_name: &'static str) -> Result<(Storage<R>, Vec<usize>)> 
    where 
        F: Fn(&mut DimArrayIter<'a, T>) -> R
    {
        let reduce_dim = reduce_dim.to_index(self.shape(), op_name)?;
        assert!(reduce_dim < self.layout().dims().len());
        let reduce_dim_stride = self.layout().stride()[reduce_dim];
        let reduce_dim_size = self.layout().dims()[reduce_dim];

        let dst_len = self.layout().element_count() / reduce_dim_size;
        let mut dst: Vec<R> = Vec::with_capacity(dst_len);
        let dst_to_set = dst.spare_capacity_mut();

        let layout = self.layout().narrow(reduce_dim, 0, 1)?;
        for (dst_index, src_index) in layout.storage_indices().enumerate() {
            let arr: DimArray<'_, T> = DimArray {
                src: self.storage_ref(src_index)?,
                size: reduce_dim_size,
                stride: reduce_dim_stride
            };
            let mut iter: DimArrayIter<'_, T> = arr.into_iter();
            dst_to_set[dst_index].write(f(&mut iter));
        }
        unsafe { dst.set_len(dst_len) };

        let storage = Storage::new(dst);
        let mut shape = self.dims().to_vec();
        // shape.remove(reduce_dim);
        shape[reduce_dim] = 1;

        Ok((storage, shape))
    }
}

pub trait ReduceOp<D: WithDType> {
    type Output: WithDType;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output;
}

pub struct ReduceAll;
impl ReduceOp<bool> for ReduceAll {
    type Output = bool;
    fn op(arr: &mut DimArrayIter<'_, bool>) -> Self::Output {
        arr.into_iter().all(|b| b)
    }
}

pub struct ReduceAny;
impl ReduceOp<bool> for ReduceAny {
    type Output = bool;
    fn op(arr: &mut DimArrayIter<'_, bool>) -> Self::Output {
        arr.into_iter().any(|b| b)
    }
}

pub struct ReduceSum;
impl<D: NumDType> ReduceOp<D> for ReduceSum {
    type Output = D;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output {
        arr.into_iter().sum::<D>()
    }
} 

pub struct ReduceMean;
impl<D: NumDType> ReduceOp<D> for ReduceMean {
    type Output = D;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output {
        let len = arr.len();
        arr.into_iter().sum::<D>() / D::from_usize(len)
    }
} 

pub struct ReduceVar;
impl<D: NumDType> ReduceOp<D> for ReduceVar {
    type Output = D;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output {
        let len = arr.len();
        if len == 0 { return D::zero(); }

        let mean = ReduceMean::op(arr);
        
        arr.reset();
        let mut sum_sq_diff = D::zero();
        while let Some(v) = arr.next() {
            let diff = v - mean;
            sum_sq_diff += diff * diff;
        }

        sum_sq_diff / D::from_usize(len)
    }
} 

pub struct ReduceProduct;
impl<D: NumDType> ReduceOp<D> for ReduceProduct {
    type Output = D;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output {
        arr.into_iter().product::<D>()
    }
} 

pub struct ReduceMin;
impl<D: NumDType> ReduceOp<D> for ReduceMin {
    type Output = D;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output {
        arr.into_iter()
            .reduce(|a, b| D::minimum(a, b)).unwrap()
    }
} 

pub struct ReduceArgMin;
impl<D: NumDType> ReduceOp<D> for ReduceArgMin {
    type Output = u32;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output {
        arr.into_iter()
            .enumerate()
            .reduce(|(ia, a), (ib, b)| {
                if a.partial_cmp(&b) == Some(std::cmp::Ordering::Less) {
                    (ia, a)
                } else {
                    (ib, b)
                }
            }).unwrap().0 as u32
    }
} 

pub struct ReduceMax;
impl<D: NumDType> ReduceOp<D> for ReduceMax {
    type Output = D;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output {
        arr.into_iter()
            .reduce(|a, b| D::maximum(a, b)).unwrap()
    }
} 

pub struct ReduceArgMax;
impl<D: NumDType> ReduceOp<D> for ReduceArgMax {
    type Output = u32;
    fn op(arr: &mut DimArrayIter<'_, D>) -> Self::Output {
        arr.into_iter()
            .enumerate()
            .reduce(|(ia, a), (ib, b)| {
                if a.partial_cmp(&b) == Some(std::cmp::Ordering::Greater) {
                    (ia, a)
                } else {
                    (ib, b)
                }
            }).unwrap().0 as u32
    }
} 

pub struct DimArray<'a, T> {
    src: StorageRef<'a, T>,
    size: usize,
    stride: usize
}

impl<'a, T: WithDType> DimArray<'a, T> {
    pub fn get(&self, index: usize) -> T {
        self.src.get_unchecked(index * self.stride)
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

impl<'a, T: WithDType> ResettableIterator for DimArrayIter<'a, T> {
    fn reset(&mut self) {
        self.index = 0
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
}
