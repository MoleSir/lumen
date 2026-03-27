use crate::{NumDType, StorageIndices, WithDType};
use super::Tensor;
use paste::paste;

macro_rules! fused_2arg_inplace_op_impl {
    ($name:ident, $f_inplace:expr, $f:expr) => {
        paste! {
            #[doc = crate::inplace_warning_doc!()]
            pub fn [< $name _ >](&self, a: &Self, b: &Self) -> crate::Result<()> {
                self.[< impl_ $name _ >](a, b)?;
                self.impl_add_mul_(a, b)
            }

            #[doc = crate::inplace_warning_doc!()]
            pub(crate) fn [< impl_ $name _ >](&self, a: &Self, b: &Self) -> crate::Result<()> {
                self.fused_2arg_inplace_op(a, b, $f_inplace, $f, stringify!($name))
            }
        }
    };
}

impl<T: NumDType> Tensor<T> {
    fused_2arg_inplace_op_impl!(add_mul, T::add, T::mul);
    fused_2arg_inplace_op_impl!(add_div, T::add, T::div);
    fused_2arg_inplace_op_impl!(sub_mul, T::sub, T::mul);
    fused_2arg_inplace_op_impl!(sub_div, T::sub, T::div);
}

impl<T: NumDType> Tensor<T> {
    pub(crate) fn fused_2arg_inplace_op<F1, F2>(
        &self,
        src1: &Tensor<T>, 
        src2: &Tensor<T>, 
        f_inplace: F2, 
        f: F1, 
        op_name: &'static str
    ) -> crate::Result<()> 
    where 
        F1: Fn(T, T) -> T,
        F2: Fn(T, T) -> T,
    {
        let dst = self;
        let _ = Tensor::<T>::same_shape_binary_op(dst, src1, op_name)?;
        let _ = Tensor::<T>::same_shape_binary_op(dst, src2, op_name)?;

        let mut dst_storage = dst.storage_write()?;
        let src1_storage = src1.storage_read()?;
        let src2_storage = src2.storage_read()?;
        
        let dst_layout = dst.layout();
        let src1_layout = src1.layout();
        let src2_layout = src2.layout();

        let dst_data = dst_storage.data_mut();
        let src1_data = src1_storage.data();
        let src2_data = src2_storage.data();        

        use StorageIndices::{Contiguous, Uncontiguous};

        match (dst_layout.storage_indices(), src1_layout.storage_indices(), src2_layout.storage_indices()) {
            // 1. (连续, 连续, 连续)
            (Contiguous(d_idx), Contiguous(s1_idx), Contiguous(s2_idx)) => {
                let d_slice = &mut dst_data[d_idx.begin_index..d_idx.end_index];
                let s1_slice = &src1_data[s1_idx.begin_index..s1_idx.end_index];
                let s2_slice = &src2_data[s2_idx.begin_index..s2_idx.end_index];
                
                d_slice.iter_mut()
                    .zip(s1_slice.iter())
                    .zip(s2_slice.iter())
                    .for_each(|((d, &s1), &s2)| {
                        *d = f_inplace(*d, f(s1, s2));
                    });
            }

            // 2. (连续, 连续, 不连续)
            (Contiguous(d_idx), Contiguous(s1_idx), Uncontiguous(s2_idx)) => {
                let d_slice = &mut dst_data[d_idx.begin_index..d_idx.end_index];
                let s1_slice = &src1_data[s1_idx.begin_index..s1_idx.end_index];

                d_slice.iter_mut()
                    .zip(s1_slice.iter())
                    .zip(s2_idx)
                    .for_each(|((d, &s1), s2_i)| {
                        *d = f_inplace(*d, f(s1, src2_data[s2_i]));
                    });
            }

            // 3. (连续, 不连续, 连续)
            (Contiguous(d_idx), Uncontiguous(s1_idx), Contiguous(s2_idx)) => {
                let d_slice = &mut dst_data[d_idx.begin_index..d_idx.end_index];
                let s2_slice = &src2_data[s2_idx.begin_index..s2_idx.end_index];

                d_slice.iter_mut()
                    .zip(s1_idx)
                    .zip(s2_slice.iter())
                    .for_each(|((d, s1_i), &s2)| {
                        *d = f_inplace(*d, f(src1_data[s1_i], s2));
                    });
            }

            // 4. (连续, 不连续, 不连续)
            (Contiguous(d_idx), Uncontiguous(s1_idx), Uncontiguous(s2_idx)) => {
                let d_slice = &mut dst_data[d_idx.begin_index..d_idx.end_index];

                d_slice.iter_mut()
                    .zip(s1_idx)
                    .zip(s2_idx)
                    .for_each(|((d, s1_i), s2_i)| {
                        *d = f_inplace(*d, f(src1_data[s1_i], src2_data[s2_i]));
                    });
            }

            // 5. (不连续, 连续, 连续)
            (Uncontiguous(d_idx), Contiguous(s1_idx), Contiguous(s2_idx)) => {
                let s1_slice = &src1_data[s1_idx.begin_index..s1_idx.end_index];
                let s2_slice = &src2_data[s2_idx.begin_index..s2_idx.end_index];

                d_idx.zip(s1_slice.iter())
                    .zip(s2_slice.iter())
                    .for_each(|((d_i, &s1), &s2)| {
                        dst_data[d_i] = f_inplace(dst_data[d_i], f(s1, s2));
                    });
            }

            // 6. (不连续, 连续, 不连续)
            (Uncontiguous(d_idx), Contiguous(s1_idx), Uncontiguous(s2_idx)) => {
                let s1_slice = &src1_data[s1_idx.begin_index..s1_idx.end_index];

                d_idx.zip(s1_slice.iter())
                    .zip(s2_idx)
                    .for_each(|((d_i, &s1), s2_i)| {
                        dst_data[d_i] = f_inplace(dst_data[d_i], f(s1, src2_data[s2_i]));
                    });
            }

            // 7. (不连续, 不连续, 连续)
            (Uncontiguous(d_idx), Uncontiguous(s1_idx), Contiguous(s2_idx)) => {
                let s2_slice = &src2_data[s2_idx.begin_index..s2_idx.end_index];

                d_idx.zip(s1_idx)
                    .zip(s2_slice.iter())
                    .for_each(|((d_i, s1_i), &s2)| {
                        dst_data[d_i] = f_inplace(dst_data[d_i], f(src1_data[s1_i], s2));
                    });
            }

            // 8. (不连续, 不连续, 不连续) - 最慢的 fallback 路径
            (Uncontiguous(d_idx), Uncontiguous(s1_idx), Uncontiguous(s2_idx)) => {
                d_idx.zip(s1_idx)
                    .zip(s2_idx)
                    .for_each(|((d_i, s1_i), s2_i)| {
                        let v = f(src1_data[s1_i], src2_data[s2_i]);
                        dst_data[d_i] = f_inplace(dst_data[d_i], v);
                    });
            }
        }
        
        Ok(())
    }

    pub(crate) fn fused_1arg_inplace_op<F1, F2>(
        &self,
        rhs: &Tensor<T>, 
        f_inplace: F1, 
        f: F2, 
        op_name: &'static str
    ) -> crate::Result<()> 
    where 
        F1: Fn(T, T) -> T,
        F2: Fn(T) -> T,
    {
        let lhs = self;
        let _ = Tensor::<T>::same_shape_binary_op(lhs, rhs, op_name)?;

        let mut lhs_storage = lhs.storage_write()?;
        let rhs_storage = rhs.storage_read()?;
        let lhs_layout = lhs.layout();
        let rhs_layout = rhs.layout();

        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");

        let lhs = lhs_storage.data_mut();
        let rhs = rhs_storage.data();
        
        match (lhs_layout.storage_indices(), rhs_layout.storage_indices()) {
            (StorageIndices::Contiguous(lhs_index), StorageIndices::Contiguous(rhs_index)) => {
                let lhs = &mut lhs[lhs_index.begin_index..lhs_index.end_index];
                let rhs = &rhs[rhs_index.begin_index..rhs_index.end_index];
                lhs.iter_mut().zip(rhs.iter())
                    .for_each(|(l, &r)| *l = f_inplace(*l, f(r)));
            }
            (StorageIndices::Contiguous(lhs_index), StorageIndices::Uncontiguous(rhs_index)) => {
                let lhs = &mut lhs[lhs_index.begin_index..lhs_index.end_index];
                lhs.iter_mut().zip(rhs_index)
                    .for_each(|(l, rhs_index)| *l = f_inplace(*l, f(rhs[rhs_index])));
            }
            (StorageIndices::Uncontiguous(lhs_index), StorageIndices::Contiguous(rhs_index)) => {
                let rhs = &rhs[rhs_index.begin_index..rhs_index.end_index];
                lhs_index.zip(rhs.iter())
                    .for_each(|(lhs_index, &r)| lhs[lhs_index] = f_inplace(lhs[lhs_index], f(r)) );
            }
            (StorageIndices::Uncontiguous(lhs_index), StorageIndices::Uncontiguous(rhs_index)) => {
                lhs_index.zip(rhs_index)
                    .for_each(|(lhs_index, rhs_index)| lhs[lhs_index] = f_inplace(lhs[lhs_index], f(rhs[rhs_index])) );
            }
        }
        
        Ok(())
    }
}

impl<T: WithDType> Tensor<T> {
    pub fn check_implace_op(&self) -> crate::Result<()> {
        if self.requires_grad() {
            return Err(crate::Error::InplaceOpInWhenRequiresGrad)
        }
        Ok(())
    }
}