mod construct;
mod indexer;
mod iter;
pub mod display;
mod shape;
mod arith;
mod matmul;
mod reduce;
mod broadcast;
mod convert;
mod boolean;

pub use construct::ToTensor;
use std::{borrow::Borrow, hash::Hash, sync::Arc};
pub use indexer::{Slice, IndexOp};
use crate::{AutogradInfo, Error, FloatDType, Op, Storage};
use super::{DType, Dim, DimCoordinates, DimNCoordinates, Layout, NumDType, Shape, StorageArc, StorageIndices, StorageMut, StorageRef, WithDType};
pub use iter::*;
pub use indexer::*;

#[derive(Clone)]
pub struct Tensor<T: WithDType>(Arc<TensorImpl<T>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

struct TensorImpl<T: WithDType> {
    id: TensorId,
    storage: Option<StorageArc<T>>,
    layout: Layout,
    meta: T::AutogradMeta,
}

impl TensorId {
    pub fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn value(&self) -> usize {
        self.0
    }
}

impl Borrow<usize> for TensorId {
    fn borrow(&self) -> &usize {
        &self.0
    }
}

impl<T: WithDType> Hash for Tensor<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.id.0.hash(state);
    }
} 

impl<T: WithDType> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id.0 == other.0.id.0
    }
}

impl<T: WithDType> Eq for Tensor<T> {}

impl<T: WithDType> Tensor<T> {
    pub fn is_scalar(&self) -> bool {
        self.shape().is_scalar()
    }

    pub fn check_scalar(&self) -> crate::Result<()> {
        if !self.is_scalar() {
            Err(Error::NotScalar)?
        } else {
            Ok(())
        }
    }

    pub fn to_scalar(&self) -> crate::Result<T> {
        self.check_scalar()?;
        let v = self.storage_read()?.get_unchecked(self.layout().start_offset());
        Ok(v)
    }

    pub fn set_scalar(&self, val: T) -> crate::Result<()> {
        self.check_scalar()?;
        self.storage_write()?.set_unchecked(self.layout().start_offset(), val);
        Ok(())
    }

    pub fn storage_ref<'a>(&'a self, start_offset: usize) -> crate::Result<StorageRef<'a, T>> {
        self.0.storage.as_ref()
            .ok_or(crate::Error::MetaTensor)
            .map(|s| s.get_ref(start_offset))
    }

    pub fn storage_mut<'a>(&'a self, start_offset: usize) -> crate::Result<StorageMut<'a, T>> {
        self.0.storage.as_ref()
            .ok_or(crate::Error::MetaTensor)
            .map(|s| s.get_mut(start_offset))
    }

    pub fn storage_ptr(&self, start_offset: usize) -> crate::Result<*mut T> {
        self.0.storage.as_ref()
            .ok_or(crate::Error::MetaTensor)
            .map(|s| s.get_ptr(start_offset))
    }

    pub fn is_meta(&self) -> bool {
        self.0.storage.is_none()
    }
}

impl<T: WithDType> Tensor<T> {
    pub fn id(&self) -> TensorId {
        self.0.id
    }

    pub fn shape(&self) -> &Shape {
        self.0.layout.shape()
    }

    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn dim<D: Dim>(&self, dim: D) -> crate::Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn storage_read(&self) -> crate::Result<std::sync::RwLockReadGuard<'_, Storage<T>>> {
        self.0.storage.as_ref()
            .ok_or(crate::Error::MetaTensor)
            .map(|s| s.read())
    }

    pub fn storage_write(&self) -> crate::Result<std::sync::RwLockWriteGuard<'_, Storage<T>>> {
        self.0.storage.as_ref()
            .ok_or(crate::Error::MetaTensor)
            .map(|s| s.write())
    }
    
    pub fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn to_vec(&self) -> crate::Result<Vec<T>> {
        self.iter().map(|i| i.collect())
    }

    /// Returns an iterator over **storage indices**.
    ///
    /// This iterator yields the linear (flat) indices as they are laid out
    /// in the underlying storage buffer. The order depends on the memory
    /// layout (e.g., row-major / column-major / with strides).
    ///
    /// Example for shape = (2, 2) in row-major layout:
    /// yields: `0, 1, 2, 3`
    pub fn storage_indices(&self) -> StorageIndices {
        self.layout().storage_indices()
    }

    /// Returns an iterator over **dimension coordinates**.
    ///
    /// This iterator yields the multi-dimensional coordinates
    /// (e.g., `[i, j, k, ...]`) of each element in the array, independent
    /// of the physical storage layout.
    ///
    /// Example for shape = (2, 2):
    /// yields: `[0, 0], [0, 1], [1, 0], [1, 1]`
    pub fn dim_coordinates(&self) -> DimCoordinates {
        self.shape().dim_coordinates()
    }

    pub fn dims_coordinates<const N: usize>(&self) -> crate::Result<DimNCoordinates<N>> {
        self.shape().dims_coordinates::<N>()
    }

    pub fn dim2_coordinates(&self) -> crate::Result<DimNCoordinates<2>> {
        self.shape().dim2_coordinates()
    }

    pub fn dim3_coordinates(&self) -> crate::Result<DimNCoordinates<3>> {
        self.shape().dim3_coordinates()
    }

    pub fn dim4_coordinates(&self) -> crate::Result<DimNCoordinates<4>> {
        self.shape().dim4_coordinates()
    }

    pub fn dim5_coordinates(&self) -> crate::Result<DimNCoordinates<5>> {
        self.shape().dim5_coordinates()
    }
}

impl<T: NumDType> Tensor<T> {
    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> crate::Result<bool> {
        if self.shape() != other.shape() {
            return Ok(false);
        }
        Ok(
            self.iter()?.zip(other.iter()?).all(|(a, b)| a.close(b, rtol, atol))
        )
    }
}

impl<T: FloatDType> Tensor<T> {
    pub fn detach(&self) -> Self {
        if !self.requires_grad() {
            self.clone()
        } else {
            Self(Arc::new(TensorImpl { 
                id: TensorId::new(), 
                storage: self.0.storage.clone(), 
                layout: self.layout().clone(), 
                meta: AutogradInfo::val(), 
            }))
        }
    }

    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.0.meta.requires_grad()
    }
    
    #[inline]
    pub fn set_requires_grad(&self, mode: bool) {
        self.0.meta.set_requires_grad(mode);
    }

    #[inline]
    pub fn op(&self) -> Option<&Op<T>> {
        self.0.meta.op()
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.0.meta.is_leaf()
    }
}
