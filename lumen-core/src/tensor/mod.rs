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
use crate::{AutogradInfo, Error, FloatDType, Op, Result};
use super::{DType, Dim, DimCoordinates, DimNCoordinates, Layout, NumDType, Shape, Storage, StorageArc, StorageIndices, StorageMut, StorageRef, WithDType};
pub use iter::*;
pub use indexer::*;

#[derive(Clone)]
pub struct Tensor<T: WithDType>(pub(crate) Arc<TensorImpl<T>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

pub struct TensorImpl<T: WithDType> {
    pub(crate) id: TensorId,
    pub(crate) storage: StorageArc<T>,
    pub(crate) layout: Layout,
    pub(crate) meta: T::AutogradMeta,
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

    pub fn check_scalar(&self) -> Result<()> {
        if !self.is_scalar() {
            Err(Error::NotScalar)?
        } else {
            Ok(())
        }
    }

    pub fn to_scalar(&self) -> Result<T> {
        self.check_scalar()?;
        let v = self.storage_ref(self.layout().start_offset()).get_unchecked(0);
        Ok(v)
    }

    pub fn item(&self) -> Result<Self> {
        let scalar = self.to_scalar()?;
        Tensor::new(scalar)
    }

    pub fn set_scalar(&self, val: T) -> Result<()> {
        self.check_scalar()?;
        self.storage_mut(self.layout().start_offset()).set_unchecked(0, val);
        Ok(())
    }

    #[inline]
    pub fn storage_ref<'a>(&'a self, start_offset: usize) -> StorageRef<'a, T> {
        self.0.storage.get_ref(start_offset)
    }

    #[inline]
    pub fn storage_mut<'a>(&'a self, start_offset: usize) -> StorageMut<'a, T> {
        self.0.storage.get_mut(start_offset)
    }

    #[inline]
    pub fn storage_ptr(&self, start_offset: usize) -> *mut T {
        self.0.storage.get_ptr(start_offset)
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

    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn storage_read(&self) -> std::sync::RwLockReadGuard<'_, Storage<T>> {
        self.0.storage.0.read().unwrap()
    }

    pub fn storage_write(&self) -> std::sync::RwLockWriteGuard<'_, Storage<T>> {
        self.0.storage.0.write().unwrap()
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

    pub fn to_vec(&self) -> Vec<T> {
        self.iter().collect()
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

    pub fn dims_coordinates<const N: usize>(&self) -> Result<DimNCoordinates<N>> {
        self.shape().dims_coordinates::<N>()
    }

    pub fn dim2_coordinates(&self) -> Result<DimNCoordinates<2>> {
        self.shape().dim2_coordinates()
    }

    pub fn dim3_coordinates(&self) -> Result<DimNCoordinates<3>> {
        self.shape().dim3_coordinates()
    }

    pub fn dim4_coordinates(&self) -> Result<DimNCoordinates<4>> {
        self.shape().dim4_coordinates()
    }

    pub fn dim5_coordinates(&self) -> Result<DimNCoordinates<5>> {
        self.shape().dim5_coordinates()
    }
}

impl<T: NumDType> Tensor<T> {
    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a.close(b, rtol, atol))
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
