
use crate::{DType, Dim, Layout, Shape, Tensor, TensorId};

#[derive(Clone)]
pub enum IntTensor {
    U8(Tensor<u8>),
    I32(Tensor<i32>),
    U32(Tensor<u32>),
}

impl IntTensor {
    pub fn id(&self) -> TensorId {
        match self {
            Self::U8(t) => t.id(),
            Self::I32(t) => t.id(),
            Self::U32(t) => t.id(),
        }
    }

    pub fn shape(&self) -> &Shape {
        match self {
            Self::U8(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::U32(t) => t.shape(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(t) => t.dtype(),
            Self::I32(t) => t.dtype(),
            Self::U32(t) => t.dtype(),
        }
    }

    pub fn layout(&self) -> &Layout {
        match self {
            Self::U8(t) => t.layout(),
            Self::I32(t) => t.layout(),
            Self::U32(t) => t.layout(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            Self::U8(t) => t.dims(),
            Self::I32(t) => t.dims(),
            Self::U32(t) => t.dims(),
        }
    }

    pub fn dim<D: Dim>(&self, dim: D) -> crate::Result<usize> {
        match self {
            Self::U8(t) => t.dim(dim),
            Self::I32(t) => t.dim(dim),
            Self::U32(t) => t.dim(dim),
        }
    }

    pub fn element_count(&self) -> usize {
        match self {
            Self::U8(t) => t.element_count(),
            Self::I32(t) => t.element_count(),
            Self::U32(t) => t.element_count(),
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            Self::U8(t) => t.is_contiguous(),
            Self::I32(t) => t.is_contiguous(),
            Self::U32(t) => t.is_contiguous(),
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::U8(t) => t.rank(),
            Self::I32(t) => t.rank(),
            Self::U32(t) => t.rank(),
        }
    }

    pub fn flatten_all(&self) -> crate::Result<Self> {
        match self {
            Self::U8(t) => t.flatten_all().map(Self::U8),
            Self::I32(t) => t.flatten_all().map(Self::I32),
            Self::U32(t) => t.flatten_all().map(Self::U32),
        }
    }
}

impl From<Tensor<u8>> for IntTensor {
    fn from(value: Tensor<u8>) -> Self {
        Self::U8(value)
    }
}

impl From<&Tensor<u8>> for IntTensor {
    fn from(value: &Tensor<u8>) -> Self {
        Self::U8(value.clone())
    }
}

impl From<Tensor<i32>> for IntTensor {
    fn from(value: Tensor<i32>) -> Self {
        Self::I32(value)
    }
}

impl From<&Tensor<i32>> for IntTensor {
    fn from(value: &Tensor<i32>) -> Self {
        Self::I32(value.clone())
    }
}

impl From<Tensor<u32>> for IntTensor {
    fn from(value: Tensor<u32>) -> Self {
        Self::U32(value)
    }
}

impl From<&Tensor<u32>> for IntTensor {
    fn from(value: &Tensor<u32>) -> Self {
        Self::U32(value.clone())
    }
}