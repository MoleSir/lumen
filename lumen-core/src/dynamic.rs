use crate::{DType, Dim, Layout, Shape, Tensor, TensorId};

#[derive(Clone)]
pub enum IntTensor {
    U8(Tensor<u8>),
    I32(Tensor<i32>),
    U32(Tensor<u32>),
    USize(Tensor<usize>),
}

#[derive(Clone)]
pub enum FloatTensor {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
}

impl IntTensor {
    pub fn id(&self) -> TensorId {
        match self {
            Self::U8(t) => t.id(),
            Self::I32(t) => t.id(),
            Self::U32(t) => t.id(),
            Self::USize(t) => t.id(),
        }
    }

    pub fn shape(&self) -> &Shape {
        match self {
            Self::U8(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::U32(t) => t.shape(),
            Self::USize(t) => t.shape(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(t) => t.dtype(),
            Self::I32(t) => t.dtype(),
            Self::U32(t) => t.dtype(),
            Self::USize(t) => t.dtype(),
        }
    }

    pub fn layout(&self) -> &Layout {
        match self {
            Self::U8(t) => t.layout(),
            Self::I32(t) => t.layout(),
            Self::U32(t) => t.layout(),
            Self::USize(t) => t.layout(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            Self::U8(t) => t.dims(),
            Self::I32(t) => t.dims(),
            Self::U32(t) => t.dims(),
            Self::USize(t) => t.dims(),
        }
    }

    pub fn dim<D: Dim>(&self, dim: D) -> crate::Result<usize> {
        match self {
            Self::U8(t) => t.dim(dim),
            Self::I32(t) => t.dim(dim),
            Self::U32(t) => t.dim(dim),
            Self::USize(t) => t.dim(dim),
        }
    }

    pub fn element_count(&self) -> usize {
        match self {
            Self::U8(t) => t.element_count(),
            Self::I32(t) => t.element_count(),
            Self::U32(t) => t.element_count(),
            Self::USize(t) => t.element_count(),
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            Self::U8(t) => t.is_contiguous(),
            Self::I32(t) => t.is_contiguous(),
            Self::U32(t) => t.is_contiguous(),
            Self::USize(t) => t.is_contiguous(),
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::U8(t) => t.rank(),
            Self::I32(t) => t.rank(),
            Self::U32(t) => t.rank(),
            Self::USize(t) => t.rank(),
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

impl From<Tensor<usize>> for IntTensor {
    fn from(value: Tensor<usize>) -> Self {
        Self::USize(value)
    }
}

impl From<&Tensor<usize>> for IntTensor {
    fn from(value: &Tensor<usize>) -> Self {
        Self::USize(value.clone())
    }
}

impl From<Tensor<f32>> for FloatTensor {
    fn from(value: Tensor<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<&Tensor<f32>> for FloatTensor {
    fn from(value: &Tensor<f32>) -> Self {
        Self::F32(value.clone())
    }
}

impl From<Tensor<f64>> for FloatTensor {
    fn from(value: Tensor<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<&Tensor<f64>> for FloatTensor {
    fn from(value: &Tensor<f64>) -> Self {
        Self::F64(value.clone())
    }
}