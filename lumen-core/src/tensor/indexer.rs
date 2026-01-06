use std::fmt::Display;
use crate::{AutogradMetaT, Dim, Error, IntTensor, NumDType, Result, WithDType};
use super::Tensor;

impl<T: WithDType> Tensor<T> {
    fn indexes(&self, indexers: &[Indexer]) -> Result<Self> {
        let mut x = self.clone();
        let mut current_dim = 0;
        for indexer in indexers.iter() {
            x = match indexer {
                Indexer::Select(n) => x.narrow(current_dim, *n, 1)?.squeeze(current_dim)?,
                Indexer::Slice(range) => {
                    let out = x.slice(current_dim, range)?;
                    current_dim += 1;
                    out
                }
            };
        }
        Ok(x)
    }

    pub fn index_select<D: Dim>(&self, indexes: impl Into<IntTensor>, dim: D) -> Result<Self> {   
        let indexes: IntTensor = indexes.into();
        let dim = dim.to_index(self.shape(), "index-select")?;
        let indexes_len = indexes.shape().dims1()?;
        let mut dims = self.dims().to_vec();
        dims[dim] = indexes_len;
        let meta = T::AutogradMeta::on_index_select_op(self, &indexes, dim);
        let storage = match indexes {
            IntTensor::I32(indexes) => self.storage_read().index_select(
                self.layout(),
                &indexes.storage_read(),
                indexes.layout(),
                dim,
            )?,
            IntTensor::U32(indexes) => self.storage_read().index_select(
                self.layout(),
                &indexes.storage_read(),
                indexes.layout(),
                dim,
            )?,
            IntTensor::U8(indexes) => self.storage_read().index_select(
                self.layout(),
                &indexes.storage_read(),
                indexes.layout(),
                dim,
            )?,
        };

        Ok(Self::build(storage, dims, meta))
    }

    /// Gather values across the target dimension.
    ///
    /// # Arguments
    ///
    /// * `self` - The input tensor.
    /// * `indexes` - The indices of elements to gather, this should have same number of dimensions as `self`
    ///   and indexes.dims()[d] <= self.dims()[d] for all dimensions d != dim
    /// * `dim` - the target dimension.
    ///
    /// The resulting tensor has the same shape as `indexes` and use values from `self` indexed on
    /// dimension `dim` by the values in `indexes`.
    pub fn gather<D: Dim>(&self, indexes: impl Into<IntTensor>, dim: D) -> Result<Self> {
        let indexes = indexes.into();
        let dim = dim.to_index(self.shape(), "gather")?;
        let self_dims = self.dims();
        let indexes_dims = indexes.dims();
        let mismatch = if indexes_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(indexes_dims.iter()).enumerate() {
                if i != dim && d1 < d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "gather",
                lhs: self.shape().clone(),
                rhs: indexes.shape().clone(),
            })?
        }

        let storage = match &indexes {
            IntTensor::I32(idx) => self.storage_read().gather(self.layout(), &idx.storage_read(), idx.layout(), dim)?,
            IntTensor::U32(idx) => self.storage_read().gather(self.layout(), &idx.storage_read(), idx.layout(), dim)?,
            IntTensor::U8(idx) => self.storage_read().gather(self.layout(), &idx.storage_read(), idx.layout(), dim)?,
        };

        let meta = T::AutogradMeta::on_gather_op(self, &indexes, dim);
        Ok(Self::build(storage, indexes.shape(), meta))
    }
}

impl<T: NumDType> Tensor<T> {
    pub fn index_add<D: Dim>(&self, indexes: impl Into<IntTensor>, source: &Tensor<T>, dim: D) -> Result<Self> {
        let indexes: IntTensor = indexes.into();
        let dim = dim.to_index(self.shape(), "index-add")?;
        
        let source_dims = source.dims();
        let self_dims = self.dims();
        if source_dims.len() != self_dims.len() {
             return Err(Error::ShapeMismatchBinaryOp { 
                 op: "index-add", 
                 lhs: self.shape().clone(), 
                 rhs: source.shape().clone() 
             }.into());
        }

        let indexes_len = indexes.shape().dims1()?;
        for (i, (&d_self, &d_src)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
            if i == dim {
                if d_src != indexes_len {
                    return Err(Error::ShapeMismatchBinaryOp { op: "index-add (dim mismatch)", lhs: self.shape().clone(), rhs: source.shape().clone() }.into());
                }
            } else if d_self != d_src {
                return Err(Error::ShapeMismatchBinaryOp { op: "index-add", lhs: self.shape().clone(), rhs: source.shape().clone() }.into());
            }
        }

        let storage = match &indexes {
            IntTensor::I32(idx) => self.storage_read().index_add(
                self.layout(),
                &idx.storage_read(),
                idx.layout(),
                &source.storage_read(),
                source.layout(),
                dim,
            )?,
            IntTensor::U32(idx) => self.storage_read().index_add(
                self.layout(),
                &idx.storage_read(),
                idx.layout(),
                &source.storage_read(),
                source.layout(),
                dim,
            )?,
            IntTensor::U8(idx) => self.storage_read().index_add(
                self.layout(),
                &idx.storage_read(),
                idx.layout(),
                &source.storage_read(),
                source.layout(),
                dim,
            )?,
        };

        let meta = T::AutogradMeta::on_index_add_op(self, &indexes, source, dim);
        Ok(Self::build(storage, self_dims.to_vec(), meta))
    } 

    pub fn scatter_add<D: Dim>(&self, indexes: impl Into<IntTensor>, source: &Self, dim: D) -> Result<Self> {
        let indexes = indexes.into();
        let dim = dim.to_index(self.shape(), "scatter-add")?;
        self.scatter_checks(&indexes, source, dim)?;

        let storage = match &indexes {
            IntTensor::I32(idx) => self.storage_read().scatter_add(
                self.layout(),
                &idx.storage_read(),
                idx.layout(),
                &source.storage_read(),
                source.layout(),
                dim,
            )?,
            IntTensor::U32(idx) => self.storage_read().scatter_add(
                self.layout(),
                &idx.storage_read(),
                idx.layout(),
                &source.storage_read(),
                source.layout(),
                dim,
            )?,
            IntTensor::U8(idx) => self.storage_read().scatter_add(
                self.layout(),
                &idx.storage_read(),
                idx.layout(),
                &source.storage_read(),
                source.layout(),
                dim,
            )?,
        };

        let meta = T::AutogradMeta::on_scatter_add_op(self, &indexes, source, dim);
        Ok(Self::build(storage, self.shape(), meta))
    }

    fn scatter_checks(&self, indexes: &IntTensor, source: &Self, dim: usize) -> Result<()> {
        let source_dims = source.dims();
        let self_dims = self.dims();
        let mismatch = if source_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter (self, src)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        if indexes.dims() != source.dims() {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter (indexes, src)",
                lhs: indexes.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        Ok(())
    }
}

impl<T: WithDType> Tensor<T> {
    pub fn matrix_get(&self, row: usize, col: usize) -> Result<T> {
        self.index((row, col))?.to_scalar()
    }

    pub fn matrix_set(&self, row: usize, col: usize, val: T) -> Result<()> {
        self.index((row, col))?.set_scalar(val)
    }

    pub fn vector_get(&self, n: usize) -> Result<T> {
        self.index(n)?.to_scalar()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indexer {
    Select(usize),
    Slice(Slice),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slice {
    pub start: usize, 
    pub end: Option<isize>, 
    pub step: usize
}

impl Slice {
    pub fn new(start: usize, end: Option<isize>, step: usize) -> Self {
        Self { start, end, step }
    }

    pub fn len(&self) -> usize {
        self.clone().count()
    }
}

impl Iterator for Slice {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        match self.end {
            Some(end) if end < 0 => {
                let value = self.start;
                self.start += self.step;
                Some(value)
            }
            Some(end) => {
                if self.start < end as usize {
                    let value = self.start;
                    self.start += self.step;
                    Some(value)
                } else {
                    None
                }
            }
            None => {
                let value = self.start;
                self.start += self.step;
                Some(value)
            }
        }
    }
}

impl Display for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let step_part = match self.step {
            1 => format!(""),
            _ => format!(":{}", self.step),
        };
        match self.end {
            Some(end) => write!(f, "{}:{}{}", self.start, end, step_part),
            None => write!(f, "{}:{}", self.start, step_part),
        }
    }
}

impl From<usize> for Indexer {
    fn from(index: usize) -> Self {
        Indexer::Select(index)
    }
}

impl From<Slice> for Indexer {
    fn from(value: Slice) -> Self {
        Indexer::Slice(value)
    }
}

impl From<std::ops::Range<usize>> for Indexer {
    fn from(value: std::ops::Range<usize>) -> Self {
        let range = Slice::new(value.start, Some(value.end as isize), 1);
        range.into()
    }
}

impl From<std::ops::RangeFrom<usize>> for Indexer {
    fn from(value: std::ops::RangeFrom<usize>) -> Self {
        let range = Slice::new(value.start, None, 1);
        range.into()
    }
}

impl From<std::ops::RangeFull> for Indexer {
    fn from(_: std::ops::RangeFull) -> Self {
        let range = Slice::new(0, None, 1);
        range.into()
    }
}

pub trait IndexOp<T, D: WithDType> {
    fn index(&self, index: T) -> Result<Tensor<D>>;
}

impl<I: Into<Indexer>, D: WithDType> IndexOp<I, D> for Tensor<D> {
    fn index(&self, index: I) -> Result<Tensor<D>> {
        self.indexes(&[index.into()])
    }
}

impl<I: Into<Indexer>, D: WithDType> IndexOp<(I,), D> for Tensor<D> {
    fn index(&self, (index,): (I,)) -> Result<Tensor<D>> {
        self.indexes(&[index.into()])
    }
}

macro_rules! index_op_tuple {
    ($($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($t),*, D: WithDType> IndexOp<($($t,)*), D> for Tensor<D>
        where
            $($t: Into<Indexer>,)*
        {
            fn index(&self, ($($t,)*): ($($t,)*)) -> Result<Tensor<D>> {
                self.indexes(&[$($t.into(),)*])
            }
        }
    };
}

index_op_tuple!(I1, I2);
index_op_tuple!(I1, I2, I3);
index_op_tuple!(I1, I2, I3, I4);
index_op_tuple!(I1, I2, I3, I4, I5);

#[macro_export]
macro_rules! s {
    // s!(start:end)
    ($start:tt : $end:expr) => {
        Slice::new($start as usize, Some($end as isize), 1)
    };
    // s!(start:end:step)
    ($start:tt : $end:tt : $step:expr) => {
        Slice::new($start as usize, Some($end as isize), $step as usize)
    };
    // s!(start:)
    ($start:tt :) => {
        Slice::new($start as usize, None, 1)
    };
    // s!(start::step)
    ($start:tt :: $step:expr) => {
        Slice::new($start as usize, None, $step as usize)
    };
    // s!(:$end)
    (: $end:tt) => {
        Slice::new(0, Some($end as isize), 1)
    };
    // s!(:$end:$step)
    (: $end:tt : $step:expr) => {
        Slice::new(0, Some($end as isize), $step as usize)
    };
    // s!(::$step)
    (:: $step:expr) => {
        Slice::new(0, None, $step as usize)
    };
    // s!(:)
    (:) => {
        Slice::new(0, None, 1)
    };
}

#[cfg(test)]
#[allow(unused)]
mod test {
    use crate::DType;
    use super::*;

    #[test]
    fn test_index_select_basic() {
        // [[ 0,  1,  2,  3],
        //  [ 4,  5,  6,  7],
        //  [ 8,  9, 10, 11]]
        let arr = Tensor::arange(0, 12).unwrap().reshape((3, 4)).unwrap();

        let indices = Tensor::new(&[0, 2]).unwrap();
        let selected = arr.index_select(indices, 0).unwrap();
        
        assert_eq!(selected.shape().dims(), &[2, 4]);
        let data = selected.to_vec();
        assert_eq!(data, vec![0, 1, 2, 3, 8, 9, 10, 11]);

        let indices_col = Tensor::new(&[1]).unwrap();
        let selected_col = arr.index_select(indices_col, 1).unwrap();

        assert_eq!(selected_col.shape().dims(), &[3, 1]);
        let data_col = selected_col.to_vec();
        assert_eq!(data_col, vec![1, 5, 9]);
    }

    #[test]
    fn test_index_select_duplicates_and_reorder() {
        let arr = Tensor::arange(0, 5).unwrap(); // [0, 1, 2, 3, 4]

        let indices = Tensor::new(&[4, 0, 0, 1]).unwrap();
        let selected = arr.index_select(indices, 0).unwrap();

        assert_eq!(selected.shape().dims(), &[4]);
        let data = selected.to_vec();
        assert_eq!(data, vec![4, 0, 0, 1]);
    }

    #[test]
    fn test_index_select_out_of_bounds() {
        let arr = Tensor::arange(0, 10).unwrap();
        let indices = Tensor::new(&[0, 10]).unwrap(); 
        
        let result = arr.index_select(indices, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_add_basic() {
        let dst = Tensor::<i32>::zeros((3, 3)).unwrap();
        let src = Tensor::<i32>::ones((2, 3)).unwrap();
        let indices = Tensor::new(&[0, 2]).unwrap();
        
        let result = dst.index_add(indices, &src, 0).unwrap();
        
        // [[1, 1, 1],
        //  [0, 0, 0],
        //  [1, 1, 1]]
        let data = result.to_vec();
        assert_eq!(data, vec![
            1, 1, 1, 
            0, 0, 0, 
            1, 1, 1
        ]);
    }

    #[test]
    fn test_index_add_accumulate() {
        let dst = Tensor::<i32>::zeros((5,)).unwrap(); // [0, 0, 0, 0, 0]        
        let src = Tensor::new(&[10, 20, 30]).unwrap();
        let indices = Tensor::new(&[1, 1, 3]).unwrap();
        
        let result = dst.index_add(indices, &src, 0).unwrap();
        
        // dst[0] = 0
        // dst[1] = 0 + 10 + 20 = 30
        // dst[2] = 0
        // dst[3] = 0 + 30 = 30
        // dst[4] = 0
        let data = result.to_vec();
        assert_eq!(data, vec![0, 30, 0, 30, 0]);
    }

    #[test]
    fn test_index_add_dim_mismatch() {
        let dst = Tensor::<i32>::zeros((3, 3)).unwrap();        
        let src = Tensor::<i32>::ones((2, 3)).unwrap();
        let indices = Tensor::new(&[0, 1, 2]).unwrap();
        
        let result = dst.index_add(indices, &src, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_add_inner_dim() {
        // Shape: 2x3
        // [[0, 0, 0],
        //  [0, 0, 0]]
        let dst = Tensor::<i32>::zeros((2, 3)).unwrap();
        // Source: 2x1
        let src = Tensor::new(&[
            [5],
            [5]
        ]).unwrap(); // Shape 2x1
        let indices = Tensor::new(&[1]).unwrap();        
        let result = dst.index_add(indices, &src, 1).unwrap();
        
        // 预期:
        // [[0, 5, 0],
        //  [0, 5, 0]]
        let data = result.to_vec();
        assert_eq!(data, vec![0, 5, 0, 0, 5, 0]);
    }


    #[test]
    fn test_gather_dim_1() {
        // Source: 2x2
        // [[1, 2],
        //  [3, 4]]
        let src = Tensor::new(&[
            [1, 2],
            [3, 4]
        ]).unwrap();
        
        // 我们想从第 1 维 (列) 取值。
        // indexes 形状必须与输出一致。
        // [[0, 0],  -> 取 src[0,0], src[0,0]
        //  [1, 0]]  -> 取 src[1,1], src[1,0]
        let indices = Tensor::new(&[
            [0, 0],
            [1, 0]
        ]).unwrap();

        let result = src.gather(&indices, 1).unwrap();

        // 预期结果:
        // [[1, 1],
        //  [4, 3]]
        let data = result.to_vec();
        assert_eq!(data, vec![1, 1, 4, 3]);
    }

    #[test]
    fn test_gather_dim_0() {
        // Source: 3x2
        // [[10, 20],
        //  [30, 40],
        //  [50, 60]]
        let src = Tensor::new(&[
            [10, 20],
            [30, 40],
            [50, 60]
        ]).unwrap();

        // 沿维度 0 (行) gather
        // [[1, 2], -> src[1,0]=30, src[2,1]=60
        //  [0, 1]] -> src[0,0]=10, src[1,1]=40
        let indices = Tensor::new(&[
            [1, 2],
            [0, 1]
        ]).unwrap();

        let result = src.gather(&indices, 0).unwrap();

        // 预期结果:
        // [[30, 60],
        //  [10, 40]]
        let data = result.to_vec();
        assert_eq!(data, vec![30, 60, 10, 40]);
    }

    #[test]
    fn test_gather_3d() {
        // Shape: 2x2x2
        // Block 0: [[0, 1], [2, 3]]
        // Block 1: [[4, 5], [6, 7]]
        let src = Tensor::new(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap().reshape((2, 2, 2)).unwrap();

        // Gather dim 1 (中间维度)
        // 我们保持 dim 0 和 dim 2 不变，只改变 dim 1 的索引
        // index shape: 2x1x2 (我们在 dim 1 上做降维提取)
        let indices = Tensor::<u32>::zeros((2, 1, 2)).unwrap(); // 全是 0

        // 逻辑:
        // out[0, 0, 0] = src[0, idx[0,0,0], 0] = src[0,0,0] = 0
        // out[0, 0, 1] = src[0, idx[0,0,1], 1] = src[0,0,1] = 1
        // out[1, 0, 0] = src[1, idx[1,0,0], 0] = src[1,0,0] = 4
        // out[1, 0, 1] = src[1, idx[1,0,1], 1] = src[1,0,1] = 5
        
        let result = src.gather(&indices, 1).unwrap();
        
        assert_eq!(result.dims(), &[2, 1, 2]);
        let data = result.to_vec();
        assert_eq!(data, vec![0, 1, 4, 5]);
    }


    #[test]
    fn test_scatter_add_1d_accumulate() {
        // 类似于直方图统计
        let dst = Tensor::<i32>::zeros((5,)).unwrap();
        
        // Source 和 Indices 形状一致
        let src = Tensor::new(&[1, 1, 1, 1]).unwrap();
        let indices = Tensor::new(&[0, 2, 0, 4]).unwrap();

        // scatter_add(dim=0)
        // dst[0] += 1
        // dst[2] += 1
        // dst[0] += 1 (累加)
        // dst[4] += 1
        let result = dst.scatter_add(indices, &src, 0).unwrap();

        // 预期: [2, 0, 1, 0, 1]
        let data = result.to_vec();
        assert_eq!(data, vec![2, 0, 1, 0, 1]);
    }

    #[test]
    fn test_scatter_add_2d_dim1() {
        // Dst: 2x3 zeros
        let dst = Tensor::<i32>::zeros((2, 3)).unwrap();
        
        // Src: 2x2
        // [[10, 20],
        //  [30, 40]]
        let src = Tensor::new(&[
            [10, 20],
            [30, 40]
        ]).unwrap();

        // Indices: 2x2
        // [[0, 2], -> 将 10 加到 dst[0,0], 20 加到 dst[0,2]
        //  [1, 1]] -> 将 30 加到 dst[1,1], 40 加到 dst[1,1] (累加)
        let indices = Tensor::new(&[
            [0, 2],
            [1, 1]
        ]).unwrap();

        let result = dst.scatter_add(indices, &src, 1).unwrap();

        // 预期:
        // Row 0: [10, 0, 20]
        // Row 1: [0, 70, 0] (30+40=70)
        let data = result.to_vec();
        assert_eq!(data, vec![10, 0, 20, 0, 70, 0]);
    }

    #[test]
    fn test_scatter_add_3d() {
        // Dst: 2x2x2 zeros
        let dst = Tensor::<i32>::zeros((2, 2, 2)).unwrap();
        
        // Src: 2x1x2 (dim 1 is smaller)
        let src = Tensor::ones((2, 1, 2)).unwrap(); // All ones
        
        // Indices: 2x1x2
        // Block 0: [[1, 0]] -> dst[0, 1, 0] += 1, dst[0, 0, 1] += 1
        // Block 1: [[0, 0]] -> dst[1, 0, 0] += 1, dst[1, 0, 1] += 1
        let indices = Tensor::new(&[1, 0, 0, 0]).unwrap().reshape((2, 1, 2)).unwrap();

        let result = dst.scatter_add(indices, &src, 1).unwrap();

        // Check specifics:
        // dst[0, 1, 0] should be 1
        // dst[0, 0, 1] should be 1
        // dst[1, 0, 0] should be 1
        // dst[1, 0, 1] should be 1
        // Others 0
        let res_vec = result.to_vec();
        // Flattened index check:
        // 2x2x2 -> stride [4, 2, 1]
        // [0,1,0] -> 2 -> val 1
        // [0,0,1] -> 1 -> val 1
        // [1,0,0] -> 4 -> val 1
        // [1,0,1] -> 5 -> val 1
        
        // Vec: [0, 1, 1, 0, 1, 1, 0, 0]
        assert_eq!(res_vec, vec![0, 1, 1, 0, 1, 1, 0, 0]);
    }

    #[test]
    fn test_scatter_add_shape_mismatch() {
        let dst = Tensor::<i32>::zeros((2, 2)).unwrap();
        let src = Tensor::<i32>::ones((2, 2)).unwrap();
        // Indices shape mismatch with Src
        let indices = Tensor::new(&[0]).unwrap(); 

        let result = dst.scatter_add(indices, &src, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_scalar_dim_reduction() {
        let arr = Tensor::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();
        let sub = arr.index(1).unwrap();
        assert_eq!(sub.shape().dims(), &[5, 5]);

        let sub = arr.index((2, 3)).unwrap();
        assert_eq!(sub.shape().dims(), &[5]); 
    }

    #[test]
    fn test_index_range_basic() {
        let arr = Tensor::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

        let sub = arr.index(s!(1:3)).unwrap();
        assert_eq!(sub.shape().dims(), &[2, 5, 5]);

        let sub = arr.index((s!(1:3), s!(3:4), 1)).unwrap();
        assert_eq!(sub.shape().dims(), &[2, 1]);
    }

    #[test]
    fn test_index_full_and_mixed() {
        let arr = Tensor::<i32>::zeros((5, 5, 5)).unwrap();

        let sub = arr.index((s!(1:3), .., 1..2)).unwrap();
        assert_eq!(sub.shape().dims(), &[2, 5, 1]);

        let sub = arr.index((2, .., s!(0:2))).unwrap();
        assert_eq!(sub.shape().dims(), &[5, 2]);

        let sub = arr.index((s!(0:2), s!(2:5), s!(1:3))).unwrap();
        assert_eq!(sub.shape().dims(), &[2, 3, 2]);
    }

    #[test]
    fn test_index_out_of_bounds() {
        let arr = Tensor::<i32>::zeros((5, 5, 5)).unwrap();
        let result = arr.index(10);
        assert!(result.is_err());

        let result = arr.index(s!(3:10));
        assert!(result.is_err());
    }
    
    #[test]
    fn test_index_scalar_and_values() {
        let arr = Tensor::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

        let sub = arr.index(1).unwrap();
        let expected = Tensor::arange(25, 50).unwrap().reshape((5, 5)).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));
    }

    #[test]
    fn test_index_range_values() {
        let arr = Tensor::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

        let sub = arr.index(s!(1:3)).unwrap();
        let expected = Tensor::arange(25, 75).unwrap().reshape((2, 5, 5)).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));
    }

    #[test]
    fn test_index_mixed_values() {
        let arr = Tensor::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();
        let sub = arr.index((2, 3)).unwrap();
        let expected = Tensor::arange(65, 70).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));

        let sub = arr.index((s!(1:3), s!(3:5), 2)).unwrap();
        let mut vals = Vec::new();
        for i in 1..3 {
            for j in 3..5 {
                vals.push(i * 25 + j * 5 + 2);
            }
        }
        let expected = Tensor::from_vec(vals, (2, 2)).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));
    }

    #[test]
    fn test_index_with_full_dim() {
        let arr = Tensor::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();
        let sub = arr.index((s!(1:3), .., 1..2)).unwrap();

        let expected = arr.index((s!(1:3), s!(0:5), s!(1:2))).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));
    }

    #[test]
    fn test_macro() {
        let t = (0..12usize);
        let t = (2usize..);
        assert_eq!(s!(1:10), Slice {start:1, end: Some(10), step:1});

        assert!(
            s!(1:20).zip((1..20))
                .all(|(a, b)| a == b)
        );
    
        assert!(
            s!(1:13:3).zip((1..13).step_by(3))
                .all(|(a, b)| a == b)
        );
    
        assert!(
            s!(1:).zip((1..).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            s!(1::2).zip((1..).step_by(2).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            s!(:20).zip((0..20usize))
                .all(|(a, b)| a == b)
        );

        assert!(
            s!(:20:5).zip((0..20usize).step_by(5))
                .all(|(a, b)| a == b)
        );

        assert!(
            s!(::2).zip((0..).step_by(2).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            s!(:).zip((0..).take(100))
                .all(|(a, b)| a == b)
        );
    }
}

