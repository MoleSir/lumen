use std::fmt::Display;
use crate::{Error, Result, Storage, UnsignedIntDType, WithDType};
use super::Tensor;

impl<T: WithDType> Tensor<T> {
    pub fn take<I: UnsignedIntDType>(&self, indices: &Tensor<I>) -> Result<Tensor<T>> {
        let self_storage = self.storage_ref(0);
        let self_storage_len = self_storage.len();

        let mut vec = vec![];
        for index in indices.iter() {
            let value = self_storage.get(index.to_usize())
                .ok_or_else(|| Error::IndexOutOfRangeTake{ storage_len: self_storage_len, index: index.to_usize() })?;
            vec.push(value);
        }
        // TODO: backward
        let storage = Storage::new(vec);
        Ok(Self::from_storage(storage, indices.shape()))
    }

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
    fn test_take_basic_usize() {
        let a = Tensor::new(&[10, 20, 30, 40, 50]).unwrap();
        let idx = Tensor::new(&[0usize, 2, 4]).unwrap();

        let result = a.take(&idx).unwrap();
        assert_eq!(result.to_vec(), [10, 30, 50]);
    }

    #[test]
    fn test_take_repeated_indices() {
        let a = Tensor::new(&[10, 20, 30]).unwrap();
        let idx = Tensor::new(&[1usize, 1, 1]).unwrap();

        let result = a.take(&idx).unwrap();
        assert_eq!(result.to_vec(), [20, 20, 20]);
    }

    #[test]
    fn test_take_multidim_indices_shape_kept() {
        let a = Tensor::new(&[10, 20, 30, 40]).unwrap();
        let idx = Tensor::new(&[[0usize, 3usize],
                                 [1usize, 2usize]]).unwrap();

        let result = a.take(&idx).unwrap();
        let expected = Tensor::new(&[[10, 40],
                                      [20, 30]]).unwrap();

        assert_eq!(result.to_vec(), expected.to_vec());
    }

    #[test]
    fn test_take_u32_indices() {
        let a = Tensor::new(&[5, 6, 7, 8, 9]).unwrap();
        let idx = Tensor::new(&[4u32, 0u32, 2u32]).unwrap();

        let result = a.take(&idx).unwrap();
        assert_eq!(result.to_vec(), [9, 5, 7]);
    }

    #[test]
    fn test_take_out_of_bounds() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let idx = Tensor::new(&[0usize, 5usize]).unwrap(); // 5 越界

        let result = a.take(&idx);
        assert!(result.is_err());
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

