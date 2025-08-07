use crate::rng;

use super::{
    error::TensorError, 
    iter::{TensorIntoIter, TensorIter, TensorIterMut}, 
    range::Range, 
    shape::Shape, 
    storage::Storage,
    layout::Layout,
    backward::{Backward, UnaryTensorBackward},
    op,
};
use rand::{self, Rng};
use rand_distr::{self, Distribution};
use anyhow::Result;
use std::{
    cell::{Ref, RefCell, RefMut}, 
    collections::HashSet, 
    hash::Hash, 
    ops::{Add, Div, Mul, Neg, Sub}, 
    rc::Rc
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TensorInner {
    /// Meta data
    pub(crate) layout: Layout,

    /// Storage area
    pub(crate) storage: Storage,

    /// Operate
    pub(crate) operate: Backward,

    /// Grad record
    pub(crate) requires_grad: bool,
    pub(crate) grad: Option<Tensor>,
}

impl Eq for TensorInner {}

impl TensorInner {
    fn new(data: Vec<f64>, shape: Shape, operate: Backward, requires_grad: bool) -> Self {
        Self {
            layout: Layout::from_shape(shape),
            storage: Storage::new(data),
            grad: None,
            requires_grad,
            operate
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor{pub(crate) inner: Rc<RefCell<TensorInner>>}

impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.as_ptr().hash(state);
    }
}

impl Tensor {
    /// Build a tensor in given data and shape.
    /// If data and shape can't match, return a `Err``
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::from([[1, 2], [3, 4]]);
    /// assert!(Tensor::build([1., 2., 3.], [2, 2]).is_err());
    /// ```
    pub fn build<D, S>(data: D, shape: S) -> Result<Self> 
    where 
        D: Into<Vec<f64>>, S: Into<Shape>
    {
        let data = data.into();
        let shape = shape.into();
        let size = shape.element_size();
        if data.len() != size {
            Err(TensorError::DataLenShapeUnmatch)?
        } else {
            Ok(Self::new_root(data, shape))
        }
    }

    /// Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::zeros([3, 4]);
    /// assert!(t.iter().all(|v| *v == 0.));
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        let data = vec![0.; shape.element_size()];
        Self::new_root(data, shape)
    }

    /// Returns a tensor filled with the scalar value 0, with the same size as input. 
    /// Equal to `Tensor::zeros(self.shape().clone())`
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::from([[1, 2], [3, 4]]);
    /// let zt = t.zeros_like();
    /// let zt = Tensor::zeros(t.shape().clone());
    /// ```
    pub fn zeros_like(&self) -> Self {
        Self::zeros(self.shape().clone())
    }

    /// Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::ones([5, 4]);
    /// assert!(t.iter().all(|v| *v == 1.));
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        let data = vec![1.; shape.element_size()];
        Self::new_root(data, shape)
    }

    /// Returns a tensor filled with the scalar value 1, with the same size as input. 
    /// Equal to `Tensor::ones(self.shape().clone())`
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::from([[1, 2], [3, 4]]);
    /// let zt = t.ones_like();
    /// let zt = Tensor::ones(t.shape().clone());
    /// ```
    pub fn ones_like(&self) -> Self {
        Self::ones(self.shape().clone())
    }

    /// Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).
    /// The shape of the tensor is defined by the variable argument size.
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::rand([5, 4]);
    /// ```
    pub fn rand<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        let size = shape.element_size();
        let mut rng = rand::rng();
        let data = (0..size).map(|_| rng.random()).collect::<Vec<f64>>();
        Self::new_root(data, shape)    
    }

    pub fn rand_range<S: Into<Shape>>(low: f64, up: f64, shape: S) -> Result<Self> {
        let shape = shape.into();
        let size = shape.element_size();
        let mut rng = rand::rng();
        let uniform = rand::distr::Uniform::new(low, up)?;
        let data = (0..size).map(|_| rng.sample::<f64, _>(uniform)).collect::<Vec<f64>>();
        Ok(Self::new_root(data, shape))
    }

    pub fn rand_normal<S: Into<Shape>>(mean: f64, stdev: f64, shape: S) -> Result<Self> {
        let shape = shape.into();
        let size = shape.element_size();
        let mut rng = rand::rng();
        let normal = rand_distr::Normal::new(mean, stdev).unwrap();
        let data = (0..size).map(|_| normal.sample(&mut rng)).collect::<Vec<f64>>();
        Ok(Self::new_root(data, shape))
    }

    /// Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1), 
    /// with the shape defined by the variable argument size
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::from([[1, 2], [3, 4]]);
    /// let rt = t.rand_like();
    /// ```
    pub fn rand_like(&self) -> Self {
        Self::rand(self.shape().clone())
    }

    /// Returns a tensor filled with the scalar `val`, with the shape defined by the variable argument size
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::fill([5, 4], 2.);
    /// assert!(t.iter().all(|v| *v == 2.));
    /// ```
    pub fn fill<S: Into<Shape>>(shape: S, val: f64) -> Self {  
        let shape = shape.into();
        let data = vec![val; shape.element_size()];
        Self::new_root(data, shape)
    }

    /// Returns a tensor filled with the scalar `val`, with the same size as input. 
    /// Equal to `Tensor::ones(self.shape().clone())`
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::from([[1, 2], [3, 4]]);
    /// let t = t.fill_like(3.);
    /// assert!(t.iter().all(|v| *v == 3.));
    /// ```
    pub fn fill_like(&self, val: f64) -> Self {  
        Self::fill(self.shape().clone(), val)
    }

    /// Retuens one-hot tensor!
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::one_hot(2, 5).unwrap();
    /// assert!(t.allclose(&Tensor::build([0., 0., 1., 0., 0.], [5]).unwrap()));
    /// ```
    pub fn one_hot(label: usize, num_classes: usize) -> Result<Self> {
        if label >= num_classes {
            Err(TensorError::IndexOutOfRange)?
        } else {
            let mut data = vec![0.; num_classes];
            data[label] = 1.;
            Self::build(data, [num_classes])
        }
    }

    /// Retuens vector tensor from `start` to `end`, with shape (`count`)
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::linspace(0., 3., 3);
    /// assert!(t.allclose(&Tensor::build([0., 1., 2.], [3]).unwrap()));
    /// ```
    pub fn linspace(start: f64, end: f64, count: usize) -> Self {
        let step = (end - start) / count as f64;
        let mut val = start;
        let mut data = Vec::new();
        for _ in 0..count {
            data.push(val);
            val += step;
        }

        Self::new_root(data, [count])
    }

    pub(crate) fn new<D, S>(data: D, shape: S, operate: Backward, requires_grad: bool) -> Self 
    where 
        D: Into<Vec<f64>>, S: Into<Shape>
    {
        let data = data.into();
        let shape = shape.into();
        Self::from_inner(TensorInner::new(
            data,
            shape,
            operate,
            requires_grad,
        ))
    }

    pub(crate) fn new_root<D, S>(data: D, shape: S) -> Self 
    where 
        D: Into<Vec<f64>>, S: Into<Shape>
    {
        let data = data.into();
        let shape = shape.into();
        Self::from_inner(TensorInner::new(
            data,
            shape,
            Backward::Root,
            false,
        ))
    }

    pub(crate) fn new_inner(layout: Layout, storage: Storage, operate: Backward, requires_grad: bool) -> Self {
        Self::from_inner( TensorInner {
            layout,
            storage,
            operate,
            requires_grad,
            grad: None,
        } )
    }

    fn from_inner(inner: TensorInner) -> Self {
        Self { inner: Rc::new(RefCell::new(inner)) }
    }
}

impl Tensor {
    /// Returns a tensor with the same data and number of elements as self but with the specified shape. 
    /// This method returns a view if shape is compatible with the current shape. 
    /// 
    /// Returns error when the element size is not equal to self
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap();
    /// let col_t = t.reshape([1, 4]).unwrap();
    /// let row_t = t.reshape([4, 1]).unwrap();
    /// 
    /// let t = Tensor::build(vec![
    ///     1., 2., 3., 
    ///     4., 5., 6.,
    ///     
    ///     7., 8., 9.,
    ///     10., 11., 12.,
    /// ], [2, 2, 3]).unwrap();
    /// let v = t.reshape([6, 2]).unwrap();
    /// ```
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if self.is_contiguous() {
            Self::view(&self, shape)
        } else {
            Self::build(self.to_vec(), shape)
        }
    }

    /// Returns a view if shape is compatible with the current shape. 
    /// 
    /// Returns error when the element size is not equal to self or self is not contiguous
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap();
    /// let col_t = t.view([1, 4]).unwrap();
    /// assert_eq!(t.storage_ptr(), col_t.storage_ptr());
    /// assert!(t.view([2, 4]).is_err());
    /// ```
    pub fn view<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if self.is_contiguous() {
            if self.element_size() != shape.element_size() {
                return Err(TensorError::DataLenShapeUnmatch)?;
            }

            assert_eq!(self.storage_offset(), 0);

            Ok(Self::new_inner(
                Layout::from_shape(shape),
                self.storage().clone(),
                if self.requires_grad() { 
                    Backward::Unary(UnaryTensorBackward::View(self.clone())) 
                } else { 
                    Backward::Root 
                },
                self.requires_grad(),
            ))
        } else {
            Err(TensorError::ViewUnContiguous)?
        }
    }

    /// Copy the storage of self
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap();
    /// let nt = t.copy();
    /// assert_ne!(t.storage_ptr(), nt.storage_ptr());
    /// ```
    pub fn copy(&self) -> Self {
        Self::new_root(self.to_vec(), self.shape().clone())
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.iter().map(|v| v.clone()).collect()
    }

    /// Get a slice from self!
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::zeros([6, 6, 6]);
    /// let a = t.slice(&[rng!(1::2), rng!(1::2), rng!(1::2)]).unwrap();
    /// let b = a.slice(&[rng!(::2), rng!(1:), rng!(:)]).unwrap();
    /// assert_eq!(*t.shape(), &[6, 6, 6]);
    /// assert_eq!(*t.stride(), &[36, 6, 1]);
    /// assert_eq!(t.storage_offset(), 0);
    /// assert_eq!(*a.shape(), &[3, 3, 3]);
    /// assert_eq!(*a.stride(), &[72, 12, 2]);
    /// assert_eq!(a.storage_offset(), 43);
    /// assert_eq!(*b.shape(), &[2, 2, 3]);
    /// assert_eq!(*b.stride(), &[144, 12, 2]);
    /// assert_eq!(b.storage_offset(), 55);
    /// assert!(b.slice(&[rng!(3:4), rng!(:), rng!(:)]).is_err());
    /// 
    /// // Slice with index to remove the dim
    /// let t = Tensor::zeros([5, 5, 5]);
    /// let a = t.slice(&[rng!(1)]).unwrap();
    /// let b = t.slice(&[rng!(:), rng!(1:2)]).unwrap();
    /// let c = a.slice(&[rng!(1), rng!(1)]).unwrap();
    /// assert_eq!(*a.shape(), &[5, 5]);
    /// assert_eq!(*a.stride(), &[5, 1]);
    /// assert_eq!(a.storage_offset(), 25);
    /// assert_eq!(*b.shape(), &[5, 1, 5]);
    /// assert_eq!(*b.stride(), &[25, 5, 1]);
    /// assert_eq!(b.storage_offset(), 5);
    /// assert_eq!(*c.shape(), &[]);
    /// assert_eq!(*c.stride(), &[]);
    /// assert_eq!(c.storage_offset(), 31); 
    /// ```
    pub fn slice(&self, ranges: &[Range]) -> Result<Self> {
        if ranges.len() > self.dim_size() {
            Err(TensorError::DimensionsUnmatch)?
        } else {
            if ranges.len() < self.dim_size() {
                let mut ranges = ranges.to_vec();
                for _ in 0..self.dim_size() - ranges.len() {
                    ranges.push(rng!(:));
                }
                self.do_slice(&ranges)
            } else {
                self.do_slice(&ranges)
            }
        }
    }

    fn do_slice(&self, ranges: &[Range]) -> Result<Self> {
        assert_eq!(self.dim_size(), ranges.len());

        let mut increate_storage_offset = 0;
        let mut new_shape = vec![0; self.dim_size()];
        let mut new_stride = vec![0; self.dim_size()];

        /*
            stride = origin_stride * step
            shape = range size
            storage_offset += origin_stride * start
        */
        for (index, (&dim_size, range)) in self.shape().iter().zip(ranges.iter()).enumerate().rev() {
            let (start, step, size) = Tensor::get_range_info(dim_size, range);
            if size == 0 {
                return Err(TensorError::SliceZeroSize)?;
            }
            new_stride[index] = self.stride()[index] * step;
            new_shape[index] = size;
            increate_storage_offset += self.stride()[index] * start;
        } 

        let remove_indices = ranges.iter().enumerate()
            .filter(|(_, r)| !r.is_range())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        let new_stride = new_stride.into_iter()
            .enumerate()
            .filter(|(i, _)| {
                !remove_indices.contains(i)
            })
            .map(|(_, ns)| ns)
            .collect();

        let new_shape = new_shape.into_iter()
            .enumerate()
            .filter(|(i, _)| {
                !remove_indices.contains(i)
            })
            .map(|(_, ns)| ns)
            .collect::<Vec<_>>();

        Ok(Self::new_inner(
            Layout::new(
                new_shape.into(), 
                new_stride, 
                self.storage_offset() + increate_storage_offset, 
                false
            ), 
            self.storage().clone(), 
            if self.requires_grad() { 
                Backward::Unary(UnaryTensorBackward::Slice(self.clone(), ranges.to_vec())) 
            } else { 
                Backward::Root
            }, 
            self.requires_grad()
        ))
    }

    /// Transpose of self
    /// 
    /// Only for two dim tensor
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// 
    /// let t = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap();
    /// let tt = t.transpose().unwrap();
    /// assert_eq!(tt.to_vec(), vec![1., 3., 2., 4.]);
    /// assert_eq!(t.storage_ptr(), tt.storage_ptr());
    /// 
    /// let t = Tensor::zeros([4, 3, 5]);
    /// let a = t.slice(&[rng!(1), rng!(:), rng!(:)]).unwrap();
    /// let at = a.transpose().unwrap();
    /// ```
    pub fn transpose(&self) -> Result<Self> {
        if self.dim_size() != 2 {
            Err(TensorError::DimensionsUnmatch)?
        } else {
            let shape = [self.shape()[1], self.shape()[0]].to_vec();
            let stride = [self.stride()[1], self.stride()[0]].to_vec();
            Ok(Self::from_inner( TensorInner {
                layout: Layout::new(
                    shape.into(),
                    stride,
                    self.storage_offset(),
                    false,
                ),
                storage: self.storage().clone(),
                requires_grad: false,
                grad: None,
                operate: Backward::Root,
            }))
        }
    }

    /// Get range info(start, step, size) from a given dim size and range
    /// 
    /// # Exmaple
    /// ```rust
    /// use lumen_core::*;
    /// assert_eq!(Tensor::get_range_info(3, &rng!(::2)), (0, 2, 2));
    /// assert_eq!(Tensor::get_range_info(3, &rng!(1:)), (1, 1, 2));
    /// assert_eq!(Tensor::get_range_info(3, &rng!(:)), (0, 1, 3));
    /// ```
    pub fn get_range_info(dim_size: usize, range: &Range) -> (usize, usize, usize) {
        match range {
            Range::Index { index, used: _ } => {
                if *index >= dim_size {
                    (*index, 1, 0)
                } else {
                    (*index, 1, 1)
                }
            }
            Range::Range { start, end, step } => {
                let range = match *end {
                    Some(end) => {
                        if end > dim_size {
                            Range::range(*start, dim_size, *step)
                        } else {
                            range.clone()
                        }
                    }
                    None => Range::range(*start, dim_size, *step),
                };
                let size = (&range).into_iter().count();
                (*start, *step, size)
            }
        }
    }
}

impl Tensor {
    /// Get value from tensor in position `indices`
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::build(vec![1., 2., 3., 4., 5., 6.], [2, 3]).unwrap();
    /// assert_eq!(1., t.get(&[0, 0]).unwrap());
    /// assert_eq!(5., t.get(&[1, 1]).unwrap());
    /// ```
    pub fn get(&self, indices: &[usize]) -> Result<f64> {
        let index = self.flat_index(indices)?;
        Ok(self.storage().get(index).unwrap().clone())
    }

    /// Set value for tensor in position `indices` with `val`
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let mut t = Tensor::build(vec![
    ///     1., 2., 3., 
    ///     4., 5., 6.,
    ///     7., 8., 9.,
    /// 
    ///     1., 2., 3., 
    ///     4., 5., 6.,
    ///     7., 8., 9.,
    /// ], [2, 3, 3]).unwrap();
    /// assert_eq!(5., t.get(&[0, 1, 1]).unwrap());
    /// assert_eq!(9., t.get(&[1, 2, 2]).unwrap());
    /// assert_eq!(7., t.get(&[1, 2, 0]).unwrap());
    /// t.set(&[1, 2, 0], 1.).unwrap();
    /// assert_eq!(1., t.get(&[1, 2, 0]).unwrap());
    /// ```
    pub fn set(&self, indices: &[usize], val: f64) -> Result<()> {
        let index = self.flat_index(indices)?;
        *self.storage().get_mut(index).unwrap() = val;
        Ok(())
    }

    pub fn increase(&self, indices: &[usize], val: f64) -> Result<()> {
        let index = self.flat_index(indices)?;
        *self.storage().get_mut(index).unwrap() += val;
        Ok(())
    }

    pub fn get_ref(&self, indices: &[usize]) -> Result<&f64> {
        let index = self.flat_index(indices)?;
        let p = self.storage().get_ptr(index).unwrap();
        unsafe { Ok(&*p) }
    }

    pub fn get_ref_mut(&mut self, indices: &[usize]) -> Result<&mut f64> {
        let index = self.flat_index(indices)?;
        let p = self.storage().get_ptr_mut(index).unwrap();
        unsafe { Ok(&mut *p) } 
    }

    pub(crate) fn flat_index(&self, indices: &[usize]) -> Result<usize> {
        assert_eq!(self.dim_size(), self.stride().len());
        if self.stride().len() != indices.len() {
            Err(TensorError::IndicesShapeUnmatch)?
        } else {
            if self.shape().iter().zip(indices.iter()).any(|(s, i)| i >= s) {
                return Err(TensorError::IndexOutOfRange)?;
            }
            Ok(self.calculate_flat_index(indices))
        }
    }

    pub(crate) fn calculate_flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(self.dim_size(), self.stride().len());
        let index = self.stride().iter().zip(indices.iter())
            .map(|(s, i)| s * i)
            .sum::<usize>();
        index + self.storage_offset()
    }
}

impl Tensor {
    /// Just like iter's map
    /// 
    /// # Example
    /// ```rust
    /// use lumen_core::*;
    /// let t1 = Tensor::zeros([3, 4]);
    /// let t2 = t1.map(|a| a + 1.);
    /// assert_eq!(vec![1.; 3 * 4], t2.to_vec());
    /// ```
    pub fn map<F>(&self, f: F) -> Tensor 
    where 
        F: Fn(f64) -> f64
    {
        let tensor = self.copy();
        tensor.iter_mut().for_each(|v| *v = f(*v));
        tensor
    } 

    /// Just like iter's map, but map to self
    /// 
    /// # Example
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::zeros([3, 4]);
    /// t.map_(|a| a + 1.);
    /// assert_eq!(vec![1.; 3 * 4], t.to_vec());
    /// ```
    pub fn map_<F>(&self, f: F) 
    where 
        F: Fn(f64) -> f64
    {
        self.iter_mut().for_each(|v| *v = f(*v));
    }

    /// Just like iter's zip with
    /// 
    /// # Example
    /// ```rust
    /// use lumen_core::*;
    /// let t1 = Tensor::from([[1, 2], [3, 4]]);
    /// let t2 = Tensor::build([5., 1., 2., 34.], [2, 2]).unwrap();
    /// let t3 = t1.zip_with(&t2, |a, b| a + b).unwrap();
    /// assert_eq!(t3, Tensor::build([1. + 5., 2. + 1., 3. + 2., 4. + 34.], [2, 2]).unwrap());
    /// ```
    pub fn zip_with<F: Fn(f64, f64)->f64>(&self, rhs: &Tensor, f: F) -> Result<Self> {
        self.check_same_shape(rhs)?;
        let data = self.iter().zip(rhs.iter())
            .map(|(a, b)| f(*a, *b))
            .collect::<Vec<_>>();
        Self::build(data, self.shape().clone())
    }

    /// Zip with a scalar
    /// 
    /// # Example
    /// ```rust
    /// use lumen_core::*;
    /// let t1 = Tensor::from([[1, 2], [3, 4]]);
    /// let t2 = t1.zip_with_scalar(3., |a, b| a + b).unwrap();
    /// assert_eq!(t2, Tensor::build([1. + 3., 2. + 3., 3. + 3., 4. + 3.], [2, 2]).unwrap());
    /// ```
    pub fn zip_with_scalar<F: Fn(f64, f64)->f64>(&self, scalar: f64, f: F) -> Result<Self> {
        let data = self.iter()
            .map(|&v| f(v, scalar))
            .collect::<Vec<_>>();
        Self::build(data, self.shape().clone())
    }

    /// Checks if self and other all close
    /// 
    /// # Example
    /// ```rust
    /// use lumen_core::*;
    /// let t1 = Tensor::from([[1, 2], [3, 4]]);
    /// let t2 = Tensor::build([5., 1., 2., 34.], [2, 2]).unwrap();
    /// assert!(!t1.allclose(&t2));
    /// ```
    pub fn allclose(&self, other: &Tensor) -> bool {
        *self.shape() == *other.shape() && 
        self.iter().zip(other.iter())
            .all(|(a, b)| {
                approx::abs_diff_eq!(*a, *b, epsilon = 1e-8) ||
                approx::relative_eq!(*a, *b, epsilon = 0.00001)
            } )
    }

    pub fn check_same_shape(&self, rhs: &Tensor) -> Result<()> {
        if *self.shape() != *rhs.shape() {
            Err(TensorError::DifferentShape)?
        } else {
            Ok(())
        }
    }
}

impl Tensor {
    pub(crate) fn try_init_grad(&self) {
        assert!(self.requires_grad());
        if self.inner.borrow().grad.is_none() {
            self.inner.borrow_mut().grad = Some(Self::zeros_like(self));
        }
    } 

    pub fn require_grad(self) -> Self {
        self.inner.borrow_mut().requires_grad = true;
        self
    }

    pub fn zero_grad(&self) {
        if let Some(grad) = self.grad() {
            grad.iter_mut().for_each(|v| *v = 0.);
        }
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        
        fn do_backward(visited: &mut HashSet<Tensor>, tensor: &Tensor) {
            if !tensor.requires_grad() || visited.contains(tensor) {
                return;
            }

            visited.insert(tensor.clone());
            
            match &tensor.inner.borrow().operate {
                Backward::Root => {},
                Backward::Binary(bin_op) => {
                    bin_op.backward(tensor);
                    let (l, r) = bin_op.operands();
                    do_backward(visited, l);
                    do_backward(visited, r);
                }
                Backward::Unary(un_op) => {
                    un_op.backward(tensor);
                    do_backward(visited, un_op.operand());
                }
                Backward::Multi(multi_op) => {
                    multi_op.backward(tensor);
                    for pre_t in multi_op.operands() {
                        do_backward(visited, pre_t);
                    }
                }
            }
        }

        self.inner.borrow_mut().grad = Some(Tensor::ones_like(self));
        do_backward(&mut visited, self);
    }
}

impl Tensor {
    pub(crate) fn inner(&self) -> Ref<'_, TensorInner> {
        self.inner.borrow()
    }

    pub(crate) fn inner_mut(&self) -> RefMut<'_, TensorInner> {
        self.inner.borrow_mut()
    }

    pub fn dim_size(&self) -> usize {
        self.inner().layout.shape.dim_size()
    }

    pub fn element_size(&self) -> usize {
        self.inner().layout.shape.element_size()
    }

    pub fn shape(&self) -> Ref<'_, Vec<usize>> {
        Ref::map(self.inner(), |inner| &inner.layout.shape.dims)
    }

    pub fn stride(&self) -> Ref<'_, Vec<usize>> {
        Ref::map(self.inner(), |inner| &inner.layout.stride)
    }

    pub fn storage_offset(&self) -> usize {
        self.inner().layout.storage_offset
    }

    pub fn storage_ptr(&self) -> *mut Vec<f64> {
        self.inner().storage.data.as_ptr()
    }

    pub fn storage<'a>(&'a self) -> Ref<'a, Storage> {
        Ref::map(self.inner(), |inner| &inner.storage)
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner().layout.is_contiguous
    }

    pub fn requires_grad(&self) -> bool {
        self.inner().requires_grad
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.inner().grad.clone()
    }

    pub(crate) fn with_operate(&self, op: Backward) {
        self.inner_mut().operate = op;
    }
}

impl Tensor {
    /// Get iter from tensor
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap();
    /// let data_copy: Vec<_> = t.iter().map(|v| *v).collect();
    /// assert_eq!(data_copy, vec![1., 2., 3., 4.]);
    /// ```
    pub fn iter(&self) -> TensorIter {
        TensorIter::new(self)
    }

    /// Get iter mut from tensor
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;
    /// let mut t = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap();
    /// t.iter_mut().for_each(|mut v| *v = *v +  1. );
    /// let data_copy = t.to_vec();
    /// assert_eq!(data_copy, vec![2., 3., 4., 5.]);
    /// ```
    pub fn iter_mut(&self) -> TensorIterMut {
        TensorIterMut::new(self)
    }
}

impl Tensor {
    /// Matrix dot operate
    /// 
    /// See `tensor::matmul` for more detail
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        op::matmul(self, rhs)
    }

    /// Returns a new tensor with the exponential of the elements of the input tensor input.
    /// 
    /// See `tensor::exp` for more detail
    pub fn exp(&self) -> Self {
        op::exp(self)
    }

    /// Takes the power of each element in input with exponent and returns a tensor with the result.
    /// 
    /// See `tensor::pow` for more detai
    pub fn pow(&self, exponent: f64) -> Self {
        op::pow(self, exponent)
    }

    /// Returns a new tensor with the hyperbolic tangent of the elements of input
    /// 
    /// See `tensor::tanh` for more detai
    pub fn tanh(&self) -> Self {
        op::tanh(self)
    }

    /// Returns a new tensor with the sigmoid of the input tensor input.
    /// 
    /// See `tensor::sigmoid` for more detai
    pub fn sigmoid(&self) -> Self {
        op::sigmoid(self)
    }

    /// Returns a new tensor with the relu of the input tensor input.
    /// 
    /// See `tensor::relu` for more detai
    pub fn relu(&self) -> Self {
        op::relu(self)
    }

    /// Returns a new tensor with the abs of the input tensor input.
    /// 
    /// See `tensor::abs` for more detai
    pub fn abs(&self) -> Self {
        op::abs(self)
    }

    /// Returns a new tensor with the sqrt of the input tensor input.
    /// 
    /// See `tensor::sqrt` for more detail
    pub fn sqrt(&self) -> Self {
        op::sqrt(self)
    }

    /// Returns a new tensor with the log of the input tensor input.
    /// 
    /// See `tensor::log` for more detail
    pub fn log(&self, base: f64) -> Self {
        op::log(self, base)
    }

    /// Returns a new tensor with the ln of the input tensor input.
    /// 
    /// See `tensor::log` for more detail
    pub fn ln(&self) -> Self {
        op::ln(self)
    }

    pub fn sum(&self) -> f64 {
        self.iter().sum::<f64>()
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.element_size() as f64
    }
}

impl Tensor {
    /// In-place version of tensor::add
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn add_(&self, other: &Self) -> Result<()> {
        self.zip_with_(other, |a, b| a + b)
    }

    /// In-place version of tensor::sub
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn sub_(&self, other: &Self) -> Result<()> {
        self.zip_with_(other, |a, b| a - b)
    }

    /// In-place version of tensor::mul
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn mul_(&self, other: &Self) -> Result<()> {
        self.zip_with_(other, |a, b| a * b)
    }

    /// In-place version of tensor::div
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn div_(&self, other: &Self) -> Result<()> {
        self.zip_with_(other, |a, b| a / b)
    }

    /// In-place version of tensor::exp
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn exp_(&self) {
        self.map_(|a| a.exp());
    }

    /// In-place version of tensor::pow
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn pow_(&self, exponent: f64) {
        self.map_(|a| a.powf(exponent));
    }

    /// In-place version of tensor::tanh
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn tanh_(&self) {
        self.map_(|a| a.tanh());
    }

    /// In-place version of tensor::sigmoid
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn sigmoid_(&self) {
        self.map_(|x| 1. / (1. + (-x).exp()));
    }

    /// In-place version of tensor::relu
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn relu_(&self) {
        self.map_(|x| x.powf(0.));
    }

    /// In-place version of tensor::abs
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn abs_(&self) {
        self.map_(|x| x.abs());
    }

    /// In-place version of tensor::log
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn log_(&self, base: f64) {
        self.map_(|x| x.log(base));
    }
    
    /// In-place version of tensor::ln
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn ln_(&self) {
        self.map_(|x| x.ln());
    }
    
    /// In-place version of tensor::neg
    /// 
    /// # Warning
    /// 
    /// If `self` alreay used as an autograd expression, call this method may make grad error!
    pub fn neg_(&self) {
        self.map_(|x| -x);
    }

    /// In-place version of Tensor::zip_with
    /// 
    /// # Example
    /// ```rust
    /// use lumen_core::*;
    /// let t1 = Tensor::from([[1, 2], [3, 4]]);
    /// let t2 = Tensor::build([5., 1., 2., 34.], [2, 2]).unwrap();
    /// t1.zip_with_(&t2, |a, b| a + b).unwrap();
    /// assert_eq!(t1, Tensor::build([1. + 5., 2. + 1., 3. + 2., 4. + 34.], [2, 2]).unwrap());
    /// ```
    pub fn zip_with_<F: Fn(f64, f64)->f64>(&self, rhs: &Tensor, f: F) -> Result<()> {
        self.check_same_shape(rhs)?;
        self.iter_mut().zip(rhs.iter())
            .for_each(|(a, b)| *a = f(*a, *b));
        Ok(())
    }

    /// In-place version of Tensor::zip_with_scalar
    /// 
    /// # Example
    /// ```rust
    /// use lumen_core::*;
    /// let t = Tensor::from([[1, 2], [3, 4]]);
    /// t.zip_with_scalar_(3., |a, b| a + b).unwrap();
    /// assert_eq!(t, Tensor::build([1. + 3., 2. + 3., 3. + 3., 4. + 3.], [2, 2]).unwrap());
    /// ```
    pub fn zip_with_scalar_<F: Fn(f64, f64)->f64>(&self, scalar: f64, f: F) -> Result<()> {
        self.iter_mut()
            .for_each(|v| *v = f(*v, scalar));
        Ok(())
    }
}

impl IntoIterator for Tensor {
    type IntoIter = TensorIntoIter;
    type Item = f64;
    fn into_iter(self) -> Self::IntoIter {
        TensorIntoIter::new(self)
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    /// Add `self` and `rhs`. See `tensor::add` for more detail
    /// 
    /// # Panic
    /// 
    /// When self's shape != rhs's shape 
    fn add(self, rhs: &Tensor) -> Self::Output {
        op::add(self, rhs).unwrap()
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        &self + &rhs
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        self + &rhs
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        &self + rhs
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    /// Sub `self` and `rhs`. See `tensor::sub` for more detail
    /// 
    /// # Panic
    /// 
    /// When self's shape != rhs's shape 
    fn sub(self, rhs: &Tensor) -> Self::Output {
        op::sub(self, rhs).unwrap()
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        &self - &rhs
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        self - &rhs
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        &self - rhs
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    /// Mul `self` and `rhs`. See `tensor::mul` for more detail
    /// 
    /// # Panic
    /// 
    /// When self's shape != rhs's shape 
    fn mul(self, rhs: &Tensor) -> Self::Output {
        op::mul(self, rhs).unwrap()
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        &self * rhs
    }
}

impl Div<&Tensor> for &Tensor {
    /// Div `self` and `rhs`. See `tensor::div` for more detail
    /// 
    /// # Panic
    /// 
    /// When self's shape != rhs's shape 
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        op::div(self, rhs).unwrap()
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        &self / &rhs
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        self / &rhs
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        &self / rhs
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        op::neg(&self)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        op::neg(self)
    }
}

/// Crate Tensor form array
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;
/// let t = Tensor::from([0., 3., 3.]);
/// assert!(t.allclose(&Tensor::build([0., 3., 3.], [3]).unwrap()));
/// let t = Tensor::from([0, 3, 3]);
/// assert!(t.allclose(&Tensor::build([0., 3., 3.], [3]).unwrap()));
/// ```
macro_rules! from_1darray {
    ($t:tt) => {
        impl<const N: usize> From<[$t; N]> for Tensor {
            fn from(value: [$t; N]) -> Self {
                let data: Vec<_> = value.into_iter().map(|v| v as f64).collect();
                Self::new_root(data, [N])
            }
        }        
    };
}

from_1darray!(f64);
from_1darray!(f32);
from_1darray!(u8);
from_1darray!(u16);
from_1darray!(u32);
from_1darray!(u64);
from_1darray!(i8);
from_1darray!(i16);
from_1darray!(i32);
from_1darray!(i64);

/// Crate Tensor form array
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;
/// let t = Tensor::from([[1., 2.], [3., 4.]]);
/// assert!(t.allclose(&Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap()));
/// let t = Tensor::from([[1, 2], [3, 4]]);
/// assert!(t.allclose(&Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap()));
/// let a1 = Tensor::build([1., 2., 1., 2.], [2, 2]).unwrap().require_grad();
/// let a2 = Tensor::from([[1, 2], [1, 2]]);
/// assert!(a1.allclose(&a2))
/// ```
macro_rules! from_2darray {
    ($t:tt) => {
        impl<const N1: usize, const N2: usize> From<[[$t; N1]; N2]> for Tensor {
            fn from(value: [[$t; N1]; N2]) -> Self {
                let mut data = Vec::with_capacity(N1 * N2);
                for r in 0..N2 {
                    for c in 0..N1 {
                        data.push(value[r][c] as f64);
                    }
                }
                Self::new_root(data, [N2, N1])
            }
        }
    };
}

from_2darray!(f64);
from_2darray!(f32);
from_2darray!(u8);
from_2darray!(u16);
from_2darray!(u32);
from_2darray!(u64);
from_2darray!(i8);
from_2darray!(i16);
from_2darray!(i32);
from_2darray!(i64);

macro_rules! from_3darray {
    ($t:tt) => {
        impl<const N1: usize, const N2: usize, const N3: usize> From<[[[$t; N1]; N2]; N3]> for Tensor {
            fn from(value: [[[$t; N1]; N2]; N3]) -> Self {
                let mut data = Vec::with_capacity(N1 * N2 * N3);
                for k in 0..N2 {
                    for r in 0..N2 {
                        for c in 0..N1 {
                            data.push(value[k][r][c] as f64);
                        }
                    }
                }
                Self::new_root(data, [N3, N2, N1])
            }
        }
    };
}

from_3darray!(f64);
from_3darray!(f32);
from_3darray!(u8);
from_3darray!(u16);
from_3darray!(u32);
from_3darray!(u64);
from_3darray!(i8);
from_3darray!(i16);
from_3darray!(i32);
from_3darray!(i64);

#[cfg(test)]
#[allow(unused)]
mod test {
    use super::*;
    use crate::rng;

    #[test]
    fn test_base_op() {
        // Mul Add
        let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
        let y = &a + &a;
        assert!(y.allclose(&Tensor::build([1. + 1., 2. + 2., 3. + 3., 4. + 4.], [2, 2]).unwrap()));
        y.backward();

        let grad_a = a.grad().unwrap();
        assert!(grad_a.allclose(&Tensor::build([2., 2., 2., 2.], [2, 2]).unwrap()));

        // Mul Add
        let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
        let y = &a * &a;
        assert!(y.allclose(&Tensor::build([1., 4., 9., 16.], [2, 2]).unwrap()));
        y.backward();

        let grad_a = a.grad().unwrap();
        assert!(grad_a.allclose(&Tensor::build([2., 4., 6., 8.], [2, 2]).unwrap()));

        // Comp
        let a = Tensor::build([1., 2., 4., 7.], [2, 2]).unwrap().require_grad();
        let b = Tensor::build([3., 4., 5., 6.], [2, 2]).unwrap().require_grad();
        let c = Tensor::build([9., 1., 8., 7.], [2, 2]).unwrap().require_grad();

        let y = (&a + &b) * &c;
        y.backward();

        let grad_a = a.grad().unwrap();
        assert!(grad_a.allclose(&Tensor::build([9., 1., 8., 7.], [2, 2]).unwrap()));

        let grad_b = b.grad().unwrap();
        assert!(grad_b.allclose(&Tensor::build([9., 1., 8., 7.], [2, 2]).unwrap()));

        let grad_c = c.grad().unwrap();
        assert!(grad_c.allclose(&Tensor::build([4., 6., 9., 13.], [2, 2]).unwrap()));

    }
}