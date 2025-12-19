use std::sync::Arc;
use rand_distr::{Distribution, StandardNormal, StandardUniform};
use crate::{AutogradInfo, Error, FloatDType, Layout, NumDType, Result, Shape, Storage, StorageArc, WithDType};
use super::{Tensor, TensorId, TensorImpl};

impl<T: WithDType> Tensor<T> {
    /// Creates a new `Tensor` from any supported Rust array or slice.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::new(&[1, 2, 3]).unwrap();
    /// println!("{}", a.shape());
    /// ```
    pub fn new<A: ToTensor<T>>(array: A) -> Result<Self> {
        Self::new_impl(array, T::AutogradMeta::default())
    }

    pub(crate) fn new_impl<A: ToTensor<T>>(array: A, meta: T::AutogradMeta) -> Result<Self> {
        let shape = array.shape()?;
        let storage = array.to_storage()?;
        Ok(Self::build(storage, shape, meta))
    }

    /// Creates an array full with a constant `value`.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::full((2, 2), 7).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn full<S: Into<Shape>>(shape: S, value: T) -> Result<Self> {
        Self::full_impl(shape, value, T::AutogradMeta::default())
    }

    pub(crate) fn full_impl<S: Into<Shape>>(shape: S, value: T, meta: T::AutogradMeta) -> Result<Self> {
        let shape: Shape = shape.into();
        let storage = Storage::new(vec![value; shape.element_count()]);
        Ok(Self::build(storage, shape, meta))
    }

    /// Creates a new `Tensor` directly from a storage buffer and shape.
    ///
    /// Typically used internally, but can also be used when you already
    /// have a `Storage<T>` prepared.
    pub(crate) fn from_storage<S: Into<Shape>>(storage: Storage<T>, shape: S) -> Self {
        let tensor_ = TensorImpl {
            id: TensorId::new(),
            storage: StorageArc::new(storage),
            layout: Layout::contiguous(shape),
            meta: Default::default(),
        };
        Tensor(Arc::new(tensor_))
    }

    pub(crate) fn from_op<S: Into<Shape>>(storage: Storage<T>, shape: S, meta: T::AutogradMeta) -> Self {
        let tensor_ = TensorImpl {
            id: TensorId::new(),
            storage: StorageArc::new(storage),
            layout: Layout::contiguous(shape),
            meta,
        };
        Tensor(Arc::new(tensor_))
    }

    pub(crate) fn build<S: Into<Shape>>(storage: Storage<T>, shape: S, meta: T::AutogradMeta) -> Self {
        let tensor_ = TensorImpl {
            id: TensorId::new(),
            storage: StorageArc::new(storage),
            layout: Layout::contiguous(shape),
            meta,
        };
        Tensor(Arc::new(tensor_))
    }
}

impl<T: NumDType> Tensor<T> {
    /// Creates an array of zeros with the given shape.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::<f32>::zeros((2, 3)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        Self::zeros_impl(shape, T::AutogradMeta::default())
    }

    pub(crate) fn zeros_impl<S: Into<Shape>>(shape: S, meta: T::AutogradMeta) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::zeros(&shape);
        Ok(Self::build(storage, shape, meta))
    }

    /// Creates a zero-filled array with the same shape as `self`.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::<i32>::ones((2, 2)).unwrap();
    /// let b = a.zero_like().unwrap();
    /// println!("{}", b);
    /// ```
    pub fn zero_like(&self) -> Result<Self> {
        Self::zeros(self.shape())
    }

    /// Creates an array of ones with the given shape.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::<f64>::ones((3, 3)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S) -> Result<Self> {
        Self::ones_impl(shape, T::AutogradMeta::default())
    }

    pub(crate) fn ones_impl<S: Into<Shape>>(shape: S, meta: T::AutogradMeta) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::ones(&shape);
        Ok(Self::build(storage, shape, meta))
    }

    /// Creates a one-filled array with the same shape as `self`.
    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape())
    }

    /// Creates a 1-D array with values from `start` up to (but not including) `end`.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::arange(0., 5.).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn arange(start: T, end: T) -> Result<Self> {
        Self::arange_impl(start, end, T::AutogradMeta::default())
    }

    pub(crate) fn arange_impl(start: T, end: T, meta: T::AutogradMeta) -> Result<Self> {
        let storage = T::to_range_storage(start, end)?;
        let shape = storage.len();
        Ok(Self::build(storage, shape, meta))
    }

    /// Creates an array from a flat `Vec<T>` and explicit shape.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1, 2, 3, 4], (2, 2)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn from_vec<V: Into<Vec<T>>, S: Into<Shape>>(vec: V, shape: S) -> Result<Self> {
        Self::from_vec_impl(vec, shape, T::AutogradMeta::default())
    }

    pub(crate) fn from_vec_impl<V: Into<Vec<T>>, S: Into<Shape>>(vec: V, shape: S, meta: T::AutogradMeta) -> Result<Self> {
        let vec = vec.into();
        let shape: Shape = shape.into();
        if shape.element_count() != vec.len() {
            Err(Error::ElementSizeMismatch { expected: vec.len(), got: shape.element_count(), op: "from_vec" })?
        }
        let storage = Storage::new(vec);
        Ok(Self::build(storage, shape, meta))
    }

    pub fn eye(size: usize) -> Result<Self> {
        Self::eye_impl(size, T::AutogradMeta::default())
    }

    pub(crate) fn eye_impl(size: usize, meta: T::AutogradMeta) -> Result<Self> {
        let mut vec = vec![T::zero(); size * size];
        for n in 0..size {
            vec[n * size + n] = T::one();
        }
        let storage = Storage::new(vec);
        Ok(Self::build(storage, (size, size), meta))
    }

    pub fn diag(diag: &[T]) -> Result<Self> {
        Self::diag_impl(diag, T::AutogradMeta::default())
    }

    pub(crate) fn diag_impl(diag: &[T], meta: T::AutogradMeta) -> Result<Self> {
        let size = diag.len();
        let mut vec = vec![T::zero(); size * size];
        for n in 0..size {
            vec[n * size + n] = diag[n];
        }
        let storage = Storage::new(vec);
        Ok(Self::build(storage, (size, size), meta))
    }
}

impl<T: WithDType + rand_distr::uniform::SampleUniform> Tensor<T> 
where 
    StandardUniform: Distribution<T>
{
    /// Creates an array with uniformly distributed random values in `[min, max)`.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::<f32>::rand(0., 1., (2, 3)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn rand<S: Into<Shape>>(min: T, max: T, shape: S) -> Result<Self> {
        Self::rand_impl(min, max, shape, T::AutogradMeta::default())
    }

    pub(crate) fn rand_impl<S: Into<Shape>>(min: T, max: T, shape: S, meta: T::AutogradMeta) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::rand_uniform(&shape, min, max)?;
        Ok(Self::build(storage, shape, meta))
    }

    /// Creates a random array with the same shape as `self`.
    pub fn rand_like(&self, min: T, max: T) -> Result<Self> {
        Self::rand(min, max, self.shape())
    }
}

impl<F: FloatDType> Tensor<F> {
    /// Generate a 1-D `Tensor` of `num` evenly spaced values over the interval [start, stop).
    /// 
    /// # Example
    ///
    /// ```
    /// # use lumen_core::Tensor;
    /// let arr = Tensor::linspace(0.0, 1.0, 5).unwrap();
    /// assert_eq!(arr.to_vec(), [0.0, 0.2, 0.4, 0.6000000000000001, 0.8]);
    /// ```
    pub fn linspace(start: F, stop: F, num: usize) -> Result<Self> {
        Self::linspace_impl(start, stop, num, F::AutogradMeta::default())
    }

    pub(crate) fn linspace_impl(start: F, stop: F, num: usize, meta: F::AutogradMeta) -> Result<Self> {
        let step = (stop - start) / F::from_usize(num);
        let vec: Vec<_> = std::iter::successors(Some(start), |&x| {
            let next = x + step;
            if next < stop { Some(next) } else { None }
        })
        .collect();

        let len = vec.len();
        let storage = Storage::new(vec);
        Ok(Self::build(storage, len, meta))
    }
}

impl<F: FloatDType> Tensor<F> 
where 
    StandardNormal: Distribution<F>
{
    /// Creates an array with normally distributed random values
    /// with given `mean` and `std`.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::<f64>::randn(0.0, 1.0, (2, 2)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn randn<S: Into<Shape>>(mean: F, std: F, shape: S) -> Result<Self> {
        Self::randn_impl(mean, std, shape, F::AutogradMeta::default())
    }

    pub(crate) fn randn_impl<S: Into<Shape>>(mean: F, std: F, shape: S, meta: F::AutogradMeta) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::rand_normal(&shape, mean, std)?;
        Ok(Self::build(storage, shape, meta))
    }

    /// Creates a normal-distributed random array with the same shape as `self`.
    pub fn randn_like(&self, mean: F, std: F) -> Result<Self> {
        Self::randn(mean, std, self.shape())
    }
}

impl<T: FloatDType> Tensor<T> {
    pub fn new_var<A: ToTensor<T>>(array: A) -> Result<Self> {
        Self::new_impl(array, AutogradInfo::var())
    }

    pub fn full_var<S: Into<Shape>>(shape: S, value: T) -> Result<Self> {
        Self::full_impl(shape, value, AutogradInfo::var())
    }

    pub fn zeros_var<S: Into<Shape>>(shape: S) -> Result<Self> {
        Self::zeros_impl(shape, AutogradInfo::var())
    }

    pub fn zero_like_var(&self) -> Result<Self> {
        Self::zeros_var(self.shape())
    }

    pub fn ones_var<S: Into<Shape>>(shape: S) -> Result<Self> {
        Self::ones_impl(shape, AutogradInfo::var())
    }

    pub fn ones_like_var(&self) -> Result<Self> {
        Self::ones_var(self.shape())
    }

    pub fn arange_var(start: T, end: T) -> Result<Self> {
        Self::arange_impl(start, end, AutogradInfo::var())
    }

    pub fn from_vec_var<V: Into<Vec<T>>, S: Into<Shape>>(vec: V, shape: S) -> Result<Self> {
        Self::from_vec_impl(vec, shape, AutogradInfo::var())
    }

    pub fn eye_var(size: usize) -> Result<Self> {
        Self::eye_impl(size, AutogradInfo::var())
    }

    pub fn diag_var(diag: &[T]) -> Result<Self> {
        Self::diag_impl(diag, AutogradInfo::var())
    }

    pub fn linspace_var(start: T, stop: T, num: usize) -> Result<Self> {
        Self::linspace_impl(start, stop, num, AutogradInfo::var())
    }
}

impl Tensor<bool> {
    /// Creates a boolean array filled with `true`.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::trues((2, 2)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn trues<S: Into<Shape>>(shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        let storage = Storage::new(vec![true; shape.element_count()]);
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates a boolean array filled with `false`.
    pub fn falses<S: Into<Shape>>(shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        let storage = Storage::new(vec![false; shape.element_count()]);
        Ok(Self::from_storage(storage, shape))
    }
}

pub trait ToTensor<T> {
    fn shape(&self) -> Result<Shape>;
    fn to_storage(self) -> Result<Storage<T>>;
}

impl<D: WithDType> ToTensor<D> for D {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::scalar())
    }

    fn to_storage(self) -> Result<Storage<D>> {
        Ok(Storage::new([self].to_vec()))
    }
}

impl<S: WithDType, const N: usize> ToTensor<S> for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        Ok(Storage::new(self.to_vec()))
    }
}

impl<S: WithDType> ToTensor<S> for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        Ok(Storage::new(self.to_vec()))
    }
}

impl<S: WithDType, const N1: usize, const N2: usize> ToTensor<S> 
    for &[[S; N2]; N1] 
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
        Ok(Storage::new(self.concat()))
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> ToTensor<S>
    for &[[[S; N3]; N2]; N1] 
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        Ok(Storage::new(vec))

    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> ToTensor<S>
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3 * N4);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3])
                }
            }
        }
        Ok(Storage::new(vec))
    }
}

impl<S: WithDType> ToTensor<S> for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        Ok(Storage::new(self))

    }
}