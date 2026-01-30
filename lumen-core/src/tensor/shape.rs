use std::sync::Arc;
use crate::{AutogradMetaT, Dim, Dims, Error, Layout, Result, Shape, Storage, WithDType, D};
use super::{Tensor, TensorId, TensorImpl, Slice};

impl<T: WithDType> Tensor<T> {
    /// Creates a new tensor with the specified dimension removed if its size was one.
    ///
    /// ```rust
    /// use lumen_core::{Tensor, DType, D};
    /// let a = Tensor::<f32>::zeros((2, 3, 1)).unwrap();
    ///
    /// let c = a.squeeze(2).unwrap();
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    /// ```
    pub fn squeeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "squeeze")?;
        if dims[dim] == 1 {
            let mut dims = dims.to_vec();
            let mut strides = self.layout().stride().to_vec();
            dims.remove(dim);
            strides.remove(dim);
            let tensor_ = TensorImpl {
                id: TensorId::new(),
                storage: self.0.storage.clone(),
                layout: Layout::new(dims, strides, self.layout().start_offset()),
                meta: T::AutogradMeta::on_reshape_op(self)
            };
            Ok(Self(Arc::new(tensor_)))
        } else {
            Err( Error::SqueezeDimNot1 { shape: self.shape().clone(), dim } )?
        }
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    ///
    /// ```rust
    /// use lumen_core::{Tensor, DType, D};
    /// let a = Tensor::<f32>::zeros((2, 3)).unwrap();
    ///
    /// let c = a.unsqueeze(0).unwrap();
    /// assert_eq!(c.shape().dims(), &[1, 2, 3]);
    ///
    /// let c = a.unsqueeze(D::Minus1).unwrap();
    /// assert_eq!(c.shape().dims(), &[2, 3, 1]);
    /// ```
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        let mut strides = self.layout().stride().to_vec();
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        dims.insert(dim, 1);
        let stride = if dim < strides.len() { strides[dim] } else { 1 };
        strides.insert(dim, stride);
        let tensor_ = TensorImpl {
            id: TensorId::new(),
            storage: self.0.storage.clone(),
            layout: Layout::new(dims, strides, self.layout().start_offset()),
            meta: T::AutogradMeta::on_reshape_op(self),
        };
        Ok(Self(Arc::new(tensor_)))
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + len`.
    /// ```
    /// use lumen_core::Tensor;
    /// let a = Tensor::new(&[
    ///     [0f32, 1., 2.],
    ///     [3.  , 4., 5.],
    ///     [6.  , 7., 8.]
    /// ]).unwrap();
    ///
    /// let b = a.narrow(0, 1, 2).unwrap();
    /// assert_eq!(b.shape().dims(), &[2, 3]);
    ///
    /// let c = a.narrow(1, 1, 1).unwrap();
    /// assert_eq!(c.shape().dims(), &[3, 1]);
    /// ```
    pub fn narrow<D: Dim>(&self, dim: D, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "narrow")?;
        let err = |msg| {
            Err::<(), _>(Error::NarrowInvalidArgs {
                shape: self.shape().clone(),
                dim,
                start,
                len,
                msg,
            })
        };

        if start > dims[dim] {
            err("start > dim_len")?;
        }
        if start.saturating_add(len) > dims[dim] {
            err("start + len > dim_len")?
        }
        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let meta = T::AutogradMeta::on_narrow_op(self, dim, start, len);
            let layout = self.layout().narrow(dim, start, len)?;
            let tensor_ = TensorImpl {
                id: TensorId::new(),
                storage: self.0.storage.clone(),
                layout,
                meta
            };
            Ok(Self(Arc::new(tensor_)))
        }
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// slice from `start` to `start : end : step`.
    /// 
    /// ```
    /// use lumen_core::{Tensor, DType, s, Slice};
    /// let a = Tensor::<i32>::zeros((5, 5, 5)).unwrap();
    ///
    /// let b = a.narrow(0, 1, 2).unwrap();
    /// assert_eq!(b.shape().dims(), &[2, 5, 5]);
    ///
    /// let c = a.slice(1, &s!(::2)).unwrap();
    /// assert_eq!(c.shape().dims(), &[5, 3, 5]);
    /// ```
    pub fn slice<D: Dim>(&self, dim: D, slice: &Slice) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "narrow")?;
        let err = |msg| {
            Err::<(), _>(Error::SliceInvalidArgs {
                shape: self.shape().clone(),
                dim,
                slice: slice.clone(),
                msg,
            })
        };

        let end = match slice.end {
            Some(end) if end >= 0 => end as usize,
            Some(end) => {
                let dis = -end as usize;
                if dis > dims[dim] {
                    0
                } else {
                    dims[dim] - dis
                }
            }
            None => dims[dim],
        };
        if slice.start > dims[dim] {
            err("start > dim_len")?;
        }
        if end > dims[dim] {
            err("end > dim_len")?
        }
        if slice.start == 0 && dims[dim] == end && slice.step == 1 {
            Ok(self.clone())
        } else {
            let meta = T::AutogradMeta::on_slice_op(self, dim, slice.start, end, slice.step);
            let layout = self.layout().slice(dim, slice.start, end, slice.step)?;
            Ok(self.share_storage(layout, meta))
        }
    }

    /// Reshape returns a tensor with the target shape provided that the number of elements of the
    /// original tensor is the same.
    /// If the input tensor is contiguous, this is a view on the original data. Otherwise this uses
    /// a new storage and copies the data over, the returned tensor is always contiguous.
    ///
    /// The shape can be specified using a tuple of `usize` and at most one `()` in which case
    /// the behavior is the same as when using `-1` in PyTorch: this dimension size is adjusted so
    /// as to match the number of elements in the tensor.
    /// 
    /// ```rust
    /// use lumen_core::{Tensor, DType, D};
    /// let a = Tensor::<f32>::zeros((2, 3)).unwrap();
    ///
    /// let c = a.reshape((1, 6)).unwrap();
    /// assert_eq!(c.shape().dims(), &[1, 6]);
    /// ```
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.element_count() != self.element_count() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: shape,
                op: "reshape",
            })?;
        }

        let meta = T::AutogradMeta::on_reshape_op(self);
        if self.is_contiguous() {
            let layout = Layout::contiguous_with_offset(shape, self.layout().start_offset());
            Ok(self.share_storage(layout, meta))
        } else {
            let storage = self.storage_read()?.copy(self.layout());
            Ok(Self::from_storage(storage, shape, meta))
        }
    }
    
    /// Returns a Tensor that is a transposed version of the input, the given dimensions are
    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Self> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        if dim1 == dim2 {
            return Ok(self.clone());
        }

        let meta = T::AutogradMeta::on_transpose_op(self, dim1, dim2);
        let layout = self.layout().transpose(dim1, dim2)?;
        Ok(self.share_storage(layout, meta))
    }

    pub fn transpose_last(&self) -> Result<Self> {
        self.transpose(D::Minus1, D::Minus2)
    }

    /// Returns a tensor with the same data as the input where the dimensions have been permuted.
    /// dims must be a permutation, i.e. include each dimension index exactly once.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    /// let tensor = Tensor::<u32>::arange(0u32, 120u32).unwrap().reshape((2, 3, 4, 5)).unwrap();
    /// assert_eq!(tensor.dims(), &[2, 3, 4, 5]);
    /// let tensor = tensor.permute((2, 3, 1, 0)).unwrap();
    /// assert_eq!(tensor.dims(), &[4, 5, 3, 2]);
    /// ```
    pub fn permute<D: Dims>(&self, dims: D) -> Result<Self> {
        let dims = dims.to_indexes(self.shape(), "permute")?;
        // O(n^2) permutation check but these arrays are small.
        let is_permutation =
            dims.len() == self.rank() && (0..dims.len()).all(|i| dims.contains(&i));
        if !is_permutation {
            crate::bail!(
                "dimension mismatch in permute, tensor {:?}, dims: {:?}",
                self.dims(),
                dims
            )
        }
        // let op = BackpropOp::new1(self, |t| Op::Permute(t, dims.clone()));
        let layout = self.layout().permute(&dims)?;
        let meta = T::AutogradMeta::on_permute_op(self, dims);
        Ok(self.share_storage(layout, meta))
    }

    /// Concatenates two or more tensors along a particular dimension.
    ///
    /// All tensors must of the same rank, and the output will have
    /// the same rank
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    /// let a = Tensor::<f32>::zeros((2, 3)).unwrap();
    /// let b = Tensor::<f32>::zeros((2, 3)).unwrap();
    ///
    /// let c = Tensor::cat(&[&a, &b], 0).unwrap();
    /// assert_eq!(c.dims(), &[4, 3]);
    ///
    /// let c = Tensor::cat(&[&a, &b], 1).unwrap();
    /// assert_eq!(c.dims(), &[2, 6]);
    /// ```
    pub fn cat<A: AsRef<Tensor<T>>, D: Dim>(arrs: &[A], dim: D) -> Result<Self> {
        // check shape
        if arrs.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" })?
        }
    
        // first arr's infomation
        let arr0 = &arrs[0];
        let rank0 = arr0.as_ref().rank();

        // cat_dim must be valid!
        let cat_dim = dim.to_index(arr0.as_ref().shape(), "cat")?;
        let mut target_dims = arr0.as_ref().dims().to_vec();
        target_dims[cat_dim] = 0;
        let mut dim_offsets = vec![];

        for (_arr_index, arr) in arrs.iter().enumerate() {
            // check shape 
            let rank = arr.as_ref().rank();
            if rank != rank0 {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arr.as_ref().rank(),
                    shape: arr.as_ref().shape().clone(),
                })?
            }

            // zip arr0's dims and arr's dims
            for (dim_index, (v1, v2)) in arr0.as_ref().dims().iter()
                                                                    .zip(arr.as_ref().dims().iter())
                                                                    .enumerate()
            {
                // accumalte the cat dim
                if dim_index == cat_dim {
                    dim_offsets.push(target_dims[cat_dim]);
                    target_dims[cat_dim] += v2;
                }

                // all other dims should be same
                if dim_index != cat_dim && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_index,
                        first_shape: arr0.as_ref().shape().clone(),
                        n: dim_index + 1,
                        nth_shape: arr0.as_ref().shape().clone(),
                    })?
                }
            }
        }
        
        // Now, all arr in arrs has same rank, and except `cat_dim`, all dims are equal
        // [ (a, n1, b, c), (a, n2, b, c), ... , (a, nk, b, c).... ]
        // target_dims = (a, n1+n2+...+nk, b, c)

        let target_shape: Shape = target_dims.into();
        
        // Create a new storgae and copy
        let mut dst: Vec<T> = Vec::with_capacity(target_shape.element_count());
        unsafe { dst.set_len(target_shape.element_count()) };
        
        let meta = T::AutogradMeta::on_cat_op(arrs, cat_dim);
        let res_arr = Self::from_storage(Storage::new(dst), target_shape, meta);

        for (arr_index, arr) in arrs.iter().enumerate() {
            // Take sub Tensor 
            let sub_res_arr = res_arr.narrow(cat_dim, dim_offsets[arr_index], arr.as_ref().dims()[cat_dim])?;
            assert_eq!(sub_res_arr.shape(), arr.as_ref().shape());
            // MARK: copy_from is no grad
            sub_res_arr.copy_from(arr.as_ref())?;
        }

        Ok(res_arr)
    }

    /// Stacks two or more tensors along a particular dimension.
    ///
    /// All tensors must have the same rank, and the output has one additional rank
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    /// let a = Tensor::<f32>::zeros((2, 3)).unwrap();
    /// let b = Tensor::<f32>::zeros((2, 3)).unwrap();
    ///
    /// let c = Tensor::stack(&[&a, &b], 0).unwrap();
    /// assert_eq!(c.dims(), &[2, 2, 3]);
    ///
    /// let c = Tensor::stack(&[&a, &b], 2).unwrap();
    /// assert_eq!(c.dims(), &[2, 3, 2]);
    /// ```
    pub fn stack<A: AsRef<Tensor<T>>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "stack" })?
        }
        let dim = dim.to_index_plus_one(args[0].as_ref().shape(), "stack")?;
        let args = args
            .iter()
            .map(|t| t.as_ref().unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        Self::cat(&args, dim)
    }

    /// Splits a tensor along a specified dimension into multiple sub-tensors.
    ///
    /// The tensor is split along the given `dim` into as many sub-tensors as
    /// the size of that dimension. Each sub-tensor has the same shape as the
    /// original tensor, except the size along `dim` becomes 1.
    ///
    /// ```rust
    /// use lumen_core::Tensor;
    ///
    /// let a = Tensor::new(&[[1, 2], [3, 4], [5, 6]]).unwrap();
    ///
    /// // Split along axis 0 (rows)
    /// let splits = a.split(0).unwrap();
    /// assert_eq!(splits.len(), 3);
    /// assert_eq!(splits[0].to_vec().unwrap(), [1, 2]);
    /// assert_eq!(splits[1].to_vec().unwrap(), [3, 4]);
    /// assert_eq!(splits[2].to_vec().unwrap(), [5, 6]);
    ///
    /// // Split along axis 1 (columns)
    /// let splits = a.split(1).unwrap();
    /// assert_eq!(splits.len(), 2);
    /// assert_eq!(splits[0].to_vec().unwrap(), [1, 3, 5]);
    /// assert_eq!(splits[1].to_vec().unwrap(), [2, 4, 6]);
    ///
    /// // 1D array
    /// let b = Tensor::new(&[10, 20, 30]).unwrap();
    /// let splits = b.split(0).unwrap();
    /// assert_eq!(splits.len(), 3);
    /// assert_eq!(splits[0].to_vec().unwrap(), [10]);
    /// assert_eq!(splits[1].to_vec().unwrap(), [20]);
    /// assert_eq!(splits[2].to_vec().unwrap(), [30]);
    /// ```
    pub fn split<D: Dim>(&self, dim: D) -> Result<Vec<Self>> {
        let split_index = dim.to_index(self.shape(), "split")?;
        let split_dim_size = self.dims()[split_index];
        let mut splited_shape = self.dims().to_vec();
        splited_shape.remove(split_index);
        let splited_shape: Shape = splited_shape.into();

        let mut vec = vec![];
        for i in 0..split_dim_size {
            // TODO: backend, use orgin memory
            let mut data: Vec<T> = Vec::with_capacity(splited_shape.element_count());
            unsafe { data.set_len(splited_shape.element_count()) };
            let storage = Storage::new(data);
            let arr = Self::from_storage(storage, splited_shape.clone(), Default::default());
            
            // Copy data
            let sub_self = self.narrow(split_index, i, 1)?.squeeze(split_index)?;
            assert_eq!(sub_self.dims(), splited_shape.dims());
            arr.assign(sub_self)?;

            vec.push(arr);
        }   

        Ok(vec)
    }

    /// Split a tensor into the specified number of chunks, this may return less chunks than
    /// specified.
    pub fn chunk<D: Dim>(&self, chunks: usize, dim: D) -> Result<Vec<Self>> {
        let dim = dim.to_index(self.shape(), "chunk")?;
        let size = self.dim(dim)?;
        if size < chunks {
            (0..size).map(|i| self.narrow(dim, i, 1)).collect()
        } else {
            let chunk_size = size / chunks;
            let cnt_additional = size % chunks;
            let mut tensors = vec![];
            let mut sum_chunk_size = 0;
            for i in 0..chunks {
                let chunk_size = if i < cnt_additional {
                    chunk_size + 1
                } else {
                    chunk_size
                };
                let tensor = self.narrow(dim, sum_chunk_size, chunk_size)?;
                tensors.push(tensor);
                sum_chunk_size += chunk_size
            }
            Ok(tensors)
        }
    }

    /// Flattens the input tensor on the dimension indexes from `start_dim` to `end_dim` (both
    /// inclusive).
    pub fn flatten<D1: Dim, D2: Dim>(&self, start_dim: D1, end_dim: D2) -> Result<Self> {
        self.flatten_(Some(start_dim), Some(end_dim))
    }

    /// Flattens the input tensor on the dimension indexes from `0` to `end_dim` (inclusive).
    pub fn flatten_to<D: Dim>(&self, end_dim: D) -> Result<Self> {
        self.flatten_(None::<usize>, Some(end_dim))
    }

    /// Flattens the input tensor on the dimension indexes from `start_dim` (inclusive) to the last
    /// dimension.
    pub fn flatten_from<D: Dim>(&self, start_dim: D) -> Result<Self> {
        self.flatten_(Some(start_dim), None::<usize>)
    }

    /// Flattens the input tensor by reshaping it into a one dimension tensor.
    /// 
    /// ```rust
    /// use lumen_core::Tensor;
    /// let arr = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]]).unwrap();
    /// let arr = arr.flatten_all().unwrap();
    /// let len = arr.dims1().unwrap();
    /// assert_eq!(len, 6);
    /// assert_eq!(arr.to_vec().unwrap(), [0., 1., 2., 3., 4., 5.]);
    /// ```
    pub fn flatten_all(&self) -> Result<Self> {
        self.flatten_(None::<usize>, None::<usize>)
    }

    /// Repeat this tensor along the specified dimensions.
    pub fn repeat<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let repeats: Shape = shape.into();
        let mut repeats = repeats.dims().to_vec();

        if repeats.len() > self.rank() {
            Err(Error::RepeatRankOutOfRange { repeats: repeats.clone().into(), shape: self.shape().into() })?;
        } else if repeats.len() > self.rank() {
            for _ in 0..(repeats.len() - self.rank()) {
                repeats.push(1);
            }
        }

        let mut arr = self.clone();

        for (idx, &repeat) in repeats.iter().enumerate() {
            if repeat > 1 {
                arr = Tensor::cat(&vec![&arr; repeat], idx)?
            }
        }
        Ok(arr)
    }

    /// Repeat this tensor along the specified dimension with specified times
    pub fn repeat_dim<D: Dim>(&self, dim: D, times: usize) -> Result<Self> {
        if times == 0 {
            self.squeeze(dim)
        } else if times == 1 {
            Ok(self.clone())
        } else {
            Tensor::cat(&vec![self; times], dim)
        }
    }

    fn flatten_<D1: Dim, D2: Dim>(
        &self,
        start_dim: Option<D1>,
        end_dim: Option<D2>,
    ) -> Result<Self> {
        if self.rank() == 0 {
            self.reshape(1)
        } else {
            let start_dim = match start_dim {
                None => 0,
                Some(dim) => dim.to_index(self.shape(), "flatten")?,
            };
            let end_dim = match end_dim {
                None => self.rank() - 1,
                Some(dim) => dim.to_index(self.shape(), "flatten")?,
            };
            if start_dim < end_dim {
                let dims = self.dims();
                let mut dst_dims = dims[..start_dim].to_vec();
                dst_dims.push(dims[start_dim..end_dim + 1].iter().product::<usize>());
                if end_dim + 1 < dims.len() {
                    dst_dims.extend(&dims[end_dim + 1..]);
                }
                self.reshape(dst_dims)
            } else {
                Ok(self.clone())
            }
        }
    }
}

impl<T: WithDType> AsRef<Tensor<T>> for Tensor<T> {
    fn as_ref(&self) -> &Tensor<T> {
        self
    }
}

#[cfg(test)]
#[allow(unused)]
mod test {
    use super::*;

    #[test]
    fn test_unsqueeze() -> Result<()> {
        let t = Tensor::<i32>::zeros((2, 1, 3))?;
        let sq = t.squeeze(1)?;
        println!("{}", sq);
        assert_eq!(sq.dims(), vec![2, 3]);

        let unsq = sq.unsqueeze(0)?;
        println!("{}", unsq);
        assert_eq!(unsq.dims(), vec![1, 2, 3]);

        Ok(())
    }

    #[test]
    fn test_cat_1d() -> Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = Tensor::new(&[4, 5, 6])?;
    
        let c = Tensor::cat(&[a, b], 0)?;
        assert_eq!(c.dims(), [6]);
        assert_eq!(c.to_vec().unwrap(), [1, 2, 3, 4, 5, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_cat_2d_axis0() -> Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4]])?;
        let b = Tensor::new(&[[5, 6]])?;
    
        let c = Tensor::cat(&[a, b], 0)?;
        assert_eq!(c.dims(), [3, 2]);
        assert_eq!(c.to_vec().unwrap(), [1, 2, 3, 4, 5, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_cat_2d_axis1() -> Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4]])?;
        let b = Tensor::new(&[[5], [6]])?;
    
        let c = Tensor::cat(&[a, b], 1)?;
        assert_eq!(c.dims(), [2, 3]);
        assert_eq!(c.to_vec().unwrap(), [1, 2, 5, 3, 4, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_cat_3d() -> Result<()> {
        let a = Tensor::full((2, 2, 2), 1)?;
        let b = Tensor::full((2, 2, 2), 2)?;
    
        let c = Tensor::cat(&[a, b], 0)?;
        assert_eq!(c.dims(), [4, 2, 2]);
    
        let c2 = Tensor::cat(&[c.clone(), c.clone()], 1)?;
        assert_eq!(c2.dims(), [4, 4, 2]);
    
        Ok(())
    }
    
    #[test]
    fn test_cat_shape_mismatch() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let b = Tensor::new(&[[1, 2], [3, 4]]).unwrap();
    
        let res = Tensor::cat(&[a, b], 0);
        assert!(res.is_err());
    }
    
    #[test]
    fn test_cat_bool() -> Result<()> {
        let a = Tensor::new(&[[true, false]])?;
        let b = Tensor::new(&[[false, true]])?;
    
        let c = Tensor::cat(&[a, b], 0)?;
        assert_eq!(c.dims(), [2, 2]);
        assert_eq!(c.to_vec().unwrap(), [true, false, false, true]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_1d_axis0() -> Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = Tensor::new(&[4, 5, 6])?;
    
        let c = Tensor::stack(&[a, b], 0)?;
        assert_eq!(c.dims(), [2, 3]); 
        assert_eq!(c.to_vec().unwrap(), [1, 2, 3, 4, 5, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_1d_axis1() -> Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = Tensor::new(&[4, 5, 6])?;
    
        let c = Tensor::stack(&[a, b], 1)?;
        assert_eq!(c.dims(), [3, 2]);
        assert_eq!(c.to_vec().unwrap(), [1, 4, 2, 5, 3, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_2d_axis0() -> Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4]])?;
        let b = Tensor::new(&[[5, 6], [7, 8]])?;
    
        let c = Tensor::stack(&[a, b], 0)?;
        assert_eq!(c.dims(), [2, 2, 2]);
        assert_eq!(c.to_vec().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_2d_axis1() -> Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4]])?;
        let b = Tensor::new(&[[5, 6], [7, 8]])?;
    
        let c = Tensor::stack(&[a, b], 1)?;
        assert_eq!(c.dims(), [2, 2, 2]);
        assert_eq!(c.to_vec().unwrap(), [1, 2, 5, 6, 3, 4, 7, 8]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_2d_axis2() -> Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4]])?;
        let b = Tensor::new(&[[5, 6], [7, 8]])?;
    
        let c = Tensor::stack(&[a, b], 2)?;
        assert_eq!(c.dims(), [2, 2, 2]);
        assert_eq!(c.to_vec().unwrap(), [1, 5, 2, 6, 3, 7, 4, 8]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_shape_mismatch() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let b = Tensor::new(&[4, 5]).unwrap();
    
        let res = Tensor::stack(&[a, b], 0);
        assert!(res.is_err());
    }
    
    #[test]
    fn test_split_1d() -> Result<()> {
        let a = Tensor::new(&[10, 20, 30, 40])?;
        let splits = a.split(0)?; // axis 0
    
        assert_eq!(splits.len(), 4);
        assert_eq!(splits[0].to_vec().unwrap(), [10]);
        assert_eq!(splits[1].to_vec().unwrap(), [20]);
        assert_eq!(splits[2].to_vec().unwrap(), [30]);
        assert_eq!(splits[3].to_vec().unwrap(), [40]);
    
        Ok(())
    }

    #[test]
    fn test_split_2d_axis0() -> Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4], [5, 6], [7, 8]])?;
        let splits = a.split(0)?;
        
        assert_eq!(splits.len(), 4);
        assert_eq!(splits[0].to_vec().unwrap(), [1, 2]);
        assert_eq!(splits[1].to_vec().unwrap(), [3, 4]);
        assert_eq!(splits[2].to_vec().unwrap(), [5, 6]);
        assert_eq!(splits[3].to_vec().unwrap(), [7, 8]);
        
        Ok(())
    }
    
    #[test]
    fn test_split_2d_axis1() -> Result<()> {
        let a = Tensor::new(&[[1, 2, 3], [4, 5, 6]])?;
        let splits = a.split(1)?;
        
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].to_vec().unwrap(), [1, 4]); 
        assert_eq!(splits[1].to_vec().unwrap(), [2, 5]); 
        assert_eq!(splits[2].to_vec().unwrap(), [3, 6]); 
        
        Ok(())
    }
    
    #[test]
    fn test_split_3d_axis2() -> Result<()> {
        let a = Tensor::new(&[
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ])?;
        let splits = a.split(2)?;
        
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].to_vec().unwrap(), [1, 3, 5, 7]); 
        assert_eq!(splits[1].to_vec().unwrap(), [2, 4, 6, 8]); 
        
        Ok(())
    }
    
    #[test]
    fn test_split_single_element() -> Result<()> {
        let a = Tensor::new(&[42])?;
        let splits = a.split(0)?;
        
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].to_vec().unwrap(), [42]);
        
        Ok(())
    }
    
    #[test]
    fn test_split_empty_array() -> Result<()> {
        let a = Tensor::<i32>::zeros((0, 2))?;
        let splits = a.split(0)?;
        
        assert!(splits.is_empty()); 
        Ok(())
    }

    #[test]
    fn test_repeat_1d() -> Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = a.repeat(3)?; // repeat each dimension 3 times
        assert_eq!(b.dims(), [3 * 3]); // shape: [9]
        assert_eq!(b.to_vec().unwrap(), [1, 2, 3, 1, 2, 3, 1, 2, 3]);
        Ok(())
    }

    #[test]
    fn test_repeat_2d() -> Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4]])?;
        let b = a.repeat((2, 3))?; // repeat 2 times along axis 0, 3 times along axis 1
        assert_eq!(b.dims(), [4, 6]);
        assert_eq!(
            b.to_vec().unwrap(),
            [
                1, 2, 1, 2, 1, 2,
                3, 4, 3, 4, 3, 4,
                1, 2, 1, 2, 1, 2,
                3, 4, 3, 4, 3, 4
            ]
        );
        Ok(())
    }

    #[test]
    fn test_repeat_dim() -> Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = a.repeat_dim(0, 2)?; // repeat along axis 0 two times
        assert_eq!(b.dims(), [6]);
        assert_eq!(b.to_vec().unwrap(), [1, 2, 3, 1, 2, 3]);

        let c = a.repeat_dim(0, 1)?; // repeat once -> same as clone
        assert_eq!(c.dims(), [3]);
        assert_eq!(c.to_vec().unwrap(), [1, 2, 3]);

        Ok(())
    }

    #[test]
    fn test_repeat_high_dim() -> Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4]])?;
        let b = a.repeat((2, 3))?; // more dims than array, extra dims should be treated as 1
        assert_eq!(b.dims(), [4, 6]);
        Ok(())
    }

    #[test]
    fn test_narrow_1d_basic() -> Result<()> {
        let a = Tensor::new(&[0, 1, 2, 3, 4, 5])?;
        
        let b = a.narrow(0, 2, 3)?;
        
        assert_eq!(b.dims(), &[3]);
        assert_eq!(b.to_vec().unwrap(), &[2, 3, 4]);

        let b = Tensor::randn(0.0, 1.0, (5, 5))?;
        println!("{:?}", b);
        Ok(())
    }

    #[test]
    fn test_narrow_2d_rows() -> Result<()> {
        // Shape: [3, 3]
        // [ 0,  1,  2 ]
        // [ 3,  4,  5 ]
        // [ 6,  7,  8 ]
        let a = Tensor::new(&[
            [0, 1, 2], 
            [3, 4, 5], 
            [6, 7, 8]
        ])?;

        let b = a.narrow(0, 1, 1)?;

        assert_eq!(b.dims(), &[1, 3]);
        assert_eq!(b.to_vec().unwrap(), &[3, 4, 5]);
        Ok(())
    }
}