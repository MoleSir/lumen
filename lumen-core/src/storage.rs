use std::sync::{Arc, RwLock};
use rand::rng;
use rand_distr::{Distribution, Uniform};
use crate::{Error, IntDType, Result};
use super::{DType, FloatDType, Layout, NumDType, Shape, WithDType};

#[derive(Clone)]
pub struct Storage<T>(Vec<T>);

impl<T: NumDType> Storage<T> {
    pub fn zeros(shape: &Shape) -> Self {
        Self(vec![T::zero(); shape.element_count()])
    }

    pub fn ones(shape: &Shape) -> Self {
        Self(vec![T::one(); shape.element_count()])
    }
}

impl<T: WithDType> Storage<T> {
    pub fn full(value: T, shape: &Shape) -> Self {
        Self(vec![value; shape.element_count()])
    }
}

impl<T: NumDType> Storage<T> {
    pub fn rand_uniform(shape: &Shape, min: T, max: T) -> Result<Self> {
        let elem_count = shape.element_count();
        let mut rng = rng();
        let uniform = Uniform::new(min, max).map_err(|e| Error::Rand(e.to_string()))?;
        let v: Vec<T> = (0..elem_count)
            .map(|_| uniform.sample(&mut rng))
            .collect();

        Ok(Self(v))
    }
}

impl<F: FloatDType> Storage<F> {
    pub fn rand_normal(shape: &Shape, mean: F, std: F) -> Result<Self> 
    {
        let elem_count = shape.element_count();
        let v = F::random_normal_vec(elem_count, mean, std)?;
        Ok(Self(v))
    }
}

impl<T: WithDType> Storage<T> {
    pub(crate) fn index_select<I: IntDType>(
        &self,
        self_layout: &Layout,
        ids: &Storage<I>,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let vec = Self::do_index_select(ids.data(), ids_layout, dim, self.data(), self_layout)?;
        Ok(Storage::new(vec))
    }

    fn do_index_select<I: IntDType>(ids: &[I], ids_l: &Layout, dim: usize, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        if !layout.is_contiguous() {
            Err(Error::RequiresContiguous { op: "index-select" })?
        } 
        let src = &src[layout.start_offset..layout.start_offset+layout.shape.element_count()];
        let n_ids = ids_l.dims();
        assert!(n_ids.len() == 1);
        let n_ids = n_ids[0];
        let stride_ids = ids_l.stride()[0];
        let mut dst_dims = layout.dims().to_vec();
        let src_dim = dst_dims[dim];
        dst_dims[dim] = n_ids;
        let dst_len: usize = dst_dims.iter().product();
        let left_len: usize = dst_dims[..dim].iter().product();
        let right_len: usize = dst_dims[dim + 1..].iter().product();
        let mut dst = vec![T::false_value(); dst_len];
        for left_i in 0..left_len {
            let start_src_idx = left_i * right_len * src_dim;
            let start_dst_idx = left_i * right_len * n_ids;
            for i in 0..n_ids {
                let start_dst_idx = start_dst_idx + i * right_len;
                let index = ids[ids_l.start_offset() + stride_ids * i];
                if index == I::max_value() {
                    dst[start_dst_idx..start_dst_idx + right_len].fill(T::false_value());
                } else {
                    let index = index.to_usize();
                    if index >= src_dim {
                        Err(Error::InvalidIndex {
                            index,
                            size: src_dim,
                            op: "index-select",
                        })?
                    }
                    let start_src_idx = start_src_idx + index * right_len;
                    dst[start_dst_idx..start_dst_idx + right_len]
                        .copy_from_slice(&src[start_src_idx..start_src_idx + right_len])
                }
            }
        }
        Ok(dst)
    }
}

impl<T: NumDType> Storage<T> {
    pub(crate) fn index_add<I: IntDType>(
        &self,
        self_layout: &Layout,
        ids: &Storage<I>,
        ids_layout: &Layout,
        source: &Storage<T>,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if !self_layout.is_contiguous() || !source_layout.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "index-add" }.into());
        }

        let new_data = Self::do_index_add(
            self.data(),
            self_layout,
            ids.data(),
            ids_layout,
            source.data(),
            dim
        )?;
        
        Ok(Storage::new(new_data))
    }

    fn do_index_add<I: IntDType>(
        dst_data: &[T],      
        dst_layout: &Layout,
        ids: &[I],           
        ids_layout: &Layout, 
        src_data: &[T],     
        dim: usize,
    ) -> Result<Vec<T>> {
        // 1. 复制一份 dst_data 作为结果，因为我们要修改它 (或者说是累加到一个新 buffer 上)
        // 注意：在大张量场景下，这里可能会有性能开销。
        // 如果你的系统支持 inplace 修改且确信引用计数为1，可以直接修改。
        // 这里为了安全采用 clone。
        let mut result = dst_data.to_vec();

        let n_ids = ids_layout.dims()[0];
        let stride_ids = ids_layout.stride()[0];

        let dst_dims = dst_layout.dims();
        let src_dim_size = dst_dims[dim]; // 原始张量在该维度的长度

        let left_len: usize = dst_dims[..dim].iter().product();
        let right_len: usize = dst_dims[dim + 1..].iter().product();
        
        // src (grad) 的结构是 [left, n_ids, right]
        // dst (self) 的结构是 [left, src_dim_size, right]

        for left_i in 0..left_len {
            let start_src_block = left_i * n_ids * right_len;
            let start_dst_block = left_i * src_dim_size * right_len;

            for i in 0..n_ids {
                // 获取索引值
                let index_val = ids[ids_layout.start_offset() + stride_ids * i];
                
                // 处理 mask (如果有 padding index)
                if index_val == I::max_value() {
                    continue; 
                }

                let idx = index_val.to_usize();
                if idx >= src_dim_size {
                    return Err(Error::InvalidIndex {
                        index: idx,
                        size: src_dim_size,
                        op: "index-add",
                    }.into());
                }

                // 计算偏移量
                let src_offset = start_src_block + i * right_len;
                let dst_offset = start_dst_block + idx * right_len;

                // 执行加法： dst[idx] += src[i]
                for k in 0..right_len {
                    let s_val = src_data[src_offset + k];
                    let d_val = result[dst_offset + k];
                    result[dst_offset + k] = d_val + s_val;
                }
            }
        }

        Ok(result)
    }
}

impl<T: WithDType> Storage<T> {
    pub(crate) fn gather<I: IntDType>(
        &self,
        self_layout: &Layout,
        ids: &Storage<I>,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let new_data = Self::do_gather(self.data(), self_layout, ids.data(), ids_layout, dim)?;
        Ok(Storage::new(new_data))
    }

    fn do_gather<I: IntDType>(
        src: &[T],
        src_layout: &Layout,
        ids: &[I],
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Vec<T>> {
        // 1. 检查连续性：为了简化多维索引的偏移量计算，强制要求 src 和 ids 均连续。
        // 如果需要支持非连续，需要根据 stride 手动计算物理偏移量，会显著增加复杂性。
        if !src_layout.is_contiguous() || !ids_layout.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "gather" }.into());
        }

        let src_dims = src_layout.dims();
        let ids_dims = ids_layout.dims();

        // 2. 检查维度一致性 (Rank check)
        if src_dims.len() != ids_dims.len() {
             return Err(Error::ShapeMismatchBinaryOp { 
                lhs: src_layout.shape().clone(), 
                rhs: ids_layout.shape().clone(),
                op: "gather" 
             }.into());
        }

        // 3. 准备结果 Buffer
        let dst_len = ids_layout.shape.element_count();
        // 类似于 index_select，预填充默认值 (如 0 或 false)，方便处理 Padding Mask
        let mut dst = vec![T::false_value(); dst_len];

        // 4. 计算三段式维度的长度：[Left, Dim, Right]
        // Left: dim 左边所有维度的乘积
        let left_len: usize = src_dims[..dim].iter().product();
        // Right: dim 右边所有维度的乘积
        let right_len: usize = src_dims[dim + 1..].iter().product();
        
        let src_dim_size = src_dims[dim];
        let ids_dim_size = ids_dims[dim];

        // 5. 执行 Gather 操作
        // 逻辑视角：src 是 [left, src_dim, right], ids 是 [left, ids_dim, right]
        // 我们遍历 ids (即遍历 dst) 的所有位置
        
        for i in 0..left_len {
            // 计算 src 在当前 Left 块的起始偏移
            let src_block_start = i * src_dim_size * right_len;
            let dst_block_start = i * ids_dim_size * right_len;

            for j in 0..ids_dim_size {
                for k in 0..right_len {
                    // 计算 ids 和 dst 的线性索引 (因为是连续内存)
                    // dst_idx = (i * ids_dim * right) + (j * right) + k
                    let dst_idx = dst_block_start + j * right_len + k;
                    
                    let index_val = ids[dst_idx];

                    // 处理 Mask (Padding Index)
                    if index_val == I::max_value() {
                        dst[dst_idx] = T::false_value();
                        continue;
                    }

                    let idx = index_val.to_usize();
                    if idx >= src_dim_size {
                        return Err(Error::InvalidIndex {
                            index: idx,
                            size: src_dim_size,
                            op: "gather",
                        }.into());
                    }

                    // 计算 src 的线性索引
                    // src 取值的逻辑：保持 Left(i) 和 Right(k) 不变，Dim 变为 ids 中读出的 idx
                    let src_idx = src_block_start + idx * right_len + k;

                    dst[dst_idx] = src[src_idx];
                }
            }
        }

        Ok(dst)
    }
}

impl<T: NumDType> Storage<T> {
    pub(crate) fn scatter_add<I: IntDType>(
        &self,
        self_layout: &Layout,
        ids: &Storage<I>,
        ids_layout: &Layout,
        source: &Storage<T>,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if !self_layout.is_contiguous() || !ids_layout.is_contiguous() || !source_layout.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "scatter-add" }.into());
        }

        // 这里的 self 是 destination (arg_grad)，source 是 gradient
        let new_data = Self::do_scatter_add(
            self.data(),
            self_layout,
            ids.data(),
            ids_layout,
            source.data(),
            dim
        )?;
        
        Ok(Storage::new(new_data))
    }

    fn do_scatter_add<I: IntDType>(
        dst: &[T],          // arg_grad 的当前数据 (通常是 zero buffer)
        dst_layout: &Layout,
        ids: &[I],          // 索引
        ids_layout: &Layout,
        src: &[T],          // incoming grad
        dim: usize,
    ) -> Result<Vec<T>> {
        // 1. 复制 dst 数据以进行累加 (inplace 模拟)
        let mut result = dst.to_vec();

        let dst_dims = dst_layout.dims();
        let src_dims = ids_layout.dims(); // src 和 ids 的形状应该一致

        // 检查维度一致性
        if dst_dims.len() != src_dims.len() {
             return Err(Error::ShapeMismatchBinaryOp { 
                lhs: dst_layout.shape().clone(), 
                rhs: ids_layout.shape().clone(),
                op: "scatter-add" 
             }.into());
        }

        // 2. 计算三段式维度
        // 我们遍历的是 source (grad) 和 ids，它们的形状是一样的
        let left_len: usize = src_dims[..dim].iter().product();
        let right_len: usize = src_dims[dim + 1..].iter().product();
        
        let src_dim_size = src_dims[dim]; // ids 的 dim 长度
        let dst_dim_size = dst_dims[dim]; // arg (dst) 的 dim 长度

        // 3. 执行 Scatter Add
        // 逻辑：遍历 src(grad) 的每一个元素，找到 ids 中对应的索引 idx，
        // 然后 result[left, idx, right] += src[left, i, right]
        
        for i in 0..left_len {
            let src_block_start = i * src_dim_size * right_len;
            let dst_block_start = i * dst_dim_size * right_len;

            for j in 0..src_dim_size {
                for k in 0..right_len {
                    // src 和 ids 是同步遍历的
                    let linear_idx = src_block_start + j * right_len + k;
                    
                    let index_val = ids[linear_idx];

                    // 处理 Mask (Padding Index)
                    if index_val == I::max_value() {
                        continue;
                    }

                    let idx = index_val.to_usize();
                    if idx >= dst_dim_size {
                        return Err(Error::InvalidIndex {
                            index: idx,
                            size: dst_dim_size,
                            op: "scatter-add",
                        }.into());
                    }

                    // 计算目标位置的索引
                    // 保持 Left(i) 和 Right(k) 不变，Dim 变为从 ids 读出的 idx
                    let dst_idx = dst_block_start + idx * right_len + k;

                    // 累加梯度
                    result[dst_idx] = result[dst_idx] + src[linear_idx];
                }
            }
        }

        Ok(result)
    }
}

impl<T: WithDType> Storage<T> {
    pub fn new<D: Into<Vec<T>>>(data: D) -> Self {
        Self(data.into())
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.0
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.0
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    #[inline]
    pub fn copy_data(&self) -> Vec<T> {
        self.0.clone()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.0.get(index).copied()
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.0[index]
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: T) -> Option<()> {
        if index >= self.len() {
            None
        } else {
            self.0[index] = value;
            Some(())
        }
    }

    #[inline]
    pub fn set_unchecked(&mut self, index: usize, value: T) {
        self.0[index] = value;
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn copy(&self, layout: &Layout) -> Self {
        let output: Vec<_> = layout.storage_indices()
            .map(|i| self.0[i])
            .collect();
        Self(output)
    }

    pub fn copy_map<F, U>(&self, layout: &Layout, f: F) -> Storage<U> 
    where 
        U: WithDType,
        F: Fn(T) -> U
    {
        let output: Vec<_> = layout.storage_indices()
            .map(|i| f(self.0[i]))
            .collect();
        Storage(output)
    }
}

#[derive(Clone)]
pub struct StorageArc<T>(pub(crate) Arc<RwLock<Storage<T>>>);

impl<T: WithDType> StorageArc<T> {
    pub fn new(storage: Storage<T>) -> Self {
        Self(Arc::new(RwLock::new(storage)))
    }

    #[inline]
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, Storage<T>> {
        self.0.read().unwrap()
    }

    #[inline]
    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, Storage<T>> {
        self.0.write().unwrap()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.read().get(index)
    }

    #[inline]
    pub fn set(&mut self, index: usize, val: T) -> Option<()> {
        self.write().set(index, val)
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.read().get_unchecked(index)
    }

    #[inline]
    pub fn set_unchecked(&self, index: usize, val: T) {
        self.write().set_unchecked(index, val)
    }

    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }

    #[inline]
    pub fn get_ref(&self, start_offset: usize) -> StorageRef<'_, T> {
        StorageRef::Guard(std::sync::RwLockReadGuard::map(self.0.read().unwrap(), |s| &s.data()[start_offset..]))
    }

    #[inline]
    pub fn get_mut(&self, start_offset: usize) -> StorageMut<'_, T> {
        StorageMut::Guard(std::sync::RwLockWriteGuard::map(self.0.write().unwrap(), |s| &mut s.data_mut()[start_offset..]))
    }

    #[inline]
    pub fn get_ptr(&self, start_offset: usize) -> *mut T {
        let mut s = self.0.write().unwrap();
        let data = &mut s.data_mut()[start_offset..];
        data.as_mut_ptr()
    }
}

pub enum StorageRef<'a, T> {
    Guard(std::sync::MappedRwLockReadGuard<'a, [T]>),
    Slice(&'a [T]),
}

// pub struct StorageMut<'a, T>(std::sync::MappedRwLockWriteGuard<'a, [T]>);

pub enum StorageMut<'a, T> {
    Guard(std::sync::MappedRwLockWriteGuard<'a, [T]>),
    Slice(&'a mut[T]),
}

impl<'a, T: WithDType> StorageRef<'a, T> {
    pub fn clone(&'a self) -> Self {
        Self::Slice(&self.data())
    }

    pub fn slice(&'a self, index: usize) -> Self {
        Self::Slice(&self.data()[index..])
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.data().get(index).copied()
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.data()[index]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data().len()
    }

    pub fn data(&self) -> &[T] {
        match self {
            Self::Guard(gurad) => &gurad,
            Self::Slice(s) => s,
        }
    }
}

impl<'a, T: WithDType> StorageMut<'a, T> {
    pub fn clone(&'a self) -> StorageRef<'a, T> {
        StorageRef::Slice(self.data())
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.data().get(index).copied()
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.data()[index]
    }

    #[inline]
    pub fn set(&mut self, index: usize, val: T) -> Option<()> {
        if index >= self.len() {
            None
        } else {
            self.set_unchecked(index, val);
            Some(())
        }
    }

    #[inline]
    pub fn set_unchecked(&mut self, index: usize, val: T) {
        self.data_mut()[index] = val;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data().len()
    }

    pub fn data(&self) -> &[T] {
        match self {
            Self::Guard(gurad) => &gurad,
            Self::Slice(s) => s,
        }
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        match self {
            Self::Guard(gurad) => &mut gurad[0..],
            Self::Slice(s) => &mut s[0..],
        }
    }
}