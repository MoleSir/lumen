use crate::{Tensor, AutogradMetaT, Dim, FloatDType, Layout, Storage, StorageIndices, D};
use super::reduce::DimArray;

// ==============================================================================================//
//                             RMS Norm
// ==============================================================================================//
impl<T: FloatDType> Tensor<T> {
    /// RMS Norm
    /// - input: (..., dim)
    /// - weight: (dim,)
    pub fn rms_norm(input: &Tensor<T>, weight: &Tensor<T>, eps: T) -> crate::Result<Tensor<T>> {
        // 1. Calculate Mean of Squares: Mean(x^2)
        let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
    
        // 2. Calculate RMS: sqrt(variance + eps)
        let rms = (variance + eps).sqrt()?;
    
        // 3. Normalize: x / rms
        let input_normalized = input.broadcast_div(&rms)?;
    
        // 4. Scale: x * weight
        let out = input_normalized.broadcast_mul(weight)?;
    
        Ok(out)
    }

    /// RMS Norm
    /// - input: (..., dim)
    /// - weight: (dim,)
    pub fn rms_norm_fused(input: &Tensor<T>, weight: &Tensor<T>, eps: T) -> crate::Result<Tensor<T>> {
        let last_dim = D::Minus1.to_index(input.shape(), "rms_norm")?;
        let input_layout = input.layout();
        let last_dim_stride = input_layout.stride()[last_dim];
        let last_dim_size = input_layout.dims()[last_dim];
        let weight_size = weight.dims1()?;
        if last_dim_size != weight_size {
            crate::bail!("input dim size != weight size in rms norm");
        }

        let input_storage = input.storage_read()?;
        let input_data = input_storage.data();

        let weight_storage = weight.storage_read()?;
        let weight_data = weight_storage.data();
        let weight_data = &weight_data[weight.layout().start_offset..];
        let weight_stride = weight.layout().stride()[0];        
        let mut output_data = vec![T::ZERO; input.element_count()];
    
        let batch_size = output_data.len() / last_dim_size;
        match input_layout.storage_indices() {
            StorageIndices::Contiguous(index) => {
                let input_data = &input_data[index.begin_index..index.end_index];
                for b in 0..batch_size {
                    let input_data = &input_data[b*last_dim_size..b*last_dim_size+last_dim_size];
                    let output_data = &mut output_data[b*last_dim_size..b*last_dim_size+last_dim_size];
                    
                    let variance = input_data.iter().cloned().map(|v| v.sqr()).sum::<T>() / T::from_usize(last_dim_size);
                    let rms = (variance + eps).sqrt();

                    for i in 0..last_dim_size {
                        output_data[i] = input_data[i] / rms * weight_data[i * weight_stride];
                    }
                }
            }
            StorageIndices::Uncontiguous(_) => {
                let layout = input_layout.narrow(last_dim, 0, 1)?;
                println!("{}", layout.storage_indices().count());
                for (b, src_index) in layout.storage_indices().enumerate() {
                    let output_data = &mut output_data[b*last_dim_size..b*last_dim_size+last_dim_size];
                    let input_arr: DimArray<'_, T> = DimArray {
                        src: &input_data[src_index..],
                        size: last_dim_size,
                        stride: last_dim_stride,
                    };

                    let variance = input_arr.clone().into_iter().map(|v| v.sqr()).sum::<T>() / T::from_usize(last_dim_size);
                    let rms = (variance + eps).sqrt();

                    for (i, v) in (0..last_dim_size).into_iter().zip(input_arr.into_iter()) {
                        output_data[i] = v / rms * weight_data[i * weight_stride];
                    }
                }
            }
        }

        let output_storage = Storage::new(output_data);
        let meta = T::AutogradMeta::on_rms_norm_op(input, weight, eps);
        let output = Tensor::from_storage(output_storage, input.shape(), meta);
        Ok(output)
    }

    pub(crate) fn rms_norm_backward_fused(
        grad_output: &Tensor<T>,
        input: &Tensor<T>,      
        weight: &Tensor<T>, 
        input_sum_grad: &Tensor<T>,
        weight_sum_grad: &Tensor<T>,
        eps: T,
    ) -> crate::Result<()> {
        let shape = input.shape();
        let last_dim = shape.dims().len() - 1;
        let last_dim_size = shape.dims()[last_dim];
        let batch_size = input.element_count() / last_dim_size;
        let d_inv = T::from_f64(1.0 / last_dim_size as f64);

        // 获取数据引用
        let g_data_storage = grad_output.storage_read()?;
        let g_data = g_data_storage.data();
        assert!(grad_output.is_contiguous());

        let x_data_storage = input.storage_read()?;
        let x_data = x_data_storage.data();

        let w_data_storage = weight.storage_read()?;
        let w_data = w_data_storage.data();
        let w_data = &w_data[weight.layout().start_offset..];
        let w_stride = weight.layout().stride()[0];

        // 准备输出结果，直接分配内存
        assert!(input_sum_grad.is_contiguous());
        assert!(weight_sum_grad.is_contiguous());
        let mut input_grad_storage = input_sum_grad.storage_write()?;
        let mut weight_gard_storage = weight_sum_grad.storage_write()?;
        let input_gard_data = input_grad_storage.data_mut();
        let weight_gard_data = weight_gard_storage.data_mut();

        if input.is_contiguous() {
            let x_data = &x_data[input.layout().start_offset..];
            // 遍历每个 Batch (每一行数据)
            for b in 0..batch_size {
                let offset = b * last_dim_size;
                let g_row = &g_data[offset..offset + last_dim_size];
                let x_row = &x_data[offset..offset + last_dim_size];

                let input_gard_row_data = &mut input_gard_data[offset..offset + last_dim_size];

                // --- 第一次遍历：计算 v (方差) 和 s (点积和) ---
                let mut sum_sq = T::ZERO;
                let mut sum_dot = T::ZERO;
                
                for i in 0..last_dim_size {
                    let xi = x_row[i];
                    let wi = w_data[i * w_stride];
                    let gi = g_row[i];
                    
                    sum_sq = sum_sq + xi * xi;
                    // 注意：这里的 gi * wi * xi 是为了计算输入的梯度
                    sum_dot = sum_dot + gi * wi * xi;
                }
                
                let variance = sum_sq * d_inv + eps;
                let rsqrt_v = T::one() / variance.sqrt(); // 1/rms
                let s_v = (sum_dot * d_inv) / variance;   // s/v

                // --- 第二次遍历：计算 dx 和累加 dw ---
                for i in 0..last_dim_size {
                    let xi = x_row[i];
                    let wi = w_data[i * w_stride];
                    let gi = g_row[i];

                    // 计算 dx: (wi / rms) * (gi - xi * s / v)
                    // input_gard_row_data[i] += (wi * rsqrt_v) * (gi - xi * s_v);
                    input_gard_row_data[i] += rsqrt_v * (wi * gi - xi * s_v);

                    // 计算 dw: gi * (xi / rms) -> 并在 batch 间累加
                    weight_gard_data[i] += gi * (xi * rsqrt_v);
                }
            }
        } else {
            unimplemented!("rms norm backward in un continue")
        }

        Ok(())
    }

    pub(crate) fn softmax_backward_fused(
        grad_output: &Tensor<T>, 
        output: &Tensor<T>, 
        input_grad: &Tensor<T>, 
        dim: usize,
    ) -> crate::Result<()> {
        let reduce_dim = dim;
        let shape = output.shape();
        let reduce_dim_size = shape.dims()[reduce_dim];
        let element_count = output.layout().element_count();

        // 获取只读数据引用
        let g_data_storage = grad_output.storage_read()?;
        let g_data = g_data_storage.data();

        let y_data_storage = output.storage_read()?;
        let y_data = y_data_storage.data();

        // 准备输出梯度的目标（获取可写引用）
        assert!(input_grad.is_contiguous());
        let mut dx_storage = input_grad.storage_write()?;
        let dx_data = dx_storage.data_mut();

        // 获取 Stride 来判断是否可以用 Fast Path
        let y_stride = output.layout().stride()[reduce_dim];
        let g_stride = grad_output.layout().stride()[reduce_dim];

        if output.is_contiguous() && grad_output.is_contiguous() && y_stride == 1 && g_stride == 1 {
            let y_data = &y_data[output.layout().start_offset..output.layout().start_offset + element_count];
            let g_data = &g_data[grad_output.layout().start_offset..grad_output.layout().start_offset + element_count];
            let dx_data = &mut dx_data[input_grad.layout().start_offset..input_grad.layout().start_offset + element_count];

            let batch_size = element_count / reduce_dim_size;

            // 遍历每个 Batch
            for b in 0..batch_size {
                let offset = b * reduce_dim_size;
                let g_row = &g_data[offset..offset + reduce_dim_size];
                let y_row = &y_data[offset..offset + reduce_dim_size];
                let dx_row = &mut dx_data[offset..offset + reduce_dim_size];

                // 计算 sum(dy * y) 
                let mut sum_gy = T::ZERO; 
                for i in 0..reduce_dim_size {
                    sum_gy = sum_gy + g_row[i] * y_row[i];
                }

                // 计算 dx 并累加 
                // 公式: dx_i = y_i * (dy_i - sum(dy * y))
                for i in 0..reduce_dim_size {
                    let dy_i = g_row[i];
                    let y_i = y_row[i];
                    dx_row[i] += y_i * (dy_i - sum_gy);
                }
            }
        } else {
            unimplemented!("softmax backward in non-contiguous memory is not implemented yet")
        }

        Ok(())
    }
}

impl<T: FloatDType> Tensor<T> {
    pub fn softmax<D: Dim>(&self, dim: D) -> crate::Result<Tensor<T>> {
        let dim = dim.to_index(self.shape(), "softmax")?;
        let max = self.max_keepdim(dim)?; // (..., 1, ...)
        let diff = self.broadcast_sub(&max)?;   // (..., D, ...)
        let num = diff.exp()?;  // (..., D, ...)
        let den = num.sum_keepdim(dim)?;
        let out = num.broadcast_div(&den)?;
        Ok(out)
    }

    pub fn softmax_fused<D: Dim>(&self, dim: D) -> crate::Result<Tensor<T>> {
        let reduce_dim = dim.to_index(self.shape(), "softmax_fused")?;
        assert!(reduce_dim < self.layout().dims().len());
    
        let reduce_dim_stride = self.layout().stride()[reduce_dim];
        let reduce_dim_size = self.layout().dims()[reduce_dim];
    
        let src_storage = self.storage_read()?;
        let src_data = src_storage.data();
        let src_start_offset = self.layout().start_offset;
    
        let element_count = self.layout().element_count();
    
        let mut dst: Vec<T> = Vec::with_capacity(element_count);
        unsafe { dst.set_len(element_count) };
    
        // fast path: contiguous
        if self.is_contiguous() && reduce_dim_stride == 1 {
            let src_data = &src_data[src_start_offset..src_start_offset + element_count];
    
            for (src_chunk, dst_chunk) in
                src_data.chunks(reduce_dim_size).zip(dst.chunks_mut(reduce_dim_size))
            {
                // ===== pass 1: online max + sum =====
                let mut m = src_chunk[0];
                let mut s = T::one();
    
                for &x in src_chunk.iter().skip(1) {
                    if x <= m {
                        s = s + (x - m).exp();
                    } else {
                        s = s * (m - x).exp() + T::one();
                        m = x;
                    }
                }
    
                // ===== pass 2: 写最终结果 =====
                for (i, &x) in src_chunk.iter().enumerate() {
                    dst_chunk[i] = (x - m).exp() / s;
                }
            }
        } else {
            let input_layout = self.layout().narrow(reduce_dim, 0, 1)?;
            let out_layout = Layout::contiguous(self.shape());
            let out_layout_narrow = out_layout.narrow(reduce_dim, 0, 1)?;
            let out_stride = out_layout.stride()[reduce_dim];
    
            for (src_offset, dst_offset) in
                input_layout.storage_indices().zip(out_layout_narrow.storage_indices())
            {
                // ===== pass 1 =====
                let mut m = src_data[src_offset];
                let mut s = T::one();
    
                for i in 1..reduce_dim_size {
                    let x = src_data[src_offset + i * reduce_dim_stride];
                    if x <= m {
                        s = s + (x - m).exp();
                    } else {
                        s = s * (m - x).exp() + T::one();
                        m = x;
                    }
                }
    
                // ===== pass 2 =====
                for i in 0..reduce_dim_size {
                    let x = src_data[src_offset + i * reduce_dim_stride];
                    dst[dst_offset + i * out_stride] = (x - m).exp() / s;
                }
            }
        }
    
        let storage = Storage::new(dst);
        let meta = T::AutogradMeta::on_softmax_op(self, reduce_dim);
        Ok(Tensor::<T>::from_storage(storage, self.shape(), meta))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Tensor, Var};

    #[test]
    fn test_rms_norm_forward() -> crate::Result<()> {
        let weight = Tensor::randn(0.0, 1.0, (64,))?;
        const EPS: f64 = 1e-4;

        let input = Tensor::randn(0.0, 1.0, (4, 10, 64))?;
        println!("{}", input.is_contiguous());
        let result1 = Tensor::rms_norm(&input, &weight, EPS)?;
        let result_fused1 = Tensor::rms_norm_fused(&input, &weight, EPS)?;
        assert!(result1.allclose(&result_fused1, 1e-5, 8e-8)?);

        let input = input.transpose_last()?.copy()?.transpose_last()?;
        println!("{}", input.is_contiguous());
        let result2 = Tensor::rms_norm(&input, &weight, EPS)?;
        let result_fused2 = Tensor::rms_norm_fused(&input, &weight, EPS)?;
        assert!(result1.allclose(&result2, 1e-5, 8e-8)?);
        assert!(result1.allclose(&result_fused2, 1e-5, 8e-8)?);

        Ok(())
    }

    use std::time::Instant;

    #[test]
    fn test_rms_norm_backward() -> crate::Result<()> {
        let input = Var::randn(0.0, 1.0, (1, 32, 128))?;
        let weight = Var::randn(0.0, 1.0, (128,))?;
        const EPS: f64 = 1e-4;
    
        {
            let _ = Tensor::rms_norm(&input, &weight, EPS)?;
            let _ = Tensor::rms_norm_fused(&input, &weight, EPS)?;
        }
    
        // baseline
        let start = Instant::now();
        let result = Tensor::rms_norm(&input, &weight, EPS)?;
        let forward_time_1 = start.elapsed();
    
        let start = Instant::now();
        let grads1 = result.backward()?;
        let backward_time_1 = start.elapsed();
    
        // fused
        let start = Instant::now();
        let result_fused = Tensor::rms_norm_fused(&input, &weight, EPS)?;
        let forward_time_2 = start.elapsed();
    
        let start = Instant::now();
        let grads2 = result_fused.backward()?;
        let backward_time_2 = start.elapsed();
    
        assert!(result.allclose(&result_fused, 1e-5, 8e-8)?);
        assert!(grads1[&input].allclose(&grads2[&input], 1e-5, 8e-8)?);
        assert!(grads1[&weight].allclose(&grads2[&weight], 1e-5, 8e-8)?);
    
        println!("=== Forward ===");
        println!("rms_norm       : {:?}", forward_time_1);
        println!("rms_norm_fused : {:?}", forward_time_2);
    
        println!("=== Backward ===");
        println!("rms_norm       : {:?}", backward_time_1);
        println!("rms_norm_fused : {:?}", backward_time_2);
    
        println!("=== Speedup ===");
        println!(
            "forward speedup : {:.2}x",
            forward_time_1.as_secs_f64() / forward_time_2.as_secs_f64()
        );
        println!(
            "backward speedup: {:.2}x",
            backward_time_1.as_secs_f64() / backward_time_2.as_secs_f64()
        );
    
        Ok(())
    }

    #[test]
    fn test_softmax_forward() -> crate::Result<()> {
        let input = Tensor::randn(0.0, 1.0, (4, 10, 64))?;
        println!("{}", input.is_contiguous());
        let result1 = Tensor::softmax(&input, 2)?;
        let result_fused1 = Tensor::softmax_fused(&input, 2)?;
        assert!(result1.allclose(&result_fused1, 1e-5, 8e-8)?);

        let input = input.transpose_last()?.copy()?.transpose_last()?;
        println!("{}", input.is_contiguous());
        let result2 = Tensor::softmax(&input, 2)?;
        let result_fused2 = Tensor::softmax_fused(&input, 2)?;
        assert!(result1.allclose(&result2, 1e-5, 8e-8)?);
        assert!(result1.allclose(&result_fused2, 1e-5, 8e-8)?);

        Ok(())
    }

    #[test]
    fn test_softmax_forward_not_last_dim() -> crate::Result<()> {
        let input = Tensor::randn(0.0, 1.0, (4, 10, 64))?;
        println!("{}", input.is_contiguous());
        let result1 = Tensor::softmax(&input, 1)?;
        let result_fused1 = Tensor::softmax_fused(&input, 1)?;
        assert!(result1.allclose(&result_fused1, 1e-5, 8e-8)?);

        let input = input.transpose_last()?.copy()?.transpose_last()?;
        println!("{}", input.is_contiguous());
        let result2 = Tensor::softmax(&input, 1)?;
        let result_fused2 = Tensor::softmax_fused(&input, 1)?;
        assert!(result1.allclose(&result2, 1e-5, 8e-8)?);
        assert!(result1.allclose(&result_fused2, 1e-5, 8e-8)?);

        Ok(())
    }


    #[test]
    fn test_softmax_backward() -> crate::Result<()> {
        let shape = (8, 32, 128);
        let dims_to_test = [2];
    
        for dim in dims_to_test {
            println!("\n========================================");
            println!("Testing Softmax along dim: {}", dim);
            println!("========================================");
    
            let input = Var::randn(0.0, 1.0, shape)?;
            {
                let _ = input.softmax(dim)?;
                let _ = input.softmax_fused(dim)?;
            }
    
            // baseline
            let start = Instant::now();
            let result = input.softmax(dim)?;
            let forward_time_1 = start.elapsed();
    
            let start = Instant::now();
            let grads1 = result.backward()?;
            let backward_time_1 = start.elapsed();
    
            // fused
            let start = Instant::now();
            let result_fused = input.softmax_fused(dim)?;
            let forward_time_2 = start.elapsed();
    
            let start = Instant::now();
            let grads2 = result_fused.backward()?;
            let backward_time_2 = start.elapsed();
    
            assert!(
                result.allclose(&result_fused, 1e-5, 8e-8)?,
                "Forward mismatch at dim {}", dim
            );
            assert!(
                grads1[&input].allclose(&grads2[&input], 1e-5, 8e-8)?,
                "Backward mismatch at dim {}", dim
            );
    
            println!("=== Forward ===");
            println!("softmax       : {:?}", forward_time_1);
            println!("softmax_fused : {:?}", forward_time_2);
    
            println!("=== Backward ===");
            println!("softmax       : {:?}", backward_time_1);
            println!("softmax_fused : {:?}", backward_time_2);
    
            println!("=== Speedup ===");
            println!(
                "forward speedup : {:.2}x",
                forward_time_1.as_secs_f64() / forward_time_2.as_secs_f64()
            );
            println!(
                "backward speedup: {:.2}x",
                backward_time_1.as_secs_f64() / backward_time_2.as_secs_f64()
            );
        }
    
        Ok(())
    }
}