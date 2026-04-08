use lumen_core::{FloatDType, Tensor, D};
use lumen_macros::Module;

use crate::{Buffer, NnResult};

#[derive(Module)]
pub struct RotaryPosEmbedding<T: FloatDType> {
    sins: Buffer<T>,
    coss: Buffer<T>,

    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub max_len: usize,
    #[module(skip)]
    pub rope_theta: f64,
}

impl<T: FloatDType> RotaryPosEmbedding<T> {
    pub fn new(d_model: usize, max_len: usize, rope_theta: f64) -> NnResult<Self> {
        // 计算 ws
        // $$
        // w_i = \frac 1 {rope_theta^{2 i / d_model}}
        // $$
        let ws = (0..d_model).into_iter().step_by(2).map(|i| {
            T::from_f64(1.0 / rope_theta.powf( i as f64 / d_model as f64 ))
        }).collect::<Vec<_>>();
        // (1, d_model / 2)
        let ws = Tensor::new(ws)?.unsqueeze(0)?;

        // 计算所有可能的 token 位置：(0, 1, 2, ... , max_len - 1)
        // (max_len, 1)
        let position = Tensor::arange(T::ZERO, T::from_usize(max_len))?.unsqueeze(1)?;

        // 计算每个 theta 值
        // (max_len, 1) @ (1, d_model / 2) = (max_len, d_model / 2)
        let theta = position.matmul(&ws)?;

        // 分别计算 sin 和 cos 保存 (max_len, d_model / 2)
        let sins = theta.sin()?;
        let coss = theta.cos()?;
        // 拷贝一份: (max_len, d_model)
        let sins = Tensor::cat(&[&sins, &sins], 1)?;
        let coss = Tensor::cat(&[&coss, &coss], 1)?;

        Ok(Self { 
            sins: Buffer::new(sins),
            coss: Buffer::new(coss),
            d_model, max_len, rope_theta
        })
    }
    
    pub fn forward(&self, input: &Tensor<T>, start_pos: usize) -> NnResult<Tensor<T>> {
        // 从缓存中截取部分 (max_len, d_model) => (seq_len, d_model)
        let seq_len = input.dim(D::Minus2)?;
        let sin = self.sins.narrow(0, start_pos, seq_len)?;
        let cos = self.coss.narrow(0, start_pos, seq_len)?;

        // 对输入进行旋转: (.., seq_len, d_model)
        let rotated_input = self.rope_input(input)?;
        
        /*
            RoPE 公式：
            $$
            x1 = x1 cos - x2 sin
            x2 = x1 sin + x2 cos
            $$
        */
        let output = input.broadcast_mul(&cos)? + rotated_input.broadcast_mul(&sin)?;
        Ok(output)
    }

    /// 将 input 分为两个部分 [x1, x2]，并且返回 [-x2, x1] 
    fn rope_input(&self, input: &Tensor<T>) -> NnResult<Tensor<T>> {
        let dim_size = input.dim(D::Minus1)?;
        let half = dim_size / 2;
        
        let x1 = input.narrow(D::Minus1, 0, half)?; // (..., dim/2)
        let x2 = input.narrow(D::Minus1, half, half)?; // (..., dim/2)
        let neg_x2 = x2.neg()?;
        
        let rotated_input = Tensor::cat(&[neg_x2, x1], D::Minus1)?; // (..., dim)
        Ok(rotated_input)
    }

}
