use lumen_core::{FloatDType, IndexOp, Tensor};
use lumen_macros::Module;

use crate::{Buffer, NnResult};

#[derive(Module)]
pub struct SinusoidalPosEmbedding<T: FloatDType> {
    pe: Buffer<T>,

    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub max_len: usize,
}

impl<T: FloatDType> SinusoidalPosEmbedding<T> {
    /// SinusoidalPosEmbedding
    /// 
    /// 在所有序列中第 t 个 token 编码为：
    /// 
    /// $$
    /// PE_t = [\sin(w_0 t), \cos(w_1 t), \cdots, \sin(w_{d_model/2-1} t), \cos(w_{d_model/2-1} t)]
    /// $$
    /// 
    /// 其中
    /// 
    /// $$
    /// w_i = \frac 1 {10000^{2 i /(d_model -1 )}}
    /// $$
    /// 
    /// - 对每行，就是一系列不同频率 sin/cos 函数的相同位置的值
    /// - 对每列，是相同频率 sin/cos 的顺序采样
    /// 列靠前，可以看出值随着 token 位置周期变化，而靠后的列由于周期太大，token 间的差异很小
    /// 
    pub fn new(d_model: usize, max_len: usize) -> NnResult<Self> {
        // (max_len,)
        let position = Tensor::arange(T::ZERO, T::from_usize(max_len))?;
        let mut pe_vec = vec![];    

        // 每次计算两列
        for i in 0..d_model / 2 {
            let p = T::from_usize(2 * i) / T::from_usize(d_model - 1);
            let w_i = T::ONE / T::from_usize(10000).powf(p);
            // 每个位置 * w_i 得到 sin/cos 的输入
            let theta = w_i * &position;
            let sins = theta.sin()?; // (max_len,)
            let coss = theta.cos()?; // (max_len,)
            pe_vec.push(sins);
            pe_vec.push(coss);
        } 

        // [(max_len,); d_model] => (max_len, d_model)
        let pe = Tensor::stack(&pe_vec, 1)?;

        Ok(Self { pe: Buffer::new(pe), d_model, max_len })
    }

    /// ## Args
    /// - `input`: (batch_size, seq_len, d_model)
    /// 
    /// ## Return
    /// input + pos_embed
    pub fn forward(&self, input: &Tensor<T>, start_pos: usize) -> NnResult<Tensor<T>> {
        let (_, seq_len, _) = input.dims3()?;
        // 从 pe 中取出 seq_len 部分返回
        // (max_len, d_model) => (seq_len, d_model)
        let pe = self.pe.index(start_pos..start_pos+seq_len)?;
        let input_with_pos = input.broadcast_add(&pe)?;
        Ok(input_with_pos)
    }
}
