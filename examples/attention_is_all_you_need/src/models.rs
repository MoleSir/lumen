use std::marker::PhantomData;
use anyhow::Context;
use lumen_core::{FloatDType, Tensor, D};
use lumen_nn::{init::Init, Buffer, Embedding, Linear, Module, ModuleForward, Parameter, Relu};

//===============================================================//
//            缩放点积注意力 (Scaled Dot-Product Attention)
//===============================================================//

#[derive(Module, Default)]
pub struct ScaledDotProductAttention<T: FloatDType> {
    #[module(skip)]
    marker: PhantomData<T>,
}

impl<T: FloatDType> ScaledDotProductAttention<T> {
    pub fn new() -> Self {
        Self { marker: PhantomData::default() }
    }

    /// ```txt
    /// \tet{Attention} =  \text{softmax} \frac {Q K^\top} {\sqrt d} V  
    /// ```
    /// 
    /// ## Args
    /// - `q`: (batch, n_heads, seq_len1, d_k)
    /// - `k`: (batch, n_heads, seq_len2, d_k)
    /// - `v`: (batch, n_heads, seq_len2, d_k)
    /// - `mask`: (batch, n_heads, 1, seq_len2) or (batch, n_heads, seq_len2, seq_len2) if len1 == len2
    pub fn forward(&self, q: &Tensor<T>, k: &Tensor<T>, v: &Tensor<T>, mask: &Tensor<bool>) -> anyhow::Result<(Tensor<T>, Tensor<T>)> {
        let dim = q.dim(D::Minus1)?;

        // 1. 计算 Q V^T: (batch, n_heads, seq_len1, seq_len2)
        let mut scores = q.matmul(&k.transpose_last()?)? / T::from_usize(dim).sqrt(); 
        
        // 2. 增加 mask
        let mask = mask.broadcast_as(scores.shape())?;
        scores = mask.if_else(&scores, T::MIN_VALUE)?;
        
        // 3. 归一话，并与 V 加权
        let attn = lumen_nn::functional::softmax(&scores, D::Minus1)?;
        let context = attn.matmul(v)?;

        Ok((context, attn))
    }
}

//===============================================================//
//           多头注意力机制 (Multi-Head Attention)
//===============================================================//

#[derive(Module)]
pub struct MultiHeadAttention<T: FloatDType> {
    pub w_q: Linear<T>,
    pub w_k: Linear<T>,
    pub w_v: Linear<T>,
    pub fc: Linear<T>,
    pub attention: ScaledDotProductAttention<T>,

    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub n_heads: usize,
    #[module(skip)]
    pub d_k: usize,
}

impl<T: FloatDType> MultiHeadAttention<T> {
    pub fn new(d_model: usize, n_heads: usize) -> anyhow::Result<Self> {
        if d_model % n_heads != 0 {
            anyhow::bail!("d_model 无法整除 n_heads");
        }
        
        Ok(Self {
            w_q: Linear::new(d_model, d_model, false, None).context("new w_q")?,
            w_k: Linear::new(d_model, d_model, false, None).context("new w_k")?,
            w_v: Linear::new(d_model, d_model, false, None).context("new w_v")?,
            fc: Linear::new(d_model, d_model, false, None)?,
            attention: ScaledDotProductAttention::new(),
            d_model,
            n_heads,
            d_k: d_model / n_heads
        })
    }

    /// ## Args
    /// - `q`: (batch, seq_len1, d_model)
    /// - `k`: (batch, seq_len2, d_model)
    /// - `v`: (batch, seq_len2, d_model)
    /// - `mask`: (batch, seq_len2, seq_len2)
    pub fn forward(&self, q: &Tensor<T>, k: &Tensor<T>, v: &Tensor<T>, mask: &Tensor<bool>) -> anyhow::Result<Tensor<T>> {
        let batch = q.dim(0)?;
        
        // 1. 计算 q, k, v 
        let q_s = self.reshape_qkv(&self.w_q.forward(q)?)?; // (batch, n_heads, seq_len1, d_k)
        let k_s = self.reshape_qkv(&self.w_k.forward(k)?)?; // (batch, n_heads, seq_len2, d_k)
        let v_s = self.reshape_qkv(&self.w_v.forward(v)?)?; // (batch, n_heads, seq_len2, d_k)

        // 2. 计算 mask
        // (batch, seq_len2, seq_len2) => (batch, 1, seq_len2, seq_len2) => (batch, n_heads, seq_len2, seq_len2) 
        let mask = mask.unsqueeze(1)?.repeat_dim(1, self.n_heads)?;

        // 3. 注意力计算
        let (context, _attn) = self.attention.forward(&q_s, &k_s, &v_s, &mask)?;
        
        // 4. 拼接多头
        let context = context.transpose(1, 2)?.contiguous()?.reshape((batch, (), self.d_model))?;
        let output = self.fc.forward(&context)?;

        Ok(output)
    }

    fn reshape_qkv(&self, x: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let batch = x.dim(0)?;
        let y = x.reshape((batch, (), self.n_heads, self.d_k))?.transpose(1, 2)?;
        Ok(y)
    }
}

//===============================================================//
//           位置编码 (Positional Encoding)
//===============================================================//

#[derive(Module)]
pub struct PositionalEncoding<T: FloatDType> {
    pub pe: Buffer<T>,
}

impl<T: FloatDType> PositionalEncoding<T> {
    pub fn new(d_model: usize, max_len: usize) -> anyhow::Result<Self> {
        
        let positions = Tensor::arange(T::ZERO, T::from_usize(max_len))?;
        let mut pe = vec![];
        
        // 对每个不同的频率
        for i in (0..d_model).step_by(2) {
            let freq = 1.0 / ( 10000.0f64.powf(2.0 * i as f64 / d_model as f64) );
            let theta = T::from_f64(freq) * &positions; // (max_len,)
            let sin = theta.sin()?;
            let cos = theta.cos()?; 
            pe.push(sin);
            pe.push(cos);
        } 

        let pe = Tensor::stack(&pe, 1)?; // (max_len, d_model)

        Ok(Self { pe: Buffer::new(pe)} )
    }

    pub fn forward(&self, x: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        // x: (batch, seq_len, d_model)
        let seq_len = x.dim(1)?;
        // 取出 pe 的部分 (1, seq_len, d_model)
        let pe = self.pe.narrow(0, 0, seq_len)?.unsqueeze(0)?;
        let out = x.broadcast_add(&pe)?;
        Ok(out)
    }
}

//===============================================================//
//          前馈神经网络 (Feed-Forward Network)
//===============================================================//

#[derive(Module)]
pub struct PoswiseFeedForwardNet<T: FloatDType> {
    pub fc: (Linear<T>, Relu, Linear<T>),
}

impl<T: FloatDType> PoswiseFeedForwardNet<T> {
    pub fn new(d_model: usize, d_ff: usize) -> anyhow::Result<Self> {
        let fc1 = Linear::new(d_model, d_ff, false, None).context("new fc1")?;
        let fc2 = Linear::new(d_ff, d_model, false, None).context("new fc2")?;
        let relu = Relu::new();
        Ok(Self {
            fc: (fc1, relu, fc2)
        })
    }

    pub fn forward(&self, x: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let out = self.fc.forward(x.clone())?;
        Ok(out)
    }
}

//===============================================================//
//          Encoder & Decoder Layer
//===============================================================//

#[derive(Module)]
pub struct EncoderLayer<T: FloatDType> {
    pub mha: MultiHeadAttention<T>,
    pub ffn: PoswiseFeedForwardNet<T>,
    pub layernorm1: LayerNorm<T>,
    pub layernorm2: LayerNorm<T>,
}

impl<T: FloatDType> EncoderLayer<T> {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> anyhow::Result<Self> {
        let mha = MultiHeadAttention::new(d_model, n_heads).context("new mha")?;
        let ffn = PoswiseFeedForwardNet::new(d_model, d_ff).context("new ffn")?;
        let layernorm1 = LayerNorm::new(d_model).context("ln1")?;
        let layernorm2 = LayerNorm::new(d_model).context("ln2")?;
        Ok(Self { mha, ffn, layernorm1, layernorm2 })
    }

    /// ## Args
    /// - `x`: (batch, src_seq_len, d_model)
    /// - `mask`: (batch, src_seq_len, src_seq_len)
    pub fn forward(&self, x: &Tensor<T>, mask: &Tensor<bool>) -> anyhow::Result<Tensor<T>> {
        /*

                +---------------+       
                |    RMSNorm    |       
                +---------------+       
                        ^
                        |
                        +---------------+
                        ^               |
                        |               |
            +-----------------------+   |
            |                       |   |
            |           Mlp         |   |
            |                       |   |
            +-----------------------+   |
                        ^               |
                        |               |
                        +---------------+
                        ^
                        |
                +---------------+      
                |    RMSNorm    |       
                +---------------+       
                        ^
                        |
                        +---------------+
                        ^               |
                        |               |
            +-----------------------+   |
            |                       |   |
            |        Attention      |   |
            |                       |   |
            +-----------------------+   |
                        ^               |
                        |               |
                        +---------------+
                        |
        
        */
        // 1. Multi-head Attention + Residual + LayerNorm
        let attn_out = self.mha.forward(x, x, x, mask)?;
        let x = self.layernorm1.forward(&(x + &attn_out))?;

        // 2. FFN + Residual + LayerNorm
        let ffn_out = self.ffn.forward(&x)?;
        let x = self.layernorm2.forward(&(&x + &ffn_out))?;

        Ok(x)
    }
}

#[derive(Module)]
pub struct DecoderLayer<T: FloatDType> {
    pub mha1: MultiHeadAttention<T>,
    pub mha2: MultiHeadAttention<T>,

    pub ffn: PoswiseFeedForwardNet<T>,
   
    pub ln1: LayerNorm<T>,
    pub ln2: LayerNorm<T>,
    pub ln3: LayerNorm<T>,
}

impl<T: FloatDType> DecoderLayer<T> {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> anyhow::Result<Self> {
        let mha1 = MultiHeadAttention::new(d_model, n_heads).context("new mha1")?; // Masked MHA
        let mha2 = MultiHeadAttention::new(d_model, n_heads).context("new mha2")?; // Encoder-Decoder 
        
        let ffn = PoswiseFeedForwardNet::new(d_model, d_ff).context("new ffn")?;
        
        let ln1 = LayerNorm::new(d_model).context("ln1")?;
        let ln2 = LayerNorm::new(d_model).context("ln2")?;
        let ln3 = LayerNorm::new(d_model).context("ln3")?;
        
        Ok(Self { mha1, mha2, ffn, ln1, ln2, ln3 })
    }

    /// ## Args
    /// - `dec_inputs`: (batch, tgt_seq_len, d_model)
    /// - `enc_outputs`: (batch, src_seq_len, d_model)
    /// - `self_mask`: (batch, tgt_seq_len, tgt_seq_len)
    /// - `cross_mask`: (batch, src_seq_len, src_seq_len)
    pub fn forward(
        &self, dec_inputs: &Tensor<T>, enc_outputs: &Tensor<T>, self_mask: &Tensor<bool>, cross_mask: &Tensor<bool>
    ) -> anyhow::Result<Tensor<T>> {
        /*

                +---------------+       
                |    RMSNorm    |       
                +---------------+       
                        ^
                        |
                        +---------------+
                        ^               |
                        |               |
            +-----------------------+   |
            |                       |   |
            |           Mlp         |   |
            |                       |   |
            +-----------------------+   |
                        ^               |
                        |               |
                        +---------------+
                        ^
                        |
                +---------------+      
                |    RMSNorm    |       
                +---------------+       
                        ^
                        |
                        +---------------+
                        ^               |
                        |               |
            +-----------------------+   |
            |                       |   |
            |        Attention      |   |
            |                       |   |
            +-----------------------+   |
                        ^               |
                        |               |
                        +---------------+
                        |
        
        */
        // 1. Masked Self-Attention
        let self_attn_out = self.mha1.forward(dec_inputs, dec_inputs, dec_inputs, self_mask)?;
        let out = self.ln1.forward(&(dec_inputs + &self_attn_out))?;

        // 2. Encoder-Decoder Attention (Q来自解码器，K, V来自编码器)
        let cross_attn_out = self.mha2.forward(&out, enc_outputs, enc_outputs, cross_mask)?;
        let out = self.ln2.forward(&(out + &cross_attn_out))?;

        // 3. FFN
        let ffn_out = self.ffn.forward(&out)?;
        let out = self.ln3.forward(&(&out + &ffn_out))?;

        Ok(out)
    }
}

#[derive(Module)]
pub struct LayerNorm<T: FloatDType> {
    pub weight: Parameter<T>,
}

impl<T: FloatDType> LayerNorm<T> {
    pub fn new(d_model: usize) -> anyhow::Result<Self> {
        let weight = Init::ones().init_param((1, 1, d_model,))?;
        Ok(Self { weight })
    }

    pub fn forward(&self, x: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        // x: (batch, seq_len, d_model)
        // 计算 平均 & std
        let mean = x.mean_keepdim(D::Minus1)?; // (batch, seq_len, 1)
        let var = x.var_keepdim(D::Minus1)?; // (batch, seq_len, 1)
        let std = (var + T::epsilon()).sqrt()?; // (batch, seq_len, 1)

        let y = x
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?;
        let y = y.broadcast_mul(&self.weight)?;

        Ok(y)
    }
}

//===============================================================//
//          Transformer
//===============================================================//

#[derive(Module)]
pub struct Transformer<T: FloatDType> {
    pub src_emb: Embedding<T>,
    pub tgt_emb: Embedding<T>,
    pub pos_emb: PositionalEncoding<T>,

    pub encoder_layers: Vec<EncoderLayer<T>>,
    pub decoder_layers: Vec<DecoderLayer<T>>,

    pub projection: Linear<T>,
}

impl<T: FloatDType> Transformer<T> {
    pub fn new(src_vocab_size: usize, tgt_vocab_size: usize, d_model: usize, n_heads: usize, n_layers: usize, d_ff: usize) -> anyhow::Result<Self> {
        let src_emb = Embedding::new(src_vocab_size, d_model, None)?;
        let tgt_emb = Embedding::new(tgt_vocab_size, d_model, None)?;
        let pos_emb = PositionalEncoding::new(d_model, 1000)?;

        let mut encoder_layers = vec![];
        let mut decoder_layers = vec![];
        for _ in 0..n_layers {
            encoder_layers.push(EncoderLayer::new(d_model, n_heads, d_ff)?);
            decoder_layers.push(DecoderLayer::new(d_model, n_heads, d_ff)?);
        }

        let projection = Linear::new(d_model, tgt_vocab_size, false, None)?;

        Ok(Self { src_emb, tgt_emb, pos_emb, encoder_layers, decoder_layers, projection })
    }

    /// ## Args
    /// - `src`: (batch, src_seq_len)
    /// - `tgt`: (batch, tgt_seq_len)
    pub fn forward(&self, src: &Tensor<u32>, tgt: &Tensor<u32>) -> anyhow::Result<Tensor<T>> {
        // 1. 获取 src 和 tgt 的 mask
        let src_mask = self.get_pad_mask(src, 0)?; // (batch, 1, src_seq_len)
        let tgt_pad_mask = self.get_pad_mask(tgt, 0)?.repeat_dim(1, tgt.dim(D::Minus1)?)?; // (batch, tgt_seq_len, tgt_seq_len)
        let tgt_causal_mask = self.get_subsequent_mask(&tgt)?.broadcast_as(tgt_pad_mask.shape())?; // (batch, tgt_seq_len, tgt_seq_len)
        let tgt_mask = tgt_causal_mask.and(&tgt_pad_mask)?; // (batch, tgt_seq_len, tgt_seq_len)

        // 2. encoder forward
        let enc_out = self.src_emb.forward(src)?; // (batch, src_seq_len, d_model)
        let mut enc_out = self.pos_emb.forward(&enc_out)?; // (batch, src_seq_len, d_model)
        for layer in self.encoder_layers.iter() {
            enc_out = layer.forward(&enc_out, &src_mask)?;
        }

        // 3. decoder forward
        let dec_out = self.tgt_emb.forward(tgt)?; // (batch, tgt_seq_len, d_model)
        let mut dec_out = self.pos_emb.forward(&dec_out)?; // (batch, tgt_seq_len, d_model)
        for layer in self.decoder_layers.iter() {
            dec_out = layer.forward(&dec_out, &enc_out, &tgt_mask, &src_mask)?;
        }

        // 4. 对 decoder 最后的输出进行预测下一个 token 概率 
        let logits = self.projection.forward(&dec_out)?;

        Ok(logits)
    }

    fn get_pad_mask(&self, x: &Tensor<u32>, pad_idx: u32) -> anyhow::Result<Tensor<bool>> {
        // (batch, seq_len) => (batch, 1, seq_len)
        let mask = x.ne(pad_idx)?.unsqueeze(1)?;
        Ok(mask)
    }

    fn get_subsequent_mask(&self, x: &Tensor<u32>) -> anyhow::Result<Tensor<bool>> {
        let (_batch_size, seq_len) = x.dims2()?;
        // 右上（包含对角）为 true
        let subsequent_mask = Tensor::<bool>::triu(seq_len, true)?.unsqueeze(0)?;
        Ok(subsequent_mask)
    }
}

#[cfg(test)]
mod tests {
    use lumen_core::Tensor;
    use super::Transformer;

    #[test]
    fn test_shape() {
        let model = Transformer::<f32>::new(5000, 5000, 512, 8, 6, 2048).unwrap();
        let src = Tensor::<u32>::new(&[[1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]]).unwrap();
        let tgt = Tensor::<u32>::new(&[[1, 2, 3, 4, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0, 0]]).unwrap();
        let output = model.forward(&src, &tgt).unwrap();
        println!("{}", output.shape());
    }
}