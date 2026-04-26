use lumen_core::{FloatDType, IndexOp, Tensor, D};
use lumen_nn::{init::Init, Dropout, Embedding, Gelu, Linear, Module, ModuleForward, Parameter};

#[derive(Module)]
pub struct BertEmbeddings<T: FloatDType> {
    pub token_embeddings: Embedding<T>,
    pub position_embeddings: Embedding<T>,
    pub token_type_embeddings: Embedding<T>,
    pub layer_norm: LayerNorm<T>,
    pub dropout: Dropout<T>,
}

impl<T: FloatDType> BertEmbeddings<T> {
    pub fn new(vocab_size: usize, hidden_size: usize, max_len: usize, token_type_size: usize) -> anyhow::Result<Self> {
        Ok(Self {
            token_embeddings: Embedding::new(vocab_size, hidden_size, None)?,
            position_embeddings: Embedding::new(max_len, hidden_size, None)?,
            token_type_embeddings: Embedding::new(token_type_size, hidden_size, None)?,
            layer_norm: LayerNorm::new(hidden_size)?,
            dropout: Dropout::new(T::from_f64(0.1)),
        })
    }

    /// ## Args
    /// - `input_ids`: (batch, seq_len)
    /// - `token_type_ids`: (batch, seq_len)
    pub fn forward(&self, input_ids: &Tensor<u32>, token_type_ids: &Tensor<u32>) -> anyhow::Result<Tensor<T>> {
        let seq_len = input_ids.dim(1)?; 
        
        // 对每个 token 生成一个输入位置，提取 position embed
        let pos = Tensor::arange(0, seq_len as u32)?;
        // (seq_len) => (1, seq_len) => (batch, seq_len)
        let pos = pos.unsqueeze(0)?.broadcast_as(input_ids.shape())?;

        // (batch, seq_len) => (batch, seq_len, hidden_size)
        let token_embed = self.token_embeddings.forward(input_ids)?; 
        let position_embed = self.position_embeddings.forward(&pos)?;
        let token_type_embed = self.token_type_embeddings.forward(token_type_ids)?;

        let emb = token_embed + position_embed + token_type_embed;
        let output = self.layer_norm.forward(&emb)?;

        Ok(output)
    }
}

#[derive(Module)]
pub struct MultiHeadAttention<T: FloatDType> {
    pub query: Linear<T>,
    pub key: Linear<T>,
    pub value: Linear<T>,
    pub output: Linear<T>,

    #[module(skip)]
    pub n_heads: usize,
    #[module(skip)]
    pub head_size: usize,
}

impl<T: FloatDType> MultiHeadAttention<T> {
    pub fn new(hidden_size: usize, n_heads: usize) -> anyhow::Result<Self> {
        assert!(hidden_size % n_heads == 0);
        let head_size = hidden_size / n_heads;

        let query = Linear::new(hidden_size, hidden_size, false, None)?;
        let key = Linear::new(hidden_size, hidden_size, false, None)?;
        let value = Linear::new(hidden_size, hidden_size, false, None)?;
        let output = Linear::new(hidden_size, hidden_size, false, None)?;

        Ok(Self {
            query, key, value, output,
            n_heads, head_size
        })
    }

    /// ## Args
    /// - `hidden_state`: (batch, seq_len, hidden_size)
    /// - `mask`: (batch, 1, 1, seq_len): bert 只需要 pad mask!
    pub fn forward(&self, hidden_state: &Tensor<T>, mask: &Tensor<bool>) -> anyhow::Result<Tensor<T>> {
        let (batch, seq_len, _) = hidden_state.dims3()?;
        
        // 1. 经过 q k v 变换
        // (batch, seq_len, hidden_size) => (batch, seq_len, n_heads, head_size) => (batch, n_heads, seq_len, head_size),
        let q = self.query.forward(hidden_state)?.reshape((batch, seq_len, self.n_heads, ()))?.transpose(1, 2)?;
        let k = self.key.forward(hidden_state)?.reshape((batch, seq_len, self.n_heads, ()))?.transpose(1, 2)?;
        let v = self.value.forward(hidden_state)?.reshape((batch, seq_len, self.n_heads, ()))?.transpose(1, 2)?;

        // 2. 计算注意力权重 (batch, n_heads, seq_len, seq_len),
        let weight = q.matmul(&k.transpose_last()?)? / T::from_usize(self.n_heads).sqrt();

        // 3. 增加 mask: (batch, seq_len) => (batch, 1, 1, seq_len) => (batch, n_heads, seq_len, seq_len)
        let mask = mask.broadcast_as(weight.shape())?;
        // mask 为 false 的位置设置为小值
        let weight = mask.if_else(&weight, T::MIN_VALUE)?;
        
        // 4. 计算得分
        let scores = weight.softmax(D::Minus1)?; // (batch, n_heads, seq_len, seq_len)

        // 5. 与 v 加权
        let context = scores.matmul(&v)?; // (batch, n_heads, seq_len, head_size)
        let context = context.transpose(1, 2)?.contiguous()?.reshape((batch, seq_len, ()))?;

        // 6. 输出
        let output = self.output.forward(&context)?;

        Ok(output)
    }
}

#[derive(Module)]
pub struct LayerNorm<T: FloatDType> {
    pub weight: Parameter<T>,
}

impl<T: FloatDType> LayerNorm<T> {
    pub fn new(hidden_size: usize) -> anyhow::Result<Self> {
        let weight = Init::ones().init_param((1, 1, hidden_size))?;
        Ok(Self { weight })
    }

    /// ## Args
    /// - `x`: (batch, seq_len, hidden_size)
    pub fn forward(&self, x: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let mean = x.mean_keepdim(D::Minus1)?; // (batch, seq_len, 1)
        let var = x.var_keepdim(D::Minus1)?; // (batch, seq_len, 1)
        let std = (var + T::epsilon()).sqrt()?; // (batch, seq_len, 1)

        let x_norm = x
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?;  // (batch, seq_len, hidden_size)

        // (1, 1, hidden_size) * (batch, seq_len, hidden_size)
        let output = self.weight.broadcast_mul(&x_norm)?;
        Ok(output)
    }
}


#[derive(Module)]
pub struct BertLayer<T: FloatDType> {
    pub attention: MultiHeadAttention<T>,
    pub norm1: LayerNorm<T>,
    pub ffn: (Linear<T>, Gelu, Linear<T>),
    pub norm2: LayerNorm<T>,
    pub dropout: Dropout<T>,
}

impl<T: FloatDType> BertLayer<T> {
    pub fn new(hidden_size: usize, n_heads: usize, intermediate_size: usize) -> anyhow::Result<Self> {
        let attention = MultiHeadAttention::new(hidden_size, n_heads)?;
        let norm1 = LayerNorm::new(hidden_size)?;
        let ffn = (
            Linear::new(hidden_size, intermediate_size, false, None)?,
            Gelu::new(),
            Linear::new(intermediate_size, hidden_size, false, None)?,
        );
        let norm2 = LayerNorm::new(hidden_size)?;
        let dropout = Dropout::new(T::from_f64(0.1));
        
        Ok(Self { attention, norm1, ffn, norm2, dropout })
    }   

    pub fn forward(&self, hidden_state: &Tensor<T>, mask: &Tensor<bool>) -> anyhow::Result<Tensor<T>> {
        // 1. attention: attn + 残差 + norm
        let att_out = self.dropout.forward(&self.attention.forward(hidden_state, mask)?)?;
        let res = att_out + hidden_state;
        let out = self.norm1.forward(&res)?;
    
        // 2. ffn
        let ffn_out = self.dropout.forward(&self.ffn.forward(out.clone())?)?;
        let res = out + ffn_out;
        let out = self.norm2.forward(&res)?;
        
        Ok(out)
    }
}

#[derive(Module)]
pub struct Bert<T: FloatDType> {
    pub embeddings: BertEmbeddings<T>,
    pub layers: Vec<BertLayer<T>>,
    pub pooler: Linear<T>,
}

impl<T: FloatDType> Bert<T> {
    pub fn new(vocab_size: usize, hidden_size: usize, n_layers: usize, n_heads: usize, max_len: usize) -> anyhow::Result<Self> {
        let embeddings = BertEmbeddings::new(vocab_size, hidden_size, max_len, 2)?;
        
        let mut layers = vec![];
        for _ in 0..n_layers {
            let layer = BertLayer::new(hidden_size, n_heads, hidden_size * 4)?;
            layers.push(layer);
        }
        
        let pooler = Linear::new(hidden_size, hidden_size, false, None)?;
        
        Ok(Self { embeddings, layers, pooler })
    }

    /// ## Args
    /// - `input_ids`: (batch, seq_len,)
    /// - `token_type_ids`: (batch, seq_len,)
    /// - `mask`: (batch, seq_len,): bert 只需要 pad mask!
    pub fn forward(&self, input_ids: &Tensor<u32>, token_type_ids: &Tensor<u32>, mask: &Tensor<bool>) -> anyhow::Result<(Tensor<T>, Tensor<T>)> {
        let mask = mask.unsqueeze(1)?.unsqueeze(2)?;

        // (batch, seq_len) => (batch, seq_len, hidden_size)
        let mut x = self.embeddings.forward(input_ids, token_type_ids)?;
        for layer in self.layers.iter() {
            x = layer.forward(&x, &mask)?;
        }

        // 取出第一个 token [CLS] 的输出作为句向量表示 (batch, hidden_size)
        let cls = x.index((.., 0))?;
        let cls_out = self.pooler.forward(&cls)?;

        Ok((x, cls_out))
    }
}

#[cfg(test)]
mod tests {
    use lumen_core::Tensor;

    use super::Bert;

    #[test]
    fn test_new_model() {
        let _ = Bert::<f32>::new(30000, 768, 12, 12, 512).unwrap();
    }

    #[test]
    fn test_new_model_forward() {
        let model = Bert::<f32>::new(30000, 768, 12, 12, 512).unwrap();
        let sample_ids = Tensor::<u32>::rand(0, 30000, (2, 10)).unwrap();
        let sample_segments = Tensor::<u32>::zeros((2, 10)).unwrap();
        let mask = Tensor::trues((2, 10)).unwrap();
        let (sequence_output, pooled_output) = model.forward(&sample_ids, &sample_segments, &mask).unwrap();
        println!("{}", sequence_output.shape());
        println!("{}", pooled_output.shape());
    }
}