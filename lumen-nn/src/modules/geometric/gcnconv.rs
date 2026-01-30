use lumen_core::{FloatDType, IndexOp, IntDType, IntTensor, Tensor};
use lumen_macros::Module;
use thiserrorctx::Context;
use crate::{init::Init, Linear, ModuleInit, NnCtxError, NnResult};

#[derive(Module)]
pub struct GCNConv<T: FloatDType> {
    linear: Linear<T>,

    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
} 

#[derive(derive_new::new)]
pub struct GCNConvConfig {
    pub in_features: usize,
    pub out_features: usize,
}

impl<T: FloatDType> ModuleInit<T> for GCNConv<T> {
    type Error = NnCtxError;
    type Config = GCNConvConfig;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let linear = Linear::new(config.in_features, config.out_features, true, init)
            .context("init linear")?;

        Ok(Self {
            linear,
            in_features: config.in_features,
            out_features: config.out_features
        })
    }
}

impl<T: FloatDType> GCNConv<T> {
    #[inline]
    pub fn new(in_features: usize, out_features: usize, init: Option<Init<T>>) -> NnResult<Self> {
        Self::init(&GCNConvConfig::new(in_features, out_features), init)
    }

    /// ## Arguments
    /// 
    /// * `x` - (N, in_features) nodes feature
    /// * `edge_index` - (2, edge_size) 
    pub fn forward(&self, x: &Tensor<T>, edge_index: impl Into<IntTensor>) -> NnResult<Tensor<T>> {
        let edge_index = edge_index.into();
        match edge_index {
            IntTensor::U8(edge_index) => self.forward_impl(x, edge_index),
            IntTensor::U32(edge_index) => self.forward_impl(x, edge_index),
            IntTensor::I32(edge_index) => self.forward_impl(x, edge_index),
        }
    }

    fn forward_impl<I: IntDType>(&self, x: &Tensor<T>, edge_index: Tensor<I>) -> NnResult<Tensor<T>> {
        let (n, _in_features) = x.dims2()?;

        // 1. Self-loop
        // (n,) => (1, n) => (2, n)
        let self_loop_index = Tensor::arange(I::zero(), I::from_usize(n))?
            .unsqueeze(0)?
            .repeat_dim(0, 2)?;

        // cat (2, edge_size) & (2, n) => (2, edge_size + n)
        let edge_index_with_loop = Tensor::cat(&[edge_index, self_loop_index], 1)?;

        // 2. Normalization
        // val = 1 / sqrt(d_i * d_j)
        let starts = edge_index_with_loop.index(0)?; // (edge_size + n)
        let ends = edge_index_with_loop.index(1)?; // (edge_size + n)
        
        // 3. Degree
        let mut degs = vec![T::zero(); n];
        for end in ends.to_vec()? {
            degs[end.to_usize()] += T::one();
        }
        let deg_inv_sqrt = Tensor::new(degs)?.pow(T::from_f64(-0.5))?; // (n,)

        // 4. Edge weight
        let edge_weights = 
            deg_inv_sqrt.index_select(I::to_inttensor(starts.clone()), 0)? 
          * deg_inv_sqrt.index_select(I::to_inttensor(ends.clone()), 0)?;
        let edge_weights = edge_weights.to_vec()?;

        // 5. Adj Matrix
        let mut adj = vec![T::zero(); n * n];
        for (i, (start, end)) in starts.to_vec()?.into_iter().zip(ends.to_vec()?.into_iter()).enumerate() {
            adj[start.to_usize() * n + end.to_usize()] = edge_weights[i];
        }
        let adj = Tensor::from_vec(adj, (n, n))?;

        // 6. Transform
        // (n, in_f) -> (n, out_f)
        let x_transform = self.linear.forward(x)?; 
        // (n, n) @ (n, out_f) => (n, out_f)
        let out = adj.matmul(&x_transform)?;
                
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::Tensor;

    #[test]
    fn test_gcn_conv_logic() -> NnResult<()> {
        let n = 3;
        let in_features = 1;
        let out_features = 1;

        let x = Tensor::from_vec(vec![1.0f32, 2.0, 4.0], (n, in_features))?;
        let edge_data = vec![
            0u32, 1, // Row 0: Starts
            1,    2  // Row 1: Ends
        ];
        let edge_index = Tensor::from_vec(edge_data, (2, 2))?;

        let gcn = GCNConv::<f32>::new(in_features, out_features, Some(Init::ones()))?;
        let output = gcn.forward(&x, edge_index)?;

        let expected_data = vec![2.41421356, 3.0, 2.0];
        let expected = Tensor::from_vec(expected_data, (3, 1))?;

        println!("Computed: {:?}", output.to_vec());
        println!("Expected: {:?}", expected.to_vec());
        
        output.allclose(&expected, 1e-5, 1e-8).unwrap();

        Ok(())
    }
}