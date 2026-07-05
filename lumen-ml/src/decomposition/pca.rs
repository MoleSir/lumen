use std::marker::PhantomData;

use lumen_core::{FloatDType, Tensor};
use thiserrorctx::Context;
use crate::{error::{MlError, MlResult}, TransformFit, TransformModel};

pub struct Pca<T> {
    pub n_components: usize,
    pub markder: PhantomData<T>,
}

pub struct PcaModel<T: FloatDType> {
    pub mean: Tensor<T>,
    pub components: Tensor<T>,
    pub n_components: usize,
}

impl<T: FloatDType> TransformFit for Pca<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;
    type Model = PcaModel<T>;

    /// ## Args
    /// - `x`: (n_samples, n_features)
    fn fit(&self, x: &Tensor<T>) -> MlResult<Self::Model> {
        let (n_samples, _) = x.dims2()
            .map_err(MlError::Core)
            .context("expect `x`: (n_samples, n_features)")?;

        // 1. 计算均值并中心化
        let mean = x.mean_keepdim(0)?; // (1, n_features)
        let x_centered = x.broadcast_sub(&mean)?; // (n_samples, n_features)

        // 2. 计算协方差并分解
        let cov = x_centered.transpose_last()?.matmul(&x_centered)?; // (n_features, n_features)
        let scale = T::from_f64(1.0 / (n_samples as f64 - 1.0));
        let cov = cov.mul_(scale)?;

        let eig = lumen_linalg::jacobi_eig(&cov, T::epsilon())?;
        
        // 3. 选取前 k 个特征向量
        let eig_values = Tensor::new(eig.eig_values)?; // (n_features,)
        let (_, idx) = eig_values.topk(self.n_components, 0)?; // (n_components)
        let eig_vectors = eig.eig_vectors.index_select(&idx, 1)?; // (n_features, n_components)

        // 4. 将状态封装进 Model
        Ok(PcaModel {
            mean,
            components: eig_vectors,
            n_components: self.n_components,
        })
    }
}

impl<T: FloatDType> TransformModel for PcaModel<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;
    
    /// ## Args
    /// - `x`: (n_samples, n_features)
    /// 
    /// ## Returns
    /// - `x_pca`: (n_samples, n_components)
    fn transform(&self, x: &Tensor<T>) -> MlResult<Tensor<T>> {
        let x_centered = x.broadcast_sub(&self.mean)?;
        // 将原始 X 投影到 eigvecs 上
        /*
            x_norm: (n_samples, n_features), n_samples 个样本
            eig_values: (n_features, n_components)，每列是一个新方向

            x_norm @ eig_values => 每个样本 x_norm，和 eig_values 的每列分布点乘
            （将原来的样本投影到某个 eig_value）
        */
        // (n_samples, n_features) @ (n_features, n_components) => (n_samples, n_components)
        let x_pca = x_centered.matmul(&self.components)?;
        Ok(x_pca)
    }
}

impl<T> Pca<T> {
    pub fn new(n_components: usize) -> Self {
        Self { n_components, markder: Default::default() }
    }
}

impl<T: FloatDType> PcaModel<T> {
    pub fn inverse_transform(&self, x_pca: &Tensor<T>) -> MlResult<Tensor<T>> {
        let x_res = x_pca.matmul(&self.components.transpose_last()?)?;
        x_res.broadcast_add(&self.mean).map_err(Into::into)
    }
}
