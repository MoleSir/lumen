use std::collections::HashSet;
use lumen_core::{FloatDType, IndexOp, Tensor};
use rand::Rng;
use thiserrorctx::Context;
use crate::{error::MlResult, pipeline::{TransformFit, TransformModel}};

pub struct KMeans<T> {
    pub k: usize,
    pub init_policy: KMeansInitPolicy,
    pub max_iters: usize,
    pub epsilon: T
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansInitPolicy {
    Random,
}

pub struct KMeansModel<T: FloatDType> {
    pub centers: Tensor<T>,
}

impl<T: FloatDType> TransformFit for KMeans<T> {
    type Input = Tensor<T>;
    type Output = Tensor<u32>;
    type Model = KMeansModel<T>;

    /// ## Args
    /// - `x`: (n_samples, n_features)
    fn fit(&self, x: &Tensor<T>) -> MlResult<Self::Model> {
        let (centers, _) = self.do_fit(x)?;
        Ok(KMeansModel { centers })
    }

    /// ## Args
    /// - `x`: (n_samples, n_features)
    /// 
    /// ## Return
    /// - `labels`: (n_samples,)
    fn fit_transform(&self, x: &Tensor<T>) -> MlResult<Tensor<u32>> {
        let (_, labels) = self.do_fit(x)?;
        Ok(labels)
    }
}

impl<T: FloatDType> TransformModel for KMeansModel<T> {
    type Input = Tensor<T>;
    type Output = Tensor<u32>;

    /// ## Args
    /// - `x`: (n_samples, n_features)
    /// 
    /// ## Return
    /// - `labels`: (n_samples,)
    fn transform(&self, x: &Tensor<T>) -> MlResult<Tensor<u32>> {
        find_closest_center(x, &self.centers)
    }
}

impl<T: FloatDType> KMeans<T> {
    /// fit a kmeans
    /// 
    /// ## Args
    /// - `x`: (n_samples, n_features)
    pub fn do_fit(&self, x: &Tensor<T>) -> MlResult<(Tensor<T>, Tensor<u32>)> {
        let mut centers = self.init_centers(x).context("init centers")?; // (k, n_features)
        let mut final_labels = None;
        if self.max_iters == 0 {
            lumen_core::bail!("no iter!");
        }

        for _ in 0..self.max_iters {
            // get closest centers: (n_samples,)
            let labels = find_closest_center(x, &centers)?;

            // (n_samples, k) => (n_samples,) 
            final_labels = Some(labels.clone());

            // 2. update centers
            let mut new_centers = vec![];
            // for each center
            for i in 0..self.k {
                // who choose center 'i'? (n_samples, )
                let mask = labels.eq(i as u32)?;
                if mask.true_count()? == 0 {
                    // no any samples in this cluster!
                    new_centers.push(centers.index(i)?);
                    continue;
                }

                // (n_samples, n_features) => (m, n_features)
                let cluster_x = x.index(&mask)?;  
                // (m, n_features) => (n_features,)
                let new_center = cluster_x.mean(0)?;
                new_centers.push(new_center);
            }
            // (k, n_features)
            let new_centers = Tensor::stack(&new_centers, 0)?;
        
            // 3. fit over?
            let delta = (&new_centers - &centers).abs()?.mean_all()?.to_scalar()?;
            if delta < self.epsilon {
                break;
            }

            centers = new_centers;
        }

        Ok((centers, final_labels.expect("must not None")))
    }

    fn init_centers(&self, x: &Tensor<T>) -> MlResult<Tensor<T>> {
        match self.init_policy {
            KMeansInitPolicy::Random => {
                let (n_samples, _) = x.dims2()?;
                let mut indexs = HashSet::new();
                let mut rng = rand::rng();

                let mut centers = vec![];
                loop {
                    if indexs.len() == self.k {
                        break;
                    }

                    let index = rng.random_range(0..n_samples);
                    if indexs.insert(index) {
                        // a new index
                        centers.push(x.index(index)?);
                    } else {
                        continue;
                    }
                }

                let centers = Tensor::stack(&centers, 0)?;
                Ok(centers)
            }
        }
    }
}

/// # Args
/// - `x`: (n_samples, n_features)
/// - `centers`: (k, n_features)
/// 
/// ## Return
/// - `labels`: (n_samples,)
fn find_closest_center<T: FloatDType>(x: &Tensor<T>, centers: &Tensor<T>) -> MlResult<Tensor<u32>> {
    // 1. select closed center for each samples
    // (n_samples, n_features) => (n_samples, 1)
    let x_norm = x.sqr()?.sum_keepdim(1)?;
    // (k, n_features) => (1, k)
    let c_norm = centers.sqr()?.sum(1)?.unsqueeze(0)?;

    // (n_samples, n_features) @ (n_features, k) => (n_samples, k)
    let xc = x.matmul(&centers.transpose(0, 1)?)?;

    // (n_samples, 1) + (1, k) => (n_samples, k) - 2 * xc
    let distances = x_norm
        .broadcast_add(&c_norm)?
        .broadcast_sub(&(xc * T::from_f64(2.0)))?;

    let labels = distances.argmin(1)?;

    Ok(labels)
}