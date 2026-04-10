use std::{collections::HashMap, marker::PhantomData};
use lumen_core::{FloatDType, IndexOp, NumDType, Tensor, WithDType};
use crate::{error::{MlError, MlResult}, pipeline::{PredictFit, PredictModel}, utils};

// =========================================================================================== //
//              Knn Regression 
// =========================================================================================== //

pub struct KnnRegression<T: FloatDType> {
    pub n_neighbors: usize,
    marker: PhantomData<T>,
}

pub struct KnnRegressionModel<T: NumDType> {
    pub n_neighbors: usize,
    pub n_features: usize,
    pub x_train: Tensor<T>,
    pub y_train: Tensor<T>,
}

impl<T: FloatDType> KnnRegression<T> {
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors, marker: Default::default() }
    }   
}

impl<T: FloatDType> PredictFit for KnnRegression<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;
    type Model = KnnRegressionModel<T>;

    /// ## Args 
    /// - `x_train`: (n_samples, n_features)
    /// - `y_train`: (n_samples,)
    fn fit(&self, x: &Tensor<T>, y: &Tensor<T>) -> crate::error::MlResult<Self::Model> {
        let (n_samples, n_features) = utils::validate_xy_shapes(x, y)?;
        if n_samples < self.n_neighbors {
            thiserrorctx::bail!(MlError::Knn(format!("not enough samples! < k {}", self.n_neighbors)));
        }

        Ok(KnnRegressionModel { n_neighbors: self.n_neighbors, n_features, x_train: x.clone(), y_train: y.clone() })
    }
}

impl<T: FloatDType> PredictModel for KnnRegressionModel<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    /// ## Args
    /// - `x`: (n_test_samples, n_features)
    /// 
    /// ## Return
    /// - `prediction`: (n_test_samples,)
    fn predict(&self, x: &Tensor<T>) -> crate::error::MlResult<Tensor<T>> {
        let neighbors = find_closed_n_neighbors(
            &self.x_train, &self.y_train, x, self.n_neighbors
        )?;

        let prediction = neighbors.mean(1)?;

        Ok(prediction)
    }
}

// =========================================================================================== //
//              Knn Classifier 
// =========================================================================================== //

pub struct KnnClassifier<T: FloatDType> {
    pub n_neighbors: usize,
    markder: PhantomData<T>,
}

pub struct KnnClassifierModel<T: NumDType> {
    pub n_neighbors: usize,
    pub n_features: usize,
    pub x_train: Tensor<T>,
    pub y_train: Tensor<u32>,
}

impl<T: FloatDType> KnnClassifier<T> {
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors, markder: Default::default() }
    }   
}

impl<T: FloatDType> PredictFit for KnnClassifier<T> {
    type Input = Tensor<T>;
    type Output = Tensor<u32>;

    type Model = KnnClassifierModel<T>;
    
    /// ## Args 
    /// - `x_train`: (n_samples, n_features)
    /// - `y_train`: (n_samples,)
    fn fit(&self, x: &Tensor<T>, y: &Tensor<u32>) -> MlResult<Self::Model> {
        let (n_samples, n_features) = utils::validate_xy_shapes(x, y)?;
        if n_samples < self.n_neighbors {
            thiserrorctx::bail!(MlError::Knn(format!("not enough samples! < k {}", self.n_neighbors)));
        }

        Ok(KnnClassifierModel { n_neighbors: self.n_neighbors, n_features, x_train: x.clone(), y_train: y.clone() })
    }
}

impl<T: FloatDType> PredictModel for KnnClassifierModel<T> {
    type Input = Tensor<T>;
    type Output = Tensor<u32>;

    /// ## Args
    /// - `x`: (n_test_samples, n_features)
    /// 
    /// ## Return
    /// - `prediction`: (n_test_samples,)
    fn predict(&self, x: &Tensor<T>) -> MlResult<Tensor<u32>> {
        let (n_test_samples, n_features) = x.dims2()?;
        if self.n_features != n_features {
            lumen_core::bail!("expect n_fetures {}, not got {}", self.n_features, n_features);
        }

        let neighbor_labels = find_closed_n_neighbors(
            &self.x_train, &self.y_train, x, self.n_neighbors
        )?;

        let mut labels = Vec::with_capacity(n_test_samples);
        for n in 0..n_test_samples {
            let sample_labels = neighbor_labels.index(n)?;
            let mut counters = HashMap::new();
            
            for label in sample_labels.iter()? {
                *counters.entry(label).or_insert(0) += 1;
            }
            
            let majority_label = counters.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(label, _)| label)
                .unwrap(); 
                
            labels.push(majority_label);
        }

        Ok(Tensor::new(labels)?)
    }
}

fn find_closed_n_neighbors<T1: FloatDType, T2: WithDType>(
    x_train: &Tensor<T1>,
    y_train: &Tensor<T2>,
    x_test: &Tensor<T1>,    
    n_neighbors: usize,
) -> lumen_core::Result<Tensor<T2>> {
    let (n_test_samples, _) = x_test.dims2()?;

    // (n_test_samples, 1, n_features) - (1, n_samples, n_features) => (n_test_samples, n_samples, n_features)
    let delta_features = x_test.unsqueeze(1)?.broadcast_sub(&x_train.unsqueeze(0)?)?;
    // (n_test_samples, n_samples, n_features) => (n_test_samples, n_samples)
    let neg_distances = delta_features.sqr()?.sum(2)?.neg()?;

    // (n_test_samples, n_samples) => (n_test_samples, k) and (n_test_samples, k)
    let (_, idx) = neg_distances.topk(n_neighbors, 1)?;

    // (n_test_samples * k)
    let flat_idx = idx.flatten_all()?;
    let flat_neighbors = y_train.index_select(&flat_idx, 0)?;
    let neighbors = flat_neighbors.reshape((n_test_samples as usize, n_neighbors))?;

    Ok(neighbors)
}

#[cfg(test)]
mod tests {
    use lumen_core::Tensor;

    use crate::{datasets::train_test_split, neighbor::KnnRegression, pipeline::{PredictFit, PredictModel}};

    #[test]
    fn test_knn_geression() {
        const N_SAMPLES: usize = 100;
        let x = Tensor::<f32>::rand(0.0, 10.0, (N_SAMPLES,)).unwrap();
        let y = x.sin().unwrap() + (0.1 * Tensor::<f32>::randn(0.0, 1.0, (N_SAMPLES,)).unwrap());
        let x = x.unsqueeze(1).unwrap();

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();

        let trainer = KnnRegression::new(5);
        let model = trainer.fit(&x_train, &y_train).unwrap();

        let y_pred = model.predict(&x_test).unwrap();

        for (pred, real) in y_pred.iter().unwrap().zip(y_test.iter().unwrap()) {
            println!("pred: {pred:.2} vs real: {real:.2}")
        } 
    }
}