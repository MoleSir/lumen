use std::collections::HashMap;

use lumen_core::{FloatDType, IndexOp, NumDType, Tensor, WithDType};

// =========================================================================================== //
//              Knn Regression 
// =========================================================================================== //

pub struct KnnRegressionTrainer {
    pub n_neighbors: usize,
}

impl KnnRegressionTrainer {
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors }
    }   
}

impl KnnRegressionTrainer {
    /// ## Args 
    /// - `x_train`: (n_samples, n_features)
    /// - `y_train`: (n_samples,)
    pub fn fit<T: NumDType>(&self, x: &Tensor<T>, y: &Tensor<T>) -> lumen_core::Result<KnnRegression<T>> {
        let (n_samples, n_features) = x.dims2()?;
        let n_samples_y = y.dims1()?; 
        if n_samples != n_samples_y {
            lumen_core::bail!("x samples {} != y samples {}", n_samples, n_samples_y);
        }
        if n_samples < self.n_neighbors {
            lumen_core::bail!("not enough samples! < k {}", self.n_neighbors);
        }

        Ok(KnnRegression { n_neighbors: self.n_neighbors, n_features, x_train: x.clone(), y_train: y.clone() })
    }
}

pub struct KnnRegression<T: NumDType> {
    pub n_neighbors: usize,
    pub n_features: usize,
    pub x_train: Tensor<T>,
    pub y_train: Tensor<T>,
}

impl<T: FloatDType> KnnRegression<T> {
    /// ## Args
    /// - `x`: (n_test_samples, n_features)
    /// 
    /// ## Return
    /// - `prediction`: (n_test_samples,)
    pub fn predict(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        let (_, n_features) = x.dims2()?;
        if self.n_features != n_features {
            lumen_core::bail!("expect n_fetures {}, not got {}", self.n_features, n_features);
        }

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

pub struct KnnClassifierTrainer {
    pub n_neighbors: usize,
}

impl KnnClassifierTrainer {
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors }
    }   
}

impl KnnClassifierTrainer {
    /// ## Args 
    /// - `x_train`: (n_samples, n_features)
    /// - `y_train`: (n_samples,)
    pub fn fit<T: NumDType>(&self, x: &Tensor<T>, y: &Tensor<u32>) -> lumen_core::Result<KnnClassifier<T>> {
        let (n_samples, n_features) = x.dims2()?;
        let n_samples_y = y.dims1()?; 
        if n_samples != n_samples_y {
            lumen_core::bail!("x samples {} != y samples {}", n_samples, n_samples_y);
        }
        if n_samples < self.n_neighbors {
            lumen_core::bail!("not enough samples! < k {}", self.n_neighbors);
        }

        Ok(KnnClassifier { n_neighbors: self.n_neighbors, n_features, x_train: x.clone(), y_train: y.clone() })
    }
}

pub struct KnnClassifier<T: NumDType> {
    pub n_neighbors: usize,
    pub n_features: usize,
    pub x_train: Tensor<T>,
    pub y_train: Tensor<u32>,
}

impl<T: FloatDType> KnnClassifier<T> {
    /// ## Args
    /// - `x`: (n_test_samples, n_features)
    /// 
    /// ## Return
    /// - `prediction`: (n_test_samples,)
    pub fn predict(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<u32>> {
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

    use crate::{model_selection::train_test_split, neighbor::KnnRegressionTrainer};

    #[test]
    fn test_knn_geression() {
        const N_SAMPLES: usize = 100;
        let x = Tensor::<f32>::rand(0.0, 10.0, (N_SAMPLES,)).unwrap();
        let y = x.sin().unwrap() + (0.1 * Tensor::<f32>::randn(0.0, 1.0, (N_SAMPLES,)).unwrap());
        let x = x.unsqueeze(1).unwrap();

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();

        let trainer = KnnRegressionTrainer::new(5);
        let model = trainer.fit(&x_train, &y_train).unwrap();

        let y_pred = model.predict(&x_test).unwrap();

        for (pred, real) in y_pred.iter().unwrap().zip(y_test.iter().unwrap()) {
            println!("pred: {pred:.2} vs real: {real:.2}")
        } 
    }
}