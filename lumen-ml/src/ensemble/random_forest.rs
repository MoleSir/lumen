use std::collections::HashMap;

use lumen_core::{FloatDType, IndexOp, Tensor};
use rand::Rng;

use crate::tree::{DecisionTreeClassifier, DecisionTreeClassifierTrainer};


pub struct RandomForestClassifierTrainer {
    pub n_estimators: usize,
    pub max_depth: usize, 
}

impl RandomForestClassifierTrainer {
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self {
            n_estimators,
            max_depth,
        }
    }

    pub fn fit<T: FloatDType>(&self, x: &Tensor<T>, y: &Tensor<u32>) -> lumen_core::Result<RandomForestClassifier<T>> {
        let mut trees = Vec::with_capacity(self.n_estimators);
        let (n_samples, _) = x.dims2()?;

        let tree_trainer = DecisionTreeClassifierTrainer::new(self.max_depth);
        for _ in 0..self.n_estimators {
            let (x_boot, y_boot) = Self::bootstrap_sample(x, y, n_samples)?;
            let tree = tree_trainer.fit(&x_boot, &y_boot)?;
            trees.push(tree);
        }

        Ok(RandomForestClassifier { trees })
    }

    fn bootstrap_sample<T: FloatDType>(x: &Tensor<T>, y: &Tensor<u32>, n_samples: usize) -> lumen_core::Result<(Tensor<T>, Tensor<u32>)> {        
        let mut x_boots = Vec::with_capacity(n_samples);
        let mut y_boot = Vec::with_capacity(n_samples);

        let mut rng = rand::rng();
        for _ in 0..n_samples {
            let idx = rng.random_range(0..n_samples);
            let x_row = x.index(idx)?;
            let y_row = y.index(idx)?;
            x_boots.push(x_row);            
            y_boot.push(y_row.to_scalar()?);
        }

        let x_boot = Tensor::stack(&x_boots, 0)?;
        let y_boot = Tensor::new(y_boot)?;

        Ok((x_boot, y_boot))
    } 
}

pub struct RandomForestClassifier<T> {
    pub trees: Vec<DecisionTreeClassifier<T>>,
}

impl<T: FloatDType> RandomForestClassifier<T> {
    pub fn predict(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<u32>> {
        let (n_samples, _) = x.dims2()?;
        let mut all_predictions = Vec::with_capacity(self.trees.len());

        for tree in &self.trees {
            let preds = tree.predict(x)?; // (n_samples)
            all_predictions.push(preds.to_vec()?);
        }

        let mut final_preds = Vec::with_capacity(n_samples);
        // for each samples, votes a best
        for i in 0..n_samples {
            let mut counter = HashMap::new();
            for tree_preds in &all_predictions {
                let label = tree_preds[i];
                *counter.entry(label).or_insert(0) += 1;
            }

            let majority_label = *counter.iter().max_by_key(|entry| entry.1).unwrap().0;
            final_preds.push(majority_label);
        }

        Ok(Tensor::new(final_preds)?)
    }
}