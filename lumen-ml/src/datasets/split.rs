use std::collections::HashSet;
use lumen_core::{IndexOp, Tensor, WithDType};

/// X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pub fn train_test_split<T1: WithDType, T2: WithDType>(
    x: &Tensor<T1>, 
    y: &Tensor<T2>,
    test_ratio: f64,
) -> lumen_core::Result<(Tensor<T1>, Tensor<T1>, Tensor<T2>, Tensor<T2>)> {
    if test_ratio < 0.0 || test_ratio >= 1.0 {
        lumen_core::bail!("invalid test_ratio {test_ratio}");
    }

    let n_samples = x.dims()[0];
    let n_samples_y = y.dims()[0];
    if n_samples != n_samples_y {
        lumen_core::bail!("x n_samples != y n_samples");
    }

    let test_count = (n_samples as f64 * test_ratio) as usize;
    let mut test_indexs = HashSet::new();
    let mut test_mask = vec![false; n_samples];
    while test_indexs.len() < test_count {
        let index = rand::random_range(0..n_samples);
        if test_indexs.insert(index) {
            test_mask[index] = true;
        }
    }
    let test_mask = Tensor::new(test_mask)?;
    let train_mask = test_mask.not()?;

    let x_train = x.index(&train_mask)?;
    let y_train = y.index(&train_mask)?;
    
    let x_test = x.index(&test_mask)?;
    let y_test = y.index(&test_mask)?;
    
    Ok((x_train, x_test, y_train, y_test))
}