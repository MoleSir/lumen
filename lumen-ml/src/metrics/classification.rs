use lumen_core::{IndexOp, Tensor, WithDType};

/// ## Args
/// - `y_ture`: (n_samples,)
/// - `y_pred`: (n_samples,)
pub fn confusion_matrix(y_true: &Tensor<u32>, y_pred: &Tensor<u32>) -> lumen_core::Result<Tensor<u32>> {
    let true_count = y_true.dims1()?;
    let pred_count = y_pred.dims1()?;
    if true_count != pred_count {
        lumen_core::bail!("true count != pred count");
    }
    if true_count == 0 {
        lumen_core::bail!("empty samples");
    }
    
    // TODO: 统计分类数量
    let n_class1 = y_true.max_all()?.to_scalar()? as usize + 1;
    let n_class2 = y_pred.max_all()?.to_scalar()? as usize + 1;
    let n_class = n_class1.max(n_class2);
    if n_class <= 1 {
        lumen_core::bail!("only one class");
    }

    // 收集数据方便统计
    let pred_true_pairs: Vec<(u32, u32)> = y_pred.iter()?.zip(y_true.iter()?).collect();

    // N x N 
    let mut matrix = vec![0; n_class * n_class];
    for row in 0..n_class {
        for col in 0..n_class {
            // row: 预测值为 row 的样本
            // col: 真实值为 col 的样本
            let count = pred_true_pairs.iter()
                .filter(|(pred, true_)| {
                    *pred == row as u32 && *true_ == col as u32
                })
                .count() as u32;
            matrix[row * n_class + col] = count;
        }
    }

    Ok(Tensor::from_vec(matrix, (n_class, n_class))?)
}

pub fn accuracy_score<T: WithDType>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> lumen_core::Result<f64> {
    if y_true.dims() != y_pred.dims() {
        lumen_core::bail!("y_true shape {:?} != y_pred shaoe {:?}", y_true.dims(), y_pred.dims());
    }

    let n_samples = y_true.element_count();
    if n_samples == 0 {
        lumen_core::bail!("no samples!");
    }

    let n_correct = y_true.iter()?.zip(y_pred.iter()?)
        .filter(|(t, p)| t == p)
        .count();

    Ok( n_correct as f64 / n_samples as f64 )
}

pub fn precision_score(y_true: &Tensor<u32>, y_pred: &Tensor<u32>) -> lumen_core::Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let (n_class, _) = cm.dims2().expect("confusion matrix must matrix!");

    if n_class == 2 {
        let tp = cm.index((0, 0))?.to_scalar()? as f64;
        let fp = cm.index((0, 1))?.to_scalar()? as f64;
        Ok(tp / (tp + fp))
    } else {
        // 为每个特征计算
        let mut scores = 0.0;
        for class in 0..n_class {
            let tp = cm.index((class, class))?.to_scalar()? as f64;
            let all_samples = cm.index(class)?.sum_all()?.to_scalar()? as f64;
            scores += tp / all_samples;
        }

        Ok( scores / n_class as f64 )
    }
}

pub fn recall_score(y_true: &Tensor<u32>, y_pred: &Tensor<u32>) -> lumen_core::Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let (n_class, _) = cm.dims2().expect("confusion matrix must matrix!");

    if n_class == 2 {
        let tp = cm.index((0, 0))?.to_scalar()? as f64;
        let fn_ = cm.index((1, 0))?.to_scalar()? as f64;
        Ok(tp / (tp + fn_))
    } else {
        let mut scores = 0.0;
        for class in 0..n_class {
            let tp = cm.index((class, class))?.to_scalar()? as f64;
            let all_samples = cm.index((.., class))?.sum_all()?.to_scalar()? as f64;
            scores += tp / all_samples;
        }

        Ok( scores / n_class as f64 )
    }
}

pub fn f1_score(y_true: &Tensor<u32>, y_pred: &Tensor<u32>) -> lumen_core::Result<f64> {
    let precision = precision_score(y_true, y_pred)?;
    let recall = recall_score(y_true, y_pred)?;
    Ok( 2.0 * (precision * recall) / (precision + recall) )
}

#[cfg(test)]
mod tests {
    use lumen_core::Tensor;
    use crate::metrics::{confusion_matrix, f1_score, recall_score};
    use super::precision_score;

    #[test]
    fn test_confusion_binary_matrix() {
        let y_true = Tensor::new(vec![1u32, 0, 1, 1, 0, 0, 1, 0, 0, 1]).unwrap();
        let y_pred = Tensor::new(vec![1, 0, 0, 1, 0, 0, 1, 1, 0, 1]).unwrap();
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        println!("{}", cm);
    }

    #[test]
    fn test_binary_scores() {
        let y_true = Tensor::new(vec![1u32, 0, 1, 1, 0, 0, 1, 0, 0, 1]).unwrap();
        let y_pred = Tensor::new(vec![1, 0, 0, 1, 0, 0, 1, 1, 0, 1]).unwrap();
        let pr = precision_score(&y_true, &y_pred).unwrap();
        assert_eq!(pr, 0.8);
        let recall = recall_score(&y_true, &y_pred).unwrap();
        assert_eq!(recall, 0.8);
        let f1 = f1_score(&y_true, &y_pred).unwrap();
        assert_eq!(f1, 0.8000000000000002);
    }
} 