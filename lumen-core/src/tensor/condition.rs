use std::sync::Arc;

use crate::{AutogradMetaT, Error, Layout, Result, Storage, StorageArc, TensorOrScalar, WithDType};
use super::{Tensor, TensorId, TensorImpl};

impl Tensor<bool> {
    pub fn if_else<T: WithDType>(&self, true_val: impl Into<TensorOrScalar<T>>, false_val: impl Into<TensorOrScalar<T>>) -> Result<Tensor<T>> {
        let true_val = true_val.into();
        let false_val = false_val.into();

        if let TensorOrScalar::Tensor(tensor) = &true_val && tensor.shape() != self.shape() {
            Err(Error::ShapeMismatchSelect { mask: self.shape().clone(), who: "true_val", })?
        }
        if let TensorOrScalar::Tensor(tensor) = &false_val && tensor.shape() != self.shape() {
            Err(Error::ShapeMismatchSelect { mask: self.shape().clone(), who: "false_val", })?
        }

        let (mut new_storage, tv) = match &true_val {
            TensorOrScalar::Tensor(tensor) => (tensor.storage_read().copy(self.layout()), Some(tensor)),
            TensorOrScalar::Scalar(v) => (Storage::full(*v, self.shape()), None),
        };
        let layout = Layout::contiguous(self.shape());

        let fv = match &false_val {
            TensorOrScalar::Tensor(false_val) => {
                for ((result_index, condition), fv) in layout.storage_indices().zip(self.iter()).zip(false_val.iter()) {
                    if !condition {
                        new_storage.set_unchecked(result_index, fv);
                    }
                }
                Some(false_val)
            }
            TensorOrScalar::Scalar(fv) => {
                for (result_index, condition) in layout.storage_indices().zip(self.iter()) {
                    if !condition {
                        new_storage.set_unchecked(result_index, *fv);
                    }
                }
                None
            }
        };

        let meta = T::AutogradMeta::on_ifelse_op(self, tv, fv);
        
        let result = TensorImpl {
            id: TensorId::new(),
            storage: StorageArc::new(new_storage),
            layout,
            meta
        };

        Ok(Tensor(Arc::new(result)))
    }
}

impl<T: WithDType> Tensor<T> {
    pub fn masked_fill(&self, mask: &Tensor<bool>, value: impl Into<TensorOrScalar<T>>) -> Result<Tensor<T>> {
        let mask = mask.not();
        mask.if_else(self, value)
    }
}

#[cfg(test)]
mod test {
    use crate::Tensor;

    #[test]
    fn test_if_else_scalar_values() {
        let mask = Tensor::new(&[true, false, true, false]).unwrap();
        let result = Tensor::if_else(&mask, 1, 0).unwrap();
        assert_eq!(result.to_vec(), [1, 0, 1, 0]);
    }
    
    #[test]
    fn test_if_else_array_values() {
        let mask = Tensor::new(&[true, false, true, false]).unwrap();
        
        let true_vals = Tensor::new(&[10, 20, 30, 40]).unwrap();
        let false_vals = Tensor::new(&[100, 200, 300, 400]).unwrap();
        
        let result = mask.if_else(&true_vals, &false_vals).unwrap();
        assert_eq!(result.to_vec(), [10, 200, 30, 400]);
    }
    
    #[test]
    fn test_if_else_mixed_values() {
        let mask = Tensor::new(&[true, false, true, false]).unwrap();
        
        let true_vals = 5;  // 标量
        let false_vals = Tensor::new(&[100, 200, 300, 400]).unwrap();
        
        let result = Tensor::if_else(&mask, true_vals, &false_vals).unwrap();
        assert_eq!(result.to_vec(), [5, 200, 5, 400]);
    }
    
    #[test]
    fn test_if_else_shape_mismatch() {
        let mask = Tensor::new(&[true, false, true]).unwrap();
        let true_vals = Tensor::new(&[1, 2, 3, 4]).unwrap();
        let false_vals = 0;
        
        let result = Tensor::if_else(&mask, &true_vals, false_vals);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_if_else_all_true_or_all_false() {
        let mask = Tensor::new(&[true, true, true]).unwrap();
        let result = Tensor::if_else(&mask, 1, 0).unwrap();
        assert_eq!(result.to_vec(), [1, 1, 1]);
    
        let mask = Tensor::new(&[false, false, false]).unwrap();
        let result = Tensor::if_else(&mask, 1, 0).unwrap();
        assert_eq!(result.to_vec(), [0, 0, 0]);
    }

    #[test]
    fn test_if_else_2d_array_values() {
        let mask = Tensor::new(&[[true, false, true], [false, true, false]]).unwrap();
        let true_vals = Tensor::new(&[[10, 20, 30], [40, 50, 60]]).unwrap();
        let false_vals = Tensor::new(&[[100, 200, 300], [400, 500, 600]]).unwrap();
    
        let result = Tensor::if_else(&mask, &true_vals, &false_vals).unwrap();
        assert_eq!(result.to_vec(), [10, 200, 30, 400, 50, 600]);
    }
    
    #[test]
    fn test_if_else_3d_mixed_values() {
        let mask = Tensor::new(&[
            [[true, false], [false, true]],
            [[true, true], [false, false]]
        ]).unwrap();
        let true_val = 1;  // 标量
        let false_vals = Tensor::new(&[
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]]
        ]).unwrap();
    
        let result = Tensor::if_else(&mask, true_val, &false_vals).unwrap();
        assert_eq!(result.to_vec(), [1, 20, 30, 1, 1, 1, 70, 80]);
    }
    
    #[test]
    fn test_if_else() {
        let scores = Tensor::new(&[
            [45., 12., 34., 90.],
            [31., 19., 84., 60.],
            [55., 34., 44., 82.],
            [85., 89., 54., 67.],
        ]).unwrap();

        // scores > 60 & scores < 85
        let mask = scores.ge(60.).unwrap().and(&scores.le(85.).unwrap()).unwrap();

        let if_elseed_scores = Tensor::if_else(&mask, &scores, -1.).unwrap();
        println!("{}", if_elseed_scores);
    }
}