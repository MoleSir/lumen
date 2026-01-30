use crate::{AutogradMetaT, DTypeConvert, Error, Result, TensorOrScalar, WithDType};
use super::Tensor;

impl<T: WithDType> Tensor<T> {
    pub fn contiguous(&self) -> crate::Result<Tensor<T>> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            self.copy()
        }
    }

    pub fn copy(&self) -> crate::Result<Self> {
        let storage = self.storage_read()?.copy(self.layout());
        let meta = T::AutogradMeta::on_copy_op(self);
        Ok(Self::from_storage(storage, self.shape(), meta))
    }

    pub fn copy_from(&self, source: &Self) -> Result<()> {
        if self.shape() != source.shape() {
            Err(Error::ShapeMismatchCopyFrom { dst: self.shape().clone(), src: source.shape().clone() })?
        }

        let mut storage = self.storage_write()?;
        for (self_storage_index, src_value) in self.layout().storage_indices().zip(source.iter()?) {
            storage.set_unchecked(self_storage_index, src_value);
        }

        Ok(())
    }

    // no grad record
    pub fn assign(&self, source: impl Into<TensorOrScalar<T>>) -> Result<()> {
        match source.into() {
            TensorOrScalar::Scalar(src) => {
                let mut storage = self.storage_write()?;
                for storage_index in self.layout().storage_indices() {
                    storage.set_unchecked(storage_index, src);
                }
                Ok(())
            }
            TensorOrScalar::Tensor(src) => {
                if src.shape() != self.shape() {
                    Err(Error::ShapeMismatchCopyFrom { dst: self.shape().clone(), src: src.shape().clone() })?
                }
        
                let mut storage = self.storage_write()?;
                for (self_storage_index, src_value) in self.layout().storage_indices().zip(src.iter()?) {
                    storage.set_unchecked(self_storage_index, src_value);
                }
        
                Ok(())
            }
        }
    }
}


impl<From: WithDType> Tensor<From> {
    pub fn cast<To: WithDType>(&self) -> crate::Result<Tensor<To>> 
    where
        From: DTypeConvert<To>,
    {
        // if TypeId::of::<From>() == TypeId::of::<To>() {
        //     let self_any = self as &dyn Any;            
        //     if let Some(same_tensor) = self_any.downcast_ref::<Tensor<To>>() {
        //         return same_tensor.clone();
        //     }
        // }
        let storage = self.storage_read()?.copy_map(self.layout(), From::convert);
        Ok(Tensor::<To>::from_storage(storage, self.shape(), Default::default()))
    }
}

#[cfg(test)]
mod tests {
    use crate::IndexOp;

    use super::*;

    #[test]
    fn test_assign() {
        let a = Tensor::new(&[[1, 2, 3], [3, 4, 5], [4, 5, 6]]).unwrap();
        a.index(0).unwrap().assign(100).unwrap();
        println!("{}", a);
        a.index((1, 1)).unwrap().assign(200).unwrap();
        println!("{}", a);
        a.index((1.., 1..)).unwrap().assign(999).unwrap();
        println!("{}", a);
    }

    #[test]
    fn test_copy_1d() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let b = a.copy().unwrap();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.to_vec().unwrap(), b.to_vec().unwrap());
    }

    #[test]
    fn test_copy_2d() {
        let a = Tensor::new(&[[1, 2], [3, 4]]).unwrap();
        let b = a.copy().unwrap();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.to_vec().unwrap(), b.to_vec().unwrap());
    }

    #[test]
    fn test_cast_i32_to_f32() {
        let a = Tensor::new(&[1i32, 2, 3]).unwrap();
        let b: Tensor<f32> = a.cast().unwrap();
        assert_eq!(b.shape(), a.shape());
        let expected = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
        assert!(b.allclose(&expected, 1e-6, 1e-6).unwrap());
    }

    #[test]
    fn test_cast_f64_to_i32() {
        let a = Tensor::new(&[1.5f64, 2.7, 3.0]).unwrap();
        let b: Tensor<i32> = a.cast().unwrap();
        assert_eq!(b.shape(), a.shape());
        let expected = Tensor::new(&[1i32, 2, 3]).unwrap(); // cast truncates
        assert_eq!(b.to_vec().unwrap(), expected.to_vec().unwrap());
    }

    #[test]
    fn test_cast_2d() {
        let a = Tensor::new(&[[1i32, 2], [3, 4]]).unwrap();
        let b: Tensor<f64> = a.cast().unwrap();
        assert_eq!(b.shape(), a.shape());
        let expected = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
        assert!(b.allclose(&expected, 1e-12, 1e-12).unwrap());
    }

    #[test]
    fn test_copy_vs_cast() {
        let a = Tensor::new(&[1i32, 2, 3]).unwrap();
        let b = a.copy().unwrap();
        let c: Tensor<f32> = a.cast().unwrap();

        assert_eq!(b.to_vec().unwrap(), a.to_vec().unwrap());

        let expected = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6).unwrap());
    }
}
