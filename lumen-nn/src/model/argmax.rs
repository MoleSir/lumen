use lumen_core::Tensor;
use anyhow::Result;

pub struct ArgMax();

impl ArgMax {
    pub fn new() -> Self {
        Self()
    }

    /// SoftMax forward!
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;;
    /// use lumen_nn::model::ArgMax;
    /// let model = ArgMax::new();
    /// let x = Tensor::build([0.5, 0.2, 0.1, 0.4, 0.12, 0.132, -1.2, -4., 4.], [3, 3]).unwrap();
    /// //let y = model.forward(&x).unwrap();
    /// //assert!(y.allclose(&Tensor::build([1., 0., 0., 1., 0., 0., 0., 0., 1.], [3, 3]).unwrap()));
    /// ```
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        todo!()
    }   
}