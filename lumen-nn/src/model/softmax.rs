use lumen_core::Tensor;
use anyhow::Result;

pub struct SoftMax();

impl SoftMax {
    pub fn new() -> Self {
        Self()
    }

    /// SoftMax forward!
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;;
    /// use lumen_nn::model::SoftMax;
    /// let softmax = SoftMax::new();
    /// let x = Tensor::build([0.5, 0.2, 0.1, 0.4, 0.12, 0.132, -1.2, -4., 4.], [3, 3]).unwrap();
    /// //let y = softmax.forward(&x).unwrap();
    /// //assert!(y.allclose(&Tensor::build([0.41474187266806956, 0.30724833615216285, 0.27800979117976765, 0.3967165323684531, 0.2998319051307709, 0.303451562500776, 0.005484469158895358, 0.00033351091301850316, 0.9941820199280862], [3, 3]).unwrap()));
    /// ```
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        todo!()
        // if x.dim_size() > 2 {
        //     return Err(TensorError::DifferentShape).with_context(|| "check input size");
        // }

        // let (x, batch_size, reshape) = if x.dim_size() == 1 {
        //     (x.view([1, x.element_size()])?, 1, true)
        // } else {
        //     let shape = x.shape();
        //     let batch_size: usize = shape[0];
        //     (x.clone(), batch_size, false)
        // };

        // let y = Tensor::zeros_like(&x);

        // for i in 0..batch_size {
        //     let batch_x = x.slice(&[rng!(i)]).unwrap();
        //     let dst = y.slice(&[rng!(i)]).unwrap();

        //     let sum = batch_x.iter().map(|&v| v.exp()).sum::<f64>();
        //     for (d, &s) in dst.iter_mut().zip(batch_x.iter()) {
        //         *d = s.exp() / sum;
        //     }
        // }

        // Ok(if reshape {
        //     y.view([y.element_size()]).unwrap()
        // } else {
        //     y
        // })
    }
}