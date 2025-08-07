use lumen_core::Tensor;
use anyhow::Result;

pub struct LogSoftMax();

impl LogSoftMax {
    pub fn new() -> Self {
        Self()
    }

    /// LogSoftMax forward!
    /// 
    /// # Exmaple
    /// 
    /// ```rust
    /// use lumen_core::*;;
    /// use lumen_nn::model::LogSoftMax;
    /// let softmax = LogSoftMax::new();
    /// //let x = Tensor::build([2.0, 1.0, 0.1], [1, 3]).unwrap();
    /// //let y = softmax.forward(&x).unwrap();
    /// //assert!(y.allclose(&Tensor::build([-0.4170300162778333, -1.4170300162778333, -2.3170300162778332], [1, 3]).unwrap()));
    /// ```
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        // if x.dim_size() > 2 {
        //     return Err(TensorError::DifferentShape);
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
        //     let batch_x = x.slice(rngs!(i)).unwrap();
        //     let dst = y.slice(rngs!(i)).unwrap();

        //     let sum = batch_x.iter().map(|&v| v.exp()).sum::<f64>();
        //     for (d, &s) in dst.iter_mut().zip(batch_x.iter()) {
        //         *d = s - sum.ln();
        //     }
        // }

        // Ok(if reshape {
        //     y.view([y.element_size()]).unwrap()
        // } else {
        //     y
        // })
        todo!()
    }
}