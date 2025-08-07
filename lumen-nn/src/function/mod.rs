use lumen_core::Tensor;
use anyhow::Result;

/// SoftMax
/// 
/// x_{i} = \frac{e^{x_{i}}}{\sum_{k = 0}^{n} e^{x_{k}}} 
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;;
/// use lumen_nn::function;
/// let x = Tensor::build([0.5, 0.2, 0.1, 0.4, 0.12, 0.132, -1.2, -4., 4.], [3, 3]).unwrap();
/// //let y = function::softmax(&x).unwrap();
/// //assert!(y.allclose(&Tensor::build([0.41474187266806956, 0.30724833615216285, 0.27800979117976765, 0.3967165323684531, 0.2998319051307709, 0.303451562500776, 0.005484469158895358, 0.00033351091301850316, 0.9941820199280862], [3, 3]).unwrap()));
/// ```
pub fn softmax(_x: &Tensor) -> Result<Tensor> {
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
    //         *d = s.exp() / sum;
    //     }
    // }

    // Ok(if reshape {
    //     y.view([y.element_size()]).unwrap()
    // } else {
    //     y
    // })
    todo!()
}

/// LogSoftMax
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;;
/// use lumen_nn::function;
/// let x = Tensor::build([2.0, 1.0, 0.1], [1, 3]).unwrap();
/// //let y = function::log_softmax(&x).unwrap();
/// //assert!(y.allclose(&Tensor::build([-0.4170300162778333, -1.4170300162778333, -2.3170300162778332], [1, 3]).unwrap()));
/// ```
pub fn log_softmax(_x: &Tensor) -> Result<Tensor> {
    todo!()
}
