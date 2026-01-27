mod activate;
mod common;
mod loss;
mod norm;

pub use activate::*;
pub use common::*;
pub use norm::*;
pub use loss::*;

#[cfg(test)]
mod test {
    use lumen_core::Tensor;

    #[test]
    fn test_nll_loss() {
        // Batch=2, Class=3
        // Sample 1: predict class 0 is high prob (-0.1)
        // Sample 2: predict class 2 is high prob (-0.2)
        let input = Tensor::new(&[
            [-0.1f32, -2.0, -3.0], 
            [-1.5, -0.5, -0.2]
        ]).unwrap();

        // Target: Sample 1 选 class 0, Sample 2 选 class 2
        let target = Tensor::new(&[0, 2]).unwrap().reshape((2, 1)).unwrap();

        let loss = crate::functional::nll_loss(&input, target).unwrap();

        // Calculation:
        // Sample 1 loss: -(-0.1) = 0.1
        // Sample 2 loss: -(-0.2) = 0.2
        // Mean: (0.1 + 0.2) / 2 = 0.15
        
        let val = loss.to_scalar().unwrap();
        assert!((val - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_soft_target() {
        // Input: Logits (unnormalized)
        let input = Tensor::new(&[
            [1.0f32, 2.0, 3.0], 
            [1.0, 2.0, 3.0]
        ]).unwrap();

        // Target: Probabilities (One-hot 模拟)
        // Sample 1: Class 2
        // Sample 2: Class 0
        let target = Tensor::new(&[
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ]).unwrap();

        let loss = crate::functional::cross_entropy(&input, &target).unwrap();
        
        // Softmax([1, 2, 3]) ≈ [0.0900, 0.2447, 0.6652]
        // LogSoftmax ≈ [-2.4076, -1.4076, -0.4076]
        
        // Sample 1 (Class 2): -1.0 * (-0.4076) = 0.4076
        // Sample 2 (Class 0): -1.0 * (-2.4076) = 2.4076
        // Mean = (0.4076 + 2.4076) / 2 = 1.4076
        
        let val = loss.to_scalar().unwrap();
        println!("{}", val);
        assert!((val - 1.4076).abs() < 1e-3);
    }
}