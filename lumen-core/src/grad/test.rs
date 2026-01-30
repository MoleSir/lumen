
#[cfg(test)]
mod test {
    use std::f64::consts::PI;
    use crate::{Tensor, Var};

    #[test]
    fn test_binary() -> crate::Result<()> {
        let a = Var::<f64>::ones((3, 3))?;
        let b = Var::<f64>::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ])?;

        let c = &a * &b;

        let grads = c.backward()?;

        assert!(grads[&b].allclose(&a, 1e-5, 8e-8)?);
        assert!(grads[&a].allclose(&b, 1e-5, 8e-8)?);

        Ok(())
    }

    #[test]
    fn test_division() -> crate::Result<()> {
        let a = Var::<f64>::new(&[6.0, 10.0])?;
        let b = Var::<f64>::new(&[2.0, 5.0])?;
    
        let c = &a / &b;
        let grads = c.backward()?;
    
        // dc/da = 1/b = [0.5, 0.2]
        let expected_a_grad = Tensor::new(&[0.5, 0.2])?;
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8)?);
    
        // dc/db = -a / b^2 = [-6/4, -10/25] = [-1.5, -0.4]
        let expected_b_grad = Tensor::new(&[-1.5, -0.4])?;
        assert!(grads[&b].allclose(&expected_b_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_chain_rule() -> crate::Result<()> {
        let a = Var::<f64>::new(&[2.0, 3.0])?;
        let b = Var::<f64>::new(&[4.0, 5.0])?;
    
        // c = (a + b) * a = a^2 + ab
        let c = (&a + &b).mul(&a)?;
        let grads = c.backward()?;
    
        // dc/da = 2a + b = [2*2+4, 2*3+5] = [8, 11]
        let expected_a_grad = Tensor::new(&[8.0, 11.0])?;
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8)?);
    
        // dc/db = a = [2, 3]
        assert!(grads[&b].allclose(&a, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_scalar_ops() -> crate::Result<()> {
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0])?;
    
        // c = a * 2.0 + 10.0
        let c = a.add(10.0)?.mul(2.0)?;
        let grads = c.backward()?;
    
        // dc/da = 2.0
        let expected_grad = Tensor::new(&[2.0, 2.0, 2.0])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_scalar_lhs_sub() -> crate::Result<()> {
        // A: [1.0, 2.0, 3.0]
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0])?;

        // y = 10.0 - a
        // forward: [9.0, 8.0, 7.0]
        let y = 10.0 - a.clone(); 
        let grads = y.backward()?;

        // expected grad: [-1.0, -1.0, -1.0]
        let expected_grad = Tensor::new(&[-1.0, -1.0, -1.0])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_scalar_lhs_div() -> crate::Result<()> {
        // A: [1.0, 2.0, 4.0]
        let a = Var::<f64>::new(&[1.0, 2.0, 4.0])?;

        // y = 1.0 / a
        // forward: [1.0, 0.5, 0.25]
        let y = 1.0 / a.clone();
        let grads = y.backward()?;

        // expected grad: -1.0 / a^2
        // a=1.0 -> -1.0
        // a=2.0 -> -1.0 / 4.0 = -0.25
        // a=4.0 -> -1.0 / 16.0 = -0.0625
        let expected_grad = Tensor::new(&[-1.0, -0.25, -0.0625])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_scalar_lhs_div_complex() -> crate::Result<()> {
        // A: [1.0, 3.0]
        let a = Var::<f64>::new(&[1.0, 3.0])?;

        // u = a + 1.0 => [2.0, 4.0]
        // y = 8.0 / u => [4.0, 2.0]
        let y = 8.0 / (a.clone() + 1.0);
        
        let grads = y.backward()?;

        // Expected grad:
        // x=1.0, u=2.0 => -8 / 4 = -2.0
        // x=3.0, u=4.0 => -8 / 16 = -0.5
        let expected_grad = Tensor::new(&[-2.0, -0.5])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_scalar_lhs_mul() -> crate::Result<()> {
        let a = Var::<f64>::new(&[1.0, -1.0])?;
        
        // y = 0.5 * a
        let y = 0.5 * a.clone();
        let grads = y.backward()?;

        let expected_grad = Tensor::new(&[0.5, 0.5])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }
    
    #[test]
    fn test_mixed_scalar_ops() -> crate::Result<()> {
        let a = Var::<f64>::new(&[0.0, 1.0, 4.0])?;

        let t1 = 10.0 - &a; // Scalar LHS Sub
        let t2 = 2.0 + &a;  // Scalar LHS Add
        let y = t1 * t2;           // Tensor Mul
        
        let grads = y.backward()?;

        // Expected: 8 - 2x
        // x=0 -> 8
        // x=1 -> 6
        // x=4 -> 0
        let expected_grad = Tensor::new(&[8.0, 6.0, 0.0])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_max_scalar_rhs_with_ties() -> crate::Result<()> {
        let a = Var::<f64>::new(&[1.0, 5.0, 10.0])?;
        let y = a.maximum(5.0)?; 
        
        let grads = y.backward()?;
        let expected_grad = Tensor::new(&[0.0, 0.5, 1.0])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_min_scalar_lhs_with_ties() -> crate::Result<()> {
        let a = Var::<f64>::new(&[1.0, 3.0, 5.0])?;        
        let result = Tensor::scalar_minimum(3.0, &a)?; 
        let grads = result.backward()?;

        let expected_grad = Tensor::new(&[1.0, 0.5, 0.0])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_binary_max_ties() -> crate::Result<()> {
        let a = Var::<f64>::new(&[1.0, 5.0, 8.0])?;
        let b = Var::<f64>::new(&[2.0, 5.0, 6.0])?;
        let y = a.maximum(&b)?;
        let grads = y.backward()?;

        let expected_grad_a = Tensor::new(&[0.0, 0.5, 1.0])?;
        assert!(grads[&a].allclose(&expected_grad_a, 1e-5, 1e-8)?);

        let expected_grad_b = Tensor::new(&[1.0, 0.5, 0.0])?;
        assert!(grads[&b].allclose(&expected_grad_b, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_unary_math() -> crate::Result<()> {
        let x = Var::<f64>::new(&[1.0, 2.0])?;
        let y = x.exp()?; // y = e^x
        let grads = y.backward()?;
        // dy/dx = e^x
        assert!(grads[&x].allclose(&y, 1e-5, 1e-8)?);

        let x2 = Var::<f64>::new(&[1.0, 10.0])?;
        let y2 = x2.ln()?; // y = ln(x)
        let grads2 = y2.backward()?;
        // dy/dx = 1/x
        let expected_grad = Tensor::new(&[1.0, 0.1])?;
        assert!(grads2[&x2].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_trig_and_tanh() -> crate::Result<()> {
        let x = Var::<f64>::new(&[0.0, PI / 3.0])?;
        
        // Sin -> Cos
        let y_sin = x.sin()?;
        let grads_sin = y_sin.backward()?;
        assert!(grads_sin[&x].allclose(&x.cos()?, 1e-5, 1e-8)?);

        // Cos -> -Sin
        let y_cos = x.cos()?;
        let grads_cos = y_cos.backward()?;
        let expected_cos_grad = x.sin()?.neg()?;
        assert!(grads_cos[&x].allclose(&expected_cos_grad, 1e-5, 1e-8)?);

        // Tanh -> 1 - tanh^2
        let x2 = Var::<f64>::new(&[0.5, -0.5])?;
        let y_tanh = x2.tanh()?;
        let grads_tanh = y_tanh.backward()?;
        // d/dx tanh(x) = 1 - tanh(x)^2
        let expected_tanh_grad = Tensor::ones_like(&x2)?.sub(&y_tanh.sqr()?)?;
        assert!(grads_tanh[&x2].allclose(&expected_tanh_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_power_ops() -> crate::Result<()> {
        // Sqr: x^2 -> 2x
        let x = Var::<f64>::new(&[2.0, 4.0])?;
        let y_sqr = x.sqr()?;
        let grads_sqr = y_sqr.backward()?;
        let expected_sqr_grad = Tensor::new(&[4.0, 8.0])?;
        assert!(grads_sqr[&x].allclose(&expected_sqr_grad, 1e-5, 1e-8)?);

        // Sqrt: sqrt(x) -> 1/(2*sqrt(x))
        let y_sqrt = x.sqrt()?;
        let grads_sqrt = y_sqrt.backward()?;
        let expected_sqrt_grad = Tensor::new(&[1.0 / (2.0 * 2.0f64.sqrt()), 1.0 / 4.0])?;
        assert!(grads_sqrt[&x].allclose(&expected_sqrt_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_abs_and_neg() -> crate::Result<()> {
        let x = Var::<f64>::new(&[-5.0, 3.0, 0.0])?;
        
        // Abs -> sign(x)
        let y_abs = x.abs()?;
        let grads_abs = y_abs.backward()?;
        let expected_abs_grad = Tensor::new(&[-1.0, 1.0, 1.0])?; // 
        assert!(grads_abs[&x].allclose(&expected_abs_grad, 1e-5, 1e-8)?);

        // Neg -> -1
        let y_neg = x.neg()?;
        let grads_neg = y_neg.backward()?;
        let expected_neg_grad = Tensor::new(&[-1.0, -1.0, -1.0])?;
        assert!(grads_neg[&x].allclose(&expected_neg_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_reciprocal() -> crate::Result<()> {
        // Recip: 1/x -> -1/x^2
        let x = Var::<f64>::new(&[2.0, 4.0])?;
        let y = x.recip()?;
        let grads = y.backward()?;
        let expected_grad = Tensor::new(&[-0.25, -0.0625])?;
        assert!(grads[&x].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_broadcast_backward() -> crate::Result<()> {
        // 情况 1: 向量广播到矩阵 (1, 3) -> (2, 3)
        let a = Var::<f64>::new(&[[1.0, 2.0, 3.0]])?; // (1, 3)
        let b = Var::<f64>::new(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])?; // (2, 3)
        
        // c = a + b (a 会被广播成两行)
        let c = a.broadcast_add(&b)?;
        let grads = c.backward()?;
        
        // dc/da 应该是 [[2.0, 2.0, 2.0]]，因为 a 的每一项都被使用了两次
        let expected_a_grad = Tensor::new(&[[2.0, 2.0, 2.0]])?;
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8)?);

        // 情况 2: 标量广播到向量 () -> (3,)
        let s = Var::<f64>::new(&[[5.0]])?; // 简化的标量 (1,1)
        let v = Var::<f64>::new(&[[1.0, 2.0, 3.0]])?;
        let c2 = s.broadcast_add(&v)?;
        let grads2 = c2.backward()?;
        
        // dc/ds 应该累加 3 次
        let expected_s_grad = Tensor::new(&[[3.0]])?;
        assert!(grads2[&s].allclose(&expected_s_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_complex_chain_rule() -> crate::Result<()> {
        // y = exp(sin(x^2))
        // dy/dx = exp(sin(x^2)) * cos(x^2) * 2x
        let x_val = 1.0f64;
        let x = Var::<f64>::new(&[x_val])?;
        let y = x.sqr()?.sin()?.exp()?;
        
        let grads = y.backward()?;
        
        let expected_val = (x_val.powi(2).sin().exp()) * (x_val.powi(2).cos()) * (2.0 * x_val);
        let expected_grad = Tensor::new(&[expected_val])?;
        
        assert!(grads[&x].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_matmul_basic_2d() -> crate::Result<()> {
        // A: (2, 3), B: (3, 2)
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])?;
        let b = Var::<f64>::new(&[
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ])?;

        let c = a.matmul(&b)?; // (2, 2)
        let grads = c.backward()?;

        // c = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // c = [[58, 64], [139, 154]]
        
        // dc/da = grad(2,2) * b^T(2,3) = [[1,1],[1,1]] * [[7,9,11],[8,10,12]]
        // dc/da = [[15, 19, 23], [15, 19, 23]]
        let expected_a_grad = Tensor::new(&[
            [15.0, 19.0, 23.0],
            [15.0, 19.0, 23.0]
        ])?;
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8)?);

        // dc/db = a^T(3,2) * grad(2,2) = [[1,4],[2,5],[3,6]] * [[1,1],[1,1]]
        // dc/db = [[5, 5], [7, 7], [9, 9]]
        let expected_b_grad = Tensor::new(&[
            [5.0, 5.0],
            [7.0, 7.0],
            [9.0, 9.0]
        ])?;
        assert!(grads[&b].allclose(&expected_b_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_matmul_vector() -> crate::Result<()> {
        let w = Var::<f64>::new(&[
            [0.5, -0.5],
            [0.1, 0.8]
        ])?;
        let x = Var::<f64>::new(&[
            [1.0],
            [2.0]
        ])?;

        let y = w.matmul(&x)?;
        let grads = y.backward()?;

        // dy/dw = grad * x^T = [[1],[1]] * [[1, 2]] = [[1, 2], [1, 2]]
        let expected_w_grad = Tensor::new(&[
            [1.0, 2.0],
            [1.0, 2.0]
        ])?;
        assert!(grads[&w].allclose(&expected_w_grad, 1e-5, 1e-8)?);

        // dy/dx = w^T * grad = [[0.5, 0.1], [-0.5, 0.8]] * [[1], [1]] = [[0.6], [0.3]]
        let expected_x_grad = Tensor::new(&[
            [0.6],
            [0.3]
        ])?;
        assert!(grads[&x].allclose(&expected_x_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_batched_matmul() -> crate::Result<()> {
        // A: (2, 2, 2), B: (2, 2, 2)
        let a = Var::<f64>::new(&[
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 0.5], [0.5, 0.5]]
        ])?;
        let b = Var::<f64>::new(&[
            [[1.0, 0.0], [0.0, 1.0]], 
            [[2.0, 2.0], [2.0, 2.0]]   
        ])?;

        let c = a.matmul(&b)?;
        let grads = c.backward()?;

        let expected_a_grad = Tensor::new(&[
            [[1.0, 1.0], [1.0, 1.0]],
            [[4.0, 4.0], [4.0, 4.0]]
        ])?;
        
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_matmul_chain_with_add() -> crate::Result<()> {
        let w = Var::<f64>::new(&[[2.0, 3.0]])?; // (1, 2)
        let x = Var::<f64>::new(&[[4.0], [5.0]])?; // (2, 1)
        let b = Var::<f64>::new(&[[10.0]])?; // (1, 1)

        let y = w.matmul(&x)?.add(&b)?;
        let grads = y.backward()?;

        // dy/dw = x^T = [[4, 5]]
        assert!(grads[&w].allclose(&Tensor::new(&[[4.0, 5.0]])?, 1e-5, 1e-8)?);
        // dy/dx = w^T = [[2], [3]]
        assert!(grads[&x].allclose(&Tensor::new(&[[2.0], [3.0]])?, 1e-5, 1e-8)?);
        // dy/db = 1
        assert!(grads[&b].allclose(&Tensor::new(&[[1.0]])?, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_sum_dim_0() -> crate::Result<()> {
        // A: (2, 2)
        // [ [1.0, 2.0],
        //   [3.0, 4.0] ]
        let a = Var::<f64>::new(&[[1.0, 2.0], [3.0, 4.0]])?;

        let s = a.sum_keepdim(0)?; 
        let grads = s.backward()?;

        let expected_grad = Tensor::new(&[[1.0, 1.0], [1.0, 1.0]])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_sum_dim_1() -> crate::Result<()> {
        // A: (2, 3)
        let a = Var::<f64>::new(&[
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ])?;
        
        let s = a.sum_keepdim(1)?; 
        let grads = s.backward()?;

        let expected_grad = Tensor::ones_like(&a)?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_max_dim_0() -> crate::Result<()> {
        // A: (2, 2)
        // [ [10.0, 2.0],
        //   [5.0,  8.0] ]
        let a = Var::<f64>::new(&[[10.0, 2.0], [5.0, 8.0]])?;
        
        // m = a.max(0) => [10.0, 8.0]
        let m = a.max_keepdim(0)?;
        let grads = m.backward()?;

        let expected_grad = Tensor::new(&[
            [1.0, 0.0],
            [0.0, 1.0]
        ])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_min_dim_1() -> crate::Result<()> {
        // A: (2, 3)
        // [ [1.0, 5.0, 0.0],
        //   [4.0, 2.0, 6.0] ]
        let a = Var::<f64>::new(&[
            [1.0, 5.0, 0.0],
            [4.0, 2.0, 6.0]
        ])?;
        
        // m = a.min(1) => [0.0, 2.0]
        let m = a.min_keepdim(1)?;
        let grads = m.backward()?;

        let expected_grad = Tensor::new(&[
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_mean_dim_0() -> crate::Result<()> {
        // A: (2, 2)
        // [ [1.0, 2.0],
        //   [3.0, 4.0] ]
        let a = Var::<f64>::new(&[[1.0, 2.0], [3.0, 4.0]])?;

        // mean(0) => [ (1+3)/2, (2+4)/2 ] = [2.0, 3.0]
        let s = a.mean_keepdim(0)?; 
        let grads = s.backward()?;

        let expected_grad = Tensor::new(&[[0.5, 0.5], [0.5, 0.5]])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_var_dim_0() -> crate::Result<()> {
        // A: (2, 2)
        // [ [10.0, 2.0],
        //   [20.0, 8.0] ]
        let a = Var::<f64>::new(&[[10.0, 2.0], [20.0, 8.0]])?;
        println!("{}", a.requires_grad());
        
        // dim 0: mean = [15.0, 5.0]
        // var = [ ((10-15)^2 + (20-15)^2)/2, ((2-5)^2 + (8-5)^2)/2 ]
        // var = [ (25+25)/2, (9+9)/2 ] = [25.0, 9.0]
        let v = a.var_keepdim(0)?;
        let grads = v.backward()?;

        // a[0,0]=10: (2/2) * (10 - 15) = -5.0
        // a[1,0]=20: (2/2) * (20 - 15) = 5.0
        // a[0,1]=2:  (2/2) * (2 - 5)   = -3.0
        // a[1,1]=8:  (2/2) * (8 - 5)   = 3.0
        let expected_grad = Tensor::new(&[
            [-5.0, -3.0],
            [ 5.0,  3.0]
        ])?;
        
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_max_with_ties() -> crate::Result<()> {
        // A: [5.0, 5.0, 2.0]
        let a = Var::<f64>::new(&[5.0, 5.0, 2.0])?;
        
        // max(0) => 5.0
        let m = a.max_keepdim(0)?;
        let grads = m.backward()?;

        let expected_grad = Tensor::new(&[1.0, 1.0, 0.0])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_reduce_with_matmul() -> crate::Result<()> {
        // y = sum( A(2,2) @ x(2,1) )
        let a = Var::<f64>::new(&[[1.0, 2.0], [3.0, 4.0]])?;
        let x = Var::<f64>::new(&[[10.0], [100.0]])?;
        
        // res = [[210], [430]]
        let res = a.matmul(&x)?;
        // y = 210 + 430 = 640
        let y = res.sum_keepdim(0)?; 
        
        let grads = y.backward()?;

        // dy/dres = [1, 1]
        // dy/dx = A^T @ [1, 1]^T = [[1, 3], [2, 4]] @ [[1], [1]] = [[4], [6]]
        let expected_x_grad = Tensor::new(&[[4.0], [6.0]])?;
        assert!(grads[&x].allclose(&expected_x_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_reshape_backward() -> crate::Result<()> {
        // A: (2, 3)
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])?;

        // Reshape to (3, 2)
        let b = a.reshape(&[3, 2])?;
        
        let loss = b.sum_keepdim(0)?.sum_keepdim(1)?;
        let grads = loss.backward()?;

        let expected_grad = Tensor::new(&[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])?;
        
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        assert_eq!(grads[&a].dims(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_transpose_backward() -> crate::Result<()> {
        // A: (2, 3)
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])?;

        // Transpose (0, 1) -> (3, 2)
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        let b = a.transpose(0, 1)?;

        let w = Tensor::new(&[
            [1.0, 10.0],
            [100.0, 1000.0],
            [10000.0, 100000.0]
        ])?;
        
        let loss = (b.mul(&w)?).sum_keepdim(0)?.sum_keepdim(1)?;
        let grads = loss.backward()?;

        let expected_grad = Tensor::new(&[
            [1.0, 100.0, 10000.0],
            [10.0, 1000.0, 100000.0]
        ])?;

        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_narrow_backward() -> crate::Result<()> {
        // A: (3, 3)
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])?;

        // Narrow: dim 0, start 1, len 1 (取中间一行: [4.0, 5.0, 6.0])
        let b = a.narrow(0, 1, 1)?;
        
        let loss = b.sum_keepdim(0)?.sum_keepdim(1)?;
        let grads = loss.backward()?;

        let expected_grad = Tensor::new(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0]
        ])?;

        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);

        let c = a.narrow(1, 2, 1)?; // 取最后一列
        let grads_c = c.sum_keepdim(0)?.sum_keepdim(1)?.backward()?;
        let expected_grad_c = Tensor::new(&[
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ])?;
        assert!(grads_c[&a].allclose(&expected_grad_c, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_cat_backward() -> crate::Result<()> {
        // A: (2, 2), B: (2, 1)
        let a = Var::<f64>::new(&[
            [1.0, 2.0],
            [3.0, 4.0]
        ])?;
        let b = Var::<f64>::new(&[
            [5.0],
            [6.0]
        ])?;

        // Cat along dim 1 -> Result: (2, 3)
        // [[1, 2, 5],
        //  [3, 4, 6]]
        let c = Tensor::cat(&[&a, &b], 1)?;
        
        let w = Tensor::new(&[
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ])?;
        
        let loss = c.mul(&w)?.sum_keepdim(0)?.sum_keepdim(1)?;
        let grads = loss.backward()?;

        let expected_grad_a = Tensor::new(&[
            [1.0, 2.0],
            [1.0, 2.0]
        ])?;
        let expected_grad_b = Tensor::new(&[
            [3.0],
            [3.0]
        ])?;

        assert!(grads[&a].allclose(&expected_grad_a, 1e-5, 1e-8)?);
        assert!(grads[&b].allclose(&expected_grad_b, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_complex_combined_ops() -> crate::Result<()> {
        // Reshape -> Transpose -> Narrow
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0, 4.0])?; // (4,)
        
        let b = a.reshape(&[2, 2])?; // [[1,2],[3,4]]
        let c = b.transpose(0, 1)?;  // [[1,3],[2,4]]
        let d = c.narrow(0, 1, 1)?;  // [[2,4]] (取第二行)
        
        let loss = d.sum_keepdim(0)?.sum_keepdim(1)?;
        let grads = loss.backward()?;

        // d = [[2, 4]] -> grad_d = [[1, 1]]
        // c = [[1, 3], [2, 4]] -> grad_c = [[0, 0], [1, 1]]
        // b = [[1, 2], [3, 4]] -> grad_b = grad_c.T = [[0, 1], [0, 1]]
        // a = [1, 2, 3, 4] -> grad_a = [0, 1, 0, 1]
        
        let expected_grad_a = Tensor::new(&[0.0, 1.0, 0.0, 1.0])?;
        assert!(grads[&a].allclose(&expected_grad_a, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_permute_backward() -> crate::Result<()> {
        // [ [[1.0], [2.0], [3.0]],
        //   [[4.0], [5.0], [6.0]] ]
        let a = Var::<f64>::new(&[
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]]
        ])?;

        // Permute (0, 1, 2) -> (2, 0, 1)
        let b = a.permute(vec![2, 0, 1])?;
        assert_eq!(b.dims(), &[1, 2, 3]);

        let w = Tensor::new(&[[
            [1.0, 2.0, 3.0], 
            [4.0, 5.0, 6.0]
        ]])?;
        
        // loss = sum(b * w)
        let loss = b.mul(&w)?
            .sum_keepdim(0)?
            .sum_keepdim(1)?
            .sum_keepdim(2)?;
            
        let grads = loss.backward()?;

        let expected_grad = Tensor::new(&[
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]]
        ])?;

        assert_eq!(grads[&a].dims(), &[2, 3, 1]);
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_permute_complex_backward() -> crate::Result<()> {
        // (0, 1, 2, 3) -> (0, 2, 3, 1) 
        let a = Var::<f64>::zeros(&[1, 2, 2, 2])?; 
        
        let b = a.permute(vec![0, 2, 3, 1])?;
        
        let loss = b.sum_keepdim(0)?.sum_keepdim(1)?.sum_keepdim(2)?.sum_keepdim(3)?;
        let grads = loss.backward()?;
        
        let expected_grad = Tensor::ones(&[1, 2, 2, 2])?;
        assert_eq!(grads[&a].dims(), &[1, 2, 2, 2]);
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_pow_backward() -> crate::Result<()> {
        // a = [2.0, 4.0]
        let a = Var::<f64>::new(&[2.0, 4.0])?;
        
        // b = a^3.0 = [8.0, 64.0]
        let b = a.pow(3.0)?;
        
        // 假设 loss = sum(b)，则 grad_b = [1.0, 1.0]
        let loss = b.sum_keepdim(0)?;
        let grads = loss.backward()?;

        // da = grad_b * (3.0 * a^(3.0 - 1.0))
        // da = 1.0 * (3.0 * [4.0, 16.0]) = [12.0, 48.0]
        let expected_grad = Tensor::new(&[12.0, 48.0])?;
        
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_pow_fractional_backward() -> crate::Result<()> {
        // 测试开根号: a^0.5
        let a = Var::<f64>::new(&[4.0, 16.0])?;
        let b = a.pow(0.5)?;
        
        let loss = b.sum_keepdim(0)?;
        let grads = loss.backward()?;

        // da = 0.5 * a^(-0.5) = 1 / (2 * sqrt(a))
        // da = [1/(2*2), 1/(2*4)] = [0.25, 0.125]
        let expected_grad = Tensor::new(&[0.25, 0.125])?;
        
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_masked_fill_grad() -> crate::Result<()> {
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0, 4.0])?;
        let mask = Tensor::new(&[false, true, false, true])?;
        let y = a.masked_fill(&mask, f64::NEG_INFINITY)?;
        
        let grads = y.backward()?;

        let expected_grad = Tensor::new(&[1.0, 0.0, 1.0, 0.0])?;
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_where_grad() -> crate::Result<()> {
        let cond = Tensor::new(&[true, false, true])?;
        let a = Var::<f64>::new(&[10.0, 10.0, 10.0])?;
        let b = Var::<f64>::new(&[20.0, 20.0, 20.0])?;
        let y = cond.if_else(&a, &b)?;
        
        let grads = y.backward()?; // 默认传入全是 1.0 的梯度

        let expected_grad_a = Tensor::new(&[1.0, 0.0, 1.0])?;
        assert!(grads[&a].allclose(&expected_grad_a, 1e-5, 1e-8)?);

        let expected_grad_b = Tensor::new(&[0.0, 1.0, 0.0])?;
        assert!(grads[&b].allclose(&expected_grad_b, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_squeeze_backward() -> crate::Result<()> {
        // A: (1, 2, 1) -> [[ [1.0], [2.0] ]]
        let a = Var::<f64>::new(&[[[1.0], [2.0]]])?;
        
        let b = a.squeeze(0)?;
        assert_eq!(b.dims(), &[2, 1]);

        let c = b.squeeze(1)?;
        assert_eq!(c.dims(), &[2]);

        let loss = c.sum_keepdim(0)?;
        let grads = loss.backward()?;

        let grad_a = &grads[&a];
        assert_eq!(grad_a.dims(), &[1, 2, 1]);
        
        let expected_grad = Tensor::new(&[[[1.0], [1.0]]])?;
        assert!(grad_a.allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_unsqueeze_backward() -> crate::Result<()> {
        // A: (2) -> [10.0, 20.0]
        let a = Var::<f64>::new(&[10.0, 20.0])?;
        
        // 1. Unsqueeze 维度 0 -> (1, 2)
        let b = a.unsqueeze(0)?;
        assert_eq!(b.dims(), &[1, 2]);

        // 2. Unsqueeze 维度 2 -> (1, 2, 1)
        let c = b.unsqueeze(2)?;
        assert_eq!(c.dims(), &[1, 2, 1]);

        // c * [[ [2.0], [3.0] ]] = [[ [20.0], [60.0] ]]
        let weights = Tensor::new(&[[[2.0], [3.0]]])?;
        let out = c.mul(&weights)?;
        let loss = out.sum_keepdim(0)?;
        
        let grads = loss.backward()?;

        let grad_a = &grads[&a];
        assert_eq!(grad_a.dims(), &[2]);
        
        let expected_grad = Tensor::new(&[2.0, 3.0])?;
        assert!(grad_a.allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_index_select_backward_basic() -> crate::Result<()> {
        // A: (4) [1.0, 2.0, 3.0, 4.0]
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0, 4.0])?;
        
        // Indices: [0, 2]
        let indices = Tensor::new(&[0, 2])?;
        
        // Forward: b = [1.0, 3.0]
        let b = a.index_select(indices, 0)?;
        
        // Loss = Sum(b)
        // dLoss/db = [1.0, 1.0]
        let loss = b.sum_keepdim(0)?;
        let grads = loss.backward()?;
        let expected_grad = Tensor::new(&[1.0, 0.0, 1.0, 0.0])?;

        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_index_select_backward_accumulation() -> crate::Result<()> {        
        // A: (3) [10.0, 20.0, 30.0]
        let a = Var::<f64>::new(&[10.0, 20.0, 30.0])?;
        
        // Indices: [1, 1, 0] -> 第 1 个元素被选中两次！
        let indices = Tensor::new(&[1, 1, 0])?;
        
        // Forward: b = [20.0, 20.0, 10.0]
        let b = a.index_select(indices, 0)?;
        
        // 我们给 b 一个加权 loss，以便更好区分梯度
        // Weights: [0.5, 2.0, 3.0]
        let w = Tensor::new(&[0.5, 2.0, 3.0])?;
        
        // Loss = Sum(b * w)
        // b[0] (来自 a[1]) * 0.5 -> 对 a[1] 贡献梯度 0.5
        // b[1] (来自 a[1]) * 2.0 -> 对 a[1] 贡献梯度 2.0
        // b[2] (来自 a[0]) * 3.0 -> 对 a[0] 贡献梯度 3.0
        let loss = b.mul(&w)?.sum_keepdim(0)?;
        let grads = loss.backward()?;

        // Backward 预期:
        // a[0]: 被索引 0 选中一次 (权重 3.0) -> grad = 3.0
        // a[1]: 被索引 1 选中两次 (权重 0.5 和 2.0) -> grad = 0.5 + 2.0 = 2.5
        // a[2]: 未被选中 -> grad = 0.0
        let expected_grad = Tensor::new(&[3.0, 2.5, 0.0])?;

        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_index_select_backward_dim1() -> crate::Result<()> {        
        // A: (2, 3)
        // [[1.0, 2.0, 3.0],
        //  [4.0, 5.0, 6.0]]
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])?;

        // Indices: [2, 0] (选第2列和第0列)
        let indices = Tensor::new(&[2, 0])?;

        // Forward (dim=1):
        // [[3.0, 1.0],
        //  [6.0, 4.0]]
        let b = a.index_select(indices, 1)?;

        // Loss = Sum(b) -> 所有选中元素的梯度由于 sum 操作都收到 1.0 的回传
        let loss = b.sum_keepdim(0)?.sum_keepdim(1)?;
        let grads = loss.backward()?;

        // Backward 预期:
        // Col 0: 被选中一次 -> grad 1.0 (对每一行)
        // Col 1: 未被选中   -> grad 0.0
        // Col 2: 被选中一次 -> grad 1.0
        let expected_grad = Tensor::new(&[
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0]
        ])?;

        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8)?);
        Ok(())
    }
}