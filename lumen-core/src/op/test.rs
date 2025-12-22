
#[cfg(test)]
mod test {
    use std::f64::consts::PI;
    use crate::{Tensor, Var};

    #[test]
    fn test_binary() {
        let a = Var::<f64>::ones((3, 3)).unwrap();
        let b = Var::<f64>::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]).unwrap();

        let c = &a * &b;

        let grads = c.backward().unwrap();

        assert!(grads[&b].allclose(&a, 1e-5, 8e-8));
        assert!(grads[&a].allclose(&b, 1e-5, 8e-8));
    }

    #[test]
    fn test_division() {
        let a = Var::<f64>::new(&[6.0, 10.0]).unwrap();
        let b = Var::<f64>::new(&[2.0, 5.0]).unwrap();
    
        let c = &a / &b;
        let grads = c.backward().unwrap();
    
        // dc/da = 1/b = [0.5, 0.2]
        let expected_a_grad = Tensor::new(&[0.5, 0.2]).unwrap();
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8));
    
        // dc/db = -a / b^2 = [-6/4, -10/25] = [-1.5, -0.4]
        let expected_b_grad = Tensor::new(&[-1.5, -0.4]).unwrap();
        assert!(grads[&b].allclose(&expected_b_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_chain_rule() {
        let a = Var::<f64>::new(&[2.0, 3.0]).unwrap();
        let b = Var::<f64>::new(&[4.0, 5.0]).unwrap();
    
        // c = (a + b) * a = a^2 + ab
        let c = (&a + &b).mul(&a).unwrap();
        let grads = c.backward().unwrap();
    
        // dc/da = 2a + b = [2*2+4, 2*3+5] = [8, 11]
        let expected_a_grad = Tensor::new(&[8.0, 11.0]).unwrap();
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8));
    
        // dc/db = a = [2, 3]
        assert!(grads[&b].allclose(&a, 1e-5, 1e-8));
    }

    #[test]
    fn test_scalar_ops() {
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0]).unwrap();
    
        // c = a * 2.0 + 10.0
        let c = a.add(10.0).unwrap().mul(2.0).unwrap();
        let grads = c.backward().unwrap();
    
        // dc/da = 2.0
        let expected_grad = Tensor::new(&[2.0, 2.0, 2.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_scalar_lhs_sub() {
        // A: [1.0, 2.0, 3.0]
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0]).unwrap();

        // y = 10.0 - a
        // forward: [9.0, 8.0, 7.0]
        let y = 10.0 - a.clone(); 
        let grads = y.backward().unwrap();

        // expected grad: [-1.0, -1.0, -1.0]
        let expected_grad = Tensor::new(&[-1.0, -1.0, -1.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_scalar_lhs_div() {
        // A: [1.0, 2.0, 4.0]
        let a = Var::<f64>::new(&[1.0, 2.0, 4.0]).unwrap();

        // y = 1.0 / a
        // forward: [1.0, 0.5, 0.25]
        let y = 1.0 / a.clone();
        let grads = y.backward().unwrap();

        // expected grad: -1.0 / a^2
        // a=1.0 -> -1.0
        // a=2.0 -> -1.0 / 4.0 = -0.25
        // a=4.0 -> -1.0 / 16.0 = -0.0625
        let expected_grad = Tensor::new(&[-1.0, -0.25, -0.0625]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_scalar_lhs_div_complex() {
        // A: [1.0, 3.0]
        let a = Var::<f64>::new(&[1.0, 3.0]).unwrap();

        // u = a + 1.0 => [2.0, 4.0]
        // y = 8.0 / u => [4.0, 2.0]
        let y = 8.0 / (a.clone() + 1.0);
        
        let grads = y.backward().unwrap();

        // Expected grad:
        // x=1.0, u=2.0 => -8 / 4 = -2.0
        // x=3.0, u=4.0 => -8 / 16 = -0.5
        let expected_grad = Tensor::new(&[-2.0, -0.5]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_scalar_lhs_mul() {
        let a = Var::<f64>::new(&[1.0, -1.0]).unwrap();
        
        // y = 0.5 * a
        let y = 0.5 * a.clone();
        let grads = y.backward().unwrap();

        let expected_grad = Tensor::new(&[0.5, 0.5]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }
    
    #[test]
    fn test_mixed_scalar_ops() {
        let a = Var::<f64>::new(&[0.0, 1.0, 4.0]).unwrap();

        let t1 = 10.0 - &a; // Scalar LHS Sub
        let t2 = 2.0 + &a;  // Scalar LHS Add
        let y = t1 * t2;           // Tensor Mul
        
        let grads = y.backward().unwrap();

        // Expected: 8 - 2x
        // x=0 -> 8
        // x=1 -> 6
        // x=4 -> 0
        let expected_grad = Tensor::new(&[8.0, 6.0, 0.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_max_scalar_rhs_with_ties() {
        let a = Var::<f64>::new(&[1.0, 5.0, 10.0]).unwrap();
        let y = a.maximum(5.0).unwrap(); 
        
        let grads = y.backward().unwrap();
        let expected_grad = Tensor::new(&[0.0, 0.5, 1.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_min_scalar_lhs_with_ties() {
        let a = Var::<f64>::new(&[1.0, 3.0, 5.0]).unwrap();        
        let result = Tensor::scalar_minimum(3.0, &a).unwrap(); 
        let grads = result.backward().unwrap();

        let expected_grad = Tensor::new(&[1.0, 0.5, 0.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_binary_max_ties() {
        let a = Var::<f64>::new(&[1.0, 5.0, 8.0]).unwrap();
        let b = Var::<f64>::new(&[2.0, 5.0, 6.0]).unwrap();
        let y = a.maximum(&b).unwrap();
        let grads = y.backward().unwrap();

        let expected_grad_a = Tensor::new(&[0.0, 0.5, 1.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad_a, 1e-5, 1e-8));

        let expected_grad_b = Tensor::new(&[1.0, 0.5, 0.0]).unwrap();
        assert!(grads[&b].allclose(&expected_grad_b, 1e-5, 1e-8));
    }

    #[test]
    fn test_unary_math() {
        let x = Var::<f64>::new(&[1.0, 2.0]).unwrap();
        let y = x.exp(); // y = e^x
        let grads = y.backward().unwrap();
        // dy/dx = e^x
        assert!(grads[&x].allclose(&y, 1e-5, 1e-8));

        let x2 = Var::<f64>::new(&[1.0, 10.0]).unwrap();
        let y2 = x2.ln(); // y = ln(x)
        let grads2 = y2.backward().unwrap();
        // dy/dx = 1/x
        let expected_grad = Tensor::new(&[1.0, 0.1]).unwrap();
        assert!(grads2[&x2].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_trig_and_tanh() {
        let x = Var::<f64>::new(&[0.0, PI / 3.0]).unwrap();
        
        // Sin -> Cos
        let y_sin = x.sin();
        let grads_sin = y_sin.backward().unwrap();
        assert!(grads_sin[&x].allclose(&x.cos(), 1e-5, 1e-8));

        // Cos -> -Sin
        let y_cos = x.cos();
        let grads_cos = y_cos.backward().unwrap();
        let expected_cos_grad = x.sin().neg();
        assert!(grads_cos[&x].allclose(&expected_cos_grad, 1e-5, 1e-8));

        // Tanh -> 1 - tanh^2
        let x2 = Var::<f64>::new(&[0.5, -0.5]).unwrap();
        let y_tanh = x2.tanh();
        let grads_tanh = y_tanh.backward().unwrap();
        // d/dx tanh(x) = 1 - tanh(x)^2
        let expected_tanh_grad = Tensor::ones_like(&x2).unwrap().sub(&y_tanh.sqr()).unwrap();
        assert!(grads_tanh[&x2].allclose(&expected_tanh_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_power_ops() {
        // Sqr: x^2 -> 2x
        let x = Var::<f64>::new(&[2.0, 4.0]).unwrap();
        let y_sqr = x.sqr();
        let grads_sqr = y_sqr.backward().unwrap();
        let expected_sqr_grad = Tensor::new(&[4.0, 8.0]).unwrap();
        assert!(grads_sqr[&x].allclose(&expected_sqr_grad, 1e-5, 1e-8));

        // Sqrt: sqrt(x) -> 1/(2*sqrt(x))
        let y_sqrt = x.sqrt();
        let grads_sqrt = y_sqrt.backward().unwrap();
        let expected_sqrt_grad = Tensor::new(&[1.0 / (2.0 * 2.0f64.sqrt()), 1.0 / 4.0]).unwrap();
        assert!(grads_sqrt[&x].allclose(&expected_sqrt_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_abs_and_neg() {
        let x = Var::<f64>::new(&[-5.0, 3.0, 0.0]).unwrap();
        
        // Abs -> sign(x)
        let y_abs = x.abs();
        let grads_abs = y_abs.backward().unwrap();
        let expected_abs_grad = Tensor::new(&[-1.0, 1.0, 1.0]).unwrap(); // 
        assert!(grads_abs[&x].allclose(&expected_abs_grad, 1e-5, 1e-8));

        // Neg -> -1
        let y_neg = x.neg();
        let grads_neg = y_neg.backward().unwrap();
        let expected_neg_grad = Tensor::new(&[-1.0, -1.0, -1.0]).unwrap();
        assert!(grads_neg[&x].allclose(&expected_neg_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_reciprocal() {
        // Recip: 1/x -> -1/x^2
        let x = Var::<f64>::new(&[2.0, 4.0]).unwrap();
        let y = x.recip();
        let grads = y.backward().unwrap();
        let expected_grad = Tensor::new(&[-0.25, -0.0625]).unwrap();
        assert!(grads[&x].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_broadcast_backward() {
        // 情况 1: 向量广播到矩阵 (1, 3) -> (2, 3)
        let a = Var::<f64>::new(&[[1.0, 2.0, 3.0]]).unwrap(); // (1, 3)
        let b = Var::<f64>::new(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).unwrap(); // (2, 3)
        
        // c = a + b (a 会被广播成两行)
        let c = a.broadcast_add(&b).unwrap();
        let grads = c.backward().unwrap();
        
        // dc/da 应该是 [[2.0, 2.0, 2.0]]，因为 a 的每一项都被使用了两次
        let expected_a_grad = Tensor::new(&[[2.0, 2.0, 2.0]]).unwrap();
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8));

        // 情况 2: 标量广播到向量 () -> (3,)
        let s = Var::<f64>::new(&[[5.0]]).unwrap(); // 简化的标量 (1,1)
        let v = Var::<f64>::new(&[[1.0, 2.0, 3.0]]).unwrap();
        let c2 = s.broadcast_add(&v).unwrap();
        let grads2 = c2.backward().unwrap();
        
        // dc/ds 应该累加 3 次
        let expected_s_grad = Tensor::new(&[[3.0]]).unwrap();
        assert!(grads2[&s].allclose(&expected_s_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_complex_chain_rule() {
        // y = exp(sin(x^2))
        // dy/dx = exp(sin(x^2)) * cos(x^2) * 2x
        let x_val = 1.0f64;
        let x = Var::<f64>::new(&[x_val]).unwrap();
        let y = x.sqr().sin().exp();
        
        let grads = y.backward().unwrap();
        
        let expected_val = (x_val.powi(2).sin().exp()) * (x_val.powi(2).cos()) * (2.0 * x_val);
        let expected_grad = Tensor::new(&[expected_val]).unwrap();
        
        assert!(grads[&x].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_matmul_basic_2d() {
        // A: (2, 3), B: (3, 2)
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]).unwrap();
        let b = Var::<f64>::new(&[
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ]).unwrap();

        let c = a.matmul(&b).unwrap(); // (2, 2)
        let grads = c.backward().unwrap();

        // c = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // c = [[58, 64], [139, 154]]
        
        // dc/da = grad(2,2) * b^T(2,3) = [[1,1],[1,1]] * [[7,9,11],[8,10,12]]
        // dc/da = [[15, 19, 23], [15, 19, 23]]
        let expected_a_grad = Tensor::new(&[
            [15.0, 19.0, 23.0],
            [15.0, 19.0, 23.0]
        ]).unwrap();
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8));

        // dc/db = a^T(3,2) * grad(2,2) = [[1,4],[2,5],[3,6]] * [[1,1],[1,1]]
        // dc/db = [[5, 5], [7, 7], [9, 9]]
        let expected_b_grad = Tensor::new(&[
            [5.0, 5.0],
            [7.0, 7.0],
            [9.0, 9.0]
        ]).unwrap();
        assert!(grads[&b].allclose(&expected_b_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_matmul_vector() {
        let w = Var::<f64>::new(&[
            [0.5, -0.5],
            [0.1, 0.8]
        ]).unwrap();
        let x = Var::<f64>::new(&[
            [1.0],
            [2.0]
        ]).unwrap();

        let y = w.matmul(&x).unwrap();
        let grads = y.backward().unwrap();

        // dy/dw = grad * x^T = [[1],[1]] * [[1, 2]] = [[1, 2], [1, 2]]
        let expected_w_grad = Tensor::new(&[
            [1.0, 2.0],
            [1.0, 2.0]
        ]).unwrap();
        assert!(grads[&w].allclose(&expected_w_grad, 1e-5, 1e-8));

        // dy/dx = w^T * grad = [[0.5, 0.1], [-0.5, 0.8]] * [[1], [1]] = [[0.6], [0.3]]
        let expected_x_grad = Tensor::new(&[
            [0.6],
            [0.3]
        ]).unwrap();
        assert!(grads[&x].allclose(&expected_x_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_batched_matmul() {
        // A: (2, 2, 2), B: (2, 2, 2)
        let a = Var::<f64>::new(&[
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 0.5], [0.5, 0.5]]
        ]).unwrap();
        let b = Var::<f64>::new(&[
            [[1.0, 0.0], [0.0, 1.0]], 
            [[2.0, 2.0], [2.0, 2.0]]   
        ]).unwrap();

        let c = a.matmul(&b).unwrap();
        let grads = c.backward().unwrap();

        let expected_a_grad = Tensor::new(&[
            [[1.0, 1.0], [1.0, 1.0]],
            [[4.0, 4.0], [4.0, 4.0]]
        ]).unwrap();
        
        assert!(grads[&a].allclose(&expected_a_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_matmul_chain_with_add() {
        let w = Var::<f64>::new(&[[2.0, 3.0]]).unwrap(); // (1, 2)
        let x = Var::<f64>::new(&[[4.0], [5.0]]).unwrap(); // (2, 1)
        let b = Var::<f64>::new(&[[10.0]]).unwrap(); // (1, 1)

        let y = w.matmul(&x).unwrap().add(&b).unwrap();
        let grads = y.backward().unwrap();

        // dy/dw = x^T = [[4, 5]]
        assert!(grads[&w].allclose(&Tensor::new(&[[4.0, 5.0]]).unwrap(), 1e-5, 1e-8));
        // dy/dx = w^T = [[2], [3]]
        assert!(grads[&x].allclose(&Tensor::new(&[[2.0], [3.0]]).unwrap(), 1e-5, 1e-8));
        // dy/db = 1
        assert!(grads[&b].allclose(&Tensor::new(&[[1.0]]).unwrap(), 1e-5, 1e-8));
    }

    #[test]
    fn test_sum_dim_0() {
        // A: (2, 2)
        // [ [1.0, 2.0],
        //   [3.0, 4.0] ]
        let a = Var::<f64>::new(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();

        let s = a.sum_keepdim(0).unwrap(); 
        let grads = s.backward().unwrap();

        let expected_grad = Tensor::new(&[[1.0, 1.0], [1.0, 1.0]]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_sum_dim_1() {
        // A: (2, 3)
        let a = Var::<f64>::new(&[
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ]).unwrap();
        
        let s = a.sum_keepdim(1).unwrap(); 
        let grads = s.backward().unwrap();

        let expected_grad = Tensor::ones_like(&a).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_max_dim_0() {
        // A: (2, 2)
        // [ [10.0, 2.0],
        //   [5.0,  8.0] ]
        let a = Var::<f64>::new(&[[10.0, 2.0], [5.0, 8.0]]).unwrap();
        
        // m = a.max(0) => [10.0, 8.0]
        let m = a.max_keepdim(0).unwrap();
        let grads = m.backward().unwrap();

        let expected_grad = Tensor::new(&[
            [1.0, 0.0],
            [0.0, 1.0]
        ]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_min_dim_1() {
        // A: (2, 3)
        // [ [1.0, 5.0, 0.0],
        //   [4.0, 2.0, 6.0] ]
        let a = Var::<f64>::new(&[
            [1.0, 5.0, 0.0],
            [4.0, 2.0, 6.0]
        ]).unwrap();
        
        // m = a.min(1) => [0.0, 2.0]
        let m = a.min_keepdim(1).unwrap();
        let grads = m.backward().unwrap();

        let expected_grad = Tensor::new(&[
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_mean_dim_0() {
        // A: (2, 2)
        // [ [1.0, 2.0],
        //   [3.0, 4.0] ]
        let a = Var::<f64>::new(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();

        // mean(0) => [ (1+3)/2, (2+4)/2 ] = [2.0, 3.0]
        let s = a.mean_keepdim(0).unwrap(); 
        let grads = s.backward().unwrap();

        let expected_grad = Tensor::new(&[[0.5, 0.5], [0.5, 0.5]]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_var_dim_0() {
        // A: (2, 2)
        // [ [10.0, 2.0],
        //   [20.0, 8.0] ]
        let a = Var::<f64>::new(&[[10.0, 2.0], [20.0, 8.0]]).unwrap();
        println!("{}", a.requires_grad());
        
        // dim 0: mean = [15.0, 5.0]
        // var = [ ((10-15)^2 + (20-15)^2)/2, ((2-5)^2 + (8-5)^2)/2 ]
        // var = [ (25+25)/2, (9+9)/2 ] = [25.0, 9.0]
        let v = a.var_keepdim(0).unwrap();
        let grads = v.backward().unwrap();

        // a[0,0]=10: (2/2) * (10 - 15) = -5.0
        // a[1,0]=20: (2/2) * (20 - 15) = 5.0
        // a[0,1]=2:  (2/2) * (2 - 5)   = -3.0
        // a[1,1]=8:  (2/2) * (8 - 5)   = 3.0
        let expected_grad = Tensor::new(&[
            [-5.0, -3.0],
            [ 5.0,  3.0]
        ]).unwrap();
        
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_max_with_ties() {
        // A: [5.0, 5.0, 2.0]
        let a = Var::<f64>::new(&[5.0, 5.0, 2.0]).unwrap();
        
        // max(0) => 5.0
        let m = a.max_keepdim(0).unwrap();
        let grads = m.backward().unwrap();

        let expected_grad = Tensor::new(&[1.0, 1.0, 0.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_reduce_with_matmul() {
        // y = sum( A(2,2) @ x(2,1) )
        let a = Var::<f64>::new(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let x = Var::<f64>::new(&[[10.0], [100.0]]).unwrap();
        
        // res = [[210], [430]]
        let res = a.matmul(&x).unwrap();
        // y = 210 + 430 = 640
        let y = res.sum_keepdim(0).unwrap(); 
        
        let grads = y.backward().unwrap();

        // dy/dres = [1, 1]
        // dy/dx = A^T @ [1, 1]^T = [[1, 3], [2, 4]] @ [[1], [1]] = [[4], [6]]
        let expected_x_grad = Tensor::new(&[[4.0], [6.0]]).unwrap();
        assert!(grads[&x].allclose(&expected_x_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_reshape_backward() {
        // A: (2, 3)
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]).unwrap();

        // Reshape to (3, 2)
        let b = a.reshape(&[3, 2]).unwrap();
        
        let loss = b.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let grads = loss.backward().unwrap();

        let expected_grad = Tensor::new(&[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ]).unwrap();
        
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
        assert_eq!(grads[&a].dims(), &[2, 3]);
    }

    #[test]
    fn test_transpose_backward() {
        // A: (2, 3)
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]).unwrap();

        // Transpose (0, 1) -> (3, 2)
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        let b = a.transpose(0, 1).unwrap();

        let w = Tensor::new(&[
            [1.0, 10.0],
            [100.0, 1000.0],
            [10000.0, 100000.0]
        ]).unwrap();
        
        let loss = (b.mul(&w).unwrap()).sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let grads = loss.backward().unwrap();

        let expected_grad = Tensor::new(&[
            [1.0, 100.0, 10000.0],
            [10.0, 1000.0, 100000.0]
        ]).unwrap();

        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_narrow_backward() {
        // A: (3, 3)
        let a = Var::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]).unwrap();

        // Narrow: dim 0, start 1, len 1 (取中间一行: [4.0, 5.0, 6.0])
        let b = a.narrow(0, 1, 1).unwrap();
        
        let loss = b.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let grads = loss.backward().unwrap();

        let expected_grad = Tensor::new(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0]
        ]).unwrap();

        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));

        let c = a.narrow(1, 2, 1).unwrap(); // 取最后一列
        let grads_c = c.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap().backward().unwrap();
        let expected_grad_c = Tensor::new(&[
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ]).unwrap();
        assert!(grads_c[&a].allclose(&expected_grad_c, 1e-5, 1e-8));
    }

    #[test]
    fn test_cat_backward() {
        // A: (2, 2), B: (2, 1)
        let a = Var::<f64>::new(&[
            [1.0, 2.0],
            [3.0, 4.0]
        ]).unwrap();
        let b = Var::<f64>::new(&[
            [5.0],
            [6.0]
        ]).unwrap();

        // Cat along dim 1 -> Result: (2, 3)
        // [[1, 2, 5],
        //  [3, 4, 6]]
        let c = Tensor::cat(&[&a, &b], 1).unwrap();
        
        let w = Tensor::new(&[
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ]).unwrap();
        
        let loss = c.mul(&w).unwrap().sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let grads = loss.backward().unwrap();

        let expected_grad_a = Tensor::new(&[
            [1.0, 2.0],
            [1.0, 2.0]
        ]).unwrap();
        let expected_grad_b = Tensor::new(&[
            [3.0],
            [3.0]
        ]).unwrap();

        assert!(grads[&a].allclose(&expected_grad_a, 1e-5, 1e-8));
        assert!(grads[&b].allclose(&expected_grad_b, 1e-5, 1e-8));
    }

    #[test]
    fn test_complex_combined_ops() {
        // Reshape -> Transpose -> Narrow
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap(); // (4,)
        
        let b = a.reshape(&[2, 2]).unwrap(); // [[1,2],[3,4]]
        let c = b.transpose(0, 1).unwrap();  // [[1,3],[2,4]]
        let d = c.narrow(0, 1, 1).unwrap();  // [[2,4]] (取第二行)
        
        let loss = d.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let grads = loss.backward().unwrap();

        // d = [[2, 4]] -> grad_d = [[1, 1]]
        // c = [[1, 3], [2, 4]] -> grad_c = [[0, 0], [1, 1]]
        // b = [[1, 2], [3, 4]] -> grad_b = grad_c.T = [[0, 1], [0, 1]]
        // a = [1, 2, 3, 4] -> grad_a = [0, 1, 0, 1]
        
        let expected_grad_a = Tensor::new(&[0.0, 1.0, 0.0, 1.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad_a, 1e-5, 1e-8));
    }

    #[test]
    fn test_permute_backward() {
        // [ [[1.0], [2.0], [3.0]],
        //   [[4.0], [5.0], [6.0]] ]
        let a = Var::<f64>::new(&[
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]]
        ]).unwrap();

        // Permute (0, 1, 2) -> (2, 0, 1)
        let b = a.permute(vec![2, 0, 1]).unwrap();
        assert_eq!(b.dims(), &[1, 2, 3]);

        let w = Tensor::new(&[[
            [1.0, 2.0, 3.0], 
            [4.0, 5.0, 6.0]
        ]]).unwrap();
        
        // loss = sum(b * w)
        let loss = b.mul(&w).unwrap()
            .sum_keepdim(0).unwrap()
            .sum_keepdim(1).unwrap()
            .sum_keepdim(2).unwrap();
            
        let grads = loss.backward().unwrap();

        let expected_grad = Tensor::new(&[
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]]
        ]).unwrap();

        assert_eq!(grads[&a].dims(), &[2, 3, 1]);
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_permute_complex_backward() {
        // (0, 1, 2, 3) -> (0, 2, 3, 1) 
        let a = Var::<f64>::zeros(&[1, 2, 2, 2]).unwrap(); 
        
        let b = a.permute(vec![0, 2, 3, 1]).unwrap();
        
        let loss = b.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap().sum_keepdim(2).unwrap().sum_keepdim(3).unwrap();
        let grads = loss.backward().unwrap();
        
        let expected_grad = Tensor::ones(&[1, 2, 2, 2]).unwrap();
        assert_eq!(grads[&a].dims(), &[1, 2, 2, 2]);
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_pow_backward() {
        // a = [2.0, 4.0]
        let a = Var::<f64>::new(&[2.0, 4.0]).unwrap();
        
        // b = a^3.0 = [8.0, 64.0]
        let b = a.pow(3.0);
        
        // 假设 loss = sum(b)，则 grad_b = [1.0, 1.0]
        let loss = b.sum_keepdim(0).unwrap();
        let grads = loss.backward().unwrap();

        // da = grad_b * (3.0 * a^(3.0 - 1.0))
        // da = 1.0 * (3.0 * [4.0, 16.0]) = [12.0, 48.0]
        let expected_grad = Tensor::new(&[12.0, 48.0]).unwrap();
        
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_pow_fractional_backward() {
        // 测试开根号: a^0.5
        let a = Var::<f64>::new(&[4.0, 16.0]).unwrap();
        let b = a.pow(0.5);
        
        let loss = b.sum_keepdim(0).unwrap();
        let grads = loss.backward().unwrap();

        // da = 0.5 * a^(-0.5) = 1 / (2 * sqrt(a))
        // da = [1/(2*2), 1/(2*4)] = [0.25, 0.125]
        let expected_grad = Tensor::new(&[0.25, 0.125]).unwrap();
        
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_masked_fill_grad() {
        let a = Var::<f64>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mask = Tensor::new(&[false, true, false, true]).unwrap();
        let y = a.masked_fill(&mask, f64::NEG_INFINITY).unwrap();
        
        let grads = y.backward().unwrap();

        let expected_grad = Tensor::new(&[1.0, 0.0, 1.0, 0.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_where_grad() {
        let cond = Tensor::new(&[true, false, true]).unwrap();
        let a = Var::<f64>::new(&[10.0, 10.0, 10.0]).unwrap();
        let b = Var::<f64>::new(&[20.0, 20.0, 20.0]).unwrap();
        let y = cond.if_else(&a, &b).unwrap();
        
        let grads = y.backward().unwrap(); // 默认传入全是 1.0 的梯度

        let expected_grad_a = Tensor::new(&[1.0, 0.0, 1.0]).unwrap();
        assert!(grads[&a].allclose(&expected_grad_a, 1e-5, 1e-8));

        let expected_grad_b = Tensor::new(&[0.0, 1.0, 0.0]).unwrap();
        assert!(grads[&b].allclose(&expected_grad_b, 1e-5, 1e-8));
    }

    #[test]
    fn test_squeeze_backward() {
        // A: (1, 2, 1) -> [[ [1.0], [2.0] ]]
        let a = Var::<f64>::new(&[[[1.0], [2.0]]]).unwrap();
        
        let b = a.squeeze(0).unwrap();
        assert_eq!(b.dims(), &[2, 1]);

        let c = b.squeeze(1).unwrap();
        assert_eq!(c.dims(), &[2]);

        let loss = c.sum_keepdim(0).unwrap();
        let grads = loss.backward().unwrap();

        let grad_a = &grads[&a];
        assert_eq!(grad_a.dims(), &[1, 2, 1]);
        
        let expected_grad = Tensor::new(&[[[1.0], [1.0]]]).unwrap();
        assert!(grad_a.allclose(&expected_grad, 1e-5, 1e-8));
    }

    #[test]
    fn test_unsqueeze_backward() {
        // A: (2) -> [10.0, 20.0]
        let a = Var::<f64>::new(&[10.0, 20.0]).unwrap();
        
        // 1. Unsqueeze 维度 0 -> (1, 2)
        let b = a.unsqueeze(0).unwrap();
        assert_eq!(b.dims(), &[1, 2]);

        // 2. Unsqueeze 维度 2 -> (1, 2, 1)
        let c = b.unsqueeze(2).unwrap();
        assert_eq!(c.dims(), &[1, 2, 1]);

        // c * [[ [2.0], [3.0] ]] = [[ [20.0], [60.0] ]]
        let weights = Tensor::new(&[[[2.0], [3.0]]]).unwrap();
        let out = c.mul(&weights).unwrap();
        let loss = out.sum_keepdim(0).unwrap();
        
        let grads = loss.backward().unwrap();

        let grad_a = &grads[&a];
        assert_eq!(grad_a.dims(), &[2]);
        
        let expected_grad = Tensor::new(&[2.0, 3.0]).unwrap();
        assert!(grad_a.allclose(&expected_grad, 1e-5, 1e-8));
    }
}