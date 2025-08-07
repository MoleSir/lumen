use crate::rng;

use crate::{Range, Tensor};
use crate::utils;

/// Backward for matmul
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build(vec![1., 2., 3., 2., 4., 7.], [2, 3]).unwrap().require_grad();
/// let b = Tensor::build(vec![3., 4., 5., 6., 0., 1.], [3, 2]).unwrap().require_grad();
/// let y = op::matmul(&a, &b).unwrap();
/// y.backward();
/// let grad_a = a.grad().unwrap();
/// assert!(grad_a.allclose(&Tensor::build(vec![7., 11.,  1., 7., 11.,  1.], [2, 3]).unwrap()));
/// let grad_b = b.grad().unwrap();
/// assert!(grad_b.allclose(&Tensor::build(vec![3.,  3., 6., 6., 10., 10.], [3, 2]).unwrap()));
/// 
/// let a = Tensor::build(vec![1., 2., 3., 2., 4., 7., 1., 2., 3., 2., 4., 7.], [2, 2, 3]).unwrap().require_grad();
/// let b = Tensor::build(vec![3., 4., 5., 6., 0., 1., 3., 4., 5., 6., 0., 1.], [2, 3, 2]).unwrap().require_grad();
/// let y = op::matmul(&a, &b).unwrap();
/// y.backward();
/// let grad_a = a.grad().unwrap();
/// assert!(grad_a.allclose(&Tensor::build(vec![7., 11., 1., 7., 11., 1., 7., 11., 1., 7., 11., 1.], [2, 2, 3]).unwrap()));
/// let grad_b = b.grad().unwrap();
/// assert!(grad_b.allclose(&Tensor::build(vec![3.,  3., 6., 6., 10., 10., 3.,  3., 6., 6., 10., 10.], [2, 3, 2]).unwrap()));
/// ```
pub(super) fn matmul_backward(y: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    if lhs.requires_grad() {
        lhs.try_init_grad();
    }
    if rhs.requires_grad() {
        rhs.try_init_grad();
    }

    /*
        p = l @ r
        dp/dl => p.g @ r.T
        dp/dr => l.T @ p.g
    */
    if y.dim_size() == 2 {
        if lhs.requires_grad() {
            utils::matmul_(&lhs.grad().unwrap(), &y.grad().unwrap(), &rhs.transpose().unwrap());
        }

        if rhs.requires_grad() {
            utils::matmul_(&rhs.grad().unwrap(), &lhs.transpose().unwrap(), &y.grad().unwrap());
        }
    } else {
        let shape1 = lhs.shape().iter().cloned().collect::<Vec<_>>();

        let high_dim_size = shape1.len() - 2;    
        for ranges in utils::generate_coordinate_ranges(&shape1[..high_dim_size]) {
            let view_pre_g = y.grad().unwrap().slice(&ranges).unwrap();

            if lhs.requires_grad() {
                let view_lhs_g = lhs.grad().unwrap().slice(&ranges).unwrap();
                let view_rhs_t = rhs.slice(&ranges).unwrap().transpose().unwrap();
                utils::matmul_(&view_lhs_g, &view_pre_g, &view_rhs_t);
            }

            if rhs.requires_grad() {
                let view_rhs_g = rhs.grad().unwrap().slice(&ranges).unwrap();
                let view_lhs_t = lhs.slice(&ranges).unwrap().transpose().unwrap();
                utils::matmul_(&view_rhs_g, &view_lhs_t, &view_pre_g);
            }
        }
    }
}

/// Backward for add
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap().require_grad();
/// let y = op::add(&a, &b).unwrap();
/// y.backward();
/// let grad_a = a.grad().unwrap();
/// assert!(grad_a.allclose(&Tensor::build([1., 1., 1., 1.], [2, 2]).unwrap()));
/// let grad_b = b.grad().unwrap();
/// assert!(grad_b.allclose(&Tensor::build([1., 1., 1., 1.], [2, 2]).unwrap()));
/// 
/// let a = Tensor::build([1., 2.], [2]).unwrap().require_grad();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap().require_grad();
/// let y = op::add(&a, &b).unwrap();
/// y.backward();
/// let grad_a = a.grad().unwrap();
/// assert!(grad_a.allclose(&Tensor::build([2., 2.], [2]).unwrap()));
/// let grad_b = b.grad().unwrap();
/// assert!(grad_b.allclose(&Tensor::build([1., 1., 1., 1.], [2, 2]).unwrap()));
/// ```
pub(super) fn add_backward(y: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    assert_eq!(*lhs.shape(), *rhs.shape());
    assert_eq!(*y.shape(), *rhs.shape());
    assert!(y.inner().grad.is_some());

    if lhs.requires_grad() {
        lhs.try_init_grad();
        for (l, p) in lhs.grad().unwrap().iter_mut().zip(y.grad().unwrap().iter()) {
            *l += *p;
        }
    }

    if rhs.requires_grad() {
        rhs.try_init_grad();
        for (r, p) in rhs.grad().unwrap().iter_mut().zip(y.grad().unwrap().iter()) {
            *r += *p;
        }
    }
}

pub(super) fn broadcast_add_backward(y: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    assert_ne!(lhs.dim_size(), rhs.dim_size());
    let (long, short, long_requires_grad, short_requires_grad) = if lhs.dim_size() > rhs.dim_size() {
        (lhs, rhs, lhs.requires_grad(), rhs.requires_grad())
    } else {
        (rhs, lhs, rhs.requires_grad(), lhs.requires_grad())
    };

    if long_requires_grad {
        long.try_init_grad();
        for (lg, yg) in long.grad().unwrap().iter_mut().zip(y.grad().unwrap().iter()) {
            *lg += *yg;
        }
    }

    if short_requires_grad {
        short.try_init_grad();
        let high_dim = y.dim_size() - short.dim_size();
        let pre_grad = y.grad().unwrap();
        for ranges in utils::generate_coordinate_ranges(&y.shape()[..high_dim]) {
            let pre_grad_view = pre_grad.slice(&ranges).unwrap();
            for (sg, g) in short.grad().unwrap().iter_mut().zip(pre_grad_view.iter()) {
                *sg += *g;
            }
        } 
    }
}

/// Backward for sub
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap().require_grad();
/// let y = op::sub(&a, &b).unwrap();
///
/// y.backward();
/// 
/// let grad_a = a.grad().unwrap();
/// assert!(grad_a.allclose(&Tensor::build([1., 1., 1., 1.], [2, 2]).unwrap()));
/// 
/// let grad_b = b.grad().unwrap();
/// assert!(grad_b.allclose(&Tensor::build([-1., -1., -1., -1.], [2, 2]).unwrap()));
/// ```
pub(super) fn sub_backward(y: &Tensor, lhs: &Tensor, rhs: &Tensor) {    
    assert_eq!(*lhs.shape(), *rhs.shape());
    assert_eq!(*y.shape(), *rhs.shape());
    assert!(y.inner().grad.is_some());

    if lhs.requires_grad() {
        lhs.try_init_grad();
        for (l, p) in lhs.grad().unwrap().iter_mut().zip(y.grad().unwrap().iter()) {
            *l += *p;
        }
    }

    if rhs.requires_grad() {
        rhs.try_init_grad();
        for (r, p) in rhs.grad().unwrap().iter_mut().zip(y.grad().unwrap().iter()) {
            *r -= *p;
        }
    }
}

/// Backward for mul
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap().require_grad();
/// let y = op::mul(&a, &b).unwrap();
/// y.backward();
/// 
/// let grad_a = a.grad().unwrap();
/// assert!(grad_a.allclose(&b));
/// 
/// let grad_b = b.grad().unwrap();
/// assert!(grad_b.allclose(&a));
/// ```
pub(super) fn mul_backward(y: &Tensor, lhs: &Tensor, rhs: &Tensor) {    
    assert_eq!(*lhs.shape(), *rhs.shape());
    assert_eq!(*y.shape(), *rhs.shape());
    assert!(y.inner().grad.is_some());

    if lhs.requires_grad() {
        lhs.try_init_grad();
        for (lg, rv, p) in utils::zip3(
            lhs.grad().unwrap().iter_mut(), 
            rhs.iter(),
            y.grad().unwrap().iter())
        {
            *lg += p * rv;
        }
    }
    if rhs.requires_grad() {
        rhs.try_init_grad();
        for (rg, lv, p) in utils::zip3(
            rhs.grad().unwrap().iter_mut(), 
            lhs.iter(),
            y.grad().unwrap().iter())
        {
            *rg += p * lv;
        }
    }
}

/// Backward for div
/// 
/// # Exmaple
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let b = Tensor::build([3., 1., 5., 7.], [2, 2]).unwrap().require_grad();
/// let y = op::div(&a, &b).unwrap();
/// 
/// y.backward();
/// 
/// let grad_a = a.grad().unwrap();
/// assert!(grad_a.allclose(&Tensor::build([0.3333333333333333, 1.0, 0.2, 0.14285714285714285], [2, 2]).unwrap()));
/// 
/// let grad_b = b.grad().unwrap();
/// assert!(grad_b.allclose(&Tensor::build([-0.1111111111111111, -2.0, -0.12, -0.08163265306122448], [2, 2]).unwrap()));
/// ```
pub(super) fn div_backward(y: &Tensor, lhs: &Tensor, rhs: &Tensor) {    
    assert_eq!(*lhs.shape(), *rhs.shape());
    assert_eq!(*y.shape(), *rhs.shape());
    assert!(y.inner().grad.is_some());
    
    if lhs.requires_grad() {
        lhs.try_init_grad();
    }
    if rhs.requires_grad() {
        rhs.try_init_grad();
    }

    for (lv, lg, rv, rg, p) in utils::zip5(
        lhs.iter(), lhs.grad().unwrap().iter_mut(), 
        rhs.iter(), rhs.grad().unwrap().iter_mut(), 
        y.grad().unwrap().iter())
    {
        if lhs.requires_grad() {
            *lg += p * (1. / rv);
        }
        if rhs.requires_grad() {
            *rg += p * (-lv * rv.powf(-2.));
        }
    }
}

/// Backward for exp
/// 
/// # Exmaple 
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let y = x.exp();
/// y.backward();
/// let grad_x = x.grad().unwrap();
/// assert!(grad_x.allclose(&Tensor::build(vec![2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236], [2, 2]).unwrap()));
/// ```
pub(super) fn exp_backward(y: &Tensor, x: &Tensor) {
    assert_eq!(*y.shape(), *x.shape());

    // y = e^x
    // dy/dx = e^x = y
    x.try_init_grad();
    for (xg, yv, yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        y.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += *yv * *yg;
    }
}

/// Backward for pow
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let y = x.pow(1.5);
/// y.backward();
/// let grad_x = x.grad().unwrap();
/// assert!(grad_x.allclose(&Tensor::build(vec![1.5, 2.121320343559643, 2.598076211353316, 3.0], [2, 2]).unwrap()));
/// ```
pub(super) fn pow_backward(y: &Tensor, x: &Tensor, e: f64) {
    assert_eq!(*y.shape(), *x.shape());
    
    // y = x**e
    // dy/dx = e * x**(e - 1)
    x.try_init_grad();
    for (xg, xv, yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        x.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += *yg * (e * xv.powf(e - 1.));
    }
}

/// Backward for tanh
/// 
/// ```rust
/// use lumen_core::*;
///
/// let x = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let y = x.tanh();
/// y.backward();
/// let grad_x = x.grad().unwrap();
/// assert!(grad_x.allclose(&Tensor::build(vec![0.41997434161402614, 0.07065082485316443, 0.009866037165440211, 0.0013409506830258655], [2, 2]).unwrap()));
/// 
/// let x = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let y = x.tanh().tanh();
/// y.backward();
/// let grad_x = x.grad().unwrap();
/// assert!(grad_x.allclose(&Tensor::build(vec![0.24686795258431515, 0.031325342296270944, 0.004174768274961411, 0.0005637403984995298], [2, 2]).unwrap()));
/// ```
pub(super) fn tanh_backward(y: &Tensor, x: &Tensor) {
    assert_eq!(*y.shape(), *x.shape());

    // y = x.tanh()
    // dy/dx = 1 - y^2
    x.try_init_grad();
    for (xg, &y, &yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        y.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += yg * (1. - y.powf(2.));
    }
}

/// Backward for sigmoid
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap().require_grad();
/// let y = x.sigmoid();
/// y.backward();
/// let grad_x = x.grad().unwrap();
/// assert!(grad_x.allclose(&Tensor::build(vec![0.19661193324148185, 0.10499358540350662, 0.045176659730912, 0.017662706213291107], [2, 2]).unwrap()));
/// ```
pub(super) fn sigmoid_backward(y: &Tensor, x: &Tensor) {
    assert_eq!(*y.shape(), *x.shape());

    x.try_init_grad();
    for (xg, &y, &yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        y.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += yg * (y * (1. - y));
    }
}

pub(super) fn abs_backward(y: &Tensor, x: &Tensor) {
    assert_eq!(*y.shape(), *x.shape());

    x.try_init_grad();
    for (xg, &x, &yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        x.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += match x {
            x if x > 0. => yg,
            x if x < 0. => -yg,
            _ => 0.
        };
    }
}

pub(super) fn log_backward(y: &Tensor, x: &Tensor, base: f64) {
    assert_eq!(*y.shape(), *x.shape());

    // y = log_{base}(y)
    // dy/dx = 1. / (x * ln(base))
    x.try_init_grad();
    for (xg, &x, &yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        x.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += yg * (1. / (x * base.ln()));
    }
}

pub(super) fn relu_backward(y: &Tensor, x: &Tensor) {
    assert_eq!(*y.shape(), *x.shape());

    x.try_init_grad();
    for (xg, &x, &yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        x.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += yg * if x > 0. { yg } else { 0. };
    }
}

pub(super) fn sin_backward(y: &Tensor, x: &Tensor) {
    assert_eq!(*y.shape(), *x.shape());

    x.try_init_grad();
    for (xg, &x, &yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        x.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += yg * x.cos();
    }
}

pub(super) fn cos_backward(y: &Tensor, x: &Tensor) {
    assert_eq!(*y.shape(), *x.shape());

    x.try_init_grad();
    for (xg, &x, &yg) in utils::zip3(
        x.grad().unwrap().iter_mut(),
        x.iter(),
        y.grad().unwrap().iter())
    {   
        *xg += yg * -x.sin();
    }
}

/// Backward for stack
/// 
/// All tensors need to be of the same size.
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*; 
/// 
/// let a = Tensor::build([1., 2., 1., 2.], [2, 2]).unwrap().require_grad();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap().require_grad();
/// let y1 = &a + &b;
/// let y2 = &a * &b;
/// assert!(y1.requires_grad());
/// let y = op::stack(&[y1.clone(), y2.clone()]).unwrap();
/// y.backward();
/// assert!(a.grad().unwrap().allclose(&Tensor::build(vec![4., 2., 6., 1.], [2, 2]).unwrap()));
/// assert!(b.grad().unwrap().allclose(&Tensor::build(vec![2., 3., 2., 3.], [2, 2]).unwrap()));
/// 
/// let a = Tensor::build([1., 2., 1., 2.], [2, 2]).unwrap().require_grad();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap().require_grad();
/// let y1 = &a + &b;
/// let y2 = &a * &b;
/// assert!(y1.requires_grad());
/// let y = op::stack(&[y1.clone(), y2.clone()]).unwrap() + &a;
/// y.backward();
/// assert!(a.grad().unwrap().allclose(&Tensor::build(vec![6., 4., 8., 3.], [2, 2]).unwrap()));
/// assert!(b.grad().unwrap().allclose(&Tensor::build(vec![2., 3., 2., 3.], [2, 2]).unwrap()));
/// ```
pub(super) fn stack_backward(y: &Tensor, xs: &[Tensor]) {
    // xs have same shape!
    let sub_tensor_size = y.shape()[0];
    let grad = y.grad().unwrap();
    for i in 0..sub_tensor_size {
        let sub_grad = grad.slice(&[rng!(i)]).unwrap();
        let x = &xs[i];
        x.try_init_grad();
        for (xg, &yg) in x.grad().unwrap().iter_mut().zip(sub_grad.iter()) {
            *xg = yg;
        }
    }
}

/// Backward for view
/// 
/// ```
/// use lumen_core::*; 
/// let a = Tensor::from([[1, 2], [1, 2]]).require_grad();
/// let b = Tensor::from([[3, 1], [5, 0]]).require_grad();
/// let y1 = &a + &b;
/// let y2 = &a * &b;
/// let y = op::stack(&[y1.clone(), y2.clone()]).unwrap();
/// let z = y.view([8]).unwrap();
/// z.backward();
/// assert!(a.grad().unwrap().allclose(&Tensor::from([[4, 2], [6, 1]])));
/// assert!(b.grad().unwrap().allclose(&Tensor::from([[2, 3], [2, 3]])));
/// ```
pub(super) fn view_backward(y: &Tensor, x: &Tensor) {
    // y = x.view(...)
    x.try_init_grad();
    for (xg, &yg) in x.grad().unwrap().iter_mut().zip(y.grad().unwrap().iter()) {
        *xg = yg;
    }
}

/// Backward for slice
/// 
/// ```
/// use lumen_core::*; 
/// let a = Tensor::from([[1, 2], [1, 2]]).require_grad();
/// let b = Tensor::from([[3, 1], [5, 0]]).require_grad();
/// let y1 = &a + &b;
/// let y2 = &a * &b;
/// let y = op::stack(&[y1.clone(), y2.clone()]).unwrap();
/// let z = y.slice(rngs!(1)).unwrap();
/// z.backward();
/// assert!(a.grad().unwrap().allclose(&Tensor::from([[3, 1], [5, 0]])));
/// assert!(b.grad().unwrap().allclose(&Tensor::from([[1, 2], [1, 2]])));
/// let z = y.slice(rngs!(0)).unwrap();
/// z.backward();
/// assert!(a.grad().unwrap().allclose(&Tensor::from([[4, 2], [6, 1]])));
/// assert!(b.grad().unwrap().allclose(&Tensor::from([[2, 3], [2, 3]])));
/// ```
pub(super) fn slice_backward(y: &Tensor, x: &Tensor, ranges: &[Range]) {
    // y = x.slice(ranges);
    x.try_init_grad();
    x.zero_grad();
    let x_slice_g = x.grad().unwrap().slice(ranges).unwrap();
    let y_g = y.grad().unwrap();
    assert_eq!(*x_slice_g.shape(), *y_g.shape());
    for (xg, yg) in x_slice_g.iter_mut().zip(y_g.iter()) {
        *xg = *yg;
    }
}