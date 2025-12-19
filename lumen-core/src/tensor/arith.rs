use crate::{op::BinaryOp, AutogradMetaT, CmpOp, Error, FloatDType, NumDType, Result, Shape, Storage, UnaryOp, WithDType};
use super::Tensor;
use paste::paste;

//////////////////////////////////////////////////////////////////////////////
///        Binary(Assign) Op with Tensor and Tensor / scalar
//////////////////////////////////////////////////////////////////////////////

impl<T: WithDType> Tensor<T> {
    fn same_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<&Shape> {
        let lhs = self.shape();
        let rhs = rhs.shape();
        if lhs != rhs {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op,
            })
        } else {
            Ok(lhs)
        }
    }
}

impl<T: WithDType> Tensor<T> {
    fn compute_binary_op<U, F>(lhs: &Tensor<T>, rhs: &Tensor<T>, mut f: F, op_name: &'static str) -> Result<(Storage<U>, Shape)>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let shape = Tensor::<T>::same_shape_binary_op(lhs, rhs, op_name)?;
        let lhs_storage = lhs.storage();
        let rhs_storage = rhs.storage();
        let lhs_layout = lhs.layout();
        let rhs_layout = rhs.layout();

        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");

        let lhs = lhs_storage.data();
        let rhs = rhs_storage.data();
        
        let output: Vec<_> = lhs_layout.storage_indices().zip(rhs_layout.storage_indices())
            .map(|(lhs_index, rhs_index)| f(lhs[lhs_index], rhs[rhs_index]))
            .collect();
        
        let storage = Storage::<U>::new(output);
        Ok((storage, shape.clone()))
    }

    fn binary_op<U, F>(lhs: &Tensor<T>, rhs: &Tensor<T>, f: F, op_name: &'static str) -> Result<Tensor<U>>
    where 
        U: WithDType, 
        F: FnMut(T, T) -> U 
    {
        let (storage, shape) = Self::compute_binary_op(lhs, rhs, f, op_name)?;
        Ok(Tensor::<U>::from_storage(storage, shape))
    }

    fn assign_op<F>(lhs: &Tensor<T>, rhs: &Tensor<T>, mut f: F, op_name: &'static str) -> Result<()>
        where 
            F: FnMut(T, T) -> T
    {
        Tensor::<T>::same_shape_binary_op(lhs, rhs, op_name)?;

        let mut lhs_storage = lhs.0.storage.write();
        let rhs_storage = rhs.storage();
        let lhs_layout = lhs.layout();
        let rhs_layout = rhs.layout();

        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");
        let lhs = lhs_storage.data_mut();
        let rhs = rhs_storage.data();
        
        for (lhs_index, rhs_index) in lhs_layout.storage_indices().zip(rhs_layout.storage_indices()) {
            lhs[lhs_index] = f(lhs[lhs_index], rhs[rhs_index]) ;
        }

        Ok(())    
    }
}

macro_rules! binary_op_impl {
    ($fn_name:ident) => {
        paste! {
            pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
                let (storage, shape) = Self::compute_binary_op(self, rhs, T::$fn_name, stringify!(fn_name))?;
                let meta = T::AutogradMeta::on_binary_op(self, rhs, BinaryOp:: [< $fn_name:camel >]);
                Ok(Self::from_op(storage, shape, meta))
            }
        }
    };
}

impl<T: NumDType> Tensor<T> {
    binary_op_impl!(add);
    binary_op_impl!(mul);
    binary_op_impl!(sub);
    binary_op_impl!(div);
    binary_op_impl!(minimum);
    binary_op_impl!(maximum);
}

macro_rules! assign_op_impl {
    ($fn_name:ident, $op:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<()> {
            Self::assign_op(self, rhs, T::$op, stringify!(fn_name))
        }
    };
}

impl<T: NumDType> Tensor<T> {
    assign_op_impl!(add_assign, add);
    assign_op_impl!(mul_assign, mul);
    assign_op_impl!(sub_assign, sub);
    assign_op_impl!(div_assign, div);
    assign_op_impl!(minimum_assign, minimum);
    assign_op_impl!(maximum_assign, maximum);
}

impl<T: NumDType> Tensor<T> {
    pub fn eq(&self, rhs: &Self) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Eq)
    }

    pub fn ne(&self, rhs: &Self) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Ne)
    }

    pub fn le(&self, rhs: &Self) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Le)
    }

    pub fn ge(&self, rhs: &Self) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Ge)
    }

    pub fn lt(&self, rhs: &Self) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Lt)
    }

    pub fn gt(&self, rhs: &Self) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Gt)
    }

    pub fn cmp(&self, rhs: &Self, op: CmpOp) -> Result<Tensor<bool>> {
        match op {
            CmpOp::Eq => Self::binary_op(self, rhs, |a, b| a == b, "eq"),
            CmpOp::Ne => Self::binary_op(self, rhs, |a, b| a != b, "nq"),
            CmpOp::Le => Self::binary_op(self, rhs, |a, b| a <= b, "le"),
            CmpOp::Ge => Self::binary_op(self, rhs, |a, b| a >= b, "ge"),
            CmpOp::Lt => Self::binary_op(self, rhs, |a, b| a <  b, "lt"),
            CmpOp::Gt => Self::binary_op(self, rhs, |a, b| a >  b, "gt"),
        }
    } 
}

impl Tensor<bool> {
    pub fn and(&self, rhs: &Self) -> Result<Tensor<bool>> {
        Self::binary_op(self, rhs, |a, b| a & b, "and")
    }

    pub fn or(&self, rhs: &Self) -> Result<Tensor<bool>> {
        Self::binary_op(self, rhs, |a, b| a | b, "or")
    }

    pub fn xor(&self, rhs: &Self) -> Result<Tensor<bool>> {
        Self::binary_op(self, rhs, |a, b| a ^ b, "xor")
    }

    pub fn and_assign(&self, rhs: &Self) -> Result<()> {
        Self::assign_op(self, rhs, |a, b| a & b, "and_assign")?;
        Ok(())
    }

    pub fn or_assign(&self, rhs: &Self) -> Result<()> {
        Self::assign_op(self, rhs, |a, b| a | b, "or_assign")?;
        Ok(())
    }

    pub fn xor_assign(&self, rhs: &Self) -> Result<()> {
        Self::assign_op(self, rhs, |a, b| a ^ b, "xor_assign")?;
        Ok(())
    }

    pub fn not(&self) -> Tensor<bool> {
        if self.element_count() == 0 {
            return self.clone();
        }
        let storage = self.compute_unary_op(|v| !v);
        Self::from_storage(storage, self.shape())
    }
}

//////////////////////////////////////////////////////////////////////////////
///        Unary Op / Unary Assign Op  for Tensor
//////////////////////////////////////////////////////////////////////////////

impl<T: WithDType> Tensor<T> {
    fn compute_unary_op<U, F>(&self, mut f: F) -> Storage<U> 
    where
        U: WithDType,
        F: FnMut(T) -> U
    {
        let storage = self.storage();
        let vec = storage.data();
        let mut output = vec![];
        for index in self.layout().storage_indices() {
            output.push( f(vec[index]) );
        }
        
        Storage::new(output)
    }

    fn unary_assign_op<F>(&self, mut f: F) 
    where
        F: FnMut(T) -> T
    {
        let mut storage = self.0.storage.write();
        let vec = storage.data_mut();
        for index in self.layout().storage_indices() {
            vec[index] = f(vec[index]);
        }
    }
}

impl<T: NumDType> Tensor<T> {
    pub fn affine(&self, mul: T, add: T) -> Result<Self> {
        if self.element_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.compute_unary_op(|v| v * mul + add);
        Ok(Self::from_storage(storage, self.shape()))
    }

    pub fn affine_assign(&self, mul: T, add: T) -> Result<()> {
        if self.element_count() == 0 {
            return Ok(());
        }
        self.unary_assign_op(|v| v * mul + add);
        Ok(())
    }
}

macro_rules! float_unary_op_impl {
    ($fn_name:ident) => {
        paste! {
            pub fn $fn_name(&self) -> Self {
                if self.element_count() == 0 {
                    return self.clone();
                }
                let storage = self.compute_unary_op(F::$fn_name);
                let meta = F::AutogradMeta::on_unray_op(self, UnaryOp:: [< $fn_name:camel >]);
                Self::from_op(storage, self.shape(), meta)
            }
        }
    };
}

macro_rules! float_unary_assign_op_impl {
    ($fn_name:ident, $op:ident) => {
        pub fn $fn_name(&self) {
            if self.element_count() == 0 {
                return;
            }
            self.unary_assign_op(F::$op);
        }
    };
}

impl<T: WithDType> Tensor<T> {
    pub fn map<F, O>(&self, f: F) -> Tensor<O>
    where 
        O: WithDType,
        F: Fn(T) -> O,
    {
        let storage = self.compute_unary_op(f);
        Tensor::from_storage(storage, self.shape())
    }

    pub fn map_assign<F>(&self, f: F)
    where 
        F: Fn(T) -> T,
    {
        if self.element_count() == 0 {
            return;
        }
        self.unary_assign_op(f);
    }
}

impl<F: FloatDType> Tensor<F> {
    float_unary_op_impl!(exp);
    float_unary_op_impl!(sin);
    float_unary_op_impl!(cos);
    float_unary_op_impl!(tanh);
    float_unary_op_impl!(sqrt);
    float_unary_op_impl!(floor);
    float_unary_op_impl!(ceil);
    float_unary_op_impl!(round);
    float_unary_op_impl!(abs);
    float_unary_op_impl!(neg);
    float_unary_op_impl!(ln);
    float_unary_op_impl!(recip);

    float_unary_assign_op_impl!(exp_assign, exp);
    float_unary_assign_op_impl!(sin_assign, sin);
    float_unary_assign_op_impl!(cos_assign, cos);
    float_unary_assign_op_impl!(sqrt_assign, sqrt);
    float_unary_assign_op_impl!(tanh_assign, tanh);
    float_unary_assign_op_impl!(floor_assign, floor);
    float_unary_assign_op_impl!(ceil_assign, ceil);
    float_unary_assign_op_impl!(round_assign, round);
    float_unary_assign_op_impl!(abs_assign, abs);
    float_unary_assign_op_impl!(neg_assign, neg);
    float_unary_assign_op_impl!(ln_assign, ln);
    float_unary_assign_op_impl!(recip_assign, recip);
}

// use std::ops::{Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

// //////////////////////////////////////////////////////////////////////////////
// ///        Add
// //////////////////////////////////////////////////////////////////////////////

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> Add<R> for &Tensor<T> {
//     type Output = Tensor<T>;
//     fn add(self, rhs: R) -> Self::Output {
//         Tensor::add(self, rhs).unwrap()
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> Add<R> for Tensor<T> {
//     type Output = Tensor<T>;
//     fn add(self, rhs: R) -> Self::Output {
//         Tensor::add(&self, rhs).unwrap()
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> AddAssign<R> for Tensor<T> {
//     fn add_assign(&mut self, rhs: R) {
//         Tensor::add_assign(self, rhs).unwrap();
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> AddAssign<R> for &Tensor<T> {
//     fn add_assign(&mut self, rhs: R) {
//         Tensor::add_assign(self, rhs).unwrap();
//     }
// }

// //////////////////////////////////////////////////////////////////////////////
// ///        Sub
// //////////////////////////////////////////////////////////////////////////////

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> Sub<R> for &Tensor<T> {
//     type Output = Tensor<T>;
//     fn sub(self, rhs: R) -> Self::Output {
//         Tensor::sub(self, rhs).unwrap()
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> Sub<R> for Tensor<T> {
//     type Output = Tensor<T>;
//     fn sub(self, rhs: R) -> Self::Output {
//         Tensor::sub(&self, rhs).unwrap()
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> SubAssign<R> for Tensor<T> {
//     fn sub_assign(&mut self, rhs: R) {
//         Tensor::sub_assign(self, rhs).unwrap();
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> SubAssign<R> for &Tensor<T> {
//     fn sub_assign(&mut self, rhs: R) {
//         Tensor::sub_assign(self, rhs).unwrap();
//     }
// }

// //////////////////////////////////////////////////////////////////////////////
// ///        Mul
// //////////////////////////////////////////////////////////////////////////////

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> Mul<R> for &Tensor<T> {
//     type Output = Tensor<T>;
//     fn mul(self, rhs: R) -> Self::Output {
//         Tensor::mul(self, rhs).unwrap()
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> Mul<R> for Tensor<T> {
//     type Output = Tensor<T>;
//     fn mul(self, rhs: R) -> Self::Output {
//         Tensor::mul(&self, rhs).unwrap()
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> MulAssign<R> for Tensor<T> {
//     fn mul_assign(&mut self, rhs: R) {
//         Tensor::mul_assign(self, rhs).unwrap();
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> MulAssign<R> for &Tensor<T> {
//     fn mul_assign(&mut self, rhs: R) {
//         Tensor::mul_assign(self, rhs).unwrap();
//     }
// }

// //////////////////////////////////////////////////////////////////////////////
// ///        Div
// //////////////////////////////////////////////////////////////////////////////

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> Div<R> for &Tensor<T> {
//     type Output = Tensor<T>;
//     fn div(self, rhs: R) -> Self::Output {
//         Tensor::div(self, rhs).unwrap()
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> Div<R> for Tensor<T> {
//     type Output = Tensor<T>;
//     fn div(self, rhs: R) -> Self::Output {
//         Tensor::div(&self, rhs).unwrap()
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> DivAssign<R> for Tensor<T> {
//     fn div_assign(&mut self, rhs: R) {
//         Tensor::div_assign(self, rhs).unwrap();
//     }
// }

// impl<T: NumDType, R: TensorBinaryOpRhs<T>> DivAssign<R> for &Tensor<T> {
//     fn div_assign(&mut self, rhs: R) {
//         Tensor::div_assign(self, rhs).unwrap();
//     }
// }

// //////////////////////////////////////////////////////////////////////////////
// ///        Bool
// //////////////////////////////////////////////////////////////////////////////

// impl<R: TensorBinaryOpRhs<bool>> BitAnd<R> for &Tensor<bool> {
//     type Output = Tensor<bool>;
//     fn bitand(self, rhs: R) -> Self::Output {
//         self.and(rhs).unwrap()
//     }
// }

// impl<R: TensorBinaryOpRhs<bool>> BitAnd<R> for Tensor<bool> {
//     type Output = Tensor<bool>;
//     fn bitand(self, rhs: R) -> Self::Output {
//         self.and(rhs).unwrap()
//     }
// }

// impl<R: TensorBinaryOpRhs<bool>> BitOr<R> for &Tensor<bool> {
//     type Output = Tensor<bool>;
//     fn bitor(self, rhs: R) -> Self::Output {
//         self.or(rhs).unwrap()
//     }
// }

// impl<R: TensorBinaryOpRhs<bool>> BitOr<R> for Tensor<bool> {
//     type Output = Tensor<bool>;
//     fn bitor(self, rhs: R) -> Self::Output {
//         self.or(rhs).unwrap()
//     }
// }

// impl<R: TensorBinaryOpRhs<bool>> BitXor<R> for &Tensor<bool> {
//     type Output = Tensor<bool>;
//     fn bitxor(self, rhs: R) -> Self::Output {
//         self.xor(rhs).unwrap()
//     }
// }

// impl<R: TensorBinaryOpRhs<bool>> BitXor<R> for Tensor<bool> {
//     type Output = Tensor<bool>;
//     fn bitxor(self, rhs: R) -> Self::Output {
//         self.xor(rhs).unwrap()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_log() {
        let a = Tensor::new(&[0.0f32, 1.0, 2.0]).unwrap();
        let exp_a = a.exp();
        let log_a = exp_a.ln();
        assert!(a.allclose(&log_a, 1e-5, 1e-8));
    }

    #[test]
    fn test_trig() {
        let a = Tensor::new(&[0.0f32, std::f32::consts::FRAC_PI_2]).unwrap();
        let sin_a = a.sin();
        let cos_a = a.cos();

        let expected_sin = Tensor::new(&[0.0f32, 1.0]).unwrap();
        let expected_cos = Tensor::new(&[1.0f32, 0.0]).unwrap();

        println!("{:?}", cos_a.iter().collect::<Vec<_>>());

        assert!(sin_a.allclose(&expected_sin, 1e-5, 1e-8));
        assert!(cos_a.allclose(&expected_cos, 1e-5, 8e-8));
    }

    #[test]
    fn test_abs_neg() {
        let a = Tensor::new(&[-1.0f32, 0.0, 2.0]).unwrap();
        let abs_a = a.abs();
        let neg_a = a.neg();

        let expected_abs = Tensor::new(&[1.0f32, 0.0, 2.0]).unwrap();
        let expected_neg = Tensor::new(&[1.0f32, 0.0, -2.0]).unwrap();

        assert!(abs_a.allclose(&expected_abs, 1e-6, 1e-6));
        assert!(neg_a.allclose(&expected_neg, 1e-6, 1e-6));
    }

    #[test]
    fn test_floor_ceil_round() {
        let a = Tensor::new(&[1.2f32, 2.7, -1.3]).unwrap();
        let floor_a = a.floor();
        let ceil_a = a.ceil();
        let round_a = a.round();

        let expected_floor = Tensor::new(&[1.0f32, 2.0, -2.0]).unwrap();
        let expected_ceil = Tensor::new(&[2.0f32, 3.0, -1.0]).unwrap();
        let expected_round = Tensor::new(&[1.0f32, 3.0, -1.0]).unwrap();

        assert!(floor_a.allclose(&expected_floor, 1e-6, 1e-6));
        assert!(ceil_a.allclose(&expected_ceil, 1e-6, 1e-6));
        assert!(round_a.allclose(&expected_round, 1e-6, 1e-6));
    }

    #[test]
    fn test_floor_recip() {
        let a = Tensor::new(&[1.2f32, 2.7, -1.3]).unwrap();
        let recip_a = a.recip();
        let expected = Tensor::new(&[1.2f32.recip(), 2.7f32.recip(), -1.3f32.recip(),]).unwrap();

        assert!(recip_a.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_add_basic() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = Tensor::new(&[4.0f32, 5.0, 6.0]).unwrap();
        let c = Tensor::add(&a, &b).unwrap();
        let expected = Tensor::new(&[5.0f32, 7.0, 9.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_add_basic_variable() {
        let a = Tensor::new_var(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = Tensor::new_var(&[4.0f32, 5.0, 6.0]).unwrap();
        let c = Tensor::add(&a, &b).unwrap();
        let expected = Tensor::new(&[5.0f32, 7.0, 9.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
        assert!(c.requires_grad());
    }

    #[test]
    fn test_sub_basic() {
        let a = Tensor::new(&[10.0f32, 20.0, 30.0]).unwrap();
        let b = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let c = Tensor::sub(&a, &b).unwrap();
        let expected = Tensor::new(&[9.0f32, 18.0, 27.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_mul_basic() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 4.0]).unwrap();
        let c = Tensor::mul(&a, &b).unwrap();
        let expected = Tensor::new(&[2.0f32, 6.0, 12.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_div_basic() {
        let a = Tensor::new(&[4.0f32, 9.0, 16.0]).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 4.0]).unwrap();
        let c = Tensor::div(&a, &b).unwrap();
        let expected = Tensor::new(&[2.0f32, 3.0, 4.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_min_max_basic() {
        let a = Tensor::new(&[1.0f32, 5.0, 3.0]).unwrap();
        let b = Tensor::new(&[2.0f32, 4.0, 6.0]).unwrap();
        let min_res = Tensor::minimum(&a, &b).unwrap();
        let max_res = Tensor::maximum(&a, &b).unwrap();
        let expected_min = Tensor::new(&[1.0f32, 4.0, 3.0]).unwrap();
        let expected_max = Tensor::new(&[2.0f32, 5.0, 6.0]).unwrap();
        assert!(min_res.allclose(&expected_min, 1e-6, 1e-6));
        assert!(max_res.allclose(&expected_max, 1e-6, 1e-6));
    }

    #[test]
    fn test_add_sub_mul_div_assign() {
        let a = Tensor::new(&[1., 2., 3.]).unwrap();
        let b = Tensor::new(&[4., 5., 6.]).unwrap();

        a.add_assign(&b).unwrap();
        assert!(a.allclose(&Tensor::new(&[5., 7., 9.]).unwrap(), 1e-6, 1e-6));

        a.sub_assign(&b).unwrap();
        assert!(a.allclose(&Tensor::new(&[1., 2., 3.]).unwrap(), 1e-6, 1e-6));

        a.mul_assign(&b).unwrap();
        assert!(a.allclose(&Tensor::new(&[4., 10., 18.]).unwrap(), 1e-6, 1e-6));

        a.div_assign(&b).unwrap();
        assert!(a.allclose(&Tensor::new(&[1., 2., 3.]).unwrap(), 1e-6, 1e-6));
    }

    #[test]
    fn test_min_max_assign() {
        let a = Tensor::new(&[1., 5., 3.]).unwrap();
        let b = Tensor::new(&[4., 2., 6.]).unwrap();

        a.minimum_assign(&b).unwrap();
        assert!(a.allclose(&Tensor::new(&[1., 2., 3.]).unwrap(), 1e-6, 1e-6));

        let a2 = Tensor::new(&[1., 5., 3.]).unwrap();
        a2.maximum_assign(&b).unwrap();
        assert!(a2.allclose(&Tensor::new(&[4., 5., 6.]).unwrap(), 1e-6, 1e-6));
    }

    #[test]
    fn test_comparisons() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let b = Tensor::new(&[1, 0, 3]).unwrap();

        assert_eq!(a.eq(&b).unwrap().to_vec(), [true, false, true]);
        assert_eq!(a.ne(&b).unwrap().to_vec(), [false, true, false]);
        assert_eq!(a.lt(&b).unwrap().to_vec(), [false, false, false]);
        assert_eq!(a.le(&b).unwrap().to_vec(), [true, false, true]);
        assert_eq!(a.gt(&b).unwrap().to_vec(), [false, true, false]);
        assert_eq!(a.ge(&b).unwrap().to_vec(), [true, true, true]);
    }

    #[test]
    fn test_add_mul_2d_3d() {
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]]).unwrap();
        let b = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]]).unwrap();
        let c = Tensor::add(&a, &b).unwrap();
        let expected = Tensor::new(&[[6., 8.], [10., 12.]]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));

        let a3 = Tensor::new(&[
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
        ]).unwrap();
        let b3 = Tensor::new(&[
            [[2., 0.5], [1., 2.]],
            [[0.5, 2.], [1.5, 1.]],
        ]).unwrap();
        let c3 = Tensor::mul(&a3, &b3).unwrap();
        let expected3 = Tensor::new(&[
            [[2., 1.], [3., 8.]],
            [[2.5, 12.], [10.5, 8.]],
        ]).unwrap();
        assert!(c3.allclose(&expected3, 1e-6, 1e-6));
    }

    #[test]
    fn test_div_high_dim() {
        let a = Tensor::full((2, 2, 2, 2), 8.0f32).unwrap();
        let b = Tensor::full((2, 2, 2, 2), 2.0f32).unwrap();
        let c = Tensor::div(&a, &b).unwrap();
        let expected = Tensor::full((2, 2, 2, 2), 4.0f32).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_affine_and_affine_assign() {
        let a = Tensor::<f64>::ones((3, 3)).unwrap();
        let b = a.affine(3., 2.).unwrap();
        let expected = Tensor::new(&[[5., 5., 5.],[5.,5.,5.],[5.,5.,5.]]).unwrap();
        assert!(b.allclose(&expected, 1e-6, 1e-6));

        let a2 = Tensor::<f64>::ones((3, 3)).unwrap();
        a2.affine_assign(3., 2.).unwrap();
        assert!(a2.allclose(&expected, 1e-6, 1e-6));
    }

    // #[test]
    // fn test_add_scalar() {
    //     let a = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
    //     let b = 10.0f32;
    //     let c = Tensor::add(&a, &b).unwrap();
    //     let expected = Tensor::new(&[11.0f32, 12.0, 13.0]).unwrap();
    //     assert!(c.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_sub_scalar() {
    //     let a = Tensor::new(&[10.0f32, 20.0, 30.0]).unwrap();
    //     let b = 5.0f32;
    //     let c = Tensor::sub(&a, &b).unwrap();
    //     let expected = Tensor::new(&[5.0f32, 15.0, 25.0]).unwrap();
    //     assert!(c.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_mul_scalar() {
    //     let a = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
    //     let b = 2.0f32;
    //     let c = Tensor::mul(&a, &b).unwrap();
    //     let expected = Tensor::new(&[2.0f32, 4.0, 6.0]).unwrap();
    //     assert!(c.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_div_scalar() {
    //     let a = Tensor::new(&[4.0f32, 9.0, 16.0]).unwrap();
    //     let b = 2.0f32;
    //     let c = Tensor::div(&a, &b).unwrap();
    //     let expected = Tensor::new(&[2.0f32, 4.5, 8.0]).unwrap();
    //     assert!(c.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_minimum_scalar() {
    //     let a = Tensor::new(&[1.0f32, 5.0, 3.0]).unwrap();
    //     let b = 4.0f32;
    //     let c = Tensor::minimum(&a, &b).unwrap();
    //     let expected = Tensor::new(&[1.0f32, 4.0, 3.0]).unwrap();
    //     assert!(c.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_maximum_scalar() {
    //     let a = Tensor::new(&[1.0f32, 5.0, 3.0]).unwrap();
    //     let b = 4.0f32;
    //     let c = Tensor::maximum(&a, &b).unwrap();
    //     let expected = Tensor::new(&[4.0f32, 5.0, 4.0]).unwrap();
    //     assert!(c.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_add_assign_scalar() {
    //     let a = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
    //     let b = 5.0f32;
    //     a.add_assign(&b).unwrap();
    //     let expected = Tensor::new(&[6.0f32, 7.0, 8.0]).unwrap();
    //     assert!(a.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_sub_assign_scalar() {
    //     let a = Tensor::new(&[10.0f32, 20.0, 30.0]).unwrap();
    //     let b = 5.0f32;
    //     a.sub_assign(&b).unwrap();
    //     let expected = Tensor::new(&[5.0f32, 15.0, 25.0]).unwrap();
    //     assert!(a.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_mul_assign_scalar() {
    //     let a = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
    //     let b = 3.0f32;
    //     a.mul_assign(&b).unwrap();
    //     let expected = Tensor::new(&[3.0f32, 6.0, 9.0]).unwrap();
    //     assert!(a.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_div_assign_scalar() {
    //     let a = Tensor::new(&[8.0f32, 12.0, 20.0]).unwrap();
    //     let b = 2.0f32;
    //     a.div_assign(&b).unwrap();
    //     let expected = Tensor::new(&[4.0f32, 6.0, 10.0]).unwrap();
    //     assert!(a.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_minimum_assign_scalar() {
    //     let a = Tensor::new(&[1.0f32, 5.0, 3.0]).unwrap();
    //     let b = 4.0f32;
    //     a.minimum_assign(&b).unwrap();
    //     let expected = Tensor::new(&[1.0f32, 4.0, 3.0]).unwrap();
    //     assert!(a.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_maximum_assign_scalar() {
    //     let a = Tensor::new(&[1.0f32, 5.0, 3.0]).unwrap();
    //     let b = 4.0f32;
    //     a.maximum_assign(&b).unwrap();
    //     let expected = Tensor::new(&[4.0f32, 5.0, 4.0]).unwrap();
    //     assert!(a.allclose(&expected, 1e-6, 1e-6));
    // }

    // #[test]
    // fn test_eq_ne_scalar() {
    //     let a = Tensor::new(&[1, 2, 3]).unwrap();
    //     let b = 2;

    //     // Tensor vs scalar
    //     let eq_res = a.eq(&b).unwrap();
    //     let expected_eq = Tensor::new(&[false, true, false]).unwrap();
    //     assert_eq!(eq_res.to_vec(), expected_eq.to_vec());

    //     let ne_res = a.ne(&b).unwrap();
    //     let expected_ne = Tensor::new(&[true, false, true]).unwrap();
    //     assert_eq!(ne_res.to_vec(), expected_ne.to_vec());
    // }

    // #[test]
    // fn test_lt_le_gt_ge_scalar() {
    //     let a = Tensor::new(&[1, 2, 3]).unwrap();
    //     let b = 2;

    //     let lt_res = a.lt(&b).unwrap();
    //     assert_eq!(lt_res.to_vec(), [true, false, false]);

    //     let le_res = a.le(&b).unwrap();
    //     assert_eq!(le_res.to_vec(), [true, true, false]);

    //     let gt_res = a.gt(&b).unwrap();
    //     assert_eq!(gt_res.to_vec(), [false, false, true]);

    //     let ge_res = a.ge(&b).unwrap();
    //     assert_eq!(ge_res.to_vec(), [false, true, true]);
    // }

    #[test]
    fn test_eq_ne_tensor() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let b = Tensor::new(&[1, 0, 3]).unwrap();

        let eq_res = a.eq(&b).unwrap();
        assert_eq!(eq_res.to_vec(), [true, false, true]);

        let ne_res = a.ne(&b).unwrap();
        assert_eq!(ne_res.to_vec(), [false, true, false]);
    }

    #[test]
    fn test_lt_le_gt_ge_tensor() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let b = Tensor::new(&[2, 2, 1]).unwrap();

        let lt_res = a.lt(&b).unwrap();
        assert_eq!(lt_res.to_vec(), [true, false, false]);

        let le_res = a.le(&b).unwrap();
        assert_eq!(le_res.to_vec(), [true, true, false]);

        let gt_res = a.gt(&b).unwrap();
        assert_eq!(gt_res.to_vec(), [false, false, true]);

        let ge_res = a.ge(&b).unwrap();
        assert_eq!(ge_res.to_vec(), [false, true, true]);
    }

    #[test]
    fn test_comparison_2d() {
        let a = Tensor::new(&[[1, 2], [3, 4]]).unwrap();
        let b = Tensor::new(&[[2, 2], [1, 5]]).unwrap();

        let eq_res = a.eq(&b).unwrap();
        let expected_eq = Tensor::new(&[[false, true], [false, false]]).unwrap();
        assert_eq!(eq_res.to_vec(), expected_eq.to_vec());

        let gt_res = a.gt(&b).unwrap();
        let expected_gt = Tensor::new(&[[false, false], [true, false]]).unwrap();
        assert_eq!(gt_res.to_vec(), expected_gt.to_vec());

        // Tensor vs scalar
        // let le_res = a.le(3).unwrap();
        // let expected_le = Tensor::new(&[[true, true], [true, false]]).unwrap();
        // assert_eq!(le_res.to_vec(), expected_le.to_vec());
    }

    // #[test]
    // fn test_std_ops() {
    //     let a = Tensor::new(&[[1., 2.], [3., 4.]]).unwrap();
    //     let b = Tensor::new(&[[2., 2.], [1., 5.]]).unwrap();
    //     let _ = a + b;

    //     let a = Tensor::new(&[[1., 2.], [3., 4.]]).unwrap();
    //     let b = Tensor::new(&[[2., 2.], [1., 5.]]).unwrap();
    //     let _ = &a + &b;

    //     let a = Tensor::new(&[[1., 2.], [3., 4.]]).unwrap();
    //     let b = Tensor::new(&[[2., 2.], [1., 5.]]).unwrap();
    //     let _ = a + b;

    //     let a = Tensor::new(&[[1., 2.], [3., 4.]]).unwrap();
    //     let b = Tensor::new(&[[2., 2.], [1., 5.]]).unwrap();
    //     let _ = a + b;
    // }
}
