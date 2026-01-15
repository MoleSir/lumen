use crate::{TensorOrScalar, grad::BinaryOp, AutogradMetaT, CmpOp, Error, FloatDType, NumDType, Result, Shape, Storage, UnaryOp, WithDType};
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
            })?
        } else {
            Ok(lhs)
        }
    }
}

impl<T: WithDType> Tensor<T> {
    fn compute_binary_scalar_rhs_op<U, F>(lhs: &Tensor<T>, rhs: T, mut f: F, _op_name: &'static str) -> Result<(Storage<U>, Shape)>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let shape = lhs.shape();
        let lhs_storage = lhs.storage_read();
        let lhs_layout = lhs.layout();

        let lhs = lhs_storage.data();
        
        let output: Vec<_> = lhs_layout.storage_indices()
            .map(|lhs_index| f(lhs[lhs_index], rhs))
            .collect();
        
        let storage = Storage::<U>::new(output);
        Ok((storage, shape.clone()))
    }

    fn compute_scalar_binary_lhs_op<U, F>(lhs: T, rhs: &Tensor<T>, mut f: F, _op_name: &'static str) -> Result<(Storage<U>, Shape)>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let shape = rhs.shape();
        let rhs_storage = rhs.storage_read();
        let rhs_layout = rhs.layout();

        let rhs = rhs_storage.data();
        
        let output: Vec<_> = rhs_layout.storage_indices()
            .map(|index| f(lhs, rhs[index]))
            .collect();
        
        let storage = Storage::<U>::new(output);
        Ok((storage, shape.clone()))
    }

    fn compute_binary_op<U, F>(lhs: &Tensor<T>, rhs: &Tensor<T>, mut f: F, op_name: &'static str) -> Result<(Storage<U>, Shape)>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let shape = Tensor::<T>::same_shape_binary_op(lhs, rhs, op_name)?;
        let lhs_storage = lhs.storage_read();
        let rhs_storage = rhs.storage_read();
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

    fn binary_scalar_op<U, F>(lhs: &Tensor<T>, rhs: T, f: F, op_name: &'static str) -> Result<Tensor<U>>
    where 
        U: WithDType, 
        F: FnMut(T, T) -> U 
    {
        let (storage, shape) = Self::compute_binary_scalar_rhs_op(lhs, rhs, f, op_name)?;
        Ok(Tensor::<U>::from_storage(storage, shape))
    }
}

macro_rules! binary_op_impl {
    ($fn_name:ident) => {
        paste! {
            pub fn [< $fn_name _tensor >](&self, rhs: &Self) -> Result<Self> {
                let (storage, shape) = Self::compute_binary_op(self, rhs, T::$fn_name, stringify!([< $fn_name _tensor >]))?;
                let meta = T::AutogradMeta::on_binary_op(self, rhs, BinaryOp::  [< $fn_name:camel >]);
                Ok(Self::build(storage, shape, meta))
            }
        
            pub fn [< $fn_name _scalar >](&self, rhs: T) -> Result<Self> {
                let (storage, shape) = Self::compute_binary_scalar_rhs_op(self, rhs, T::$fn_name, stringify!([< $fn_name _scalar >]))?;
                let meta = T::AutogradMeta::on_binary_scalar_rhs_op(self, rhs, BinaryOp::  [< $fn_name:camel >]);
                Ok(Self::build(storage, shape, meta))
            } 
        
            pub fn [< scalar_ $fn_name >](lhs: T, rhs: &Tensor<T>) -> Result<Tensor<T>> {
                let (storage, shape) = Self::compute_scalar_binary_lhs_op(lhs, rhs, T::$fn_name, stringify!([< scalar_ $fn_name >]))?;
                let meta = T::AutogradMeta::on_binary_scalar_lhs_op(lhs, rhs, BinaryOp::  [< $fn_name:camel >]);
                Ok(Self::build(storage, shape, meta))
            }

            pub fn $fn_name(&self, rhs: impl Into<TensorOrScalar<T>>) -> Result<Self> {
                match rhs.into() {
                    TensorOrScalar::Tensor(t) => self.[< $fn_name _tensor >](&t),
                    TensorOrScalar::Scalar(s) => self.[< $fn_name _scalar >](s)
                }
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

    pub fn clamp(&self, min: T, max: T) -> Result<Self> {
        self.maximum(min)?.minimum(max)
    }
}

impl<T: NumDType> Tensor<T> {
    fn binary_op_inplace<F>(lhs: &Tensor<T>, rhs: &Tensor<T>, mut f: F, op_name: &'static str) -> Result<()> 
    where 
        F: FnMut(T, T) -> T
    {
        let _ = Tensor::<T>::same_shape_binary_op(lhs, rhs, op_name)?;

        let mut lhs_storage = lhs.storage_write();
        let rhs_storage = rhs.storage_read();
        let lhs_layout = lhs.layout();
        let rhs_layout = rhs.layout();

        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");

        let lhs = lhs_storage.data_mut();
        let rhs = rhs_storage.data();
        
        lhs_layout.storage_indices().zip(rhs_layout.storage_indices())
            .for_each(|(lhs_index, rhs_index)| lhs[lhs_index] = f(lhs[lhs_index], rhs[rhs_index]));
        
        Ok(())
    }

    fn binary_op_scalar_inplace<F>(lhs: &Tensor<T>, rhs: T, mut f: F, _op_name: &'static str) -> Result<()> 
    where 
        F: FnMut(T, T) -> T
    {
        let mut lhs_storage = lhs.storage_write();
        let lhs_layout = lhs.layout();


        let lhs = lhs_storage.data_mut();
        
        lhs_layout.storage_indices()
            .for_each(|lhs_index| lhs[lhs_index] = f(lhs[lhs_index], rhs));
        
        Ok(())
    }
}

macro_rules! binary_inplace_op_impl {
    ($fn_name:ident) => {
        paste! {
            pub fn [< $fn_name _ >](&self, rhs: impl Into<TensorOrScalar<T>>) -> Result<()> {
                let rhs = rhs.into();
                match rhs {
                    TensorOrScalar::Scalar(rhs) => Self::binary_op_scalar_inplace(self, rhs, T::$fn_name, stringify!([< $fn_name _scalar_ >])),
                    TensorOrScalar::Tensor(rhs) => Self::binary_op_inplace(self, &rhs, T::$fn_name, stringify!([< $fn_name _scalar >])),
                }
            }
        }
    };
}

#[allow(unused)]
impl<T: NumDType> Tensor<T> {
    binary_inplace_op_impl!(add);
    binary_inplace_op_impl!(sub);
    binary_inplace_op_impl!(mul);
    binary_inplace_op_impl!(div);
}

impl<T: NumDType> Tensor<T> {
    pub fn eq(&self, rhs: impl Into<TensorOrScalar<T>>) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Eq)
    }

    pub fn ne(&self, rhs: impl Into<TensorOrScalar<T>>) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Ne)
    }

    pub fn le(&self, rhs: impl Into<TensorOrScalar<T>>) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Le)
    }

    pub fn ge(&self, rhs: impl Into<TensorOrScalar<T>>) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Ge)
    }

    pub fn lt(&self, rhs: impl Into<TensorOrScalar<T>>) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Lt)
    }

    pub fn gt(&self, rhs: impl Into<TensorOrScalar<T>>) -> Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Gt)
    }

    pub fn cmp(&self, rhs: impl Into<TensorOrScalar<T>>, op: CmpOp) -> Result<Tensor<bool>> {
        match rhs.into() {
            TensorOrScalar::Tensor(rhs) => {
                match op {
                    CmpOp::Eq => Self::binary_op(self, &rhs, |a, b| a == b, "eq"),
                    CmpOp::Ne => Self::binary_op(self, &rhs, |a, b| a != b, "nq"),
                    CmpOp::Le => Self::binary_op(self, &rhs, |a, b| a <= b, "le"),
                    CmpOp::Ge => Self::binary_op(self, &rhs, |a, b| a >= b, "ge"),
                    CmpOp::Lt => Self::binary_op(self, &rhs, |a, b| a <  b, "lt"),
                    CmpOp::Gt => Self::binary_op(self, &rhs, |a, b| a >  b, "gt"),
                }
            }
            TensorOrScalar::Scalar(rhs) => {
                match op {
                    CmpOp::Eq => Self::binary_scalar_op(self, rhs, |a, b| a == b, "eq"),
                    CmpOp::Ne => Self::binary_scalar_op(self, rhs, |a, b| a != b, "nq"),
                    CmpOp::Le => Self::binary_scalar_op(self, rhs, |a, b| a <= b, "le"),
                    CmpOp::Ge => Self::binary_scalar_op(self, rhs, |a, b| a >= b, "ge"),
                    CmpOp::Lt => Self::binary_scalar_op(self, rhs, |a, b| a <  b, "lt"),
                    CmpOp::Gt => Self::binary_scalar_op(self, rhs, |a, b| a >  b, "gt"),
                }
            }
        }
    } 
}

impl Tensor<bool> {
    pub fn and(&self, rhs: impl Into<TensorOrScalar<bool>>) -> Result<Tensor<bool>> {
        match rhs.into() {
            TensorOrScalar::Tensor(rhs) => Self::binary_op(self, &rhs, |a, b| a & b, "and"),
            TensorOrScalar::Scalar(rhs) => Self::binary_scalar_op(self, rhs, |a, b| a & b, "and"),
        }
    }

    pub fn or(&self, rhs: impl Into<TensorOrScalar<bool>>) -> Result<Tensor<bool>> {
        match rhs.into() {
            TensorOrScalar::Tensor(rhs) => Self::binary_op(self, &rhs, |a, b| a | b, "or"),
            TensorOrScalar::Scalar(rhs) => Self::binary_scalar_op(self, rhs, |a, b| a | b, "or"),
        }
    }

    pub fn xor(&self, rhs: impl Into<TensorOrScalar<bool>>) -> Result<Tensor<bool>> {
        match rhs.into() {
            TensorOrScalar::Tensor(rhs) => Self::binary_op(self, &rhs, |a, b| a ^ b, "xor"),
            TensorOrScalar::Scalar(rhs) => Self::binary_scalar_op(self, rhs, |a, b| a ^ b, "xor"),
        }
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
        let storage = self.storage_read();
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
                Self::build(storage, self.shape(), meta)
            }
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

impl<T: NumDType + Neg<Output = T>> Tensor<T> {
    pub fn neg(&self) -> Self {
        if self.element_count() == 0 {
            return self.clone();
        }
        let storage = self.compute_unary_op(Neg::neg);
        let meta = T::AutogradMeta::on_unray_op(self, UnaryOp::Neg);
        Self::build(storage, self.shape(), meta)
    }
}

impl<F: FloatDType> Tensor<F> {
    float_unary_op_impl!(floor);
    float_unary_op_impl!(ceil);
    float_unary_op_impl!(round);

    float_unary_op_impl!(exp);
    float_unary_op_impl!(ln);

    float_unary_op_impl!(sin);
    float_unary_op_impl!(cos);
    float_unary_op_impl!(tanh);

    float_unary_op_impl!(sqrt);
    float_unary_op_impl!(sqr);
    float_unary_op_impl!(abs);
    // float_unary_op_impl!(neg);

    float_unary_op_impl!(recip);
    float_unary_op_impl!(gelu);
    float_unary_op_impl!(gelu_erf);
    float_unary_op_impl!(erf);
    float_unary_op_impl!(relu);
    float_unary_op_impl!(silu);
    float_unary_op_impl!(sigmoid);

    pub fn leaky_relu(&self, negative_slope: F) -> Self {
        if self.element_count() == 0 {
            return self.clone();
        }
        let f = |v: F| F::leaky_relu(v, negative_slope);
        let storage = self.compute_unary_op(f);
        let meta = F::AutogradMeta::on_unray_op(self, UnaryOp::LeakyRelu(negative_slope));
        Self::build(storage, self.shape(), meta)
    }
}

impl<F: FloatDType> Tensor<F> {
    pub fn pow(&self, e: F) -> Self {
        if self.element_count() == 0 {
            return self.clone();
        }
        let f = |v: F| v.powf(e); 
        let storage = self.compute_unary_op(f);
        let meta = F::AutogradMeta::on_pow_op(self, e);
        Self::build(storage, self.shape(), meta)
    }
}

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Sub};

//////////////////////////////////////////////////////////////////////////////
///        Add
//////////////////////////////////////////////////////////////////////////////

impl<'a, T: NumDType, R: Into<TensorOrScalar<T>>> Add<R> for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: R) -> Self::Output {
        Tensor::add(self, rhs).expect("Tensor::add failed")
    }
}

impl<'a, T: NumDType, R> Add<R> for Tensor<T> 
where R: Into<TensorOrScalar<T>> 
{
    type Output = Tensor<T>;
    fn add(self, rhs: R) -> Self::Output {
        Tensor::add(&self, rhs).expect("Tensor::add failed")
    }
}

//////////////////////////////////////////////////////////////////////////////
///        Sub
//////////////////////////////////////////////////////////////////////////////

impl<'a, T: NumDType, R: Into<TensorOrScalar<T>>> Sub<R> for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: R) -> Self::Output {
        Tensor::sub(self, rhs).expect("Tensor::sub failed")
    }
}

impl<'a, T: NumDType, R: Into<TensorOrScalar<T>>> Sub<R> for Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: R) -> Self::Output {
        Tensor::sub(&self, rhs).expect("Tensor::sub failed")
    }
}

//////////////////////////////////////////////////////////////////////////////
///        Mul
//////////////////////////////////////////////////////////////////////////////

impl<'a, T: NumDType, R: Into<TensorOrScalar<T>>> Mul<R> for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: R) -> Self::Output {
        Tensor::mul(self, rhs).expect("Tensor::mul failed")
    }
}

impl<'a, T: NumDType, R: Into<TensorOrScalar<T>>> Mul<R> for Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: R) -> Self::Output {
        Tensor::mul(&self, rhs).expect("Tensor::mul failed")
    }
}

//////////////////////////////////////////////////////////////////////////////
///        Div
//////////////////////////////////////////////////////////////////////////////

impl<'a, T: NumDType, R: Into<TensorOrScalar<T>>> Div<R> for &Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: R) -> Self::Output {
        Tensor::div(self, rhs).expect("Tensor::div failed")
    }
}

impl<'a, T: NumDType, R: Into<TensorOrScalar<T>>> Div<R> for Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: R) -> Self::Output {
        Tensor::div(&self, rhs).expect("Tensor::div failed")
    }
}

//////////////////////////////////////////////////////////////////////////////
///        Bool
//////////////////////////////////////////////////////////////////////////////

impl<'a, R: Into<TensorOrScalar<bool>>> BitAnd<R> for &Tensor<bool> {
    type Output = Tensor<bool>;
    fn bitand(self, rhs: R) -> Self::Output {
        self.and(rhs).expect("Tensor::and failed")
    }
}

impl<'a, R: Into<TensorOrScalar<bool>>> BitAnd<R> for Tensor<bool> {
    type Output = Tensor<bool>;
    fn bitand(self, rhs: R) -> Self::Output {
        self.and(rhs).expect("Tensor::and failed")
    }
}

impl<'a, R: Into<TensorOrScalar<bool>>> BitOr<R> for &Tensor<bool> {
    type Output = Tensor<bool>;
    fn bitor(self, rhs: R) -> Self::Output {
        self.or(rhs).expect("Tensor::or failed")
    }
}

impl<'a, R: Into<TensorOrScalar<bool>>> BitOr<R> for Tensor<bool> {
    type Output = Tensor<bool>;
    fn bitor(self, rhs: R) -> Self::Output {
        self.or(rhs).expect("Tensor::or failed")
    }
}

impl<'a, R: Into<TensorOrScalar<bool>>> BitXor<R> for &Tensor<bool> {
    type Output = Tensor<bool>;
    fn bitxor(self, rhs: R) -> Self::Output {
        self.xor(rhs).expect("Tensor::xor failed")
    }
}

impl<'a, R: Into<TensorOrScalar<bool>>> BitXor<R> for Tensor<bool> {
    type Output = Tensor<bool>;
    fn bitxor(self, rhs: R) -> Self::Output {
        self.xor(rhs).expect("Tensor::xor failed")
    }
}

//////////////////////////////////////////////////////////////////////////////
///        Scalar & op
//////////////////////////////////////////////////////////////////////////////

macro_rules! impl_scalar_tensor_binary {
    ($($t:ty),*) => {
        $(
            impl Add<Tensor<$t>> for $t {
                type Output = Tensor<$t>;

                fn add(self, rhs: Tensor<$t>) -> Self::Output {
                    Tensor::add(&rhs, self).expect("Tensor::add failed")
                }
            }

            impl Add<&Tensor<$t>> for $t {
                type Output = Tensor<$t>;

                fn add(self, rhs: &Tensor<$t>) -> Self::Output {
                    Tensor::add(rhs, self).expect("Tensor::add failed")
                }
            }

            impl Mul<Tensor<$t>> for $t {
                type Output = Tensor<$t>;

                fn mul(self, rhs: Tensor<$t>) -> Self::Output {
                    Tensor::mul(&rhs, self).expect("Tensor::mul failed")
                }
            }

            impl Mul<&Tensor<$t>> for $t {
                type Output = Tensor<$t>;

                fn mul(self, rhs: &Tensor<$t>) -> Self::Output {
                    Tensor::mul(rhs, self).expect("Tensor::mul failed")
                }
            }

            impl Sub<&Tensor<$t>> for $t {
                type Output = Tensor<$t>;

                fn sub(self, rhs: &Tensor<$t>) -> Self::Output {
                    Tensor::scalar_sub(self, rhs).expect("Tensor::scalar_sub failed")
                }
            }

            impl Sub<Tensor<$t>> for $t {
                type Output = Tensor<$t>;

                fn sub(self, rhs: Tensor<$t>) -> Self::Output {
                    Tensor::scalar_sub(self, &rhs).expect("Tensor::scalar_sub failed")
                }
            }

            impl Div<&Tensor<$t>> for $t {
                type Output = Tensor<$t>;

                fn div(self, rhs: &Tensor<$t>) -> Self::Output {
                    Tensor::scalar_div(self, rhs).expect("Tensor::scalar_div failed")
                }
            }

            impl Div<Tensor<$t>> for $t {
                type Output = Tensor<$t>;

                fn div(self, rhs: Tensor<$t>) -> Self::Output {
                    Tensor::scalar_div(self, &rhs).expect("Tensor::scalar_div failed")
                }
            }
        )*
    };
}

impl_scalar_tensor_binary!(f32, f64, u8, i32, u32);

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

    #[test]
    fn test_add_scalar() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = 10.0f32;
        let c = Tensor::add(&a, b).unwrap();
        let expected = Tensor::new(&[11.0f32, 12.0, 13.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_sub_scalar() {
        let a = Tensor::new(&[10.0f32, 20.0, 30.0]).unwrap();
        let b = 5.0f32;
        let c = Tensor::sub(&a, b).unwrap();
        let expected = Tensor::new(&[5.0f32, 15.0, 25.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = 2.0f32;
        let c = Tensor::mul(&a, b).unwrap();
        let expected = Tensor::new(&[2.0f32, 4.0, 6.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_div_scalar() {
        let a = Tensor::new(&[4.0f32, 9.0, 16.0]).unwrap();
        let b = 2.0f32;
        let c = Tensor::div(&a, b).unwrap();
        let expected = Tensor::new(&[2.0f32, 4.5, 8.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_minimum_scalar() {
        let a = Tensor::new(&[1.0f32, 5.0, 3.0]).unwrap();
        let b = 4.0f32;
        let c = Tensor::minimum(&a, b).unwrap();
        let expected = Tensor::new(&[1.0f32, 4.0, 3.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_maximum_scalar() {
        let a = Tensor::new(&[1.0f32, 5.0, 3.0]).unwrap();
        let b = 4.0f32;
        let c = Tensor::maximum(&a, b).unwrap();
        let expected = Tensor::new(&[4.0f32, 5.0, 4.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_eq_ne_scalar() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let b = 2;

        // Tensor vs scalar
        let eq_res = a.eq(b).unwrap();
        let expected_eq = Tensor::new(&[false, true, false]).unwrap();
        assert_eq!(eq_res.to_vec(), expected_eq.to_vec());

        let ne_res = a.ne(b).unwrap();
        let expected_ne = Tensor::new(&[true, false, true]).unwrap();
        assert_eq!(ne_res.to_vec(), expected_ne.to_vec());
    }

    #[test]
    fn test_lt_le_gt_ge_scalar() {
        let a = Tensor::new(&[1, 2, 3]).unwrap();
        let b = 2;

        let lt_res = a.lt(b).unwrap();
        assert_eq!(lt_res.to_vec(), [true, false, false]);

        let le_res = a.le(b).unwrap();
        assert_eq!(le_res.to_vec(), [true, true, false]);

        let gt_res = a.gt(b).unwrap();
        assert_eq!(gt_res.to_vec(), [false, false, true]);

        let ge_res = a.ge(b).unwrap();
        assert_eq!(ge_res.to_vec(), [false, true, true]);
    }

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
        let le_res = a.le(3).unwrap();
        let expected_le = Tensor::new(&[[true, true], [true, false]]).unwrap();
        assert_eq!(le_res.to_vec(), expected_le.to_vec());
    }

    #[test]
    fn test_std_ops() {
        let a = Tensor::new(&[[1., 2.], [3., 4.]]).unwrap();
        let b = Tensor::new(&[[2., 2.], [1., 5.]]).unwrap();
        let _ = a + b;

        let a = Tensor::new(&[[1., 2.], [3., 4.]]).unwrap();
        let b = Tensor::new(&[[2., 2.], [1., 5.]]).unwrap();
        let _ = &a + &b;

        let a = Tensor::new(&[[1., 2.], [3., 4.]]).unwrap();
        let b = Tensor::new(&[[2., 2.], [1., 5.]]).unwrap();
        let _ = a + b;

        let a = Tensor::new(&[[1., 2.], [3., 4.]]).unwrap();
        let b = Tensor::new(&[[2., 2.], [1., 5.]]).unwrap();
        let _ = a + b;
    }
}
