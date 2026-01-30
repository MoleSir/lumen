use crate::{TensorOrScalar, grad::BinaryOp, AutogradMetaT, CmpOp, Error, FloatDType, NumDType, Shape, Storage, UnaryOp, WithDType};
use super::Tensor;
use paste::paste;

//////////////////////////////////////////////////////////////////////////////
///        Binary(Assign) Op with Tensor and Tensor / scalar
//////////////////////////////////////////////////////////////////////////////

impl<T: WithDType> Tensor<T> {
    fn same_shape_binary_op(&self, rhs: &Self, op: &'static str) -> crate::Result<&Shape> {
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
    fn compute_binary_scalar_rhs_op<U, F>(lhs: &Tensor<T>, rhs: T, mut f: F, _op_name: &'static str) -> crate::Result<(Storage<U>, Shape)>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let shape = lhs.shape();
        let lhs_storage = lhs.storage_read()?;
        let lhs_layout = lhs.layout();

        let lhs = lhs_storage.data();
        
        let output: Vec<_> = lhs_layout.storage_indices()
            .map(|lhs_index| f(lhs[lhs_index], rhs))
            .collect();
        
        let storage = Storage::<U>::new(output);
        Ok((storage, shape.clone()))
    }

    fn compute_binary_scalar_lhs_op<U, F>(lhs: T, rhs: &Tensor<T>, mut f: F, _op_name: &'static str) -> crate::Result<(Storage<U>, Shape)>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let shape = rhs.shape();
        let rhs_storage = rhs.storage_read()?;
        let rhs_layout = rhs.layout();

        let rhs = rhs_storage.data();
        
        let output: Vec<_> = rhs_layout.storage_indices()
            .map(|index| f(lhs, rhs[index]))
            .collect();
        
        let storage = Storage::<U>::new(output);
        Ok((storage, shape.clone()))
    }

    fn compute_binary_op<U, F>(lhs: &Tensor<T>, rhs: &Tensor<T>, mut f: F, op_name: &'static str) -> crate::Result<(Storage<U>, Shape)>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let shape = Tensor::<T>::same_shape_binary_op(lhs, rhs, op_name)?;
        let lhs_storage = lhs.storage_read()?;
        let rhs_storage = rhs.storage_read()?;
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

    fn binary_op<U, F>(lhs: &Tensor<T>, rhs: &Tensor<T>, f: F, meta: U::AutogradMeta, op_name: &'static str) -> crate::Result<Tensor<U>>
    where 
        U: WithDType, 
        F: FnMut(T, T) -> U 
    {
        let (storage, shape) = Self::compute_binary_op(lhs, rhs, f, op_name)?;
        Ok(Tensor::<U>::from_storage(storage, shape, meta))
    }

    fn binary_scalar_rhs_op<U, F>(lhs: &Tensor<T>, rhs: T, f: F, meta: U::AutogradMeta, op_name: &'static str) -> crate::Result<Tensor<U>>
    where 
        U: WithDType, 
        F: FnMut(T, T) -> U 
    {
        let (storage, shape) = Self::compute_binary_scalar_rhs_op(lhs, rhs, f, op_name)?;
        Ok(Tensor::<U>::from_storage(storage, shape, meta))
    }

    fn binary_scalar_lhs_op<U, F>(lhs: T, rhs: &Tensor<T>, f: F, meta: U::AutogradMeta, op_name: &'static str) -> crate::Result<Tensor<U>>
    where 
        U: WithDType, 
        F: FnMut(T, T) -> U 
    {
        let (storage, shape) = Self::compute_binary_scalar_lhs_op(lhs, rhs, f, op_name)?;
        Ok(Tensor::<U>::from_storage(storage, shape, meta))
    }
}

macro_rules! binary_op_impl {
    ($fn_name:ident) => {
        paste! {
            pub fn [< $fn_name _tensor >](&self, rhs: &Self) -> crate::Result<Self> {
                let meta = T::AutogradMeta::on_binary_op(self, rhs, BinaryOp::  [< $fn_name:camel >]);
                Self::binary_op(self, rhs, T::$fn_name, meta, stringify!([< $fn_name _tensor >]))
            }
        
            pub fn [< $fn_name _scalar >](&self, rhs: T) -> crate::Result<Self> {
                let meta = T::AutogradMeta::on_binary_scalar_rhs_op(self, rhs, BinaryOp::  [< $fn_name:camel >]);
                Self::binary_scalar_rhs_op(self, rhs, T::$fn_name, meta, stringify!([< $fn_name _scalar >]))
            } 
        
            pub fn [< scalar_ $fn_name >](lhs: T, rhs: &Tensor<T>) -> crate::Result<Tensor<T>> {
                let meta = T::AutogradMeta::on_binary_scalar_lhs_op(lhs, rhs, BinaryOp::  [< $fn_name:camel >]);
                Self::binary_scalar_lhs_op(lhs, rhs, T::$fn_name, meta, stringify!([< scalar_ $fn_name >]))
            }

            pub fn $fn_name(&self, rhs: impl Into<TensorOrScalar<T>>) -> crate::Result<Self> {
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

    pub fn clamp(&self, min: T, max: T) -> crate::Result<Self> {
        self.maximum(min)?.minimum(max)
    }
}

impl<T: NumDType> Tensor<T> {
    fn binary_op_inplace<F>(lhs: &Tensor<T>, rhs: &Tensor<T>, mut f: F, op_name: &'static str) -> crate::Result<()> 
    where 
        F: FnMut(T, T) -> T
    {
        let _ = Tensor::<T>::same_shape_binary_op(lhs, rhs, op_name)?;

        let mut lhs_storage = lhs.storage_write()?;
        let rhs_storage = rhs.storage_read()?;
        let lhs_layout = lhs.layout();
        let rhs_layout = rhs.layout();

        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");

        let lhs = lhs_storage.data_mut();
        let rhs = rhs_storage.data();
        
        lhs_layout.storage_indices().zip(rhs_layout.storage_indices())
            .for_each(|(lhs_index, rhs_index)| lhs[lhs_index] = f(lhs[lhs_index], rhs[rhs_index]));
        
        Ok(())
    }

    fn binary_op_scalar_inplace<F>(lhs: &Tensor<T>, rhs: T, mut f: F, _op_name: &'static str) -> crate::Result<()> 
    where 
        F: FnMut(T, T) -> T
    {
        let mut lhs_storage = lhs.storage_write()?;
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
            pub fn [< $fn_name _ >](&self, rhs: impl Into<TensorOrScalar<T>>) -> crate::Result<Self> {
                let rhs = rhs.into();
                match rhs {
                    TensorOrScalar::Scalar(rhs) => Self::binary_op_scalar_inplace(self, rhs, T::$fn_name, stringify!([< $fn_name _scalar_ >]))?,
                    TensorOrScalar::Tensor(rhs) => Self::binary_op_inplace(self, &rhs, T::$fn_name, stringify!([< $fn_name _scalar >]))?,
                }
                Ok(self.clone())
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
    pub fn eq(&self, rhs: impl Into<TensorOrScalar<T>>) -> crate::Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Eq)
    }

    pub fn ne(&self, rhs: impl Into<TensorOrScalar<T>>) -> crate::Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Ne)
    }

    pub fn le(&self, rhs: impl Into<TensorOrScalar<T>>) -> crate::Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Le)
    }

    pub fn ge(&self, rhs: impl Into<TensorOrScalar<T>>) -> crate::Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Ge)
    }

    pub fn lt(&self, rhs: impl Into<TensorOrScalar<T>>) -> crate::Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Lt)
    }

    pub fn gt(&self, rhs: impl Into<TensorOrScalar<T>>) -> crate::Result<Tensor<bool>> {
        self.cmp(rhs, CmpOp::Gt)
    }

    pub fn cmp(&self, rhs: impl Into<TensorOrScalar<T>>, op: CmpOp) -> crate::Result<Tensor<bool>> {
        match rhs.into() {
            TensorOrScalar::Tensor(rhs) => {
                match op {
                    CmpOp::Eq => Self::binary_op(self, &rhs, |a, b| a == b, Default::default(), "eq"),
                    CmpOp::Ne => Self::binary_op(self, &rhs, |a, b| a != b, Default::default(), "nq"),
                    CmpOp::Le => Self::binary_op(self, &rhs, |a, b| a <= b, Default::default(), "le"),
                    CmpOp::Ge => Self::binary_op(self, &rhs, |a, b| a >= b, Default::default(), "ge"),
                    CmpOp::Lt => Self::binary_op(self, &rhs, |a, b| a <  b, Default::default(), "lt"),
                    CmpOp::Gt => Self::binary_op(self, &rhs, |a, b| a >  b, Default::default(), "gt"),
                }
            }
            TensorOrScalar::Scalar(rhs) => {
                match op {
                    CmpOp::Eq => Self::binary_scalar_rhs_op(self, rhs, |a, b| a == b, Default::default(), "eq"),
                    CmpOp::Ne => Self::binary_scalar_rhs_op(self, rhs, |a, b| a != b, Default::default(), "nq"),
                    CmpOp::Le => Self::binary_scalar_rhs_op(self, rhs, |a, b| a <= b, Default::default(), "le"),
                    CmpOp::Ge => Self::binary_scalar_rhs_op(self, rhs, |a, b| a >= b, Default::default(), "ge"),
                    CmpOp::Lt => Self::binary_scalar_rhs_op(self, rhs, |a, b| a <  b, Default::default(), "lt"),
                    CmpOp::Gt => Self::binary_scalar_rhs_op(self, rhs, |a, b| a >  b, Default::default(), "gt"),
                }
            }
        }
    } 
}

impl Tensor<bool> {
    pub fn and(&self, rhs: impl Into<TensorOrScalar<bool>>) -> crate::Result<Tensor<bool>> {
        match rhs.into() {
            TensorOrScalar::Tensor(rhs) => Self::binary_op(self, &rhs, |a, b| a & b, Default::default(), "and"),
            TensorOrScalar::Scalar(rhs) => Self::binary_scalar_rhs_op(self, rhs, |a, b| a & b, Default::default(), "and"),
        }
    }

    pub fn or(&self, rhs: impl Into<TensorOrScalar<bool>>) -> crate::Result<Tensor<bool>> {
        match rhs.into() {
            TensorOrScalar::Tensor(rhs) => Self::binary_op(self, &rhs, |a, b| a | b, Default::default(), "or"),
            TensorOrScalar::Scalar(rhs) => Self::binary_scalar_rhs_op(self, rhs, |a, b| a | b, Default::default(), "or"),
        }
    }

    pub fn xor(&self, rhs: impl Into<TensorOrScalar<bool>>) -> crate::Result<Tensor<bool>> {
        match rhs.into() {
            TensorOrScalar::Tensor(rhs) => Self::binary_op(self, &rhs, |a, b| a ^ b, Default::default(), "xor"),
            TensorOrScalar::Scalar(rhs) => Self::binary_scalar_rhs_op(self, rhs, |a, b| a ^ b, Default::default(), "xor"),
        }
    }

    pub fn not(&self) -> crate::Result<Tensor<bool>> {
        if self.element_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.compute_unary_op(|v| !v)?;
        Ok(Self::from_storage(storage, self.shape(), Default::default()))
    }
}

//////////////////////////////////////////////////////////////////////////////
///        Unary Op / Unary Assign Op  for Tensor
//////////////////////////////////////////////////////////////////////////////

impl<T: WithDType> Tensor<T> {
    fn compute_unary_op<U, F>(&self, mut f: F) -> crate::Result<Storage<U>> 
    where
        U: WithDType,
        F: FnMut(T) -> U
    {
        let storage = self.storage_read()?;
        let vec = storage.data();
        let mut output = vec![];
        for index in self.layout().storage_indices() {
            output.push( f(vec[index]) );
        }
        
        Ok(Storage::new(output))
    }

    fn unary_assign_op<F>(&self, mut f: F) -> crate::Result<()>
    where
        F: FnMut(T) -> T
    {
        let mut storage = self.storage_write()?;
        let vec = storage.data_mut();
        for index in self.layout().storage_indices() {
            vec[index] = f(vec[index]);
        }
        Ok(())
    }
}

impl<T: NumDType> Tensor<T> {
    pub fn affine(&self, mul: T, add: T) -> crate::Result<Self> {
        if self.element_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.compute_unary_op(|v| v * mul + add)?;
        Ok(Self::from_storage(storage, self.shape(), Default::default()))
    }

    pub fn affine_assign(&self, mul: T, add: T) -> crate::Result<()> {
        if self.element_count() == 0 {
            return Ok(());
        }
        self.unary_assign_op(|v| v * mul + add)
    }
}

macro_rules! float_unary_op_impl {
    ($fn_name:ident) => {
        paste! {
            pub fn $fn_name(&self) -> crate::Result<Self> {
                if self.element_count() == 0 {
                    return Ok(self.clone());
                }
                let storage = self.compute_unary_op(F::$fn_name)?;
                let meta = F::AutogradMeta::on_unray_op(self, UnaryOp:: [< $fn_name:camel >]);
                Ok(Self::from_storage(storage, self.shape(), meta))
            }
        }
    };
}

impl<T: WithDType> Tensor<T> {
    pub fn map<F, O>(&self, f: F) -> crate::Result<Tensor<O>>
    where 
        O: WithDType,
        F: Fn(T) -> O,
    {
        let storage = self.compute_unary_op(f)?;
        Ok(Tensor::from_storage(storage, self.shape(), Default::default()))
    }

    pub fn map_assign<F>(&self, f: F) -> crate::Result<()>
    where 
        F: Fn(T) -> T,
    {
        if self.element_count() == 0 {
            return Ok(());
        }
        self.unary_assign_op(f)
    }
}

impl<T: NumDType + Neg<Output = T>> Tensor<T> {
    pub fn neg(&self) -> crate::Result<Self> {
        if self.element_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.compute_unary_op(Neg::neg)?;
        let meta = T::AutogradMeta::on_unray_op(self, UnaryOp::Neg);
        Ok(Self::from_storage(storage, self.shape(), meta))
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

    pub fn leaky_relu(&self, negative_slope: F) -> crate::Result<Self> {
        if self.element_count() == 0 {
            return Ok(self.clone());
        }
        let f = |v: F| F::leaky_relu(v, negative_slope);
        let storage = self.compute_unary_op(f)?;
        let meta = F::AutogradMeta::on_unray_op(self, UnaryOp::LeakyRelu(negative_slope));
        Ok(Self::from_storage(storage, self.shape(), meta))
    }
}

impl<F: FloatDType> Tensor<F> {
    pub fn pow(&self, e: F) -> crate::Result<Self> {
        if self.element_count() == 0 {
            return Ok(self.clone());
        }
        let f = |v: F| v.powf(e); 
        let storage = self.compute_unary_op(f)?;
        let meta = F::AutogradMeta::on_pow_op(self, e);
        Ok(Self::from_storage(storage, self.shape(), meta))
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
    fn test_exp_log() -> crate::Result<()> {
        let a = Tensor::new(&[0.0f32, 1.0, 2.0])?;
        let exp_a = a.exp()?;
        let log_a = exp_a.ln()?;
        assert!(a.allclose(&log_a, 1e-5, 1e-8)?);
        Ok(())
    }

    #[test]
    fn test_trig() -> crate::Result<()> {
        let a = Tensor::new(&[0.0f32, std::f32::consts::FRAC_PI_2])?;
        let sin_a = a.sin()?;
        let cos_a = a.cos()?;

        let expected_sin = Tensor::new(&[0.0f32, 1.0])?;
        let expected_cos = Tensor::new(&[1.0f32, 0.0])?;

        println!("{:?}", cos_a.iter()?.collect::<Vec<_>>());

        assert!(sin_a.allclose(&expected_sin, 1e-5, 1e-8)?);
        assert!(cos_a.allclose(&expected_cos, 1e-5, 8e-8)?);

        Ok(())
    }

    #[test]
    fn test_abs_neg() -> crate::Result<()> {
        let a = Tensor::new(&[-1.0f32, 0.0, 2.0])?;
        let abs_a = a.abs()?;
        let neg_a = a.neg()?;

        let expected_abs = Tensor::new(&[1.0f32, 0.0, 2.0])?;
        let expected_neg = Tensor::new(&[1.0f32, 0.0, -2.0])?;

        assert!(abs_a.allclose(&expected_abs, 1e-6, 1e-6)?);
        assert!(neg_a.allclose(&expected_neg, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_floor_ceil_round() -> crate::Result<()> {
        let a = Tensor::new(&[1.2f32, 2.7, -1.3])?;
        let floor_a = a.floor()?;
        let ceil_a = a.ceil()?;
        let round_a = a.round()?;

        let expected_floor = Tensor::new(&[1.0f32, 2.0, -2.0])?;
        let expected_ceil = Tensor::new(&[2.0f32, 3.0, -1.0])?;
        let expected_round = Tensor::new(&[1.0f32, 3.0, -1.0])?;

        assert!(floor_a.allclose(&expected_floor, 1e-6, 1e-6)?);
        assert!(ceil_a.allclose(&expected_ceil, 1e-6, 1e-6)?);
        assert!(round_a.allclose(&expected_round, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_floor_recip() -> crate::Result<()> {
        let a = Tensor::new(&[1.2f32, 2.7, -1.3])?;
        let recip_a = a.recip()?;
        let expected = Tensor::new(&[1.2f32.recip(), 2.7f32.recip(), -1.3f32.recip(),])?;

        assert!(recip_a.allclose(&expected, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_add_basic() -> crate::Result<()> {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0])?;
        let b = Tensor::new(&[4.0f32, 5.0, 6.0])?;
        let c = Tensor::add(&a, &b)?;
        let expected = Tensor::new(&[5.0f32, 7.0, 9.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_add_basic_variable() -> crate::Result<()> {
        let a = Tensor::new_var(&[1.0f32, 2.0, 3.0])?;
        let b = Tensor::new_var(&[4.0f32, 5.0, 6.0])?;
        let c = Tensor::add(&a, &b)?;
        let expected = Tensor::new(&[5.0f32, 7.0, 9.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_sub_basic() -> crate::Result<()> {
        let a = Tensor::new(&[10.0f32, 20.0, 30.0])?;
        let b = Tensor::new(&[1.0f32, 2.0, 3.0])?;
        let c = Tensor::sub(&a, &b)?;
        let expected = Tensor::new(&[9.0f32, 18.0, 27.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_mul_basic() -> crate::Result<()> {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0])?;
        let b = Tensor::new(&[2.0f32, 3.0, 4.0])?;
        let c = Tensor::mul(&a, &b)?;
        let expected = Tensor::new(&[2.0f32, 6.0, 12.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_div_basic() -> crate::Result<()> {
        let a = Tensor::new(&[4.0f32, 9.0, 16.0])?;
        let b = Tensor::new(&[2.0f32, 3.0, 4.0])?;
        let c = Tensor::div(&a, &b)?;
        let expected = Tensor::new(&[2.0f32, 3.0, 4.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_min_max_basic() -> crate::Result<()> {
        let a = Tensor::new(&[1.0f32, 5.0, 3.0])?;
        let b = Tensor::new(&[2.0f32, 4.0, 6.0])?;
        let min_res = Tensor::minimum(&a, &b)?;
        let max_res = Tensor::maximum(&a, &b)?;
        let expected_min = Tensor::new(&[1.0f32, 4.0, 3.0])?;
        let expected_max = Tensor::new(&[2.0f32, 5.0, 6.0])?;
        assert!(min_res.allclose(&expected_min, 1e-6, 1e-6)?);
        assert!(max_res.allclose(&expected_max, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_comparisons() -> crate::Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = Tensor::new(&[1, 0, 3])?;

        assert_eq!(a.eq(&b).unwrap().to_vec()?, [true, false, true]);
        assert_eq!(a.ne(&b).unwrap().to_vec()?, [false, true, false]);
        assert_eq!(a.lt(&b).unwrap().to_vec()?, [false, false, false]);
        assert_eq!(a.le(&b).unwrap().to_vec()?, [true, false, true]);
        assert_eq!(a.gt(&b).unwrap().to_vec()?, [false, true, false]);
        assert_eq!(a.ge(&b).unwrap().to_vec()?, [true, true, true]);

        Ok(())
    }

    #[test]
    fn test_add_mul_2d_3d() -> crate::Result<()> {
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]])?;
        let b = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]])?;
        let c = Tensor::add(&a, &b)?;
        let expected = Tensor::new(&[[6., 8.], [10., 12.]])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);

        let a3 = Tensor::new(&[
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
        ])?;
        let b3 = Tensor::new(&[
            [[2., 0.5], [1., 2.]],
            [[0.5, 2.], [1.5, 1.]],
        ])?;
        let c3 = Tensor::mul(&a3, &b3)?;
        let expected3 = Tensor::new(&[
            [[2., 1.], [3., 8.]],
            [[2.5, 12.], [10.5, 8.]],
        ])?;
        assert!(c3.allclose(&expected3, 1e-6, 1e-6)?);

        Ok(())
    }

    #[test]
    fn test_div_high_dim() -> crate::Result<()> {
        let a = Tensor::full((2, 2, 2, 2), 8.0f32)?;
        let b = Tensor::full((2, 2, 2, 2), 2.0f32)?;
        let c = Tensor::div(&a, &b)?;
        let expected = Tensor::full((2, 2, 2, 2), 4.0f32)?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);
    
        Ok(())
    }

    #[test]
    fn test_affine_and_affine_assign() -> crate::Result<()> {
        let a = Tensor::<f64>::ones((3, 3))?;
        let b = a.affine(3., 2.)?;
        let expected = Tensor::new(&[[5., 5., 5.],[5.,5.,5.],[5.,5.,5.]])?;
        assert!(b.allclose(&expected, 1e-6, 1e-6)?);

        let a2 = Tensor::<f64>::ones((3, 3))?;
        a2.affine_assign(3., 2.)?;
        assert!(a2.allclose(&expected, 1e-6, 1e-6)?);
        Ok(())
    }

    #[test]
    fn test_add_scalar() -> crate::Result<()> {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0])?;
        let b = 10.0f32;
        let c = Tensor::add(&a, b)?;
        let expected = Tensor::new(&[11.0f32, 12.0, 13.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);
        Ok(())
    }

    #[test]
    fn test_sub_scalar() -> crate::Result<()> {
        let a = Tensor::new(&[10.0f32, 20.0, 30.0])?;
        let b = 5.0f32;
        let c = Tensor::sub(&a, b)?;
        let expected = Tensor::new(&[5.0f32, 15.0, 25.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);
        Ok(())
    }

    #[test]
    fn test_mul_scalar() -> crate::Result<()> {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0])?;
        let b = 2.0f32;
        let c = Tensor::mul(&a, b)?;
        let expected = Tensor::new(&[2.0f32, 4.0, 6.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);
        Ok(())
    }

    #[test]
    fn test_div_scalar() -> crate::Result<()> {
        let a = Tensor::new(&[4.0f32, 9.0, 16.0])?;
        let b = 2.0f32;
        let c = Tensor::div(&a, b)?;
        let expected = Tensor::new(&[2.0f32, 4.5, 8.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);
        Ok(())
    }

    #[test]
    fn test_minimum_scalar() -> crate::Result<()> {
        let a = Tensor::new(&[1.0f32, 5.0, 3.0])?;
        let b = 4.0f32;
        let c = Tensor::minimum(&a, b)?;
        let expected = Tensor::new(&[1.0f32, 4.0, 3.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);
        Ok(())
    }

    #[test]
    fn test_maximum_scalar() -> crate::Result<()> {
        let a = Tensor::new(&[1.0f32, 5.0, 3.0])?;
        let b = 4.0f32;
        let c = Tensor::maximum(&a, b)?;
        let expected = Tensor::new(&[4.0f32, 5.0, 4.0])?;
        assert!(c.allclose(&expected, 1e-6, 1e-6)?);
        Ok(())
    }

    #[test]
    fn test_eq_ne_scalar() -> crate::Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = 2;

        // Tensor vs scalar
        let eq_res = a.eq(b)?;
        let expected_eq = Tensor::new(&[false, true, false])?;
        assert_eq!(eq_res.to_vec()?, expected_eq.to_vec()?);

        let ne_res = a.ne(b)?;
        let expected_ne = Tensor::new(&[true, false, true])?;
        assert_eq!(ne_res.to_vec()?, expected_ne.to_vec()?);
        Ok(())
    }

    #[test]
    fn test_lt_le_gt_ge_scalar() -> crate::Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = 2;

        let lt_res = a.lt(b)?;
        assert_eq!(lt_res.to_vec()?, [true, false, false]);

        let le_res = a.le(b)?;
        assert_eq!(le_res.to_vec()?, [true, true, false]);

        let gt_res = a.gt(b)?;
        assert_eq!(gt_res.to_vec()?, [false, false, true]);

        let ge_res = a.ge(b)?;
        assert_eq!(ge_res.to_vec()?, [false, true, true]);

        Ok(())
    }

    #[test]
    fn test_eq_ne_tensor() -> crate::Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = Tensor::new(&[1, 0, 3])?;

        let eq_res = a.eq(&b)?;
        assert_eq!(eq_res.to_vec()?, [true, false, true]);

        let ne_res = a.ne(&b)?;
        assert_eq!(ne_res.to_vec()?, [false, true, false]);

        Ok(())
    }

    #[test]
    fn test_lt_le_gt_ge_tensor() -> crate::Result<()> {
        let a = Tensor::new(&[1, 2, 3])?;
        let b = Tensor::new(&[2, 2, 1])?;

        let lt_res = a.lt(&b)?;
        assert_eq!(lt_res.to_vec()?, [true, false, false]);

        let le_res = a.le(&b)?;
        assert_eq!(le_res.to_vec()?, [true, true, false]);

        let gt_res = a.gt(&b)?;
        assert_eq!(gt_res.to_vec()?, [false, false, true]);

        let ge_res = a.ge(&b)?;
        assert_eq!(ge_res.to_vec()?, [false, true, true]);

        Ok(())
    }

    #[test]
    fn test_comparison_2d() -> crate::Result<()> {
        let a = Tensor::new(&[[1, 2], [3, 4]])?;
        let b = Tensor::new(&[[2, 2], [1, 5]])?;

        let eq_res = a.eq(&b)?;
        let expected_eq = Tensor::new(&[[false, true], [false, false]])?;
        assert_eq!(eq_res.to_vec()?, expected_eq.to_vec()?);

        let gt_res = a.gt(&b)?;
        let expected_gt = Tensor::new(&[[false, false], [true, false]])?;
        assert_eq!(gt_res.to_vec()?, expected_gt.to_vec()?);

        // Tensor vs scalar
        let le_res = a.le(3)?;
        let expected_le = Tensor::new(&[[true, true], [true, false]])?;
        assert_eq!(le_res.to_vec()?, expected_le.to_vec()?);

        Ok(())
    }

    #[test]
    fn test_std_ops() -> crate::Result<()> {
        let a = Tensor::new(&[[1., 2.], [3., 4.]])?;
        let b = Tensor::new(&[[2., 2.], [1., 5.]])?;
        let _ = a + b;

        let a = Tensor::new(&[[1., 2.], [3., 4.]])?;
        let b = Tensor::new(&[[2., 2.], [1., 5.]])?;
        let _ = &a + &b;

        let a = Tensor::new(&[[1., 2.], [3., 4.]])?;
        let b = Tensor::new(&[[2., 2.], [1., 5.]])?;
        let _ = a + b;

        let a = Tensor::new(&[[1., 2.], [3., 4.]])?;
        let b = Tensor::new(&[[2., 2.], [1., 5.]])?;
        let _ = a + b;

        Ok(())
    }
}
