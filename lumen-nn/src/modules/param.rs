use std::ops::{Deref, DerefMut};
use lumen_core::{FloatDType, Tensor};

use super::{Module, ModuleVisitor, ModuleVisitorMut, TensorVisitor, TensorVisitorMut};

#[derive(Clone)]
pub struct Parameter<T: FloatDType>(Tensor<T>);

impl<T: FloatDType> Deref for Parameter<T> {
    type Target = Tensor<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.tensor()
    }
} 

impl<T: FloatDType> DerefMut for Parameter<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor_mut()
    }
} 

impl<T: FloatDType> Parameter<T> {
    pub fn new(tensor: Tensor<T>) -> Self {
        tensor.set_requires_grad(true);
        Self(tensor)
    }

    pub fn tensor(&self) -> &Tensor<T> {
        &self.0
    }

    pub fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.0
    }
}

impl<T: FloatDType> Module<T> for Parameter<T> {
    #[inline]
    fn visit_param<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param(self.tensor())
    }

    #[inline]
    fn visit_param_mut<Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param_mut(self.tensor_mut())
    }

    #[allow(unused_variables)]
    fn visit_buffer<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_buffer_mut<Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_state<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param(self.tensor())
    }

    #[allow(unused_variables)]
    fn visit_state_mut<Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param_mut(self.tensor_mut())
    }

    #[inline]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, _visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[inline]
    fn visit_module_mut<Visitor: ModuleVisitorMut<T>>(&mut self, _visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[inline]
    fn set_train(&mut self, mode: bool) {
        self.0.set_requires_grad(mode);
    }
}
