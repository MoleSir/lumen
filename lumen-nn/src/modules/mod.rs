use std::{collections::HashMap, convert::Infallible, marker::PhantomData};
use lumen_core::{DynTensor, FloatDType, NumDType, Tensor, WithDType};
mod linear;
mod dropout;
mod embedding;
mod rnn;
mod attention;
mod softmax;
pub use linear::*;
pub use dropout::*;
pub use embedding::*;
pub use rnn::*;
pub use attention::*;
pub use softmax::*;
use thiserrorctx::Context;
use crate::{init::Initialize, NnCtxError, NnError, NnResult};

pub trait Module<T: FloatDType> {
    #[allow(unused_variables)]
    fn visit<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_mut<Visitor: ModuleVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    fn param_count(&self) -> usize {
        let mut visitor = ParamCountVisitor::new();
        self.visit(&mut visitor).unwrap();
        visitor.count
    }

    fn named_params(&self) -> HashMap<String, Tensor<T>> {
        let mut visitor = NamedParamsCountVisitor::new();
        self.visit(&mut visitor).unwrap();
        visitor.map
    }

    fn params(&self) -> Vec<Tensor<T>> {
        let mut visitor = ParamsVisitor::new();
        self.visit(&mut visitor).unwrap();
        visitor.params
    }

    fn resign(&mut self, init: &Initialize<T>) -> NnResult<()> {
        let mut visitor = InitVisitor::new(init);
        self.visit_mut(&mut visitor)
    }

    fn load_named_params(&mut self, params: &HashMap<String, DynTensor>, strict: bool) -> NnResult<()> {   
        let mut visitor = LoadParamsVisitor::new(params, strict);
        self.visit_mut(&mut visitor)
    }


}

pub trait ModuleVisitor<T: WithDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_module(&mut self, name: &str) {}

    #[allow(unused_variables)]
    fn exit_module(&mut self, name: &str) {}
}

pub trait ModuleVisitorMut<T: WithDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_module(&mut self, name: &str) {}

    #[allow(unused_variables)]
    fn exit_module(&mut self, name: &str) {}
}

//==================================================//
//         Util Modules
//==================================================//

impl<T: FloatDType> Module<T> for Tensor<T> {
    #[inline]
    fn visit<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit(self)
    }

    #[inline]
    fn visit_mut<Visitor: ModuleVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_mut(self)
    }
}

impl<T: FloatDType, M: Module<T>> Module<T> for Option<M> {
    fn visit<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(module) = self {
            module.visit(visitor)?
        }
        Ok(())
    }

    fn visit_mut<Visitor: ModuleVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(module) = self {
            module.visit_mut(visitor)?
        }
        Ok(())
    }
}

impl<T: FloatDType, M: Module<T>> Module<T> for Vec<M> {
    fn visit<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, module) in self.iter().enumerate() {
            visitor.enter_module(&i.to_string());
            module.visit(visitor)?;
            visitor.exit_module(&i.to_string());
        }
        Ok(())
    }

    fn visit_mut<Visitor: ModuleVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, module) in self.iter_mut().enumerate() {
            visitor.enter_module(&i.to_string());
            module.visit_mut(visitor)?;
            visitor.exit_module(&i.to_string());
        }
        Ok(())
    }
}

//==================================================//
//         Util Visitor
//==================================================//

struct ParamCountVisitor {
    count: usize,
}

impl ParamCountVisitor {
    fn new() -> Self {
        Self { count: 0 }
    }
}

impl<T: NumDType> ModuleVisitor<T> for ParamCountVisitor {
    type Error = Infallible;
    fn visit(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.count += param.element_count();
        Ok(())
    }
}

struct NamedParamsCountVisitor<T: NumDType> {
    map: HashMap<String, Tensor<T>>,
    path: Vec<String>,
}

impl<T: NumDType> NamedParamsCountVisitor<T> {
    fn new() -> Self {
        Self { map: HashMap::new(), path: vec![] }
    }
}

impl<T: NumDType> ModuleVisitor<T> for NamedParamsCountVisitor<T> {
    type Error = Infallible;

    fn visit(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), param.clone());
        Ok(())
    }

    fn enter_module(&mut self, name: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _: &str) {
        self.path.pop();
    }

}

struct ParamsVisitor<T: NumDType> {
    params: Vec<Tensor<T>>,
}

impl<T: NumDType> ParamsVisitor<T> {
    fn new() -> Self {
        Self { params: vec![] }
    }
}

impl<T: NumDType> ModuleVisitor<T> for ParamsVisitor<T> {
    type Error = Infallible;

    fn visit(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.params.push(param.clone());
        Ok(())
    }
}

struct InitVisitor<'a, T: FloatDType> {
    init: &'a Initialize<T>,
}

impl<'a, T: FloatDType> InitVisitor<'a, T> {
    fn new(init: &'a Initialize<T>) -> Self {
        Self { init }
    }
}

impl<'a, T: FloatDType> ModuleVisitorMut<T> for InitVisitor<'a, T> {
    type Error = NnCtxError;
    fn visit_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        let shape = param.shape();
        let new_param = self.init.init(shape)?;
        *param = new_param;
        Ok(())
    }
}

struct LoadParamsVisitor<'a, T: FloatDType> {
    params: &'a HashMap<String, DynTensor>,
    path: Vec<String>,
    strict: bool,
    _data: PhantomData<T>,
}

impl<'a, T: FloatDType> LoadParamsVisitor<'a, T> {
    fn new(params: &'a HashMap<String, DynTensor>, strict: bool,) -> Self {
        Self {
            params,
            path: Vec::new(),
            strict,
            _data: PhantomData::default(),
        }
    }
}

impl<'a, T: FloatDType> ModuleVisitorMut<T> for LoadParamsVisitor<'a, T> {
    type Error = NnCtxError;

    fn enter_module(&mut self, name: &str) {
        self.path.push(name.to_string());
    }
    
    fn exit_module(&mut self, _name: &str) {
        self.path.pop();
    }

    fn visit_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        let name = self.path.join(".");
        
        match self.params.get(&name) {
            Some(src) => {
                let src_tensor = src.as_tensor::<T>()
                    .with_context(|| format!("loading param {}", name))?;
                if param.shape() != src.shape() {
                    Err(NnError::ShapeUnmatchWhenLoadParam(param.shape().clone(), src.shape().clone()))?;
                }

                *param = src_tensor.clone();
                param.set_requires_grad(true); 
                Ok(())
            }
            None => {
                if self.strict {
                    Err(NnError::ParamNotFound(name, "load_state_dict"))?
                } else {
                    Ok(())
                }
            }
        }
    }
}