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

use std::fmt;
use std::marker::PhantomData;
use std::{collections::HashMap, convert::Infallible, path::Path};
use std::any::type_name;
use lumen_core::{DynTensor, FloatDType, NumDType, Tensor};
use thiserrorctx::Context;
use crate::{init::Init, NnCtxError, NnError, NnResult};

pub trait Module<T: FloatDType> : Sized {
    #[allow(unused_variables)]
    fn visit_param<Visitor: ParamVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_param_mut<Visitor: ParamVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    fn module_name() -> &'static str {
        let full_name = type_name::<Self>();
        full_name.split("::").last().unwrap_or(full_name)
    }

    fn extra_repr(&self) -> String {
        String::new()
    }

    fn display(&self) -> ModuleDisplayer<'_, Self, T> {
        ModuleDisplayer { module: self, indent: 0, _marker: std::marker::PhantomData }
    }

    fn param_count(&self) -> usize {
        let mut visitor = ParamCountVisitor::new();
        self.visit_param(&mut visitor).unwrap();
        visitor.count
    }

    fn submodule_count(&self) -> usize {
        let mut visitor = SubModuleCountVisitor::new();
        self.visit_module(&mut visitor).unwrap();
        visitor.count - 1
    }

    fn submodule_names(&self) -> Vec<String> {
        let mut visitor = SubModuleNamesVisitor::new();
        self.visit_module(&mut visitor).unwrap();
        visitor.names
    }

    fn named_params(&self) -> HashMap<String, Tensor<T>> {
        let mut visitor = NamedParamsCountVisitor::new();
        self.visit_param(&mut visitor).unwrap();
        visitor.map
    }

    fn named_dyn_params(&self) -> HashMap<String, DynTensor> {
        let mut visitor = NamedDynParamsCountVisitor::new();
        self.visit_param(&mut visitor).unwrap();
        visitor.map
    }

    fn params(&self) -> Vec<Tensor<T>> {
        let mut visitor = ParamsVisitor::new();
        self.visit_param(&mut visitor).unwrap();
        visitor.params
    }

    fn dyn_params(&self) -> Vec<DynTensor> {
        let mut visitor = DynParamsVisitor::new();
        self.visit_param(&mut visitor).unwrap();
        visitor.params
    }

    fn copy(&self) -> Self  
    where 
        Self: Clone
    {
        let mut new_module = self.clone();
        let mut visitor = CopyVisitor;
        new_module.visit_param_mut(&mut visitor).unwrap();
        new_module
    }

    fn reinit(&mut self, init: &Init<T>) -> NnResult<()> {
        let mut visitor = InitVisitor::new(init);
        self.visit_param_mut(&mut visitor)
    }

    fn load_named_params(&mut self, params: &HashMap<String, DynTensor>, strict: bool) -> NnResult<()> {   
        let mut visitor = LoadParamsVisitor::new(params, strict);
        self.visit_param_mut(&mut visitor)
    }

    fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> NnResult<()> {
        let named_params = self.named_dyn_params();
        lumen_io::safetensors::save_file(&named_params, None, path)
            .map_err(NnError::SafeTensors)
            .context("save model to safetensors")?;
        Ok(())
    }

    fn load_safetensors<P: AsRef<Path>>(&mut self, path: P, strict: bool) -> NnResult<()> {
        let params = lumen_io::safetensors::load_file(path)
            .map_err(NnError::SafeTensors)
            .context("load model param")?;
        self.load_named_params(&params.tensors, strict)
    }
}

pub trait ModuleInit<T: FloatDType> : Module<T> {
    type Error: From<NnCtxError>;
    type Config;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error>;

    fn init_default(config: &Self::Config) -> Result<Self, Self::Error> {
        Self::init(config, None)
    }
    
    fn init_with(config: &Self::Config, init: Init<T>) -> Result<Self, Self::Error> {
        Self::init(config, Some(init))
    }

    fn from_safetensors<P: AsRef<Path>>(config: &Self::Config, path: P) -> Result<Self, Self::Error> {
        let mut model = Self::init_with(config, Init::Empty)?;
        model.load_safetensors(path, true)?;
        Ok(model)
    }
}

pub trait ModuleVisitor<T: FloatDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit_module<M: Module<T>>(&mut self, module: &M) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_module_end(&mut self, module: &impl Module<T>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule(&mut self, name: &str, submodule: &impl Module<T>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn exit_submodule(&mut self, name: &str, submodule: &impl Module<T>) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub trait ParamVisitor<T: FloatDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule(&mut self, name: &str, module: &impl Module<T>) {}

    #[allow(unused_variables)]
    fn exit_submodule(&mut self, name: &str, module: &impl Module<T>) {}
}

pub trait ParamVisitorMut<T: FloatDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit_param_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule(&mut self, name: &str, module: &mut impl Module<T>) {}

    #[allow(unused_variables)]
    fn exit_submodule(&mut self, name: &str, module: &mut impl Module<T>) {}
}

//==================================================//
//         Util Modules
//==================================================//

impl<T: FloatDType> Module<T> for Tensor<T> {
    #[inline]
    fn visit_param<Visitor: ParamVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param(self)
    }

    #[inline]
    fn visit_param_mut<Visitor: ParamVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param_mut(self)
    }

    #[inline]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, _visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }
}

impl<T: FloatDType, M: Module<T>> Module<T> for Option<M> {
    fn visit_param<Visitor: ParamVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(module) = self {
            module.visit_param(visitor)?
        }
        Ok(())
    }

    fn visit_param_mut<Visitor: ParamVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(module) = self {
            module.visit_param_mut(visitor)?
        }
        Ok(())
    }

    #[inline]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(module) = self {
            module.visit_module(visitor)?
        }
        Ok(())
    }
}

impl<T: FloatDType, M: Module<T>> Module<T> for Vec<M> {
    fn visit_param<Visitor: ParamVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, module) in self.iter().enumerate() {
            visitor.enter_submodule(&i.to_string(), module);
            module.visit_param(visitor)?;
            visitor.exit_submodule(&i.to_string(), module);
        }
        Ok(())
    }

    fn visit_param_mut<Visitor: ParamVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, module) in self.iter_mut().enumerate() {
            visitor.enter_submodule(&i.to_string(), module);
            module.visit_param_mut(visitor)?;
            visitor.exit_submodule(&i.to_string(), module);
        }
        Ok(())
    }

    #[inline]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, module) in self.iter().enumerate() {
            visitor.enter_submodule(&i.to_string(), module)?;
            module.visit_module(visitor)?;
            visitor.exit_submodule(&i.to_string(), module)?;
        }
        Ok(())
    }
}

//===========================================================================//
//         Util Visitor
//===========================================================================//

/// ParamCount
struct ParamCountVisitor {
    count: usize,
}

impl ParamCountVisitor {
    fn new() -> Self {
        Self { count: 0 }
    }
}

impl<T: FloatDType> ParamVisitor<T> for ParamCountVisitor {
    type Error = Infallible;
    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.count += param.element_count();
        Ok(())
    }
}

/// NamedParams
struct NamedParamsCountVisitor<T: FloatDType> {
    map: HashMap<String, Tensor<T>>,
    path: Vec<String>,
}

impl<T: FloatDType> NamedParamsCountVisitor<T> {
    fn new() -> Self {
        Self { map: HashMap::new(), path: vec![] }
    }
}

impl<T: FloatDType> ParamVisitor<T> for NamedParamsCountVisitor<T> {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), param.clone());
        Ok(())
    }

    fn enter_submodule(&mut self, name: &str, _module: &impl Module<T>) {
        self.path.push(name.to_string());
    }

    fn exit_submodule(&mut self, _: &str, _module: &impl Module<T>) {
        self.path.pop();
    }
}

/// NamedParams
struct NamedDynParamsCountVisitor {
    map: HashMap<String, DynTensor>,
    path: Vec<String>,
}

impl NamedDynParamsCountVisitor {
    fn new() -> Self {
        Self { map: HashMap::new(), path: vec![] }
    }
}

impl<T: FloatDType> ParamVisitor<T> for NamedDynParamsCountVisitor {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), T::into_dyn(param.clone()));
        Ok(())
    }

    fn enter_submodule(&mut self, name: &str, _module: &impl Module<T>) {
        self.path.push(name.to_string());
    }

    fn exit_submodule(&mut self, _: &str, _module: &impl Module<T>) {
        self.path.pop();
    }
}

/// ParamsVisitor
struct ParamsVisitor<T: NumDType> {
    params: Vec<Tensor<T>>,
}

impl<T: NumDType> ParamsVisitor<T> {
    fn new() -> Self {
        Self { params: vec![] }
    }
}

impl<T: FloatDType> ParamVisitor<T> for ParamsVisitor<T> {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.params.push(param.clone());
        Ok(())
    }
}

/// ParamsVisitor
struct DynParamsVisitor {
    params: Vec<DynTensor>,
}

impl DynParamsVisitor {
    fn new() -> Self {
        Self { params: vec![] }
    }
}

impl<T: FloatDType> ParamVisitor<T> for DynParamsVisitor {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.params.push(T::into_dyn(param.clone()));
        Ok(())
    }
}

/// InitVisitor
struct InitVisitor<'a, T: FloatDType> {
    init: &'a Init<T>,
}

impl<'a, T: FloatDType> InitVisitor<'a, T> {
    fn new(init: &'a Init<T>) -> Self {
        Self { init }
    }
}

impl<'a, T: FloatDType> ParamVisitorMut<T> for InitVisitor<'a, T> {
    type Error = NnCtxError;
    fn visit_param_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        let shape = param.shape();
        let new_param = self.init.init(shape)?;
        *param = new_param;
        Ok(())
    }
}

/// Copy Visitor {
struct CopyVisitor;

impl<T: FloatDType> ParamVisitorMut<T> for CopyVisitor {
    type Error = Infallible;

    fn visit_param_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        *param = param.copy();
        Ok(())    
    }
}

/// LoadParamsVisitor
struct LoadParamsVisitor<'a> {
    params: &'a HashMap<String, DynTensor>,
    path: Vec<String>,
    strict: bool,
}

impl<'a> LoadParamsVisitor<'a> {
    fn new(params: &'a HashMap<String, DynTensor>, strict: bool,) -> Self {
        Self {
            params,
            path: Vec::new(),
            strict,
        }
    }
}

impl<'a, T: FloatDType> ParamVisitorMut<T> for LoadParamsVisitor<'a> {
    type Error = NnCtxError;

    fn enter_submodule(&mut self, name: &str, _module: &mut impl Module<T>) {
        self.path.push(name.to_string());
    }
    
    fn exit_submodule(&mut self, _name: &str, _module: &mut impl Module<T>) {
        self.path.pop();
    }

    fn visit_param_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        let name = self.path.join(".");
        
        match self.params.get(&name) {
            Some(src) => {
                let src_tensor = src.as_tensor::<T>()
                    .map_err(NnError::Core)
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

pub struct ModuleDisplayer<'a, M, T> {
    pub module: &'a M,
    pub indent: usize,
    pub _marker: PhantomData<T>,
}

impl<'a, M, T> fmt::Display for ModuleDisplayer<'a, M, T>
where
    M: Module<T>,
    T: FloatDType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut visitor = DisplayVisitor {
            f,
            indent: self.indent,
            child_count_stack: vec![], 
            name: String::new()
        };
        
        self.module.visit_module(&mut visitor)
    }
}

struct DisplayVisitor<'a, 'b> {
    f: &'a mut fmt::Formatter<'b>,
    indent: usize,
    child_count_stack: Vec<usize>, 
    name: String,
}

impl<'a, 'b, T: FloatDType> ModuleVisitor<T> for DisplayVisitor<'a, 'b> {
    type Error = fmt::Error;

    fn visit_module<M: Module<T>>(&mut self, module: &M) -> Result<(), Self::Error> {
        // is sub module?
        if !self.name.is_empty() {
            let parent_child_count = self.child_count_stack.last_mut().unwrap();        
            *parent_child_count += 1;
            write!(self.f, "\n{:indent$}({}): ", "", self.name, indent = self.indent)?;
        }

        write!(self.f, "{}", M::module_name())?;

        // print Module Name
        let extra = module.extra_repr();
        if !extra.is_empty() {
            write!(self.f, "({}", extra)?;
        } else {
            write!(self.f, "(")?;
        }
        
        self.child_count_stack.push(0);
        Ok(())
    }

    fn enter_submodule(&mut self, name: &str, _submodule: &impl Module<T>) -> Result<(), Self::Error> {
        self.indent += 2;
        self.name = name.to_string();
        Ok(())
    }

    fn exit_submodule(&mut self, _name: &str, _submodule: &impl Module<T>) -> Result<(), Self::Error> {
        self.indent -= 2;
        Ok(())
    }

    fn visit_module_end(&mut self, _module: &impl Module<T>) -> Result<(), Self::Error> {
        let child_count = self.child_count_stack.pop().unwrap_or(0);
        if child_count > 0 {
            write!(self.f, "\n{:indent$})", "", indent = self.indent)?;
        } else {
            write!(self.f, ")")?;
        }
        Ok(())
    }
}

// Module Count
struct SubModuleCountVisitor {
    count: usize,
}

impl SubModuleCountVisitor {
    fn new() -> Self {
        Self { count: 0 }
    }
}

impl<T: FloatDType> ModuleVisitor<T> for SubModuleCountVisitor {
    type Error = Infallible;
    fn visit_module<M: Module<T>>(&mut self, _module: &M) -> Result<(), Self::Error> {
        self.count += 1;
        Ok(())        
    }
}

// Module Named
struct SubModuleNamesVisitor {
    path: Vec<String>,
    names: Vec<String>,
}

impl SubModuleNamesVisitor {
    fn new() -> Self {
        Self { names: Vec::new(), path: Vec::new(), }
    }
}

impl<T: FloatDType> ModuleVisitor<T> for SubModuleNamesVisitor {
    type Error = Infallible;
    fn visit_module<M: Module<T>>(&mut self, _module: &M) -> Result<(), Self::Error> {
        if !self.path.is_empty() {
            self.names.push(self.path.join("."));
        }
        Ok(())        
    }

    fn enter_submodule(&mut self, name: &str, _submodule: &impl Module<T>) -> Result<(), Self::Error> {
        self.path.push(name.to_string());
        Ok(())
    }

    fn exit_submodule(&mut self, _name: &str, _submodule: &impl Module<T>) -> Result<(), Self::Error> {
        self.path.pop();
        Ok(())
    }
}