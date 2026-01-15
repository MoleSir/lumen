mod common;
mod activation;
mod rnn;
mod attention;
mod norm;
mod geometric;
mod param;
mod buffer;

pub use common::*;
pub use activation::*;
pub use rnn::*;
pub use attention::*;
pub use norm::*;
pub use param::*;
pub use buffer::*;
pub use geometric::*;

use std::fmt;
use std::marker::PhantomData;
use std::{collections::HashMap, convert::Infallible, path::Path};
use std::any::type_name;
use lumen_core::{DynTensor, FloatDType, NumDType, Tensor};
use thiserrorctx::Context;
use crate::{init::Init, NnCtxError, NnError, NnResult};
use paste::paste;

macro_rules! impl_tensor_count {
    ( $($kind:ident),* ) => {$(paste! {
        fn  [< $kind _count >] (&self) -> usize {
            let mut visitor = TensorCountVisitor::new();
            self. [< visit_ $kind  >](&mut visitor).unwrap();
            visitor.count
        }
    })*};
}

macro_rules! impl_tensor_element_count {
    ( $($kind:ident),* ) => {$(paste! {
        fn  [< $kind _element_count >] (&self) -> usize {
            let mut visitor = TensorElementCountVisitor::new();
            self. [< visit_ $kind  >](&mut visitor).unwrap();
            visitor.count
        }
    })*};
}

macro_rules! impl_named_tensors {
    ( $($kind:ident),* ) => {$(paste! {
        fn  [< named_ $kind s >] (&self) -> HashMap<String, Tensor<T>> {
            let mut visitor = NamedTensorsVisitor::new();
            self. [< visit_ $kind  >](&mut visitor).unwrap();
            visitor.map
        }
    })*};
}

macro_rules! impl_named_dyn_tensors {
    ( $($kind:ident),* ) => {$(paste! {
        fn  [< named_dyn_ $kind s >] (&self) -> HashMap<String, DynTensor> {
            let mut visitor = NamedDynTensorsCountVisitor::new();
            self. [< visit_ $kind  >](&mut visitor).unwrap();
            visitor.map
        }
    })*};
}

macro_rules! impl_tensors {
    ( $($kind:ident),* ) => {$(paste! {
        fn  [< $kind s >] (&self) -> Vec<Tensor<T>> {
            let mut visitor = TensorsVisitor::new();
            self. [< visit_ $kind  >](&mut visitor).unwrap();
            visitor.params
        }
    })*};
}

macro_rules! impl_dyn_tensors {
    ( $($kind:ident),* ) => {$(paste! {
        fn  [< dyn_ $kind s >] (&self) -> Vec<DynTensor> {
            let mut visitor = DynTensorsVisitor::new();
            self. [< visit_ $kind  >](&mut visitor).unwrap();
            visitor.params
        }
    })*};
}

macro_rules! impl_reinit_tensors {
    ( $($kind:ident),* ) => {$(paste! {
        fn  [< reinit_ $kind s >] (&mut self, init: Init<T>) -> NnResult<()> {
            let mut visitor = InitTensorVisitor::new(init);
            self. [< visit_ $kind _mut >](&mut visitor)
        }
    })*};
}

macro_rules! impl_apply_tensors {
    ( $($kind:ident),* ) => {$(paste! {
        fn  [< apply_ $kind >] (&self, f: impl Fn(&Tensor<T>)) {
            let mut visitor = ParamApplyVisitor::new(f);
            self. [< visit_ $kind  >](&mut visitor).unwrap();
        }

        fn  [< apply_ $kind _mut >] (&mut self, f: impl Fn(&mut Tensor<T>)) {
            let mut visitor = ParamApplyVisitor::new(f);
            self. [< visit_ $kind _mut >](&mut visitor).unwrap();
        }
    })*};
}

// ============================================================================================ // 
//                        Module and ModuleInit trait
// ============================================================================================ // 

pub trait Module<T: FloatDType> : Sized {
    // ================================================================= // 
    //                           Visitor
    // ================================================================= // 

    #[allow(unused_variables)]
    fn visit_param<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_param_mut<Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
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
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_state_mut<Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_module_mut<Visitor: ModuleVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    // ================================================================= // 
    //                     Module Visitor Method
    // ================================================================= // 

    fn module_name() -> &'static str {
        let full_name = type_name::<Self>();
        full_name.split("::").last().unwrap_or(full_name)
    }

    fn display(&self) -> ModuleDisplayer<'_, Self, T> {
        ModuleDisplayer { module: self, indent: 0, _marker: std::marker::PhantomData }
    }

    fn train(&mut self, mode: bool) {
        let mut visitor = TrainModeVisitor::new(mode);
        self.visit_module_mut(&mut visitor).unwrap(); 
    }
    
    fn eval(&mut self) {
        self.train(false);
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

    // ================================================================= // 
    //                     Tensor Visitor Method
    // ================================================================= // 

    fn requires_grad(&self, mode: bool) {
        let mut visitor = RequiresGradVisitor::new(mode);
        self.visit_param(&mut visitor).unwrap();
    }

    impl_tensor_count!(param, buffer, state);
    impl_tensor_element_count!(param, buffer, state);

    impl_named_tensors!(param, buffer, state);
    impl_named_dyn_tensors!(param, buffer, state);

    impl_tensors!(param, buffer, state);
    impl_dyn_tensors!(param, buffer, state);

    impl_apply_tensors!(param, buffer, state);

    impl_reinit_tensors!(param, buffer, state);

    fn copy(&self) -> Self  
    where 
        Self: Clone
    {
        let mut new_module = self.clone();
        let mut visitor = CopyVisitor;
        new_module.visit_state_mut(&mut visitor).unwrap();
        new_module
    }

    fn load_named_states(&mut self, params: &HashMap<String, DynTensor>, strict: bool) -> NnResult<()> {   
        let mut visitor = LoadTensorsVisitor::new(params, strict);
        self.visit_state_mut(&mut visitor)
    }

    fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> NnResult<()> {
        let named_states = self.named_dyn_states();
        lumen_io::safetensors::save_file(&named_states, None, path)
            .map_err(NnError::SafeTensors)
            .context("save model to safetensors")?;
        Ok(())
    }

    fn load_safetensors<P: AsRef<Path>>(&mut self, path: P, strict: bool) -> NnResult<()> {
        let states = lumen_io::safetensors::load_file(path)
            .map_err(NnError::SafeTensors)
            .context("load model param")?;
        self.load_named_states(&states.tensors, strict)
    }

    // ================================================================= // 
    //                     Override method
    // ================================================================= // 

    fn extra_repr(&self) -> String {
        String::new()
    }

    #[allow(unused_variables)]
    fn set_train(&mut self, mode: bool) {
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

// ============================================================================================ // 
//                        Visitor  traits
// ============================================================================================ // 

pub trait ModuleVisitor<T: FloatDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit_module<M: Module<T>>(&mut self, module: &M) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_module_end<M: Module<T>>(&mut self, module: &M) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule<M: Module<T>>(&mut self, name: &str, submodule: &M) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn exit_submodule<M: Module<T>>(&mut self, name: &str, submodule: &M) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub trait ModuleVisitorMut<T: FloatDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit_module_mut<M: Module<T>>(&mut self, module: &mut M) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_module_mut_end<M: Module<T>>(&mut self, module: &mut M) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule<M: Module<T>>(&mut self, name: &str, submodule: &mut M) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn exit_submodule<M: Module<T>>(&mut self, name: &str, submodule: &mut M) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub trait TensorVisitor<T: FloatDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule<M: Module<T>>(&mut self, name: &str, module: &M) {}

    #[allow(unused_variables)]
    fn exit_submodule<M: Module<T>>(&mut self, name: &str, module: &M) {}
}

pub trait TensorVisitorMut<T: FloatDType> {
    type Error;

    #[allow(unused_variables)]
    fn visit_param_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn enter_submodule<M: Module<T>>(&mut self, name: &str, module: &mut M) {}

    #[allow(unused_variables)]
    fn exit_submodule<M: Module<T>>(&mut self, name: &str, module: &mut M) {}
}

// ============================================================================================ // 
//                        Module impl
// ============================================================================================ // 

macro_rules! impl_tensor_visit_for_option {
    ( $($kind:ident),* ) => {$(paste!{
        fn  [< visit_ $kind >]<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
            if let Some(module) = self {
                module. [< visit_ $kind >](visitor)?
            }
            Ok(())
        }
    
        fn [< visit_ $kind _mut >] <Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
            if let Some(module) = self {
                module. [< visit_ $kind _mut >](visitor)?
            }
            Ok(())
        }
    })*};
}

impl<T: FloatDType, M: Module<T>> Module<T> for Option<M> {
    impl_tensor_visit_for_option!(param, state, buffer);

    #[inline]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(module) = self {
            module.visit_module(visitor)?
        }
        Ok(())
    }

    #[inline]
    fn visit_module_mut<Visitor: ModuleVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        if let Some(module) = self {
            module.visit_module_mut(visitor)?
        }
        Ok(())
    }
}

macro_rules! impl_tensor_visit_for_vec {
    ( $($kind:ident),* ) => {$(paste!{
        fn  [< visit_ $kind >]<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
            for (i, module) in self.iter().enumerate() {
                visitor.enter_submodule(&i.to_string(), module);
                module.[< visit_ $kind >](visitor)?;
                visitor.exit_submodule(&i.to_string(), module);
            }
            Ok(())
        }
    
        fn [< visit_ $kind _mut >] <Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
            for (i, module) in self.iter_mut().enumerate() {
                visitor.enter_submodule(&i.to_string(), module);
                module.[< visit_ $kind _mut >](visitor)?;
                visitor.exit_submodule(&i.to_string(), module);
            }
            Ok(())
        }
    })*};
}

impl<T: FloatDType, M: Module<T>> Module<T> for Vec<M> {
    impl_tensor_visit_for_vec!(param, state, buffer);

    #[inline]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, module) in self.iter().enumerate() {
            visitor.enter_submodule(&i.to_string(), module)?;
            module.visit_module(visitor)?;
            visitor.exit_submodule(&i.to_string(), module)?;
        }
        Ok(())
    }

    #[inline]
    fn visit_module_mut<Visitor: ModuleVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        for (i, module) in self.iter_mut().enumerate() {
            visitor.enter_submodule(&i.to_string(), module)?;
            module.visit_module_mut(visitor)?;
            visitor.exit_submodule(&i.to_string(), module)?;
        }
        Ok(())
    }
}

//===========================================================================//
//         Util Visitor
//===========================================================================//

/// ParamCount
struct TensorCountVisitor {
    count: usize,
}

impl TensorCountVisitor {
    fn new() -> Self {
        Self { count: 0 }
    }
}

impl<T: FloatDType> TensorVisitor<T> for TensorCountVisitor {
    type Error = Infallible;
    fn visit_param(&mut self, _param: &Tensor<T>) -> Result<(), Self::Error> {
        self.count += 1;
        Ok(())
    }
}

// TensorElementCountVisitor
struct TensorElementCountVisitor {
    count: usize,
}

impl TensorElementCountVisitor {
    fn new() -> Self {
        Self { count: 0 }
    }
}

impl<T: FloatDType> TensorVisitor<T> for TensorElementCountVisitor {
    type Error = Infallible;
    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.count += param.element_count();
        Ok(())
    }
}

/// NamedParams
struct NamedTensorsVisitor<T: FloatDType> {
    map: HashMap<String, Tensor<T>>,
    path: Vec<String>,
}

impl<T: FloatDType> NamedTensorsVisitor<T> {
    fn new() -> Self {
        Self { map: HashMap::new(), path: vec![] }
    }
}

impl<T: FloatDType> TensorVisitor<T> for NamedTensorsVisitor<T> {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), param.clone());
        Ok(())
    }

    fn enter_submodule<M: Module<T>>(&mut self, name: &str, _module: &M) {
        self.path.push(name.to_string());
    }

    fn exit_submodule<M: Module<T>>(&mut self, _: &str, _module: &M) {
        self.path.pop();
    }
}

/// NamedParams
struct NamedDynTensorsCountVisitor {
    map: HashMap<String, DynTensor>,
    path: Vec<String>,
}

impl NamedDynTensorsCountVisitor {
    fn new() -> Self {
        Self { map: HashMap::new(), path: vec![] }
    }
}

impl<T: FloatDType> TensorVisitor<T> for NamedDynTensorsCountVisitor {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.map.insert(self.path.join("."), T::into_dyn(param.clone()));
        Ok(())
    }

    fn enter_submodule<M: Module<T>>(&mut self, name: &str, _module: &M) {
        self.path.push(name.to_string());
    }

    fn exit_submodule<M: Module<T>>(&mut self, _: &str, _module: &M) {
        self.path.pop();
    }
}

/// TensorsVisitor
struct TensorsVisitor<T: NumDType> {
    params: Vec<Tensor<T>>,
}

impl<T: NumDType> TensorsVisitor<T> {
    fn new() -> Self {
        Self { params: vec![] }
    }
}

impl<T: FloatDType> TensorVisitor<T> for TensorsVisitor<T> {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.params.push(param.clone());
        Ok(())
    }
}

/// DynTensorsVisitor
struct DynTensorsVisitor {
    params: Vec<DynTensor>,
}

impl DynTensorsVisitor {
    fn new() -> Self {
        Self { params: vec![] }
    }
}

impl<T: FloatDType> TensorVisitor<T> for DynTensorsVisitor {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        self.params.push(T::into_dyn(param.clone()));
        Ok(())
    }
}

/// InitTensorVisitor
struct InitTensorVisitor<T: FloatDType> {
    init: Init<T>,
}

impl<T: FloatDType> InitTensorVisitor<T> {
    fn new(init: Init<T>) -> Self {
        Self { init }
    }
}

impl<T: FloatDType> TensorVisitorMut<T> for InitTensorVisitor<T> {
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

impl<T: FloatDType> TensorVisitorMut<T> for CopyVisitor {
    type Error = Infallible;

    fn visit_param_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        *param = param.copy();
        Ok(())    
    }
}

/// LoadTensorsVisitor
struct LoadTensorsVisitor<'a> {
    params: &'a HashMap<String, DynTensor>,
    path: Vec<String>,
    strict: bool,
}

impl<'a> LoadTensorsVisitor<'a> {
    fn new(params: &'a HashMap<String, DynTensor>, strict: bool,) -> Self {
        Self {
            params,
            path: Vec::new(),
            strict,
        }
    }
}

impl<'a, T: FloatDType> TensorVisitorMut<T> for LoadTensorsVisitor<'a> {
    type Error = NnCtxError;

    fn enter_submodule<M: Module<T>>(&mut self, name: &str, _module: &mut M) {
        self.path.push(name.to_string());
    }
    
    fn exit_submodule<M: Module<T>>(&mut self, _name: &str, _module: &mut M) {
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

    fn enter_submodule<M: Module<T>>(&mut self, name: &str, _submodule: &M) -> Result<(), Self::Error> {
        self.indent += 2;
        self.name = name.to_string();
        Ok(())
    }

    fn exit_submodule<M: Module<T>>(&mut self, _name: &str, _submodule: &M) -> Result<(), Self::Error> {
        self.indent -= 2;
        Ok(())
    }

    fn visit_module_end<M: Module<T>>(&mut self, _module: &M) -> Result<(), Self::Error> {
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

    fn enter_submodule<M: Module<T>>(&mut self, name: &str, _submodule: &M) -> Result<(), Self::Error> {
        self.path.push(name.to_string());
        Ok(())
    }

    fn exit_submodule<M: Module<T>>(&mut self, _name: &str, _submodule: &M) -> Result<(), Self::Error> {
        self.path.pop();
        Ok(())
    }
}

// TrainModeVisitor
#[derive(derive_new::new)]
struct TrainModeVisitor {
    mode: bool,
}

impl<T: FloatDType> ModuleVisitorMut<T> for TrainModeVisitor {
    type Error = Infallible;
    
    fn visit_module_mut<M: Module<T>>(&mut self, module: &mut M) -> Result<(), Self::Error> {
        module.set_train(self.mode);
        Ok(())
    }
}

// RQVisitor
#[derive(derive_new::new)]
struct RequiresGradVisitor {
    mode: bool,
}

impl<T: FloatDType> TensorVisitor<T> for RequiresGradVisitor {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        param.set_requires_grad(self.mode);
        Ok(())
    }
}

#[derive(derive_new::new)]
pub struct ParamApplyVisitor<F> {
    f: F,
}

impl<T: FloatDType, F: Fn(&Tensor<T>)> TensorVisitor<T> for ParamApplyVisitor<F> {
    type Error = Infallible;

    fn visit_param(&mut self, param: &Tensor<T>) -> Result<(), Self::Error> {
        (self.f)(param);
        Ok(())
    }
}

impl<T: FloatDType, F: Fn(&mut Tensor<T>)> TensorVisitorMut<T> for ParamApplyVisitor<F> {
    type Error = Infallible;

    fn visit_param_mut(&mut self, param: &mut Tensor<T>) -> Result<(), Self::Error> {
        (self.f)(param);
        Ok(())
    }
}


