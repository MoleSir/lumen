use std::collections::HashMap;
use lumen_core::{Tensor, NumDType};
mod linear;
mod dropout;
mod embedding;
pub mod activation;
pub use activation::*;
pub use linear::*;
pub use dropout::*;
pub use embedding::*;

pub trait Module<T: NumDType> {
    #[allow(unused_variables)]
    fn visit<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) {}

    fn param_count(&self) -> usize {
        let mut visitor = ParamCountVisitor::new();
        self.visit(&mut visitor);
        visitor.count
    }

    fn named_params(&self) -> HashMap<String, Tensor<T>> {
        let mut visitor = NamedParamsCountVisitor::new();
        self.visit(&mut visitor);
        visitor.map
    }

    fn params(&self) -> Vec<Tensor<T>> {
        let mut visitor = ParamsVisitor::new();
        self.visit(&mut visitor);
        visitor.params
    }
}

pub trait ModuleVisitor<T: NumDType> {
    #[allow(unused_variables)]
    fn visit(&mut self, param: &Tensor<T>) {}

    #[allow(unused_variables)]
    fn enter_module(&mut self, name: &str) {}

    #[allow(unused_variables)]
    fn exit_module(&mut self, name: &str) {}
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
    fn visit(&mut self, param: &Tensor<T>) {
        self.count += param.element_count();
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
    fn visit(&mut self, param: &Tensor<T>) {
        self.map.insert(self.path.join("."), param.clone());
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
    fn visit(&mut self, param: &Tensor<T>) {
        self.params.push(param.clone());
    }
}