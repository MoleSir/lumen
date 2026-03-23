mod gelu_erf;
mod gelu;
mod silu;
mod leaky_relu;
mod recip;
mod relu;
mod sigmoid;
mod tanh;

pub use gelu_erf::*;
pub use gelu::*;
pub use silu::*;
pub use leaky_relu::*;
pub use recip::*;
pub use relu::*;
pub use sigmoid::*;
pub use tanh::*;
use std::str::FromStr;
use paste::paste;
use crate::{NnCtxError, NnError};
use super::ModuleForward;
use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;

#[derive(Module, Clone)]
pub enum Activate {
    GeluErf(GeluErf),
    Gelu(Gelu),
    Recip(Recip),
    Relu(Relu),
    Sigmoid(Sigmoid),
    Silu(Silu),
    Tanh(Tanh),
}

macro_rules! impl_activate {
    ($($name:ident),*) => {
        paste! {
            $(
                impl Activate {
                    #[inline]
                    pub fn $name() -> Self {
                        Self::[< $name:camel >]([< $name:camel >]::new())
                    } 
                }
            )*

            impl FromStr for Activate {
                type Err = NnCtxError;
            
                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    match s {
                        $(
                            stringify!($name) => Ok(Self::$name()),
                        )*
                        _ => Err(NnError::UnsupportActivate(s.to_string()))?,
                    }
                }
            }

            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub enum ActivateKind {
                $(
                    [< $name:camel >],
                )*
            }

            impl Activate {
                pub fn new(kind: ActivateKind) -> Self {
                    match kind {
                        $(
                            ActivateKind::[< $name:camel >] => Self::$name(),
                        )*
                    }
                }
            }

            $(
                impl Into<Activate> for [< $name:camel >] {
                    fn into(self) -> Activate {
                        Activate::$name()
                    }
                }
            )*
        }
    };
}

impl_activate!(
    gelu_erf,
    gelu,
    recip,
    relu,
    sigmoid,
    silu,
    tanh
);

impl<T: FloatDType> ModuleForward<T> for Activate {
    type Error = NnCtxError;
    type Input = Tensor<T>;
    type Output = Tensor<T>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        match self {
            Self::GeluErf(a) => ModuleForward::forward(a, input),
            Self::Gelu(a) => ModuleForward::forward(a, input),
            Self::Recip(a) => ModuleForward::forward(a, input),
            Self::Relu(a) => ModuleForward::forward(a, input),
            Self::Sigmoid(a) => ModuleForward::forward(a, input),
            Self::Silu(a) => ModuleForward::forward(a, input),
            Self::Tanh(a) => ModuleForward::forward(a, input),
        }
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use super::Activate;

    #[test]
    fn test_new() {
        let _ = Activate::from_str("gelu").unwrap();
        let _ = Activate::from_str("silu").unwrap();
    }
}