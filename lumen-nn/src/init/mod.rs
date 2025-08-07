mod init;

pub use init::*;

use lumen_core::{Shape, Tensor};
use anyhow::Result;

pub fn zero<S: Into<Shape>>(s: S) -> Tensor {
    Init::Const(0.).var(s).unwrap()
}

pub fn one<S: Into<Shape>>(s: S) -> Tensor {
    Init::Const(1.).var(s).unwrap()
}

pub fn constant<S: Into<Shape>>(cst: f64, s: S) -> Tensor {
    Init::Const(cst).var(s).unwrap()
}

pub fn rand_normal<S: Into<Shape>>(mean: f64, stdev: f64, s: S) -> Result<Tensor> {
    Init::Randn { mean, stdev }.var(s)
}

pub fn uniform<S: Into<Shape>>(lo: f64, up: f64, s: S) -> Result<Tensor> {
    Init::Uniform { lo, up }.var(s)
}

pub fn kaiming_default_uniform<S: Into<Shape>>(s: S) -> Tensor {
    DEFAULT_KAIMING_UNIFORM.var(s).unwrap()
} 

pub fn kaiming_uniform<S: Into<Shape>>(s: S, fan: FanInOut, non_linearity: NonLinearity) -> Tensor {
    let init = Init::Kaiming { 
        dist: NormalOrUniform::Uniform, 
        fan,
        non_linearity
    };
    init.var(s).unwrap()
} 

pub fn kaiming_default_normal<S: Into<Shape>>(s: S) -> Tensor {
    DEFAULT_KAIMING_NORMAL.var(s).unwrap()
} 

pub fn kaiming_normal<S: Into<Shape>>(s: S, fan: FanInOut, non_linearity: NonLinearity) -> Tensor {
    let init = Init::Kaiming { 
        dist: NormalOrUniform::Normal, 
        fan,
        non_linearity
    };
    init.var(s).unwrap()
}

const DEFAULT_KAIMING_UNIFORM: Init = Init::Kaiming {
    dist: NormalOrUniform::Uniform,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

const DEFAULT_KAIMING_NORMAL: Init = Init::Kaiming {
    dist: NormalOrUniform::Normal,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

