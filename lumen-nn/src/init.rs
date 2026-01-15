use lumen_core::{FloatDType, Shape, Tensor};

use crate::{Buffer, NnError, NnResult, Parameter};

#[derive(Debug, Clone, Copy)]
pub enum Init<T: FloatDType> {
    Uninit,
    Empty,

    /// Fills tensor with specified value everywhere
    Constant {
        /// The value to fill the tensor with
        value: T,
    },
    /// Fills tensor with 1s everywhere
    Ones,
    /// Fills tensor with 0s everywhere
    Zeros,
    /// Fills tensor with values drawn uniformly between specified values
    Uniform {
        /// The minimum value to draw from
        min: T,

        /// The maximum value to draw from
        max: T,
    },
    /// Fills tensor with values drawn from normal distribution with specified mean and std
    Normal {
        /// The mean of the normal distribution
        mean: T,

        /// The standard deviation of the normal distribution
        std: T,
    },
    /// Fills tensor with values according to the uniform version of Kaiming initialization
    KaimingUniform {
        /// The gain to use in initialization formula
        gain: T,

        /// Whether to use fan out only in initialization formula
        fan_out_only: bool,
    },
    /// Fills tensor with values according to the uniform version of Kaiming initialization
    KaimingNormal {
        /// The gain to use in initialization formula
        gain: T,

        /// Whether to use fan out only in initialization formula
        fan_out_only: bool,
    },
    /// Fills tensor with values according to the uniform version of Xavier Glorot initialization
    /// described in [Understanding the difficulty of training deep feedforward neural networks
    /// ](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    XavierUniform {
        /// The gain to use in initialization formula
        gain: T,
    },
    /// Fills tensor with values according to the normal version of Xavier Glorot initialization
    /// described in [Understanding the difficulty of training deep feedforward neural networks
    /// ](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    XavierNormal {
        /// The gain to use in initialization formula
        gain: T,
    },
}

impl<T: FloatDType> Init<T> {
    /// Returns an init that fills the tensor with 0s.
    pub fn zeros() -> Self {
        Self::Zeros
    }

    /// Returns an init that fills the tensor with 1s.
    pub fn ones() -> Self {
        Self::Ones
    }

    /// Returns an init that fills the tensor with a constant value.
    pub fn constant(value: T) -> Self {
        Self::Constant { value }
    }

    /// Returns an init that generates values from a standard normal distribution
    /// (mean = 0.0, std = 1.0).
    /// Alias: randn
    pub fn standard_normal() -> Self {
        Self::Normal {
            mean: T::zero(),
            std: T::one(),
        }
    }
    
    /// Shortcut for standard_normal (PyTorch naming convention).
    pub fn randn() -> Self {
        Self::standard_normal()
    }

    /// Returns an init that generates values from a normal distribution.
    pub fn normal(mean: T, std: T) -> Self {
        Self::Normal { mean, std }
    }

    /// Returns an init that generates values from a uniform distribution between min and max.
    pub fn uniform(min: T, max: T) -> Self {
        Self::Uniform { min, max }
    }

    /// Returns an init that generates values from a uniform distribution between 0 and 1.
    /// Alias: rand
    pub fn standard_uniform() -> Self {
        Self::Uniform {
            min: T::zero(),
            max: T::one(),
        }
    }

    /// Shortcut for standard_uniform (PyTorch naming convention).
    pub fn rand() -> Self {
        Self::standard_uniform()
    }

    /// Kaiming (He) Uniform Initialization.
    /// Recommended for layers with ReLU/Leaky ReLU activations.
    pub fn kaiming_uniform(gain: T, fan_out_only: bool) -> Self {
        Self::KaimingUniform {
            gain,
            fan_out_only,
        }
    }

    /// Kaiming (He) Normal Initialization.
    /// Recommended for layers with ReLU/Leaky ReLU activations.
    /// Defaults to `fan_out_only = false` (fan_in mode).
    pub fn kaiming_normal(gain: T, fan_out_only: bool) -> Self {
        Self::KaimingNormal {
            gain,
            fan_out_only,
        }
    }

    /// Xavier (Glorot) Uniform Initialization.
    /// Recommended for layers with Sigmoid/Tanh activations.
    pub fn xavier_uniform(gain: T) -> Self {
        Self::XavierUniform { gain }
    }

    /// Xavier (Glorot) Normal Initialization.
    /// Recommended for layers with Sigmoid/Tanh activations.
    pub fn xavier_normal(gain: T) -> Self {
        Self::XavierNormal { gain }
    }
}

impl<T: FloatDType> Init<T> {
    /// Inits a tensor parameter of given shape with values depending on init kind.
    ///
    /// # Params
    ///
    /// - shape: Shape of the initiated tensor.
    pub fn init(&self, shape: impl Into<Shape>) -> NnResult<Tensor<T>> {
        self.do_init_with(shape, None, None)
    }

    #[inline]
    pub fn init_param(&self, shape: impl Into<Shape>) -> NnResult<Parameter<T>> {
        self.init(shape).map(Parameter::new)
    }

    #[inline]
    pub fn init_buffer(&self, shape: impl Into<Shape>) -> NnResult<Buffer<T>> {
        self.init(shape).map(Buffer::new)
    }

    pub fn init_with(&self, shape: impl Into<Shape>, fan_in: usize, fan_out: usize) -> NnResult<Tensor<T>> {
        self.do_init_with(shape, Some(fan_in), Some(fan_out))
    }

    #[inline]
    pub fn init_with_param(&self, shape: impl Into<Shape>, fan_in: usize, fan_out: usize) -> NnResult<Parameter<T>> {
        self.init_with(shape, fan_in, fan_out).map(Parameter::new)
    }

    #[inline]
    pub fn init_with_buffer(&self, shape: impl Into<Shape>, fan_in: usize, fan_out: usize) -> NnResult<Buffer<T>> {
        self.init_with(shape, fan_in, fan_out).map(Buffer::new)
    }

    fn do_init_with(&self, shape: impl Into<Shape>, fan_in: Option<usize>, fan_out: Option<usize>) -> NnResult<Tensor<T>> {
        let shape = shape.into();
        let result = match self {
            Init::Uninit => Tensor::uninit(shape),
            Init::Empty => Tensor::empty(shape),
            Init::Constant { value } => Tensor::full(shape, *value),
            Init::Ones => Tensor::ones(shape),
            Init::Zeros => Tensor::zeros(shape),
            Init::Uniform { min, max } => Tensor::rand(*min, *max, shape),
            Init::Normal { mean, std } => Tensor::randn(*mean, *std, shape),
            Init::KaimingUniform { gain, fan_out_only } => {
                let a = T::from_f64(3.0).sqrt() * *gain * self.kaiming_std(*fan_out_only, fan_in, fan_out);
                Tensor::rand(-a, a, shape)
            }
            Init::KaimingNormal { gain, fan_out_only } => {
                let std = *gain * self.kaiming_std(*fan_out_only, fan_in, fan_out);
                Tensor::randn(T::zero(), std, shape)
            }
            Init::XavierUniform { gain } => {
                let a = T::from_f64(3.0).sqrt() * *gain * self.xavier_std(fan_in, fan_out);
                Tensor::rand(-a, a, shape)
            }
            Init::XavierNormal { gain } => {
                let std = *gain * self.xavier_std(fan_in, fan_out);
                Tensor::randn(T::zero(), std, shape)
            }
        };

        let tensor = result.map_err(NnError::Core)?;
        Ok(tensor)
    }

    fn kaiming_std(
        &self,
        fan_out_only: bool,
        fan_in: Option<usize>,
        fan_out: Option<usize>,
    ) -> T {
        let fan = if fan_out_only { fan_out } else { fan_in };
        let fan = fan.expect(
            "Can't use Kaiming initialization without specifying fan. Use init_with method.",
        );

        T::one() / T::from_usize(fan).sqrt()
    }

    fn xavier_std(&self, fan_in: Option<usize>, fan_out: Option<usize>) -> T {
        let fan_in = fan_in.expect(
            "Can't use Xavier initialization without specifying fan in. Use init_with method and \
             provide fan_in.",
        );
        let fan_out = fan_out.expect(
            "Can't use Xavier initialization without specifying fan out. Use init_with method and \
             provide fan_out.",
        );
        T::two() / T::from_usize(fan_in + fan_out).sqrt()
    }
}
