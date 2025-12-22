# lumen-core

A lightweight, statically typed Tensor library for Rust, featuring a PyTorch-like API and built-in Automatic Differentiation.

Unlike many dynamic tensor libraries, lumen leverages Rust's type system with **Static DTypes** (`Tensor<T>`). This ensures strict type safety at compile time and allows for optimized storage layouts.



## Usage

### 1. Basic Creation & Shapes

Initialize tensors using constructors like new, zeros, rand, or arange.

````rust
use lumen_core::{Tensor};

fn main() {
    // Create from array
    let a = Tensor::new(&[1, 2, 3]).unwrap();
    println!("Shape: {:?}", a.shape());

    // Create zeros
    let b = Tensor::<f32>::zeros((2, 3)).unwrap();
    println!("{}", b);

    // Random values between 0 and 1
    let c = Tensor::<f32>::rand(0., 1., (2, 3)).unwrap();
}
````

### 2. Indexing and Slicing

The library supports powerful indexing capabilities similar to Python's NumPy, allowing for efficient views of data.

````ruby
let arr = Tensor::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

// Simple index
let sub = arr.index(1).unwrap();

// Range slicing using s! macro
let sub = arr.index(s!(1:3)).unwrap();
assert_eq!(sub.shape().dims(), &[2, 5, 5]);

// Complex slicing with strides and dimensions
let sub = arr.index((s!(1:3), s!(3:4), 1)).unwrap();
assert_eq!(sub.shape().dims(), &[2, 1]);

// Using unbounded ranges (..)
let sub = arr.index((s!(1:3), .., 1..2)).unwrap();
````

### 3. Matrix Operations

Perform matrix multiplication and reshaping with ease.

````rust
let a = Tensor::arange(0., 12.).unwrap().reshape((2, 2, 3)).unwrap();
let b = Tensor::arange(0., 12.).unwrap().reshape((2, 3, 2)).unwrap();

// Matrix multiplication
let c = a.matmul(&b).unwrap();
````

### 4. Math & Activation Functions

A wide array of unary and floating-point operations are supported directly on Tensor<F>:

- **Basic:** abs, sqrt, sqr, recip, exp, ln
- **Trig:** sin, cos, tanh
- **Neural Network Activations:** relu, gelu, gelu_erf, silu, erf
- **Rounding:** floor, ceil, round

### 5. Automatic Differentiation (Autograd)

The library includes a Var<T> type for tracking gradients. Here is an example of a simple backpropagation pass (Perceptron):

````rust
// Define inputs and weights as Variables (Var) to track gradients
let w = Var::<f64>::new(&[[2.0, 3.0]]).unwrap(); // Shape (1, 2)
let x = Var::<f64>::new(&[[4.0], [5.0]]).unwrap(); // Shape (2, 1)
let b = Var::<f64>::new(&[[10.0]]).unwrap();       // Shape (1, 1)

// Forward pass: y = w * x + b
let y = w.matmul(&x).unwrap().add(&b).unwrap();

// Backward pass: Compute gradients
let grads = y.backward().unwrap();

// Verify Gradients
// dy/dw = x^T
assert!(grads[&w].allclose(&Tensor::new(&[[4.0, 5.0]]).unwrap(), 1e-5, 1e-8));

// dy/dx = w^T
assert!(grads[&x].allclose(&Tensor::new(&[[2.0], [3.0]]).unwrap(), 1e-5, 1e-8));

// dy/db = 1
assert!(grads[&b].allclose(&Tensor::new(&[[1.0]]).unwrap(), 1e-5, 1e-8));
````



## License

MIT
