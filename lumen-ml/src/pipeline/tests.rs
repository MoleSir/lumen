use lumen_core::Tensor;
use crate::{linear::LinearRegression,  preprocessing::{MinMaxScaler, StandardScaler}, pipelines};
use super::{PredictFit, TransformFit};

#[test]
fn test_pipe_transform() {
    let f1 = StandardScaler::<f32>::default();
    let f2 = MinMaxScaler::<f32>::default();

    let fit = pipelines!(f1, f2);

    let x = Tensor::rand(-10.0f32, 10.0, (100, 25)).unwrap();
    fit.fit_transform(&x).unwrap();
}

#[test]
fn test_stand_linear() {
    let pre = StandardScaler::<f64>::default();
    let linear = LinearRegression::default();
    let fit = pipelines!(pre, linear);

    const N_SAMPLES: usize = 50;
    const N_FEATURES: usize = 5;
    let weight = Tensor::rand(-2.0, 3.0, (N_FEATURES, 1)).unwrap();
    const B: f64 = 2.5;

    let x_train = Tensor::rand(-1.0, 1.0, (N_SAMPLES, N_FEATURES)).unwrap();
    let y_train = x_train.matmul(&weight).unwrap().squeeze(1).unwrap() + B;
    let y_train = y_train + 0.1 * Tensor::randn(0.0, 1.0, (N_SAMPLES,)).unwrap();
    fit.fit(&x_train, &y_train).unwrap();
}

#[test]
fn test_pipeline3() {
    let f1 = StandardScaler::<f32>::default();
    let f2 = StandardScaler::<f32>::default();
    let f3 = StandardScaler::<f32>::default();
    pipelines!(f1, f2, f3);
}