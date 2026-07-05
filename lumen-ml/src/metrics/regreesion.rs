use lumen_core::{FloatDType, Tensor};

pub fn mean_squar_error<T: FloatDType>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> lumen_core::Result<T> {
    (y_true - y_pred).sqr()?.sum_all()?.to_scalar()
}

pub fn mean_squared_error<T: FloatDType>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> lumen_core::Result<T> {
    Ok((y_true - y_pred).sqr()?.sum_all()?.to_scalar()?.sqrt())
}

pub fn mean_absolute_error<T: FloatDType>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> lumen_core::Result<T> {
    (y_true - y_pred).abs()?.sum_all()?.to_scalar()
}

pub fn r2_score<T: FloatDType>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> lumen_core::Result<T> {
    let y_mean = y_true.mean_all()?.to_scalar()?;

    Ok(T::ONE - ( 
        (y_true - y_pred).sqr()?.sum_all()?.to_scalar()?
    ) / (
        (y_mean - y_pred).sqr()?.sum_all()?.to_scalar()?
    ))
}