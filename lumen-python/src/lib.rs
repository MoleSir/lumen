mod core;
mod nn;
mod io;
use pyo3::wrap_pymodule;
use pyo3::prelude::*;

#[pymodule]
fn _lumen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<core::PyTensor>()?;
    m.add_class::<core::PyDType>()?;
    m.add_class::<core::PyGradStore>()?;
    
    m.add_function(wrap_pyfunction!(core::set_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(core::is_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(core::no_grad, m)?)?;

    m.add_wrapped(wrap_pymodule!(nn::nn))?;
    m.add_wrapped(wrap_pymodule!(io::io))?;
    Ok(())
}
