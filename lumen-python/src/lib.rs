mod core;
mod nn;
mod dataset;
use pyo3::wrap_pymodule;
use pyo3::prelude::*;

#[pymodule]
fn lumen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<core::PyTensor>()?;
    m.add_class::<core::PyDType>()?;
    m.add_wrapped(wrap_pymodule!(nn::nn))?;
    m.add_wrapped(wrap_pymodule!(dataset::dataset))?;
    Ok(())
}
