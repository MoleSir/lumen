mod functional;
mod init;
mod param;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::pymodule;

#[pymodule]
pub fn nn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(functional::functional))?;
    m.add_class::<init::PyInit>()?;
    m.add_class::<init::PyMetaInitGuard>()?;
    m.add_class::<param::PyParameter>()?;
    m.add_class::<param::PyBuffer>()?;
    Ok(())
}
