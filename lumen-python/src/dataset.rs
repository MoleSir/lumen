use pyo3::prelude::*;

#[pymodule]
pub fn dataset(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
