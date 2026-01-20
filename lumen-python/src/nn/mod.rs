mod functional;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::wrap_pymodule;
use pyo3::pymodule;

#[pyfunction]
fn hello() {
    println!("hhhhhhh");
}

#[pymodule]
pub fn nn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_wrapped(wrap_pymodule!(functional::functional))?;
    Ok(())
}
