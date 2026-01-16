use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn hello() {
    println!("hhhhhhh");
}

#[pymodule]
pub fn nn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
