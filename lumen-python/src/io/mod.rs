mod safetensors;
use pyo3::prelude::*;
use pyo3::pymodule;

#[pymodule]
pub fn io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(safetensors::load_safetensors_file, m)?)?;
    m.add_function(wrap_pyfunction!(safetensors::save_safetensors_file, m)?)?;
    Ok(())
}
