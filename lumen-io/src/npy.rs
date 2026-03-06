use lumen_core::Shape;
use lumen_core::Tensor;
use lumen_core::WithDType;
use thiserrorctx::Context;
use zip::result::ZipError;
use zip::write::SimpleFileOptions;
use zip::ZipArchive;
use zip::ZipWriter;
use std::collections::HashMap;
use std::io::Cursor;
use std::io::Seek;
use std::str::Utf8Error;
use std::{fs::File, io::{BufReader, BufWriter, Read, Write}, path::Path};
use lumen_core::{DynTensor, DType};
use crate::utils;

#[thiserrorctx::context_error]
pub enum NpyError {
    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error(transparent)]
    Zip(#[from] ZipError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utf8(#[from] Utf8Error),

    // === Npy Error ===
    #[error("not a npy file for can't read correct magic")]
    NotNpyFile,

    #[error("unsupported NPY version {major}.{minor}")]
    UnsupportVersion {
        major: u8,
        minor: u8,
    },

    #[error("{error} in {field} field")]
    Header {
        error: String,
        field: &'static str 
    },

    #[error("no exit {field} field")]
    LackField {
        field: &'static str 
    },

    #[error("unsupport descr {0}")]
    UnsupportedDescr(String),

    #[error("Numpy does't support usize dtype")]
    UnsupportUSize,
}

const NPY_MAGIC: &[u8] = b"\x93NUMPY";


/// Load a single `.npy` file into a [`DynTensor`].
///
/// # Arguments
/// * `path` - Path to the `.npy` file.
///
/// # Errors
/// Returns an error if the file cannot be opened, read, or parsed.
pub fn load_npy_file<P: AsRef<Path>>(path: P) -> NpyResult<DynTensor> {
    let file = File::open(path).map_err(NpyError::Io).context("open file")?;
    let mut reader = BufReader::new(file);
    load_npy(&mut reader)
}

pub fn load_npy<R: Read + Seek>(mut reader: &mut R) -> NpyResult<DynTensor> {
    // Read magic
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic != NPY_MAGIC {
        Err( NpyError::NotNpyFile )?;
    }

    // Read version
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;
    let version = (version[0], version[1]);

    // Read header
    let header_len = match version {
        (1, 0) => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            u16::from_le_bytes(buf) as usize
        }
        (2, 0) | (3, 0) => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        }
        _ => Err( NpyError::UnsupportVersion { major: version.0, minor: version.1 })?,
    };

    // Read Header
    let mut header_bytes = vec![0u8; header_len];   
    reader.read_exact(&mut header_bytes)?;
    let header_str = str::from_utf8(&header_bytes)?.trim();

    // Parse header
    // "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }"
    let header_str = header_str.trim_matches(|c| c == '{' || c == '}');

    // 'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), 
    let mut descr_opt: Option<&str> = None;
    let mut fortran_order_opt: Option<bool> = None;
    let mut shape_opt: Option<Vec<usize>> = None;

    let mut header_str = header_str;
    while !header_str.is_empty() {
        if header_str.starts_with("'descr'") {
            let colon_index = header_str.find(",")
                .ok_or_else(|| NpyError::Header { 
                    field: "descr", 
                    error: "No colon".into() 
                })?;
            // ": '<f8'"
            let descr = &header_str[7..colon_index];
            let ref1_index = descr.find("'")
                .ok_or_else(|| NpyError::Header { 
                    field: "descr", 
                    error: "No start ' in 'descr' field value".into() 
                })?;
            // "<f8'"
            let descr = &descr[ref1_index+1..];
            let ref2_index = descr.find("'")
                .ok_or_else(|| NpyError::Header { 
                    field: "descr", 
                    error: "No end ' in 'descr' field value".into() 
                })?;
            // "<f8"
            let descr = &descr[..ref2_index];
            descr_opt = Some(descr);

            header_str = header_str[colon_index+1..].trim_start();
        } else if header_str.starts_with("'fortran_order'") {
            let colon_index = header_str.find(",")
                .ok_or_else(|| NpyError::Header { 
                    field: "fortran_order", 
                    error: "No colon in 'fortran_order' field".into() 
                })?;
            // ": False"
            let fortran_order = &header_str[15..colon_index];
            let index = fortran_order.find(":")
                .ok_or_else(|| NpyError::Header { 
                    field: "fortran_order", 
                    error: "No : in 'fortran_order' field".into() 
                })?;
            // "False"
            let fortran_order = fortran_order[index+1..].trim();
            match fortran_order {
                "False" => fortran_order_opt = Some(false),
                "True" => fortran_order_opt = Some(true),
                _ => Err(NpyError::Header { 
                    field: "fortran_order", 
                    error: format!("Unsupported value '{}'", fortran_order) 
                })?,
            };

            header_str = header_str[colon_index+1..].trim_start();
        } else if header_str.starts_with("'shape'") {
            // "'shape': (3, 4), "
            let left_brace_index = header_str.find("(")
                .ok_or_else(|| NpyError::Header { 
                    field: "shape", 
                    error: "No ( in 'shape' field".into() 
                })?;
            let right_brace_index = header_str.find(")")
                .ok_or_else(|| NpyError::Header { 
                    field: "shape", 
                    error: "No ) in 'shape' field".into() 
                })?;
            // 3, 4
            let shape = &header_str[left_brace_index + 1..right_brace_index];
            let shape: Vec<usize> = shape
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            shape_opt = Some(shape);

            header_str = header_str[right_brace_index+2..].trim_start();                
        }
    }

    // Check header
    let descr = descr_opt.ok_or_else(|| NpyError::LackField { field: "descr" } )?;
    let _ = fortran_order_opt.ok_or_else(|| NpyError::LackField { field: "fortran_order" } )?;
    let shape = shape_opt.ok_or_else(|| NpyError::LackField { field: "shape" } )?;
    let shape: Shape = shape.into();

    match descr {
        "|b1" => Ok(utils::load_tensor_reader(DType::Bool, shape, &mut reader).map_err(NpyError::Core)?),
        "<u1" => Ok(utils::load_tensor_reader(DType::U8, shape, &mut reader).map_err(NpyError::Core)?),
        "<i4" => Ok(utils::load_tensor_reader(DType::I32, shape, &mut reader).map_err(NpyError::Core)?),
        "<u4" => Ok(utils::load_tensor_reader(DType::U32, shape, &mut reader).map_err(NpyError::Core)?),
        "<f4" => Ok(utils::load_tensor_reader(DType::F32, shape, &mut reader).map_err(NpyError::Core)?),
        "<f8" => Ok(utils::load_tensor_reader(DType::F64, shape, &mut reader).map_err(NpyError::Core)?),
        _ => Err(NpyError::UnsupportedDescr(descr.to_string()))?
    }
}

/// Load multiple arrays from a `.npz` archive into a `HashMap`.
///
/// Each entry in the returned map corresponds to one `.npy` file inside the archive,
/// where the key is the file stem (without extension) and the value is the array.
///
/// # Arguments
/// * `path` - Path to the `.npz` archive.
///
/// # Errors
/// Returns an error if the archive cannot be opened, read, or parsed.
pub fn load_npz_file<P: AsRef<Path>>(path: P) -> NpyResult<HashMap<String, DynTensor>> {
    let file = std::fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut arrays = HashMap::new();

    for i in 0..archive.len() {
        let mut zip_file = archive.by_index(i)?;
        let mut buffer = Vec::new();
        use std::io::Read;
        zip_file.read_to_end(&mut buffer)?;
        
        let mut cursor = Cursor::new(buffer);
        let array = load_npy(&mut cursor)?;

        let name = Path::new(zip_file.name())
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(zip_file.name())
            .to_string();
        arrays.insert(name, array);
    }

    Ok(arrays)
}

/// Save a [`DynTensor`] into a `.npy` file.
///
/// # Arguments
/// * `tensor` - The array to save.
/// * `path` - Path to the output `.npy` file.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn save_npy_file<P: AsRef<Path>>(tensor: impl Into<DynTensor>, path: P) -> NpyResult<()> {
    let file = File::create(path).map_err(NpyError::Io).context("create file")?;
    let mut writer = BufWriter::new(file);
    save_npy(tensor, &mut writer)
}

pub fn save_npy<W: Write>(tensor: impl Into<DynTensor>, writer: &mut W) -> NpyResult<()> {
    let tensor = tensor.into();
    match tensor {
        DynTensor::Bool(t) => _save_npy(t, writer),
        DynTensor::U8(t) => _save_npy(t, writer),
        DynTensor::U32(t) => _save_npy(t, writer),
        DynTensor::I32(t) => _save_npy(t, writer),
        DynTensor::F32(t) => _save_npy(t, writer),
        DynTensor::F64(t) => _save_npy(t, writer),
    }
}

pub fn save_npz_file<P: AsRef<Path>>(tensors: &HashMap<String, DynTensor>, path: P) -> NpyResult<()> {
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);

    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .unix_permissions(0o755);
    
    for (name, tensor) in tensors {
        zip.start_file(format!("{}.npy", name), options)?;
        save_npy(tensor.clone(), &mut zip)?
    }

    zip.finish()?;
    Ok(())
}

fn _save_npy<D: WithDType, W: Write>(tensor: Tensor<D>, writer: &mut W) -> NpyResult<()> {
    let descr = dtype_to_descr(D::DTYPE)?;
    let version = (1, 0);
    let fortran_order = false;
    let shape = tensor.dims();

    let storage = tensor.storage_read()?;
    let data_vec = storage.data();
    let bytes: Vec<u8> = bytemuck::cast_slice(data_vec).to_vec();

    writer.write_all(NPY_MAGIC)?;

    writer.write_all(&[version.0, version.1])?;

    let dict = format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': ({}), }}",
        descr,
        if fortran_order { "True" } else { "False" },
        shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")
    );

    // header padding 
    let header_len = dict.len() + 1; // \n
    if header_len > u16::MAX as usize {
        panic!("Header too long for version 1.0 NPY");
    }
    let padding = (64 - ((10 + 2 + header_len) % 64)) % 64;
    let mut header_bytes = dict.into_bytes();
    header_bytes.push(b'\n');
    header_bytes.extend(vec![b' '; padding]);

    // header_len
    let header_len_u16: u16 = header_bytes.len().try_into().unwrap();
    writer.write_all(&header_len_u16.to_le_bytes())?;

    // header
    writer.write_all(&header_bytes)?;

    // data
    writer.write_all(&bytes)?;

    writer.flush()?;
    Ok(())
}

fn dtype_to_descr(dtype: DType) -> NpyResult<&'static str> {
    match dtype {
        DType::Bool => Ok("|b1"),
        DType::U8 => Ok("<u1"),
        DType::U32 => Ok("<u4"),
        DType::I32 => Ok("<i4"),
        DType::F32 => Ok("<f4"),
        DType::F64 => Ok("<f8"),
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use lumen_core::{DynTensor, Tensor};
    use tempfile::NamedTempFile;

    use crate::npy::{load_npy_file, save_npy_file};

    use super::{load_npz_file, save_npz_file};

    #[test]
    fn test_to_tensor() {
        let tensor = load_npy_file("./bench/test1.npy").unwrap().as_f32().unwrap();
        println!("{}", tensor);

        let tensor = load_npy_file("./bench/test3.npy").unwrap().as_bool().unwrap();
        println!("{}", tensor);

        let tensor = load_npy_file("./bench/test4.npy").unwrap().as_i32().unwrap();
        println!("{}", tensor);

        let _tensor = load_npy_file("./bench/test5.npy").unwrap().as_f64().unwrap();
    }

    #[test]
    fn test_write_npy() {
        let tmpfile = NamedTempFile::new().unwrap();
        let tensor = Tensor::<f32>::randn(0., 1., (4, 5)).unwrap();
        save_npy_file(&tensor, tmpfile.path()).unwrap();

        let loaded_tensor = load_npy_file(tmpfile.path()).unwrap().as_f32().unwrap();
        assert!(loaded_tensor.allclose(&tensor, 1e-6, 1e-6).unwrap());
    }

    #[test]
    fn test_load_npz() {
        let tensors = load_npz_file("./bench/test1.npz").unwrap();
        for (name, t) in tensors {
            println!("{}: {}", name, t.shape());
        }
    }

    #[test]
    fn test_save_npz() {
        let tmpfile = NamedTempFile::new().unwrap();

        let scalar = Tensor::new(1).unwrap();
        let vector_f32 = Tensor::new(&[1.0f32, 2., 3.]).unwrap();
        let matrix_f32 = Tensor::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let ones_f32 = Tensor::<f32>::ones((2, 9)).unwrap();
        let randn_f64 = Tensor::randn(0.0f64, 1., (1, 2, 3)).unwrap();
        let fill_f64 = Tensor::full((2, 3, 4), 1.2).unwrap();
        let arange_f64 = Tensor::arange(0., 10.).unwrap();
        let trues = Tensor::trues((3, 4)).unwrap();
        let booleans = Tensor::new(&[[true, false], [false, true]]).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("scalar".to_string(), DynTensor::I32(scalar));
        tensors.insert("vector_f32".to_string(), DynTensor::F32(vector_f32));
        tensors.insert("matrix_f32".to_string(), DynTensor::I32(matrix_f32));
        tensors.insert("ones_f32".to_string(), DynTensor::F32(ones_f32));
        tensors.insert("randn_f64".to_string(), DynTensor::F64(randn_f64));
        tensors.insert("fill_f64".to_string(), DynTensor::F64(fill_f64));
        tensors.insert("arange_f64".to_string(), DynTensor::F64(arange_f64));
        tensors.insert("trues".to_string(), DynTensor::Bool(trues));
        tensors.insert("booleans".to_string(), DynTensor::Bool(booleans));

        save_npz_file(&tensors, tmpfile.path()).unwrap();

        let tensors = load_npz_file(tmpfile.path()).unwrap();
        for (name, t) in tensors {
            println!("{}: {}", name, t.shape());
            match name.as_str() {
                "scalar" => assert!(t.as_i32().unwrap().is_scalar()),
                "vector_f32" => assert_eq!(t.as_f32().unwrap().rank(), 1),
                "matrix_f32" => assert_eq!(t.as_i32().unwrap().dims(), [2, 3]),
                _ => {}
            }
        }
    }
}