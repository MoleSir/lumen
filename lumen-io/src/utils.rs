use std::io::Write;
use lumen_core::{DType, DynTensor, Shape, Tensor};

pub fn write_tensor<W: Write>(writer: &mut W, tensor: &DynTensor) -> std::io::Result<()> {
    match tensor {
        DynTensor::Bool(t) => {
            for b in t.iter() {
                writer.write_all(&[if b {1} else {0}])?;
            }
        }
        DynTensor::U8(t) => {
            for b in t.iter() {
                writer.write_all(&[b])?;
            }
        }
        DynTensor::F32(t) => {
            for v in t.iter() {
                writer.write_all(&f32::to_le_bytes(v))?;
            }
        }
        DynTensor::F64(t) => {
            for v in t.iter() {
                writer.write_all(&f64::to_le_bytes(v))?;
            }
        }
        DynTensor::U32(t) => {
            for v in t.iter() {
                writer.write_all(&u32::to_le_bytes(v))?;
            }
        }
        DynTensor::I32(t) => {
            for v in t.iter() {
                writer.write_all(&i32::to_le_bytes(v))?;
            }
        }
    }

    Ok(())
}

pub fn load_tensor(dtype: DType, shape: impl Into<Shape>, bytes: &[u8]) -> lumen_core::Result<DynTensor> {
    let shape: Shape = shape.into();
    let element_count = shape.element_count();
    let type_size = dtype.size_of();

    if bytes.len() != element_count * type_size {
        return Err(lumen_core::Error::Msg(format!(
            "Byte length mismatch. Expected {}, got {}", 
            element_count * type_size, 
            bytes.len()
        )));
    }

    match dtype {
        DType::Bool => {
            let data: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
            Ok(DynTensor::Bool(Tensor::<bool>::from_vec(data, shape)?))
        }
        DType::U8 => {
            let data: Vec<u8> = bytes.to_vec();
            Ok(DynTensor::U8(Tensor::<u8>::from_vec(data, shape)?))
        }
        DType::F32 => {
            let data: Vec<f32> = bytes
                .chunks_exact(4) 
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap(); 
                    f32::from_le_bytes(arr)
                })
                .collect();
            Ok(DynTensor::F32(Tensor::<f32>::from_vec(data, shape)?))
        }
        DType::F64 => {
            let data: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    let arr: [u8; 8] = chunk.try_into().unwrap();
                    f64::from_le_bytes(arr)
                })
                .collect();
            Ok(DynTensor::F64(Tensor::<f64>::from_vec(data, shape)?))
        }
        DType::I32 => {
            let data: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap();
                    i32::from_le_bytes(arr)
                })
                .collect();
            Ok(DynTensor::I32(Tensor::<i32>::from_vec(data, shape)?))
        }
        DType::U32 => {
            let data: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap();
                    u32::from_le_bytes(arr)
                })
                .collect();
            Ok(DynTensor::U32(Tensor::<u32>::from_vec(data, shape)?))
        }
    }
} 