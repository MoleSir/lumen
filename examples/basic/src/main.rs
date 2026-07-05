use lumen_core::Tensor;

fn main() -> anyhow::Result<()> {
    let a = Tensor::randn(0f32, 1., (2, 3))?;
    let b = Tensor::randn(0f32, 1., (3, 4))?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}