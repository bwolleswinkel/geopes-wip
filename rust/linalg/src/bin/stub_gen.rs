use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = rlinalg::stub_info()?;
    stub.generate()?;
    Ok(())
}
