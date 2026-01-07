use std::io::{Read, Write};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use reqwest::blocking::Client;

#[derive(Debug, thiserror::Error)]
pub enum UtilError {
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}   

#[allow(unused)]
/// Download the file at the specified url.
/// File download progress is reported with the help of a [progress bar](indicatif).
///
/// # Arguments
///
/// * `url` - The file URL to download.
/// * `message` - The message to display on the progress bar during download.
///
/// # Returns
///
/// A vector of bytes containing the downloaded file data.
pub fn download_file_as_bytes(url: &str, messgae: &str) -> Result<Vec<u8>, UtilError> {
    let mut response = Client::new().get(url).send()?;
    
    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    let msg = messgae.to_owned();
    pb.set_style(
        ProgressStyle::with_template("{msg}\n    {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({eta})").unwrap()
            .with_key(
                "eta",
                |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                },
            )
            .progress_chars("▬  "),
    );
    pb.set_message(msg.clone());

    let mut bytes: Vec<u8> = Vec::with_capacity(total_size as usize);
    
    let mut buffer = [0; 8192];
    loop {
        let read_bytes = response.read(&mut buffer)?;        
        if read_bytes == 0 {
            break;
        }
        bytes.write_all(&buffer[0..read_bytes])?;        
        pb.inc(read_bytes as u64);
    }
    
    pb.finish_with_message(msg);

    Ok(bytes)
}