use std::{fs::File, io::{BufRead, BufReader}, path::{Path, PathBuf}};
use lumen_core::Tensor;
use crate::{utils, Batcher, Dataset, InMemoryDataset};

// UCI Machine Learning Repository mirror
const URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/";
const FILE_NAME: &str = "iris.data";

const FEATURE_DIM: usize = 4;

#[derive(Debug, Clone)]
pub struct IrisItem {
    // [sepal_length, sepal_width, petal_length, petal_width]
    pub features: [f32; FEATURE_DIM],
    // 0: Setosa, 1: Versicolor, 2: Virginica
    pub label: u32,
}

pub struct IrisDataset {
    dataset: InMemoryDataset<IrisItem>,
}

impl Dataset<IrisItem> for IrisDataset {
    fn get(&self, index: usize) -> Option<IrisItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl IrisDataset {
    pub fn load<P: AsRef<Path>>(cache_dir: Option<P>) -> IrisResult<Self> {
        let file_path = Self::download(cache_dir)?;
        let items = Self::read_data(&file_path)?;
        
        let dataset = InMemoryDataset::new(items);
        Ok(Self { dataset })
    }

    fn read_data<P: AsRef<Path>>(path: &P) -> IrisResult<Vec<IrisItem>> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let mut items = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            // CSV: 5.1,3.5,1.4,0.2,Iris-setosa
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != 5 {
                eprintln!("Warning: Skipping malformed line {}: {}", line_idx, line);
                return Err(IrisError::InvalidLine(line.to_string()))
            }

            let features = [
                parts[0].parse::<f32>().map_err(|_| IrisError::ParseFormat(line_idx))?,
                parts[1].parse::<f32>().map_err(|_| IrisError::ParseFormat(line_idx))?,
                parts[2].parse::<f32>().map_err(|_| IrisError::ParseFormat(line_idx))?,
                parts[3].parse::<f32>().map_err(|_| IrisError::ParseFormat(line_idx))?,
            ];

            let label = match parts[4] {
                "Iris-setosa" => 0,
                "Iris-versicolor" => 1,
                "Iris-virginica" => 2,
                _ => return Err(IrisError::UnknownLabel(parts[4].to_string())),
            };

            items.push(IrisItem { features, label });
        }

        Ok(items)
    }

    fn download<P: AsRef<Path>>(cache_dir: Option<P>) -> IrisResult<PathBuf> {
        match cache_dir {
            Some(p) => Self::do_download(p.as_ref()),
            None => {
                let cache_dir = dirs::home_dir()
                    .expect("Could not get home directory")
                    .join(".cache")
                    .join("lumen-dataset");
                Self::do_download(&cache_dir)
            }
        }
    }

    fn do_download(cache_dir: &Path) -> IrisResult<PathBuf> {
        let dest_dir = cache_dir.join("iris");
        let file_path = dest_dir.join(FILE_NAME);

        if !file_path.exists() {
            let _ = utils::download_file_as_bytes(&format!("{URL}{FILE_NAME}"), FILE_NAME)?;            
            let bytes = utils::download_file_as_bytes(&format!("{URL}{FILE_NAME}"), FILE_NAME)?;
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file_path, bytes)?;
        }

        Ok(file_path)
    }
}

#[derive(Clone)]
pub struct IrisBatch {
    pub features: Tensor<f32>,
    pub targets: Tensor<u32>,
}

pub struct IrisBatcher;

impl Batcher<IrisItem, IrisBatch> for IrisBatcher {
    type Error = IrisError;
    
    fn batch(&self, items: Vec<IrisItem>) -> IrisResult<IrisBatch> {
        let mut features_vec = vec![];
        let mut targets_vec = vec![];

        for item in items.iter() {
            let feature = Tensor::new(&item.features)?;
            let feature = feature.reshape((1, FEATURE_DIM))?;
            features_vec.push(feature);
            targets_vec.push(item.label);
        }

        let features = Tensor::cat(&features_vec, 0)?;
        let len = targets_vec.len();
        let targets = Tensor::new(targets_vec)?.reshape((len, 1))?;

        Ok(IrisBatch { features, targets })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IrisError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utils(#[from] crate::utils::UtilError),

    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error("Failed to parse float at line {0}")]
    ParseFormat(usize),

    #[error("Unknown label found: {0}")]
    UnknownLabel(String),

    #[error("Invalid line: {0}")]
    InvalidLine(String),
}

pub type IrisResult<T> = Result<T, IrisError>;