use std::{fs::File, io::{Read, Seek, SeekFrom}, path::{Path, PathBuf}};
use flate2::bufread::GzDecoder;
use lumen_core::Tensor;
use crate::{transform::{Map, MapDataset}, utils, Batcher, Dataset, InMemoryDataset};

// CVDF mirror of http://yann.lecun.com/exdb/mnist/
const URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(Debug, Clone)]
pub struct MnistItem {
    pub image: [[f32; WIDTH]; HEIGHT],
    pub label: u8,
}

#[derive(Debug, Clone)]
struct MnistItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}

struct BytesToImage;

impl Map<MnistItemRaw, MnistItem> for BytesToImage {
    fn map(&self, item: &MnistItemRaw) -> MnistItem {
        assert_eq!(item.image_bytes.len(), WIDTH * HEIGHT);
        
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in item.image_bytes.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            image_array[y][x] = *pixel as f32;
        }

        MnistItem {
            image: image_array,
            label: item.label
        }
    }
}

type MnistDatasetImpl = MapDataset<InMemoryDataset<MnistItemRaw>, BytesToImage, MnistItemRaw>;

pub struct MnistDataset {
    dataset: MnistDatasetImpl,
}

impl Dataset<MnistItem> for MnistDataset {
    fn get(&self, index: usize) -> Option<MnistItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl MnistDataset {
    pub fn train<P: AsRef<Path>>(cache_dir: Option<P>) -> MnistResult<Self> {
        Self::new(MnistSplit::Train, cache_dir)
    }

    pub fn test<P: AsRef<Path>>(cache_dir: Option<P>) -> MnistResult<Self> {
        Self::new(MnistSplit::Test, cache_dir)
    }

    fn new<P: AsRef<Path>>(split: MnistSplit, cache_dir: Option<P>) -> MnistResult<Self> {
        let root = Self::download(split, cache_dir)?;
        let images = Self::read_images(&root, split)?;
        let labels = Self::read_labels(&root, split)?;

        let items: Vec<_> = images
            .into_iter()
            .zip(labels)
            .map(|(image_bytes, label)| MnistItemRaw { image_bytes, label })
            .collect();

        let dataset = InMemoryDataset::new(items);
        let dataset = MapDataset::new(dataset, BytesToImage);
    
        Ok( Self { dataset } )
    }

    fn read_images<P: AsRef<Path>>(root: &P, split: MnistSplit) -> MnistResult<Vec<Vec<u8>>> {
        let file_path = root.as_ref().join(split.image_file_name());

        // Read number of images from 16-byte header metadata
        let mut f = File::open(file_path)?;
        let mut buf = [0u8; 4];
        f.seek(SeekFrom::Start(4))?;
        f.read_exact(&mut buf)?;
        let size = u32::from_be_bytes(buf);

        let mut buf_image: Vec<u8> = vec![0u8; WIDTH * HEIGHT * (size as usize)];
        f.seek(SeekFrom::Start(16))?;
        f.read_exact(&mut buf_image)?;

        let images = buf_image
            .chunks(WIDTH * HEIGHT)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(images)
    }

    fn read_labels<P: AsRef<Path>>(root: &P, split: MnistSplit) -> MnistResult<Vec<u8>> {
        let file_path = root.as_ref().join(split.label_file_name());
        
        // Read number of labels from 8-byte header metadata
        let mut f = File::open(file_path)?;
        let mut buf = [0u8; 4];
        f.seek(SeekFrom::Start(4))?;
        f.read_exact(&mut buf)?;
        let size = u32::from_be_bytes(buf);

        let mut buf_labels: Vec<u8> = vec![0u8; size as usize];
        f.seek(SeekFrom::Start(8))?;
        f.read_exact(&mut buf_labels)?;

        Ok(buf_labels)
    }

    fn download<P: AsRef<Path>>(split: MnistSplit, cache_dir: Option<P>) -> MnistResult<PathBuf> {
        match cache_dir {
            Some(p) => Self::do_download(split, p.as_ref()),
            None => {
                let cache_dir = dirs::home_dir()
                    .expect("Could not get home directory")
                    .join(".cache")
                    .join("lumen-dataset");
                Self::do_download(split, &cache_dir)
            }
        }
    }

    fn do_download(split: MnistSplit, cache_dir: &Path) -> MnistResult<PathBuf> {
        let split_dir = cache_dir.join("mnist").join(split.as_str());

        match split {
            MnistSplit::Train => {
                Self::download_file(TRAIN_IMAGES, &split_dir)?;
                Self::download_file(TRAIN_LABELS, &split_dir)?;
            }
            MnistSplit::Test => {
                Self::download_file(TEST_IMAGES, &split_dir)?;
                Self::download_file(TEST_LABELS, &split_dir)?;
            }
        }

        Ok(split_dir)
    }

    fn download_file<P: AsRef<Path>>(name: &str, dest_dir: &P) -> MnistResult<PathBuf> {
        let file_name = dest_dir.as_ref().join(name);

        if !file_name.exists() {
            // download gzip file
            let bytes = utils::download_file_as_bytes(&format!("{URL}{name}.gz"), name)?;
            // create file to write the downloaded content to
            let mut output_file = File::create(&file_name)?;
            // Decode gzip file content and write to disk
            let mut gz_buffer = GzDecoder::new(&bytes[..]);
            std::io::copy(&mut gz_buffer, &mut output_file)?;
        }

        Ok(file_name)
    }
}

#[derive(Clone)]
pub struct MnistBatch {
    pub images: Tensor<f32>,
    pub targets: Tensor<u32>,
}

pub struct MnistBatcher;

impl Batcher<MnistItem, MnistBatch> for MnistBatch {
    type Error = MnistError;
    fn batch(&self, items: Vec<MnistItem>) -> MnistResult<MnistBatch> {
        let mut images = vec![];
        let mut targets = vec![];

        for item in items.iter() {
            let image = Tensor::new(&item.image)?;
            let image = image.reshape((1, HEIGHT, WIDTH))?;
            let image = ((image / 255.0) - 0.1307) / 0.3081;
            images.push(image);

            let target = Tensor::new(item.label as u32)?;
            let target = target.reshape((1,))?;
            targets.push(target); 
        }

        let images = Tensor::cat(&images, 0)?;
        let targets = Tensor::cat(&targets, 0)?;

        Ok(MnistBatch { images, targets })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MnistSplit {
    Train,
    Test,
}

impl MnistSplit {
    fn image_file_name(&self) -> &'static str {
        match self {
            Self::Test => TEST_IMAGES,
            Self::Train => TRAIN_IMAGES
        }
    } 

    fn label_file_name(&self) -> &'static str {
        match self {
            Self::Test => TEST_LABELS,
            Self::Train => TRAIN_LABELS
        }
    } 

    fn as_str(&self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MnistError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utils(#[from] crate::utils::UtilError),

    #[error(transparent)]
    Core(#[from] lumen_core::Error),
}

pub type MnistResult<T> = Result<T, MnistError>;