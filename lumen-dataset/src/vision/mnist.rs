use std::{fs::File, io::{Read, Seek, SeekFrom}, path::Path};

use crate::{transform::{Map, MapDataset}, Dataset, InMemoryDataset};


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
    // fn new(split: &str) -> Self {

    // }

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
}

#[derive(Debug, thiserror::Error)]
pub enum MnistError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

pub type MnistResult<T> = Result<T, MnistError>;