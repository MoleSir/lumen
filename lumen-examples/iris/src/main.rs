use lumen_core::*;
use lumen_dataset::DataLoader;
use lumen_dataset::DataSet;
use lumen_nn::criterion::MSELoss;
use lumen_nn::optim::SDG;
use lumen_nn::{Criterion, Model, Optim};
use lumen_nn::model::MLP;

pub struct IrisDataSet {
    feature_labels: Vec<(Tensor, Tensor)>,
}

impl IrisDataSet {
    pub fn new() -> Self {
        let context = std::fs::read_to_string("./data/iris/iris.data").unwrap();
        let feature_labels = context.lines().filter(|line| !line.is_empty()).map(|line| {
            // 6.0,2.7,5.1,1.6,Iris-versicolor
            let mut tokens = line.split(',');
            let sepal_length: f64 = tokens.next().unwrap().parse().unwrap();
            let sepal_width:  f64 = tokens.next().unwrap().parse().unwrap();
            let petal_length: f64 = tokens.next().unwrap().parse().unwrap();
            let petal_width:  f64 = tokens.next().unwrap().parse().unwrap();
            let label = IrisDataSet::str_to_classification(tokens.next().unwrap());
            let feature = Tensor::build([sepal_length, sepal_width, petal_length, petal_width], [4]).unwrap();
            let label = Tensor::one_hot(label, 3).unwrap();
            (feature, label)
        })
        .collect();

        Self { feature_labels }
    }

    fn str_to_classification(s: &str) -> usize {
        match s {
            "Iris-setosa" => 0,
            "Iris-versicolor" => 1,
            "Iris-virginica" => 2,
            _ => panic!(),
        }
    }
}

impl DataSet for IrisDataSet {
    fn get(&self, index: usize) -> (Tensor, Tensor) {
        let (features, label) = self.feature_labels.get(index).unwrap();
        (features.clone(), label.clone())
    }

    fn len(&self) -> usize {
        self.feature_labels.len()
    }
}

fn main() {
    let dataset = IrisDataSet::new();
    let dataloader = DataLoader::new(dataset, 16, true);

    let model = MLP::from_archs([4, 10, 3]).unwrap();
    let criterion = MSELoss::new();
    let mut optimizer = SDG::new(model.parameters(), 0.01);

    const EPOCHS: usize = 100;
    for epoch in 0..EPOCHS {
        for (i, (x, y)) in dataloader.iter().enumerate() {
            optimizer.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = criterion.loss(&pred, &y).unwrap();
            loss.backward();
            optimizer.step();

            if i == 0 && epoch % 10 == 0 {
                println!("Epoch: {epoch}/{EPOCHS}, Loss: {:?}", loss.mean());
            }
        }
    }
}