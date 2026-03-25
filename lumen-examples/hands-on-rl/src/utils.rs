use rand::RngExt;



pub fn arg_max_vec<T: PartialOrd>(vec: &[T]) -> usize {
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

pub fn epsilon_policy(epsilon: f64) -> bool {
    rand::rng().random_range(0.0..1.0) < epsilon
}