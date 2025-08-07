use super::Tensor;

pub struct TensorIter<'a> {
    tensor: &'a Tensor,
    indices: Vec<usize>,
    shapes: Vec<usize>,
    counter: usize,
}

impl<'a> TensorIter<'a> {
    pub fn new(tensor: &'a Tensor) -> Self {
        Self {
            tensor,
            indices: vec![0; tensor.dim_size()],
            shapes: tensor.shape().clone(),
            counter: 0,
        }
    }
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = &'a f64;

    /// I am fucked up by Rust borror checker...
    /// So I just use unsafe
    /// You need to make sure no other Tensor mut borrow this tensor..
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter == self.tensor.element_size() {
            None
        } else {
            let index = self.tensor.calculate_flat_index(&self.indices);
            increment_counter(&mut self.indices, &self.shapes);
            self.counter += 1;

            let p = self.tensor.storage().get_ptr(index).unwrap();
            unsafe { Some( &*p ) }
        }
    }
}

pub struct TensorIterMut<'a> {
    tensor: &'a Tensor,
    indices: Vec<usize>,
    shapes: Vec<usize>,
    counter: usize,
}

impl<'a> TensorIterMut<'a> {
    pub fn new(tensor: &'a Tensor) -> Self {
        let dim_size = tensor.dim_size();
        let shapes = tensor.shape().clone();
        Self {
            tensor,
            indices: vec![0; dim_size],
            shapes,
            counter: 0,
        }
    }
}

impl<'a> Iterator for TensorIterMut<'a> {
    type Item = &'a mut f64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter == self.tensor.element_size() {
            None
        } else {
            let index = self.tensor.calculate_flat_index(&self.indices);
            increment_counter(&mut self.indices, &self.shapes);
            self.counter += 1;
            let p = self.tensor.storage().get_ptr_mut(index).unwrap();
            unsafe { Some( &mut *p ) }
        }
    }
}

fn increment_counter(indices: &mut [usize], shape: &[usize]) {
    assert_eq!(indices.len(), shape.len());
    for i in (0..indices.len()).rev() {
        indices[i] += 1;
        if indices[i] < shape[i] {
            break;
        }
        indices[i] = 0;
    }
}

pub struct TensorIntoIter {
    tensor: Tensor,
    indices: Vec<usize>,
    shapes: Vec<usize>,
    counter: usize,
}

impl TensorIntoIter {
    pub fn new(tensor: Tensor) -> Self {
        let indices = vec![0; tensor.dim_size()];
        let shapes = tensor.shape().clone();
        Self {
            tensor,
            indices,
            shapes,
            counter: 0,
        }
    }
}

impl Iterator for TensorIntoIter {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter == self.tensor.element_size() {
            None
        } else {
            let index = self.tensor.calculate_flat_index(&self.indices);
            increment_counter(&mut self.indices, &self.shapes);
            self.counter += 1;
            let p = self.tensor.storage().get_ptr_mut(index).unwrap();
            unsafe  { Some(*p) }
        }
    }
}