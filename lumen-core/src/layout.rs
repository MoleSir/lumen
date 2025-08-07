use super::shape::Shape;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Layout {
    pub(crate) shape: Shape,
    pub(crate) stride: Vec<usize>,
    pub(crate) storage_offset: usize,
    pub(crate) is_contiguous: bool,
}

impl Layout {
    pub(crate) fn from_shape(shape: Shape) -> Self {
        let stride = Layout::calculate_stride(&shape);
        Self {
            shape,
            stride,
            storage_offset: 0,
            is_contiguous: true,
        }
    }

    pub(crate) fn new(shape: Shape, stride: Vec<usize>, storage_offset: usize, is_contiguous: bool) -> Self {
        Self {
            shape, stride, storage_offset, is_contiguous
        }
    }

    fn calculate_stride(shape: &Shape) -> Vec<usize> {
        let mut strides = vec![1; shape.dims.len()];
        let mut stride = 1;
        for i in (0..shape.dims.len()).rev() {
            strides[i] = stride;
            stride *= shape.dims[i];
        }
        strides
    }
}