use crate::{Range, Tensor};

/// Calculate matmul with `lhs` and `rhs`, save result to increase `dst`
/// 
/// Ensure the shape of three Tensor's shape.
/// 
/// lhs: (r1, c1)
/// rhs: (r2, c2)
/// dst: (r1, c2)
pub(crate) fn matmul_(dst: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    let (r1, c1) = (lhs.shape()[0], lhs.shape()[1]);
    let (_, c2) = (rhs.shape()[0], rhs.shape()[1]);

    for r in 0..r1 {
        for c in 0..c2 {
            let val = (0..c1).map(|k| {
                lhs.get(&[r, k]).unwrap() * rhs.get(&[k, c]).unwrap()
            }).sum::<f64>();
            dst.increase(&[r, c], val).unwrap();
        } 
    }
}

pub(crate) fn add_(dst: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    for (dst, lhs, rhs) in zip3(dst.iter_mut(), lhs.iter(), rhs.iter()) {
        *dst = lhs + rhs;
    }
}

pub(super) fn generate_coordinates(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut coordinates = Vec::new();
    let mut indices = vec![0; shape.len()];
    let n = shape.len(); 

    loop {
        coordinates.push(indices.clone());
        let mut i = n - 1;
        while i != usize::MAX {
            indices[i] += 1;
            if indices[i] < shape[i] {
                break;
            } else {
                indices[i] = 0;
                if i == 0 {
                    return coordinates;
                }
                i -= 1;
            }
        }
    }
}

pub(super) fn generate_coordinate_ranges(shape: &[usize]) -> Vec<Vec<Range>> {
    generate_coordinates(shape).into_iter().map(|idx| -> Vec<_> {
        idx.into_iter().map(|i| Range::index(i)).collect()
    }).collect()
}

pub(super) fn zip3<A, B, C>(
    i1: impl Iterator<Item = A>, 
    i2: impl Iterator<Item = B>,
    i3: impl Iterator<Item = C>,
) -> impl Iterator<Item = (A, B, C)>
{
    i1.zip(i2).zip(i3)
        .map(|((a, b), c)| (a, b, c))
}

pub(super) fn zip5<A, B, C, D, E>(
    i1: impl Iterator<Item = A>, 
    i2: impl Iterator<Item = B>,
    i3: impl Iterator<Item = C>,
    i4: impl Iterator<Item = D>,
    i5: impl Iterator<Item = E>,
) -> impl Iterator<Item = (A, B, C, D, E)>
{
    i1.zip(i2).zip(i3).zip(i4).zip(i5)
        .map(|((((a, b), c), d), e)| (a, b, c, d, e))
}