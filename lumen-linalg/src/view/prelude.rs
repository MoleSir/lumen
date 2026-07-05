pub use super::tensor_read_guard;
pub use super::tensor_write_guard;
pub use super::{VectorView, MatrixView};
pub use crate::matrix_view;
pub use crate::matrix_view_mut;
pub use crate::vector_view;
pub use crate::vector_view_mut;

#[macro_export]
macro_rules! matrix_view {
    ($view:ident = $t:ident) => {
        paste::paste! {
            let [<$view _guard>] = tensor_read_guard(&$t)?;
            let $view = [<$view _guard>].as_matrix()?;
        }
    };
    ($t:ident) => {
        paste::paste! {
            let [<$t _guard>] = tensor_read_guard(&$t)?;
            let $t = [<$t _guard>].as_matrix()?;
        }
    };
}

#[macro_export]
macro_rules! matrix_view_mut {
    ($view:ident = $t:ident) => {
        paste::paste! {
            let mut [<$view _guard>] = tensor_write_guard(&$t)?;
            let mut $view = [<$view _guard>].as_matrix_mut()?;
        }
    };
    ($t:ident) => {
        paste::paste! {
            let mut [<$t _guard>] = tensor_write_guard(&$t)?;
            let mut $t = [<$t _guard>].as_matrix_mut()?;
        }
    };
}

#[macro_export]
macro_rules! vector_view {
    ($view:ident = $t:ident) => {
        paste::paste! {
            let [<$view _guard>] = tensor_read_guard(&$t)?;
            let $view = [<$view _guard>].as_vector()?;
        }
    };
    ($t:ident) => {
        paste::paste! {
            let [<$t _guard>] = tensor_read_guard(&$t)?;
            let $t = [<$t _guard>].as_vector()?;
        }
    };
}

#[macro_export]
macro_rules! vector_view_mut {
    ($view:ident = $t:ident) => {
        paste::paste! {
            let mut [<$view _guard>] = tensor_write_guard(&$t)?;
            let mut $view = [<$view _guard>].as_vector_mut()?;
        }
    };
    ($t:ident) => {
        paste::paste! {
            let mut [<$t _guard>] = tensor_write_guard(&$t)?;
            let mut $t = [<$t _guard>].as_vector_mut()?;
        }
    };
}