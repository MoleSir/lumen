mod base;
mod dtype;
mod ops;
mod error;
mod grad;
mod dynamic;
pub mod utils;
pub use base::*;
pub use dtype::*;
pub use ops::*;
pub use error::*;
pub use grad::*;
pub use dynamic::*;
pub use half::bf16;

#[macro_export]
macro_rules! inplace_warning_doc {
    () => {
        "> ** WARNING: Bypasses Autograd**\n\
         > \n\
         > This is an unsafe operation that bypasses the autograd engine.\n\
         > It does NOT track computation history and cannot compute gradients.\n\
         > Attempting to call this method on a tensor with `requires_grad=True`\n\
         > will raise an Error."
    };
}