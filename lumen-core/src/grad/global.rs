use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = Cell::new(true);
}

pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|c| c.set(enabled));
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|c| c.get())
}

pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = is_grad_enabled();
        set_grad_enabled(false);
        Self { prev }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}