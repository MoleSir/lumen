use std::cell::{RefCell, Ref, RefMut};
use std::rc::Rc;

pub type DataRef<'a> = Ref<'a, f64>;
pub type DataRefMut<'a> = RefMut<'a, f64>;

#[derive(Debug, Clone, PartialEq)]
pub struct Storage {
    pub(crate) data: Rc<RefCell<Vec<f64>>>,
}

impl Storage {
    pub fn new(data: Vec<f64>) -> Self {
        Storage {
            data: Rc::new(RefCell::new(data)),
        }
    }

    pub fn get(&self, index: usize) -> Option<DataRef> {
        let data = self.data.borrow();
        if index < data.len() {
            Some(Ref::map(data, |data| &data[index]))
        } else {
            None
        }
    }

    pub fn get_mut<'a>(&'a self, index: usize) -> Option<DataRefMut<'a>> {
        let data = self.data.borrow_mut();
        if index < data.len() {
            Some(RefMut::map(data, |data| &mut data[index]))
        } else {
            None
        }
    }

    pub fn get_ptr(&self, index: usize) -> Option<*const f64> {
        self.data.borrow().get(index).map(|r| r as *const f64)
    }

    pub fn get_ptr_mut(&self, index: usize) -> Option<*mut f64> {
        self.data.borrow_mut().get_mut(index).map(|r| r as *mut  f64)
    }

    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }
}
