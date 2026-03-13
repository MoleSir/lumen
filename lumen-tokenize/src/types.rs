#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32, 
    pub value: String,
    pub offset: (usize, usize),
}

pub struct PreToken {
    pub value: String,
    pub offset: (usize, usize),
}

impl Token {
    pub fn new(id: u32, value: impl Into<String>, offset: (usize, usize)) -> Self {
        Self { id, value: value.into(), offset }
    }
}

impl PreToken {
    pub fn new(value: impl Into<String>, offset: (usize, usize)) -> Self {
        Self { value: value.into(), offset }
    }
}

impl Into<PreToken> for &str {
    fn into(self) -> PreToken {
        let len = self.len();
        PreToken::new(self, (0, len))
    }
}