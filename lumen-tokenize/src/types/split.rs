use super::token::Token;

pub enum Split {
    AddedToken(Token),
    Origin(String),
}
