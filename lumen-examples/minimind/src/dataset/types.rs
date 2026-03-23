use std::fmt::Display;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "role")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub functions: Vec<ToolDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    pub content: String,  
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMessage {
    pub content: String,
    pub tool_call_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
	pub name: String,
    pub arguments: String,
}


impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema
        }
    }
}

impl Message {
    pub fn role(&self) -> Role {
        match self {
            Self::Assistant(_) => Role::Assistant,
            Self::User(_) => Role::User,
            Self::Tool(_) => Role::Tool,
            Self::System(_) => Role::System,
        }
    }

    pub fn system(content: impl Into<String>, functions: impl Into<Vec<ToolDefinition>>) -> Self {
        Self::System(SystemMessage {
            content: content.into(), functions: functions.into(),
        })
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User(UserMessage {
            content: content.into(),
        })
    }

    pub fn assistant(content: impl Into<String>, tool_calls: impl Into<Vec<ToolCall>>) -> Self {
        Self::Assistant(AssistantMessage {
            content: content.into(),
            tool_calls: tool_calls.into(),
        })
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self::Tool(ToolMessage {
            content: content.into(),
            tool_call_id: tool_call_id.into()
        })
    }

    pub fn is_system(&self) -> bool {
        matches!(self, Self::System(_))
    }

    pub fn is_user(&self) -> bool {
        matches!(self, Self::User(_))
    }

    pub fn is_assistant(&self) -> bool {
        matches!(self, Self::Assistant(_))
    }

    pub fn is_tool(&self) -> bool {
        matches!(self, Self::Tool(_))
    }

}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

impl Into<Message> for SystemMessage {
    fn into(self) -> Message {
        Message::System(self)
    }
}

impl Into<Message> for UserMessage {
    fn into(self) -> Message {
        Message::User(self)
    }
}

impl Into<Message> for AssistantMessage {
    fn into(self) -> Message {
        Message::Assistant(self)
    }
}

impl Into<Message> for ToolMessage {
    fn into(self) -> Message {
        Message::Tool(self)
    }
}
