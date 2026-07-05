use crate::dataset::types::Message;
use super::Tokenizer;

impl Tokenizer {
    // TODO: minijina??
    pub fn apply_chat_template(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();
        if messages.len() == 0 {
            return "".to_string();
        }

    
        // --- 1. 处理 System & Tools 头部 ---
        let messages = if let Message::System(msg) = &messages[0] {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(&msg.content);                    

            if msg.functions.len() != 0 {
                prompt.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
                for tool in &msg.functions {
                    prompt.push_str("\n");
                    prompt.push_str(&serde_json::to_string(tool).unwrap_or_default());
                }
                prompt.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
            }
            &messages[1..]
        } else {
            messages
        };

        // --- 2. 遍历消息 ---
        for (i, msg) in messages.iter().enumerate() {    
            match msg {
                Message::System(m) => {
                    prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", m.content));
                }
                Message::User(m) => {
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", m.content));
                }
                Message::Assistant(m) => {
                    prompt.push_str("<|im_start|>assistant\n");
                    prompt.push_str(&m.content);
                    
                    for tc in &m.tool_calls {
                        // 如果有内容且是第一个 tool_call，换行；后续 tool_call 也换行
                        prompt.push_str("\n<tool_call>\n");
                        // 这里的 arguments 已经是 JSON 字符串
                        prompt.push_str(&format!("{{\"name\": \"{}\", \"arguments\": {}}}", tc.name, tc.arguments));
                        prompt.push_str("\n</tool_call>");
                    }
                    prompt.push_str("<|im_end|>\n");
                }
                Message::Tool(m) => {
                    // --- Tool 消息特殊处理：包装在 user 角色中 ---
                    // 判断是否是连续 tool 消息的开头
                    let prev_is_not_tool = i == 0 || !matches!(messages[i-1], Message::Tool(_));
                    if prev_is_not_tool {
                        prompt.push_str("<|im_start|>user");
                    }

                    prompt.push_str(&format!("\n<tool_response>\n{}\n</tool_response>", m.content));

                    // 判断是否是连续 tool 消息的结尾
                    let next_is_not_tool = i + 1 == messages.len() || !matches!(messages[i+1], Message::Tool(_));
                    if next_is_not_tool {
                        prompt.push_str("<|im_end|>\n");
                    }
                }
            }
        }
        
        // --- 3. 生成提示符 ---
        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
        }
    
        prompt
    }
}
