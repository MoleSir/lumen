import regex as re
import collections
from typing import List, Dict, Tuple

class ByteLevelBPE:
    def __init__(self, special_tokens: List[str] = None):
        """
        初始化 BPE Tokenizer
        """
        # 1. 基础映射表：Byte(0~255) -> 可见 Unicode 字符
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 2. 词表 (vocab): token_str -> token_id
        # 前 256 个 ID 永远留给基础字节映射的字符
        self.vocab: Dict[str, int] = {ch: i for i, ch in enumerate(self.byte_encoder.values())}
        self.decoder: Dict[int, str] = {i: ch for ch, i in self.vocab.items()}
        
        # 3. BPE 合并规则优先级 (merges)
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
        # 4. 预分词正则表达式 (照搬 GPT-2/3 的官方正则)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # 5. 提前编译 Special Tokens 正则，用于 Train 和 Encode 阶段的切分隔离
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_vocab: Dict[str, int] = {}
        
        if self.special_tokens:
            # Capturing group (...) 确保 re.split 时，特殊 token 本身也能保留在列表里
            escaped_special_tokens = [re.escape(t) for t in self.special_tokens]
            self.special_regex = re.compile(f"({'|'.join(escaped_special_tokens)})")
        else:
            self.special_regex = None

    def _bytes_to_unicode(self) -> Dict[int, str]:
        """将 0-255 的 bytes 映射到可见 Unicode，避免 BPE 处理不可见控制符"""
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))

    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """统计词表中所有相邻 token pair 的出现频次"""
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[word[i], word[i+1]] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], v_in: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """将语料(v_in)中出现了 pair 的地方合并成一个新的 token"""
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_str = ' '.join(word)
            w_out = p.sub(''.join(pair), w_str)
            v_out[tuple(w_out.split(' '))] = v_in[word]
        return v_out

    def train(self, text: str, vocab_size: int):
        """
        训练阶段：提取 BPE merges 和构建词表
        """
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256
        
        # 1. 按 Special Tokens 隔离切分，防止污染 BPE 统计
        if self.special_regex:
            chunks = self.special_regex.split(text)
        else:
            chunks = [text]

        # 2. 对非特殊 token 的纯净文本进行预分词
        word_counts = collections.Counter()
        for chunk in chunks:
            if not chunk or chunk in self.special_tokens: 
                # 空字符或者特殊 token 绝对不计入统计！
                continue
            words = re.findall(self.pat, chunk)
            word_counts.update(words)

        # 3. 将单词转化为基础的 byte-level unicode 元组
        bpe_vocab = {}
        for word, count in word_counts.items():
            byte_word = tuple(self.byte_encoder[b] for b in word.encode("utf-8"))
            bpe_vocab[byte_word] = count

        # 4. BPE 核心训练循环
        for i in range(num_merges):
            pairs = self._get_stats(bpe_vocab)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            self.bpe_ranks[best_pair] = i # 记录合并优先级 (rank)
            
            # 更新词表
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.decoder[new_id] = new_token
            
            # 合并语料中的 pair
            bpe_vocab = self._merge_vocab(best_pair, bpe_vocab)

        # 5. 训练结束，给 Special Tokens 分配词表尾部的独立 ID
        current_max_id = max(self.vocab.values()) if self.vocab else -1
        for i, st in enumerate(self.special_tokens):
            st_id = current_max_id + 1 + i
            self.special_tokens_vocab[st] = st_id
            self.decoder[st_id] = st 

    def _get_bpe_word(self, word_tuple: Tuple[str, ...]) -> List[str]:
        """
        推理核心：给定字节序列，严格按照训练时的 rank 优先级进行合并。
        不能从左向右合并，必须从全局 rank 最小的开始！
        """
        word = list(word_tuple)
        
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            
            # 找到当前所有的 pair 中，在 bpe_ranks 中 rank 值最小（优先级最高）的
            best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            
            if best_pair not in self.bpe_ranks:
                break # 没有任何可合并的 pair，退出
                
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            
        return word

    def encode(self, text: str) -> List[int]:
        """
        编码阶段：文本 -> Token IDs
        """
        bpe_ids = []
        
        # 1. 按照 special tokens 将文本切块 (保持与 train 时绝对一致的切分逻辑)
        if self.special_regex:
            chunks = self.special_regex.split(text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk: 
                continue
                
            # 2. 如果这部分是特殊 token，直接查 special_vocab 转 ID
            if chunk in self.special_tokens_vocab:
                bpe_ids.append(self.special_tokens_vocab[chunk])
                continue
                
            # 3. 如果是普通文本，走 BBPE 切分逻辑
            for word in re.findall(self.pat, chunk):
                # 3.1 字符串 -> utf8 bytes -> unicode 可见字符序列
                byte_word = tuple(self.byte_encoder[b] for b in word.encode("utf-8"))
                
                # 3.2 根据 rank 推断 BPE tokens
                bpe_tokens = self._get_bpe_word(byte_word)
                
                # 3.3 Token -> ID
                for bpe_token in bpe_tokens:
                    bpe_ids.append(self.vocab[bpe_token])
                    
        return bpe_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        解码阶段：Token IDs -> 文本
        """
        text_bytes = bytearray()
        special_texts = []
        
        for tid in token_ids:
            if tid not in self.decoder:
                raise ValueError(f"Unknown token id: {tid}")
                
            token_str = self.decoder[tid]
            
            # 判断是否是特殊 token
            if token_str in self.special_tokens_vocab:
                # 遇到特殊 token 前，先把累积的普通 bytes 解码
                if text_bytes:
                    special_texts.append(text_bytes.decode("utf-8", errors="replace"))
                    text_bytes = bytearray()
                # 特殊 token 直接当做普通字符串拼接进去
                special_texts.append(token_str)
            else:
                # 普通 BPE token：把可见的 unicode 字符还原成底层 0-255 的 bytes
                for char in token_str:
                    text_bytes.append(self.byte_decoder[char])
                    
        # 循环结束后，处理剩余的 bytes
        if text_bytes:
            special_texts.append(text_bytes.decode("utf-8", errors="replace"))
            
        return "".join(special_texts)


# ==============================================================================
# 测试代码：模拟完整的数据生命周期
# ==============================================================================
if __name__ == "__main__":
    # 1. 准备富含不同语言、缩写、标点和特殊 Token 的语料
    corpus = (
        "Hello world! This is an LLM BPE implementation example. "
        "AI is shaping the world's future.\n"
        "测试一下中文能否正常切分。<pad>"
    )
    
    # 2. 实例化并声明 special tokens
    tokenizer = ByteLevelBPE(special_tokens=["<|endoftext|>", "[MASK]", "<pad>"])
    
    # 3. 训练 (基础256 + 合并100次 = 词表大小356)
    print(">>> 开始训练 Tokenizer...")
    tokenizer.train(corpus, vocab_size=356)
    print(f"训练完成！普通词表大小: {len(tokenizer.vocab)}, Special词表大小: {len(tokenizer.special_tokens_vocab)}\n")

    # 4. 测试 Encode 和 Decode
    # 构造一段模型从未见过，但包含特殊 Token 和中英文的句子
    test_text = "Hello [MASK] future! 这是一个BPE测试。<|endoftext|>"
    print(f">>> [测试输入文本]:\n{test_text}\n")
    
    # Encode
    encoded_ids = tokenizer.encode(test_text)
    print(f">>> [Encode 结果 (Token IDs)]:\n{encoded_ids}\n")
    
    # 验证一下切分出来的 token 都是什么
    print(">>> [切分出的 Token 序列映射]:")
    for tid in encoded_ids:
        token_str = tokenizer.decoder[tid]
        is_special = " (Special)" if token_str in tokenizer.special_tokens_vocab else ""
        print(f"  ID {tid:3d} -> '{token_str}'{is_special}")
    print()

    # Decode
    decoded_text = tokenizer.decode(encoded_ids)
    print(f">>> [Decode 结果 (还原文本)]:\n{decoded_text}\n")
    
    # 终极断言验证
    assert test_text == decoded_text, "❌ 解码文本与输入不一致！"
    print("✅ 测试通过！Train -> Encode -> Decode 形成完美闭环！")