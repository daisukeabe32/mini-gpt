import torch
from src.tokenizer import CharTokenizer 
from src.embedding import TokenEmbedding

text = "hello world!"
tok = CharTokenizer(text)
vocab_size = tok.vocab_size

ids = tok.encode("hello")
print("ids:", ids)          # for example :[4, 3, 5, 5, 6] 

x = torch.tensor([ids], dtype=torch.long)  # (B=1, T=5)
emb = TokenEmbedding(vocab_size, d_model=8)
out = emb(x)

print("out.shape:", out.shape)  # -> torch.Size([1, 5, 8])
print(out)