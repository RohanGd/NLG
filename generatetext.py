import tiktoken 
import torch
import torch.nn as nn
from torch.nn import functional as F
from languagemodel import *
from hyperparameters2 import *

tokenizer = tiktoken.get_encoding("gpt2")
with open('corpus\poetry.txt', 'r', encoding='latin-1') as f:
    _tokens = tokenizer.encode_ordinary(f.read())

num_tokens = len(_tokens)
vocab = list(set(_tokens))
vocab_size = len(vocab)
# ordinal_encodings

otoe = {i : vocab[i] for i in range(vocab_size)}
etoo = {vocab[i] : i for i in range(vocab_size)}
# otoe = {i : _tokens[i] for i in range(num_tokens)}
# etoo = {_tokens[i] : i for i in range(num_tokens)}
ordinalize = lambda t : etoo[t]
deordinalize = lambda t : otoe[t]

tokens = [ordinalize(t) for t in _tokens]
assert(_tokens == [deordinalize(t) for t in tokens])
assert(max(tokens) == vocab_size - 1)




data = torch.tensor(tokens, dtype=torch.long, device=device)

train_data = data[:int(num_tokens * 0.9)]
val_data = data[int(num_tokens * 0.9): ]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch("train")
print(xb.shape)
print(yb.shape)

model = BigramLanguageModel(vocab_size)
m = model.to(device)


# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

m.load_state_dict(torch.load("models/Model_iter_50000", map_location=torch.device(device)))


xb, yb = get_batch('val')
_idx = m.generate(xb, 100)

for batch in _idx:
    res = []
    for num in batch:
        num2 = deordinalize(int(num))
        res.append(num2)
    resstr = tokenizer.decode(res)
    print(resstr)
