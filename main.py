from generatetext import tokens, num_tokens, vocab_size, deordinalize, tokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F
from hyperparameters import *
from languagemodel import BigramLanguageModel

data = torch.tensor(tokens, dtype=torch.long)
print(data.shape, data.dtype)
# vocab_size = len(data.unique())


# this is a mistake, CORRECT IT
# vocab_size = num_tokens

train_data = data[:int(num_tokens * 0.9)]
val_data = data[int(num_tokens * 0.9): ]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch("train")

print(xb.shape)
print(yb.shape)

model = BigramLanguageModel(vocab_size)
m = model.to(device)

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









# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

xb, yb = get_batch('val')
_idx = model.generate(xb, 100)

print(_idx.shape)

for batch in _idx:
    res = []
    for num in batch:
        num2 = deordinalize(int(num))
        res.append(num2)
    resstr = tokenizer.decode(res)
    print(resstr)





