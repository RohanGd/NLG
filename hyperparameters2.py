import torch
batch_size = 16
block_size = 32
max_iters = 10000
eval_iters = 500
eval_interval = 500
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 32
n_head = 4
n_layer = 4
print("device:" + device)
dropout = 0.5
