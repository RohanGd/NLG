import torch
batch_size = 4
block_size = 8 
max_iters = 3000
eval_iters = 100
eval_interval = 100
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 32
n_head = 8
n_layer = 4

dropout = 0
