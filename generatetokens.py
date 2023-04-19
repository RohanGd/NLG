import nltk 
from nltk.corpus import gutenberg

# with open('gutenbergtext.txt', 'w') as textfile:
#     textfile.write(nltk.corpus.genesis.raw('english-kjv.txt'))
#     # textfile.write(gutenberg.raw(gutenberg.fileids()[-6]))
#     # textfile.write(gutenberg.raw(gutenberg.fileids()[-7]))
#     # textfile.write(gutenberg.raw(gutenberg.fileids()[-8]))
#     # textfile.write(gutenberg.raw(gutenberg.fileids()[1]))
#     # textfile.write(gutenberg.raw(gutenberg.fileids()[0]))

import tiktoken 
tokenizer = tiktoken.get_encoding("gpt2")
# with open('gutenbergtext.txt', 'r') as f:
#     _tokens = tokenizer.encode_ordinary(f.read())

with open('corpus/poetry.txt', 'r',encoding="latin-1") as f:
    _tokens = tokenizer.encode_ordinary(f.read())

# sample tokens
print("\nSamples from tokenization: ")
print([tokenizer.decode_single_token_bytes(token) for token in _tokens[150:170]])

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
print(f'number of tokens = {len(tokens)}')
assert(max(tokens) == vocab_size - 1)




