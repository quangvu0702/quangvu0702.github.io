# 2023-01-30 self attention layers


```python
# download data
```


```python
import torch
torch.cuda.device_count(), torch.cuda.current_device()
```




    (1, 0)




```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

    cuda



```python
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

    --2023-02-05 14:53:31--  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 2606:50c0:8000::154, 2606:50c0:8003::154, 2606:50c0:8002::154, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|2606:50c0:8000::154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1115394 (1,1M) [text/plain]
    Saving to: ‘input.txt.5’
    
    input.txt.5         100%[===================>]   1,06M   484KB/s    in 2,3s    
    
    2023-02-05 14:53:34 (484 KB/s) - ‘input.txt.5’ saved [1115394/1115394]
    



```python
# read on review data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```


```python
# Here is all unique character that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
```


```python
# create a mapping from characters to integers
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}

encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

print(encode("hi there"))

print(decode(encode("hi there")))
```

    [46, 47, 1, 58, 46, 43, 56, 43]
    hi there



```python
# Let now encode the entire text dataset and store it into torch.Tensor
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

data[0:100]
```

    torch.Size([1115394]) torch.int64





    tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 14, 43, 44,
            53, 56, 43,  1, 61, 43,  1, 54, 56, 53, 41, 43, 43, 42,  1, 39, 52, 63,
             1, 44, 59, 56, 58, 46, 43, 56,  6,  1, 46, 43, 39, 56,  1, 51, 43,  1,
            57, 54, 43, 39, 49,  8,  0,  0, 13, 50, 50, 10,  0, 31, 54, 43, 39, 49,
             6,  1, 57, 54, 43, 39, 49,  8,  0,  0, 18, 47, 56, 57, 58,  1, 15, 47,
            58, 47, 64, 43, 52, 10,  0, 37, 53, 59])




```python
# Let's now split up the data into train set and validation set
n = round(len(data) * 0.9);
train_data = data[:n]
val_data   = data[n:]
len(train_data), len(val_data)
```




    (1003855, 111539)




```python
block_size = 8
train_data[:block_size]
```




    tensor([18, 47, 56, 57, 58,  1, 15, 47])




```python
x = train_data[:block_size]
y = train_data[1:block_size + 1]

for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    print(f"when context is {context} the target: {target}.")
```

    when context is tensor([18]) the target: 47.
    when context is tensor([18, 47]) the target: 56.
    when context is tensor([18, 47, 56]) the target: 57.
    when context is tensor([18, 47, 56, 57]) the target: 58.
    when context is tensor([18, 47, 56, 57, 58]) the target: 1.
    when context is tensor([18, 47, 56, 57, 58,  1]) the target: 15.
    when context is tensor([18, 47, 56, 57, 58,  1, 15]) the target: 47.
    when context is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target: 58.



```python
torch.manual_seed(1337)

batch_size = 4
block_size = 8

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(train_data) - block_size, (batch_size, ))
    xb = torch.stack([train_data[i:i+block_size] for i in ix])
    yb = torch.stack([train_data[i+1:i+1+block_size] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

xb, yb = get_batch()
print(xb)
print(yb)
for b in range(batch_size):
    for i in range(block_size):
        x = xb[b,:i+1]
        y = yb[b,i]
        print(f"when context is {x} the target: {y}.")
```

    tensor([[56,  6,  0, 24, 43, 58,  1, 61],
            [39, 47, 51,  1, 58, 46, 39, 58],
            [52, 45,  1, 58, 53,  1, 57, 39],
            [43, 47, 52, 45,  1, 46, 53, 50]], device='cuda:0')
    tensor([[ 6,  0, 24, 43, 58,  1, 61, 46],
            [47, 51,  1, 58, 46, 39, 58,  1],
            [45,  1, 58, 53,  1, 57, 39, 63],
            [47, 52, 45,  1, 46, 53, 50, 47]], device='cuda:0')
    when context is tensor([56], device='cuda:0') the target: 6.
    when context is tensor([56,  6], device='cuda:0') the target: 0.
    when context is tensor([56,  6,  0], device='cuda:0') the target: 24.
    when context is tensor([56,  6,  0, 24], device='cuda:0') the target: 43.
    when context is tensor([56,  6,  0, 24, 43], device='cuda:0') the target: 58.
    when context is tensor([56,  6,  0, 24, 43, 58], device='cuda:0') the target: 1.
    when context is tensor([56,  6,  0, 24, 43, 58,  1], device='cuda:0') the target: 61.
    when context is tensor([56,  6,  0, 24, 43, 58,  1, 61], device='cuda:0') the target: 46.
    when context is tensor([39], device='cuda:0') the target: 47.
    when context is tensor([39, 47], device='cuda:0') the target: 51.
    when context is tensor([39, 47, 51], device='cuda:0') the target: 1.
    when context is tensor([39, 47, 51,  1], device='cuda:0') the target: 58.
    when context is tensor([39, 47, 51,  1, 58], device='cuda:0') the target: 46.
    when context is tensor([39, 47, 51,  1, 58, 46], device='cuda:0') the target: 39.
    when context is tensor([39, 47, 51,  1, 58, 46, 39], device='cuda:0') the target: 58.
    when context is tensor([39, 47, 51,  1, 58, 46, 39, 58], device='cuda:0') the target: 1.
    when context is tensor([52], device='cuda:0') the target: 45.
    when context is tensor([52, 45], device='cuda:0') the target: 1.
    when context is tensor([52, 45,  1], device='cuda:0') the target: 58.
    when context is tensor([52, 45,  1, 58], device='cuda:0') the target: 53.
    when context is tensor([52, 45,  1, 58, 53], device='cuda:0') the target: 1.
    when context is tensor([52, 45,  1, 58, 53,  1], device='cuda:0') the target: 57.
    when context is tensor([52, 45,  1, 58, 53,  1, 57], device='cuda:0') the target: 39.
    when context is tensor([52, 45,  1, 58, 53,  1, 57, 39], device='cuda:0') the target: 63.
    when context is tensor([43], device='cuda:0') the target: 47.
    when context is tensor([43, 47], device='cuda:0') the target: 52.
    when context is tensor([43, 47, 52], device='cuda:0') the target: 45.
    when context is tensor([43, 47, 52, 45], device='cuda:0') the target: 1.
    when context is tensor([43, 47, 52, 45,  1], device='cuda:0') the target: 46.
    when context is tensor([43, 47, 52, 45,  1, 46], device='cuda:0') the target: 53.
    when context is tensor([43, 47, 52, 45,  1, 46, 53], device='cuda:0') the target: 50.
    when context is tensor([43, 47, 52, 45,  1, 46, 53, 50], device='cuda:0') the target: 47.



```python
print(xb) # input to our transformer
```

    tensor([[56,  6,  0, 24, 43, 58,  1, 61],
            [39, 47, 51,  1, 58, 46, 39, 58],
            [52, 45,  1, 58, 53,  1, 57, 39],
            [43, 47, 52, 45,  1, 46, 53, 50]], device='cuda:0')



```python
# self attention layer
```


```python
import torch

torch.manual_seed(1338)

B, T, C = 4, 8, 2 ; # batch, time, channel

x = torch.randn((B, T, C))

x[0]
```




    tensor([[-1.3113, -1.0017],
            [-1.2342,  0.1297],
            [-0.5150, -1.2666],
            [-0.6719,  0.1851],
            [ 0.9367,  0.3139],
            [-1.3950,  0.1132],
            [ 0.3622,  2.5192],
            [-0.7672, -0.9529]])




```python
# version 1
xbow = torch.zeros_like(x)
for b in range(B):
    for t in range(T):
        x_prev = x[b, :t+1] # t, C
        xbow[b,t] = torch.mean(x_prev, dim=0)
xbow[0]
```




    tensor([[-1.3113, -1.0017],
            [-1.2728, -0.4360],
            [-1.0202, -0.7129],
            [-0.9331, -0.4884],
            [-0.5591, -0.3279],
            [-0.6985, -0.2544],
            [-0.5469,  0.1418],
            [-0.5745,  0.0050]])




```python
# version 2
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)
xbow2 = wei @ x # (B, T, T) x (B, T, C) -> (B, T, C)
torch.allclose(xbow, xbow2)
```




    False




```python
# version 3
# using solfmax
wei = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(wei==0, float('-inf'))
wei = torch.softmax(wei, dim=-1)
xbow3 = wei @ x # (B, T, T) x (B, T, C) -> (B, T, C)
torch.allclose(xbow, xbow3)
```




    False




```python
torch.__version__
```




    '1.7.0'




```python
import torch
import torch.nn as nn

torch.manual_seed(1337)

B, T, C = 4, 8, 32 ; # batch, time, channel
head_size = 16

x = torch.randn((B, T, C))

query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

q = query(x) # (B, T, head_size)
k = key(x) # (B, T, head_size)
v = value(x) # (B, T, head_size)

wei = q@k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, head_size, T) -> (B, T, T)
wei = wei / head_size**0.5
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = torch.softmax(wei, dim=-1)

out = wei @ v # (B, T, T) x (B, T, C) -> (B, T, C)

q.shape, k.shape, v.shape, wei.shape, out.shape
```




    (torch.Size([4, 8, 16]),
     torch.Size([4, 8, 16]),
     torch.Size([4, 8, 16]),
     torch.Size([4, 8, 8]),
     torch.Size([4, 8, 16]))




```python
drop_out = 0.2
```


```python
# create a class for self attention layer
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(C, head_size)
        self.key = nn.Linear(C, head_size)
        self.value = nn.Linear(C, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))
        self.dropout = nn.Dropout(drop_out)
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # B, T, head_size
        k = self.key(x) # B, T, head_size
        v = self.value(x) # B, T, head_size
        # computer attention score
        wei = q @ v.transpose(-2, -1) * head_size ** -0.5 # (B, T, head_size) x (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei,dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation
        out = wei@v # (B, T, T) x (B, T, head_size) -> (B, T, head_size)
        return out
    
# class MultipleHeadAttention(nn.Module):
#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         n_embd = head_size * num_heads
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.dropout = nn.Dropout(drop_out)
#     def forward(self, x):
#         out = torch.cat([head(x) for head in self.heads], -1)
#         out = self.dropout(self.proj(out))
#         return out

class MultipleHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.query = nn.Linear(C, head_size * num_heads)
        self.key = nn.Linear(C, head_size * num_heads)
        self.value = nn.Linear(C, head_size * num_heads)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))
        self.proj = nn.Linear(num_heads*head_size, num_heads*head_size)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        
    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, 1, T, C)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1,2) # B, num_heads, T, head_size
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1,2) # B, num_heads, T, head_size
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1,2) # B, num_heads, T, head_size
        # computer attention score
        wei = q @ v.transpose(-2, -1) * self.head_size ** -0.5 # (B, num_heads, T, head_size) x (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei,dim=-1) # (B, num_heads, T, T)
        wei = self.dropout1(wei)
        # perform the weighted aggregation
        out = wei@v # (B, num_heads, T, T) x (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        out = out.transpose(1,2).reshape(B, T, -1) # B, T, head_size * n_head
        out = self.dropout2(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
                        nn.Linear(n_embd, n_embd * 4),
                        nn.ReLU(),
                        nn.Linear(n_embd * 4, n_embd),
                        nn.Dropout(drop_out)
                    )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
class Block(nn.Module):
    def __init__(self, num_heads, n_embd):
        super().__init__()
        head_size = n_embd//num_heads
        self.sa_head = MultipleHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = self.sa_head(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x

B, T, C = x.shape
    
head = Head(2)
out = head(x)
print(out.shape)

print(x.shape)
head_size = 8
num_heads = int(C / head_size)
multi_heads = MultipleHeadAttention(num_heads, head_size)
out = multi_heads(x)
print(out.shape)

ffwd = FeedForward(C)
out = ffwd(out)
print(out.shape)

x = torch.randn((4, 8, 32))
block = Block(4, 32)
out = block(x)
print(out.shape)

blocks = nn.Sequential(*nn.ModuleList([Block(4, 32) for _ in range(4)]))
blocks(x).shape
```

    torch.Size([4, 8, 2])
    torch.Size([4, 8, 32])
    torch.Size([4, 8, 32])
    torch.Size([4, 8, 32])
    torch.Size([4, 8, 32])





    torch.Size([4, 8, 32])




```python
torch.manual_seed(1338)
x = torch.randn((4, 8, 32))
print(x[0, 0].mean(), x[0, 0].var())

layer_norm = nn.LayerNorm(32)
x = layer_norm(x)
print(x[1, 1].mean(), x[2, 2].var())
```

    tensor(-0.2145) tensor(1.0175)
    tensor(-1.6764e-08, grad_fn=<MeanBackward0>) tensor(1.0322, grad_fn=<VarBackward0>)



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    out = {}
    for split in ['train', 'val']:
        for i in range(eval_iters):
            xb, yb = get_batch()
            loss, logits = model(xb, yb)
            losses[i] = loss
        out[split] = losses.mean().item()
    return out

# bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(T, n_embd)
        self.blocks = nn.Sequential(*nn.ModuleList([Block(num_heads, n_embd) for _ in range(num_blocks)] + [nn.LayerNorm(n_embd)]))
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T).to(device))
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, n_embd, vocab_size) x (B, T, n_embd) -> (B, T, vocab_size)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return loss, logits

    @torch.no_grad()
    def generate(self, idx, max_new_token):
        for i in range(max_new_token):
            loss, logits = self(idx[:, -block_size:])
            logits = logits[:, -1,:]
            probs = F.softmax(logits, -1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], 1)
        return idx

def train(lr=0.001):
    optimizer = AdamW(model.parameters())
    for i in range(max_iter):
        if i % eval_iters == 0:
            out = estimate_loss()
            print(f"Train loss: {out['train']}. Val loss: {out['val']}. ")
        xb, yb = get_batch()
        loss, logits = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


```


```python
# B, T, C = 32, 8, 32 ; # batch, time, channel
# n_embd = 32
# batch_size, block_size = B, T
# max_iter = 3000
# num_heads = 4
# num_blocks = 4
# eval_iters = 200
# eval_interval = 300
# drop_out = 0.2
# head_size = 16
# lr = 0.001
# xb, yb = get_batch()
# print(xb.shape, yb.shape)

# model = BigramLanguageModel(vocab_size).to(device)
# m = model.to(device)
# # print the number of parameters in the model
# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# loss, logits = model(xb, yb)
# print(loss)

# train(lr=lr)

# sent = model.generate(idx=torch.zeros((1,1), dtype=torch.long).to(device), max_new_token=500)
# print(decode(sent[0].tolist()))

# train(lr=lr / 10)
    
# sent = model.generate(idx=torch.zeros((1,1), dtype=torch.long).to(device), max_new_token=500)
# print(decode(sent[0].tolist()))
```


```python
# hyperparameters
# batch_size = 16 # how many independent sequences will we process in parallel?
# block_size = 32 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 100
# learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 64
# n_head = 4
# n_layer = 4
# dropout = 0.0
# ------------


B, T, C = 64, 128, 384 ; # batch, time, channel
n_embd = C
batch_size, block_size = B, T
max_iter = 5000
num_heads = 6
num_blocks = 6
eval_iters = 200
eval_interval = 100
drop_out = 0.2
head_size = n_embd / num_heads
xb, yb = get_batch()
lr = 0.0001
print(xb.shape, yb.shape)

model = BigramLanguageModel(vocab_size).to(device)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
loss, logits = model(xb, yb)
print(loss)
```

    torch.Size([64, 128]) torch.Size([64, 128])
    10.746689 M parameters
    tensor(4.3245, device='cuda:0', grad_fn=<NllLossBackward>)



```python
train(lr=lr)

sent = model.generate(idx=torch.zeros((1,1), dtype=torch.long).to(device), max_new_token=500)
print(decode(sent[0].tolist()))
```

    Train loss: 4.319405555725098. Val loss: 4.31920051574707. 
    Train loss: 2.07173228263855. Val loss: 2.071382522583008. 
    Train loss: 1.658880352973938. Val loss: 1.6577978134155273. 
    Train loss: 1.4907934665679932. Val loss: 1.4930508136749268. 
    Train loss: 1.40459406375885. Val loss: 1.4055858850479126. 
    Train loss: 1.3399468660354614. Val loss: 1.3428082466125488. 
    Train loss: 1.2919952869415283. Val loss: 1.2917110919952393. 
    Train loss: 1.2531120777130127. Val loss: 1.255987524986267. 
    Train loss: 1.2189946174621582. Val loss: 1.2206335067749023. 
    Train loss: 1.1781675815582275. Val loss: 1.1791486740112305. 
    Train loss: 1.1486330032348633. Val loss: 1.1486282348632812. 
    Train loss: 1.111915111541748. Val loss: 1.1131877899169922. 
    Train loss: 1.0789719820022583. Val loss: 1.0793486833572388. 
    Train loss: 1.0437376499176025. Val loss: 1.0418130159378052. 
    Train loss: 1.001774787902832. Val loss: 1.0007587671279907. 
    Train loss: 0.9658932685852051. Val loss: 0.966268002986908. 
    Train loss: 0.9201828241348267. Val loss: 0.9186395406723022. 
    Train loss: 0.8731043338775635. Val loss: 0.8724583983421326. 
    Train loss: 0.8258631825447083. Val loss: 0.8270233869552612. 
    Train loss: 0.7863552570343018. Val loss: 0.7853423357009888. 
    Train loss: 0.7316440343856812. Val loss: 0.7323237657546997. 
    Train loss: 0.6888072490692139. Val loss: 0.690226674079895. 
    Train loss: 0.6393572092056274. Val loss: 0.6397681832313538. 
    Train loss: 0.601671040058136. Val loss: 0.6048049926757812. 
    Train loss: 0.5630137920379639. Val loss: 0.5605818629264832. 
    
    Unless vexange hurd! I will not drunk a crutch,
    To try if her great children, grandam,
    Are very soldiers for the ground.
    
    FRIAR LAURENCE:
    Romeo shall thank thee, daughter, and with flowers,
    As to thy other prediction.
    
    JOHN OF GAUNT:
    Catal stars, passing shock, Edward my eye,
    Comes have been their war: my woman was a better
    metings to the offic and my house, if I slewd.
    
    LUCENTIO:
    O thou virlain keepest detect I saw them,
    From time hath to made them all to live.
    
    JULIET:
    Madam, I come!
    Brisk it 


### Convert this file to md


```python
from IPython.core.display import Javascript
```


```python
%%js
IPython.notebook.kernel.execute('this_notebook = "' + IPython.notebook.notebook_name + '"')
```


    <IPython.core.display.Javascript object>



```python
this_notebook
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-31-0a5671918a34> in <module>
    ----> 1 this_notebook
    

    NameError: name 'this_notebook' is not defined



```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```

    [NbConvertApp] WARNING | pattern '{this_notebook}' matched no files
    This application is used to convert notebook files (*.ipynb) to various other
    formats.
    
    WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    -------
    
    Arguments that take values are actually convenience aliases to full
    Configurables, whose aliases are listed on the help line. For more information
    on full configurables, see '--help-all'.
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
    --generate-config
        generate default config file
    -y
        Answer yes to any questions instead of prompting.
    --execute
        Execute the notebook prior to export.
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
    --stdout
        Write notebook output to stdout instead of files.
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only 
        relevant when converting to notebook format)
    --clear-output
        Clear output of current file and save in place, 
        overwriting the existing notebook.
    --no-prompt
        Exclude input and output prompts from converted document.
    --no-input
        Exclude input cells and output prompts from converted document. 
        This mode is ideal for generating code-free reports.
    --log-level=<Enum> (Application.log_level)
        Default: 30
        Choices: (0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL')
        Set the log level by value or name.
    --config=<Unicode> (JupyterApp.config_file)
        Default: ''
        Full path of a config file.
    --to=<Unicode> (NbConvertApp.export_format)
        Default: 'html'
        The export format to be used, either one of the built-in formats
        ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf',
        'python', 'rst', 'script', 'slides'] or a dotted object name that represents
        the import path for an `Exporter` class
    --template=<Unicode> (TemplateExporter.template_file)
        Default: ''
        Name of the template file to use
    --writer=<DottedObjectName> (NbConvertApp.writer_class)
        Default: 'FilesWriter'
        Writer class used to write the  results of the conversion
    --post=<DottedOrNone> (NbConvertApp.postprocessor_class)
        Default: ''
        PostProcessor class used to write the results of the conversion
    --output=<Unicode> (NbConvertApp.output_base)
        Default: ''
        overwrite base name use for output files. can only be used when converting
        one notebook at a time.
    --output-dir=<Unicode> (FilesWriter.build_directory)
        Default: ''
        Directory to write output(s) to. Defaults to output to the directory of each
        notebook. To recover previous default behaviour (outputting to the current
        working directory) use . as the flag value.
    --reveal-prefix=<Unicode> (SlidesExporter.reveal_url_prefix)
        Default: ''
        The URL prefix for reveal.js (version 3.x). This defaults to the reveal CDN,
        but can be any url pointing to a copy  of reveal.js.
        For speaker notes to work, this must be a relative path to a local  copy of
        reveal.js: e.g., "reveal.js".
        If a relative path is given, it must be a subdirectory of the current
        directory (from which the server is run).
        See the usage documentation
        (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-
        slideshow) for more details.
    --nbformat=<Enum> (NotebookExporter.nbformat_version)
        Default: 4
        Choices: [1, 2, 3, 4]
        The nbformat version to write. Use this to downgrade notebooks.
    
    To see all available configurables, use `--help-all`
    
    Examples
    --------
    
        The simplest way to use nbconvert is
        
        > jupyter nbconvert mynotebook.ipynb
        
        which will convert mynotebook.ipynb to the default format (probably HTML).
        
        You can specify the export format with `--to`.
        Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides'].
        
        > jupyter nbconvert --to latex mynotebook.ipynb
        
        Both HTML and LaTeX support multiple output templates. LaTeX includes
        'base', 'article' and 'report'.  HTML includes 'basic' and 'full'. You
        can specify the flavor of the format used.
        
        > jupyter nbconvert --to html --template basic mynotebook.ipynb
        
        You can also pipe the output to stdout, rather than a file
        
        > jupyter nbconvert mynotebook.ipynb --stdout
        
        PDF is generated via latex
        
        > jupyter nbconvert mynotebook.ipynb --to pdf
        
        You can get (and serve) a Reveal.js-powered slideshow
        
        > jupyter nbconvert myslides.ipynb --to slides --post serve
        
        Multiple notebooks can be given at the command line in a couple of 
        different ways:
        
        > jupyter nbconvert notebook*.ipynb
        > jupyter nbconvert notebook1.ipynb notebook2.ipynb
        
        or you can specify the notebooks list in a config file, containing::
        
            c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
        
        > jupyter nbconvert --config mycfg.py
    

