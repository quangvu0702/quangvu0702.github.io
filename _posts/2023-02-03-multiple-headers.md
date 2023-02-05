# Multiple headers


```python
# version 1: for and concat
import torch

torch.manual_seed(123)

B, T, C = 2, 4, 8 # batch, time, channel
q = torch.randn((B, T, C)) # B, T, C

k = torch.randn((B, T, C))

v = torch.randn((B, T, C))

def head(q, k, v):
    att = (q @ k.transpose(-2, -1)) / C**0.5 # B, T, T
    tril = torch.tril(torch.ones(T, T)) # T, T
    att = att.masked_fill(tril == 0, float('-inf')) # B, T, T
    att = torch.softmax(att, dim=-1) # B, T, T

    out = att @ v # B, T, C
    return out

out = head(q, k, v) # B, T, C
```


```python
v[0]
```




    tensor([[-0.2582, -2.0407, -0.8016, -0.8183, -1.1820, -0.2877, -0.6043,  0.6002],
            [-1.4053, -0.5922, -0.2548,  1.1517, -0.0179,  0.4264, -0.7657, -0.0545],
            [-1.2743,  0.4513, -0.2280,  0.9224,  0.2056, -0.4970,  0.5821,  0.2053],
            [-0.3018, -0.6703, -0.6171, -0.8334,  0.4839, -0.1349,  0.2119, -0.8714]])




```python
out[0]
```




    tensor([[-0.2582, -2.0407, -0.8016, -0.8183, -1.1820, -0.2877, -0.6043,  0.6002],
            [-0.5085, -1.7247, -0.6823, -0.3885, -0.9280, -0.1319, -0.6395,  0.4574],
            [-1.2056, -0.2033, -0.3026,  0.8066, -0.0315, -0.1442, -0.0328,  0.1576],
            [-0.8482, -0.1931, -0.4107,  0.1548,  0.2657, -0.2460,  0.2601, -0.2675]])




```python
torch.allclose(v[0][0], out[0][0])
```




    True




```python
torch.allclose(0.7818 * v[0][0] + 0.2182 * v[0][1], out[0][1], atol=1e-02)
```




    True




```python
q1, q2 = torch.randn((B, T, C)), torch.randn((B, T, C)) # B, T, C

k1, k2 = torch.randn((B, T, C)), torch.randn((B, T, C))

v1, v2 = torch.randn((B, T, C)), torch.randn((B, T, C))

head1 = head(q1, k1, v1) # B, T, C
head2 = head(q2, k2, v2) # B, T, C

heads = torch.cat([head1, head2], -1) # B, T, 2*C
```


```python
torch.allclose(head1, heads[:, :, :C]), torch.allclose(head2, heads[:, :, C:])
```




    (True, True)




```python
# version 2 multiple
a = torch.arange(B*T*C*2).view(2, B, T, C).float()
b = torch.ones(2, B, T, C)
```


```python
c = a @ b.transpose(-1, -2) # 2, B, T, T
```


```python
c0 = a[0] @ b[0].transpose(-1, -2) # B, T, T
c1 = a[1] @ b[1].transpose(-1, -2) # B, T, T
torch.stack([c0, c1])
```




    tensor([[[[ 28.,  28.,  28.,  28.],
              [ 92.,  92.,  92.,  92.],
              [156., 156., 156., 156.],
              [220., 220., 220., 220.]],
    
             [[284., 284., 284., 284.],
              [348., 348., 348., 348.],
              [412., 412., 412., 412.],
              [476., 476., 476., 476.]]],
    
    
            [[[540., 540., 540., 540.],
              [604., 604., 604., 604.],
              [668., 668., 668., 668.],
              [732., 732., 732., 732.]],
    
             [[796., 796., 796., 796.],
              [860., 860., 860., 860.],
              [924., 924., 924., 924.],
              [988., 988., 988., 988.]]]])




```python
torch.allclose(c, torch.stack([c0, c1]))
```




    True




```python
q = torch.stack([q1, q2], 1) # B, 2, T, C
k = torch.stack([k1, k2], 1) # B, 2, T, C
v = torch.stack([v1, v2], 1) # B, 2, T, C

heads2 = head(q, k, v)

torch.allclose(heads[:,:,0:C], heads2[0])
torch.allclose(heads[:,:,C:], heads2[1])
```




    False




```python
torch.allclose(heads ,heads2.transpose(1,2).reshape(B, T, C*2))
```




    True




```python
def multiple_head(q, k, v):
    # shape B, n_head, T, C
    att = (q @ k.transpose(-2, -1)) / C**0.5 # B, n_head, T, T
    tril = torch.tril(torch.ones(T, T)) # T, T
    att = att.masked_fill(tril == 0, float('-inf')) # B, n_head, T, T
    att = torch.softmax(att, dim=-1) # B, n_head, T, T

    out = att @ v # B, n_head, T, C
    out = out.transpose(1,2).reshape(B, T, -1) # B, T, C * n_head
    return out

heads3 = multiple_head(q, k, v)
torch.allclose(heads ,heads3)
```




    True




```python
x = torch.stack([torch.ones(B*T*C).view(B, T, C), torch.ones(B*T*C).view(B, T, C) * 2.0], 1)
```


```python
x = x.transpose(1,2).reshape(B, T, C*2)
```


```python
x
```




    tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
             [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
             [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
             [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.]],
    
            [[1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
             [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
             [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
             [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.]]])




```python
import torch
import torch.nn as nn
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

class MultipleHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        n_embd = head_size * num_heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(drop_out)
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], -1)
        out = self.dropout(self.proj(out))
        return out
```


```python
class MultipleHeadAttention_v2(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(C, head_size * num_heads)
        self.key = nn.Linear(C, head_size * num_heads)
        self.value = nn.Linear(C, head_size * num_heads)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))
        self.proj = nn.Linear(num_heads*head_size, num_heads*head_size)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        
    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, T, C)
        q = self.query(x).view(B, T, num_heads, head_size).transpose(1,2) # B, num_heads, T, head_size
        k = self.key(x).view(B, T, num_heads, head_size).transpose(1,2) # B, num_heads, T, head_size
        v = self.value(x).view(B, T, num_heads, head_size).transpose(1,2) # B, num_heads, T, head_size
        # computer attention score
        wei = q @ v.transpose(-2, -1) * head_size ** -0.5 # (B, num_heads, T, head_size) x (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei,dim=-1) # (B, num_heads, T, T)
        wei = self.dropout1(wei)
        # perform the weighted aggregation
        out = wei@v # (B, num_heads, T, T) x (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        out = out.transpose(1,2).reshape(B, T, -1) # B, T, head_size * n_head
        out = self.dropout2(self.proj(out))
        return out
```


```python
B, T, C = 4, 8, 32
head_size = 16
num_heads = int(32 / head_size)
x = torch.randn(B, T, C)
Q1 = torch.randn(C, head_size)
Q2 = torch.randn(C, head_size)

q1 = x @ Q1 # (B, T, C) @ (C, head_size) -> (B, T, head_size)
q2 = x @ Q2 # (B, T, C) @ (C, head_size) -> (B, T, head_size)

q = torch.stack([q1, q2], 1) # B, 2, T, head_size
assert(torch.allclose(q1, q[:,0,:,:]))
```


```python
Q = torch.cat([Q1, Q2], 1) # C, head_size * num_heads
q = x @ Q # B, T, head_size * num_heads
q = q.view(B, T, num_heads, head_size).transpose(1,2) # B, num_heads, T, head_size
assert torch.allclose(q1, q[:, 0, :, :])
```


```python
Q = torch.stack([Q1, Q2]) # num_heads, C, head_size
q_v2 = x.view(B, 1, T, C) @ Q # B, num_heads, T, head_size

assert(torch.allclose(q, q_v2, atol=0.001))
```


```python
drop_out = 0
num_heads = 2
att = MultipleHeadAttention_v2(head_size)

out = att(x)

out.shape
```




    torch.Size([4, 8, 32])



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

    <ipython-input-26-0a5671918a34> in <module>
    ----> 1 this_notebook
    

    NameError: name 'this_notebook' is not defined



```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```
