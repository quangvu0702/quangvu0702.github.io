# Bigram model


```python
# reading and exploring the dataset
```


```python
words = open('../data/names.txt', 'r').read().splitlines()
words[0:5]
```




    ['emma', 'olivia', 'ava', 'isabella', 'sophia']




```python
len(words)
```




    32033




```python
min([len(w) for w in words]), max([len(w) for w in words])
```




    (2, 15)




```python
# explore bigram in the dataset
```


```python
for w in words[0:3]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)
```

    <S> e
    e m
    m m
    m a
    a <E>
    <S> o
    o l
    l i
    i v
    v i
    i a
    a <E>
    <S> a
    a v
    v a
    a <E>



```python
# counting data in dictionary
```


```python
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
# b.items()
```


```python
sorted(b.items(), key=lambda kv: -kv[1])[0:3]
```




    [(('n', '<E>'), 6763), (('a', '<E>'), 6640), (('a', 'n'), 5438)]




```python
# counting bigram in 2D tensor
```


```python
import torch
import numpy as np
```


```python
chars = sorted(set(list(''.join(words)))) + ['.']
itoc = dict(enumerate(chars))
ctoi = {c:i for i,c in itoc.items()}
N = torch.zeros((27, 27), dtype=torch.int)
```


```python
b = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[ctoi[ch1], ctoi[ch2]] += 1
# b.items()
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = f'{itoc[i]}{itoc[j]}'
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i,j].item(), ha='center', va='top', color='gray')
```


```python
# sampling from model
```


```python
P = (N + 1) / (N + 1).sum(1, keepdim=True)
```


```python
g = torch.Generator(device='cpu').manual_seed(2147483647)
for _ in range(5):
    chs = ['.']
    while True:
        i = ctoi[chs[-1]]
        i = torch.multinomial(P[i], 1, replacement = True, generator=g).item()
        chs.append(itoc[i])
        if chs[-1] == '.':
            print(''.join(chs))
            break
```


```python
# loss - negative log likelikehood
```


```python
loss = []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        loss.append(P[ctoi[ch1], ctoi[ch2]])
loss = -torch.tensor(loss).log().mean(); loss
```


```python
loss = []
for w in ['fm']:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        loss.append(P[ctoi[ch1], ctoi[ch2]])
loss = -torch.tensor(loss).log().mean(); loss
```


```python
# creating bigram dataset for neural net
```


```python
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for x, y in zip(chs, chs[1:]):
        xs.append(ctoi[x])
        ys.append(ctoi[y])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
xs, ys
```


```python
# one hot encoding
```


```python
import torch.nn.functional as F
xenc = F.one_hot(xs).to(torch.float32)
```


```python
# plt.imshow(xenc)
```


```python
W = torch.randn(size=(27, 27), requires_grad=True)
```


```python
for _ in range(100):
    # forward
    out = (xenc @ W).exp()
    out = out / out.sum(1, keepdim=True)

    # vetorized loss
    loss = -out[torch.arange(0, len(xs)), ys].log().mean() + 0.01 * (W**2).mean()
    print(loss)

    # backward and update

    W.grad = None
    loss.backward()

    W.data += -50 * W.grad
```

### Convert this file to md


```python
from IPython.core.display import Javascript
```


```python
%%js
IPython.notebook.kernel.execute('this_notebook = "' + IPython.notebook.notebook_name + '"')
```


```python
this_notebook
```


```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```
