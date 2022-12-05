# Trigram py-torch-ify model


```python
# build training data set
```


```python
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
words = open('../data/names.txt', 'r').read().splitlines()
vocabs = sorted(set(list(''.join(words)))) + ['.']
itoc = dict(enumerate(vocabs))
ctoi = {v:k for k,v in itoc.items()}
itoc[0], ctoi['a']
block_size = 3
```


```python
# build training data set
def build_dataset(words):
    random.shuffle(words)
    X, Y = [], []
    for w in words:
        w = list(w) + ['.']
        context = [ctoi['.']] * block_size
        for y in w:
            X.append(context)
            Y.append(ctoi[y])
            context = context[1: ] + [ctoi[y]]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

X, Y = build_dataset(words)
n1 = int(X.shape[0] * 0.8) 
n2 = int(X.shape[0] * 0.9)
Xtr, Ytr = X[:n1], Y[:n1]
Xdev, Ydev = X[n1:n2], Y[n1:n2]
Xtest, Ytest = X[n2:], Y[n2:]
```


```python
# build Linear and Tanh and BatchNorm class
class Linear():
    def __init__(self, fan_in, fan_out, bias=None):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.randn((fan_out)) * 0.0 if bias else None
        if bias == True:
            self.bias = torch.randn((fan_out)) * 0.0
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias != None: self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + [self.bias] if self.bias != None else [self.weight]

class Tanh():
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

class BatchNorm():
    def __init__(self, size, eps=10**-5):
        self.training = True
        self.running_mean = torch.ones((1, size))
        self.running_var = torch.zeros((1, size))
        self.bn_shift = torch.ones((1, size))
        self.bn_bias = torch.zeros((1, size))
        self.eps = eps
    def __call__(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True)
            self.running_mean = 0.999 * self.running_mean + 0.001 * mean
            self.running_var = 0.999 * self.running_var + 0.001 * var
        else:
            with torch.no_grad():
                mean = self.running_mean
                var = self.running_var
        self.out = self.bn_shift * (x - mean)/torch.sqrt(var + self.eps) + self.bn_bias
        return self.out
    def parameters(self):
        return [self.bn_shift, self.bn_bias]
```


```python
emb_dim = 10
h_dim = 100
n_word = 27 #len(vocabs) # 27

lockup_table = torch.randn((n_word, emb_dim))

layers = [Linear(emb_dim * block_size, h_dim), BatchNorm(h_dim), Tanh(),
          Linear(h_dim, h_dim), BatchNorm(h_dim), Tanh(),
          Linear(h_dim, h_dim), BatchNorm(h_dim), Tanh(),
          Linear(h_dim, h_dim), BatchNorm(h_dim), Tanh(),
          Linear(h_dim, h_dim), BatchNorm(h_dim), Tanh(),
          Linear(h_dim, n_word, bias=True)]
with torch.no_grad():
    layers[-1].weight *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight = layer.weight * 5/3
    parameters = [lockup_table] + [p for layer in layers for p in layer.parameters()]
    for p in parameters: p.requires_grad = True
    print(sum([p.nelement() for p in parameters]))
```

    46997



```python
max_steps = 200000
batch_size = 32
lossi = []
ratios = []
for i in range(max_steps):
    # forward
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    x = lockup_table[Xtr[ix]].view((-1, emb_dim * block_size))
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Ytr[ix])
    lossi.append(loss.log10().item())
    # backward
    for l in layers: l.out.retain_grad()
    for p in parameters: p.grad = None
    loss.backward()
    lr = 0.1 if i < (max_steps/2) else 0.01
    for p in parameters: p.data += - lr * p.grad
    with torch.no_grad():
        ratios.append([(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters])
    #track stats
    if i%1000==0:
        print(i, loss.item())
    if i > 10000:
        break
#     break
```

    0 2.406104326248169
    10000 2.2499120235443115



```python
# plot activation
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers):
    if isinstance(layer, Tanh):
        print(f"Layer {i}, mean {round(layer.out.mean().abs().item(), 2)}, std : {round(layer.out.std().abs().item(), 2)}, saturated : {round((layer.out.abs() > 0.97).float().mean().item() * 100, 2)}%")
        legends.append(f"layer_{i}")
        hy, hx = torch.histogram(layer.out, density=True)
        plt.plot(hx.tolist()[:-1], hy.tolist())
plt.legend(legends)
```

    Layer 2, mean 0.0, std : 0.64, saturated : 4.19%
    Layer 5, mean 0.01, std : 0.66, saturated : 3.34%
    Layer 8, mean 0.01, std : 0.67, saturated : 3.06%
    Layer 11, mean 0.0, std : 0.68, saturated : 2.16%
    Layer 14, mean 0.0, std : 0.24, saturated : 0.0%





    <matplotlib.legend.Legend at 0x7fc780bd9430>




    
![png](2022-12-02-trigram-part3_files/2022-12-02-trigram-part3_8_2.png)
    



```python
# plot grad of activation
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers):
    if isinstance(layer, Tanh):
        print(f"Layer {i}, mean {round(layer.out.grad.mean().abs().item(), 4)}, std : {round(layer.out.grad.std().abs().item(), 4)}, saturated : {round((layer.out.grad.abs() > 0.97).float().mean().item() * 100, 2)}%")
        legends.append(f"layer_{i}")
        hy, hx = torch.histogram(layer.out.grad, density=True)
        plt.plot(hx.tolist()[:-1], hy.tolist())
plt.legend(legends)
```

    Layer 2, mean 0.0, std : 0.0019, saturated : 0.0%
    Layer 5, mean 0.0, std : 0.0017, saturated : 0.0%
    Layer 8, mean 0.0, std : 0.0016, saturated : 0.0%
    Layer 11, mean 0.0, std : 0.0016, saturated : 0.0%
    Layer 14, mean 0.0001, std : 0.0041, saturated : 0.0%





    <matplotlib.legend.Legend at 0x7fc7807144f0>




    
![png](2022-12-02-trigram-part3_files/2022-12-02-trigram-part3_9_2.png)
    



```python
# plot activation / grad through time
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters[:-1]):
    if p.shape[0] > 1:
        print(f"layer_{i}, shape {p.shape}, mean {round(p.mean().item(), 4)}, std {round(p.std().item(), 4)}, grad:data {round((p.grad.std()/p.std()).item(), 4)}")
        hy, hx = torch.histogram(p.grad, density=True)
        plt.plot(hx.tolist()[:-1], hy.tolist())
plt.legend(legends)
```

    layer_0, shape torch.Size([27, 10]), mean -0.0057, std 0.998, grad:data 0.0074
    layer_1, shape torch.Size([30, 100]), mean -0.0086, std 0.3318, grad:data 0.0112
    layer_4, shape torch.Size([100, 100]), mean -0.0026, std 0.1863, grad:data 0.0122
    layer_7, shape torch.Size([100, 100]), mean 0.003, std 0.1831, grad:data 0.0118
    layer_10, shape torch.Size([100, 100]), mean -0.0015, std 0.182, grad:data 0.0118
    layer_13, shape torch.Size([100, 100]), mean -0.0004, std 0.1775, grad:data 0.0114
    layer_16, shape torch.Size([100, 27]), mean -0.0, std 0.1352, grad:data 0.0548





    <matplotlib.legend.Legend at 0x7fc78084f190>




    
![png](2022-12-02-trigram-part3_files/2022-12-02-trigram-part3_10_2.png)
    



```python
# plot activation / grad through time
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
    if p.shape[0] > 1:
        legends.append(f"param {i}")
        ys = [o[i] for o in ratios]
        plt.plot(range(len(ys)), ys)

plt.plot(range(len(ys)), [-3] * len(ys))
plt.legend(legends)
```




    <matplotlib.legend.Legend at 0x7fc4c4d09f40>




    
![png](2022-12-02-trigram-part3_files/2022-12-02-trigram-part3_11_1.png)
    


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




    '2022-12-02-trigram-part3.ipynb'




```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```

    [NbConvertApp] Converting notebook 2022-12-02-trigram-part3.ipynb to markdown
    [NbConvertApp] Support files will be in 2022-12-02-trigram-part3_files/
    [NbConvertApp] Making directory ../_posts/2022-12-02-trigram-part3_files
    [NbConvertApp] Making directory ../_posts/2022-12-02-trigram-part3_files
    [NbConvertApp] Making directory ../_posts/2022-12-02-trigram-part3_files
    [NbConvertApp] Making directory ../_posts/2022-12-02-trigram-part3_files
    [NbConvertApp] Making directory ../_posts/2022-12-02-trigram-part3_files
    [NbConvertApp] Making directory ../_posts/2022-12-02-trigram-part3_files
    [NbConvertApp] Making directory ../_posts/2022-12-02-trigram-part3_files
    [NbConvertApp] Making directory ../_posts/2022-12-02-trigram-part3_files
    [NbConvertApp] Writing 11968 bytes to ../_posts/2022-12-02-trigram-part3.md

