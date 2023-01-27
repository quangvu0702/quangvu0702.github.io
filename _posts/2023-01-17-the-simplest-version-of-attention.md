# 2023-01-17 the simplest version of attention

Task: tranlate english to vietnamese

Which word is in your head when you see:
  - `I`? --> `tôi`
  - `know`? --> `biết`
  - `don't`? --> `không`

The idea of creating a model that can query
  - `I` return `tôi`
  - `know` return `biết`
  - `don't` return `không`
  
To let's translate `I don't know` to `Tôi không biết`

### Idea 1
  - We can have a dictionary
  ```
  {"I": "tôi", 'know' : "biết", "don't":"không"}
  
  ```
  - We all know that language is more complex than that. One word can have multiple meaning and the order of word is different.

### Idea 2
  - Attention layer with (query, key, value) as we have discused.

  - every word has one embeeding value. X
  - from X we can generate [(q1, k1, v1), (q2, k2, v2), ..., (qn, kn, vn)] deal to n contexts
  - a tuple (q1, k1, v1) will be
    - q1 = tranform_to_q1(X) = X * Mq_1
    - k1 = tranform_to_k1(X) = X * Mk_1
    - v1 = tranform_to_v1(X) = X * Mv_1
    
  - the attention of `I` to `tôi` on context 1 will be attention(`I`, `tôi`) = `I`.q_1 * `tôi`.k_1
  
  - the attention of `I` to `tôi` on context 2 will be attention(`I`, `tôi`) = `I`.q_2 * `tôi`.k_2
  
  - In this example we only use one context. 
  - Our expectation is that:
    - value of attention(`I`, `tôi`), attention(`know`, `biết`), and attention(`don't`, `không`) will be high.
    - other attentions will be lower.
    
  - We will adjust the value of X_I, X_know, X_don't, X_tôi, X_biết, X_không and Mq_1, Mk_1, Mv_1. Because we only have 1 context, let's remove the index in Mq_1, Mk_1, Mv_1 and rename them into Mq, Mk, Mv.


```python
import torch
from torch import nn
torch.manual_seed(5)

# step 1: inital value

# step 2: attention function

# step 3a: review attention before adjust value

# step 3b: adjust value to get what we expect

# step 3c: review attention after adjust value

```




    <torch._C.Generator at 0x7f988c53ae50>




```python
# step 1: inital value
ws = dict()
ms = dict()
vocabs = ['I', 'know', "don't", "tôi", "biết", "không"]
n_word = len(vocabs)
for w in vocabs:
    ws[w] = torch.randn(10 ,requires_grad=True)
    
for m in ['q', 'k']: # don't use 'v' now
    ms[m] = torch.randn((10,10) ,requires_grad=True)
parameters = list(ws.values()) + list(ms.values())
for w in vocabs:
    print(f'{w} \t {ws[w]}')
```

    I 	 tensor([-0.4868, -0.6038, -0.5581,  0.6675, -0.1974,  1.9428, -1.4017, -0.7626,
             0.6312, -0.8991], requires_grad=True)
    know 	 tensor([-0.5578,  0.6907,  0.2225, -0.6662,  0.6846,  0.5740, -0.5829,  0.7679,
             0.0571, -1.1894], requires_grad=True)
    don't 	 tensor([-0.5659, -0.8327,  0.9014,  0.2116,  0.4479, -0.6088,  0.2389,  0.4699,
            -1.9540, -0.5587], requires_grad=True)
    tôi 	 tensor([ 0.4295, -2.2643, -0.2017,  1.0677,  0.3246, -0.0684, -0.9959,  1.1563,
            -0.3992,  1.2153], requires_grad=True)
    biết 	 tensor([-0.8115, -0.8848, -0.0070, -1.7700, -1.1698, -0.2593,  0.2692,  0.0837,
            -0.5490, -0.0838], requires_grad=True)
    không 	 tensor([-0.1387, -0.5289, -0.4919, -0.4646, -0.0588,  1.2624,  1.1935,  1.5696,
            -0.8977, -0.1139], requires_grad=True)



```python
parameters
```




    [tensor([-0.4868, -0.6038, -0.5581,  0.6675, -0.1974,  1.9428, -1.4017, -0.7626,
              0.6312, -0.8991], requires_grad=True),
     tensor([-0.5578,  0.6907,  0.2225, -0.6662,  0.6846,  0.5740, -0.5829,  0.7679,
              0.0571, -1.1894], requires_grad=True),
     tensor([-0.5659, -0.8327,  0.9014,  0.2116,  0.4479, -0.6088,  0.2389,  0.4699,
             -1.9540, -0.5587], requires_grad=True),
     tensor([ 0.4295, -2.2643, -0.2017,  1.0677,  0.3246, -0.0684, -0.9959,  1.1563,
             -0.3992,  1.2153], requires_grad=True),
     tensor([-0.8115, -0.8848, -0.0070, -1.7700, -1.1698, -0.2593,  0.2692,  0.0837,
             -0.5490, -0.0838], requires_grad=True),
     tensor([-0.1387, -0.5289, -0.4919, -0.4646, -0.0588,  1.2624,  1.1935,  1.5696,
             -0.8977, -0.1139], requires_grad=True),
     tensor([[ 0.8417, -0.6211,  1.4462,  0.4473, -0.6523, -2.0344,  1.1931,  1.1670,
               1.1824,  0.5183],
             [ 1.2896,  0.7412,  0.3150, -1.4139,  0.7605, -0.1033, -1.8593,  0.0541,
              -1.7767, -0.4437],
             [-0.4252,  0.1495,  0.5522, -0.6166, -0.7675, -0.2601, -0.4379,  0.4993,
              -0.1160,  2.6246],
             [ 0.6367, -0.6582, -1.2152,  0.3816, -0.7237, -0.0239,  0.9237, -0.1613,
              -0.9628, -0.4818],
             [-0.3635, -1.6170, -0.4804, -0.1052, -0.2997, -0.0814, -0.5548,  1.5795,
               1.4283,  1.9547],
             [ 0.2882, -1.7719,  1.2488,  0.9230, -0.2765,  0.0223,  1.1403,  1.1029,
               0.4984,  2.3024],
             [ 0.7382,  0.5227,  0.1008,  0.0121,  0.9717, -0.0611,  0.2503, -1.7103,
              -1.4517, -0.3249],
             [-0.3524,  1.1095, -0.5187, -1.3252,  1.3208,  0.6606, -0.9261, -0.6246,
               0.8256,  0.4491],
             [ 0.2775,  1.1933,  0.5628, -0.9629,  0.5114,  1.5773,  0.1181, -0.5082,
              -0.9963,  0.2322],
             [ 0.5533, -0.4774, -0.8092, -0.1280,  0.2360,  1.9578, -0.7626, -1.1713,
               0.3057,  1.1343]], requires_grad=True),
     tensor([[-0.4295,  0.2088, -0.8169, -1.7940, -0.8204,  1.1981, -0.1011,  1.5047,
               0.1800, -2.5233],
             [ 0.3629, -0.0887, -1.5128,  0.3801,  1.3454,  1.1067, -0.6922, -1.1087,
               1.0303,  0.4553],
             [ 0.1949,  1.5408, -0.6603,  0.7536,  0.6132, -0.1875,  1.1591,  0.0550,
               0.3635, -1.1753],
             [-0.8516,  0.0108, -0.6256,  1.9270,  2.7857,  1.1867, -1.6556, -0.3342,
              -1.6154,  0.4106],
             [-0.3648,  0.7173,  0.1441, -0.6492, -0.5191, -1.2785,  0.7221,  0.2498,
               0.9954,  0.0522],
             [-0.5776,  1.2181, -0.6359, -1.2286,  0.1054, -0.2691,  1.8014, -0.2985,
               0.2094,  0.8249],
             [ 0.5919,  0.7107, -0.2622,  1.5016,  1.2452, -0.5602,  1.0424, -0.4483,
               1.5554, -1.2570],
             [-0.1336,  0.1117, -0.1468, -0.5753, -0.6084, -2.3205, -0.3038,  0.5376,
               0.2686, -1.7700],
             [ 0.9437,  0.1021,  1.0349,  0.3142,  1.2781,  0.5857,  1.3827,  0.6941,
              -1.1786,  0.2777],
             [-0.8117, -0.2776, -0.5490, -1.0008, -0.3259, -0.4088, -0.0407, -2.4222,
              -0.4039, -0.4381]], requires_grad=True)]




```python
# step 2: attention function
def Q(w): return ws[w] @ ms['q']
def K(w): return ws[w] @ ms['k']
def V(w): return ws[w] @ ms['v']
    
def attention(w1, w2):
    return Q(w1) @ K(w2)

attention('I', 'tôi').item()
```




    -9.358345031738281




```python
# step 3a: review attention before adjust value
import matplotlib.pyplot as plt
%matplotlib inline

def ploat_heatmap(N):
    plt.figure(figsize=(8,8))
    plt.imshow(N, cmap='Blues')
    for i in range(n_word):
        for j in range(n_word):
            w1, w2 = vocabs[i], vocabs[j]
            chstr = f"{w2}-{w1}"
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, round(N[i,j].item(),2), ha="center", va="top", color='gray')
    plt.axis('off');

N = torch.zeros((n_word, n_word), dtype=torch.float)
for i in range(n_word):
    for j in range(n_word):
        w2, w1 = vocabs[i], vocabs[j]
        N[i,j] = attention(w1, w2).item()
        
ploat_heatmap(N)
```


    
![png](2023-01-17-the-simplest-version-of-attention_files/2023-01-17-the-simplest-version-of-attention_9_0.png)
    



```python
attention('I', 'tôi').item(), attention('know', "biết").item(), attention("don't", "không").item()
```




    (-9.358345031738281, -3.9177298545837402, 27.493515014648438)




```python
# step 3b: adjust value to get what we expected.
# - Our expectation is that:
#   - value of attention(`I`, `tôi`), attention(`know`, `biết`), and attention(`don't`, `không`) will be high.
#   - other attentions will be lower.

# let set values of attention(`I`, `tôi`), attention(`know`, `biết`), and attention(`don't`, `không`) are 1
# and others are zero
# for i in range(3):
loss = 0
N = torch.zeros((n_word, n_word), dtype=torch.float)
for i in range(n_word):
    for j in range(n_word):
        w1, w2 = vocabs[i], vocabs[j]
        if (w1, w2) in [('I', "tôi"), ('know', "biết"), ("don't", "không")]:
            loss += (1 - attention(w1, w2))**2
            print(w1, w2)
        else:
            loss += (0 - attention(w1, w2))**2
loss = loss / n_word**2
print(loss.item())
for p in parameters: p.grad = None
loss.backward()
for p in parameters:
    p.data -= p.grad * 0.01
print(attention('I', 'tôi').item(), attention('know', "biết").item(), attention("don't", "không").item())
```

    I tôi
    know biết
    don't không
    261.2638854980469
    2.6664211750030518 -7.058520317077637 -19.68987274169922



```python
for p in parameters:
    print(p.data, p.grad)
```

    tensor([-0.9359, -0.1554, -0.7149,  0.7381, -0.3860,  1.5995, -1.0444, -0.5659,
             0.4357, -0.5331]) tensor([ 44.9136, -44.8463,  15.6789,  -7.0604,  18.8613,  34.3284, -35.7294,
            -19.6673,  19.5554, -36.5994])
    tensor([-0.4857,  0.6613,  0.0900, -0.3526,  0.4924,  0.4200, -0.5507,  0.8069,
             0.0284, -0.9084]) tensor([ -7.2066,   2.9390,  13.2456, -31.3591,  19.2269,  15.4033,  -3.2172,
             -3.8952,   2.8673, -28.0966])
    tensor([-1.1266, -0.6443,  1.1219,  0.3079,  0.3964, -0.3271,  0.4423,  0.5943,
            -1.1669,  0.0448]) tensor([ 56.0702, -18.8334, -22.0443,  -9.6297,   5.1496, -28.1674, -20.3331,
            -12.4404, -78.7153, -60.3507])
    tensor([-0.2729, -1.3634, -0.0075,  0.8225,  0.2344, -0.1267, -0.4623,  0.7054,
            -0.3406,  0.9588]) tensor([ 70.2396, -90.0879, -19.4226,  24.5210,   9.0204,   5.8325, -53.3596,
             45.0875,  -5.8574,  25.6537])
    tensor([-0.5511, -0.7286,  0.4007, -1.3859, -1.0879,  0.1073,  0.2343,  0.0385,
            -0.4960,  0.2140]) tensor([-26.0469, -15.6203, -40.7730, -38.4058,  -8.1892, -36.6579,   3.4871,
              4.5187,  -5.2962, -29.7826])
    tensor([ 0.0222, -0.4750, -0.4965, -0.2808, -0.2954,  1.0003,  1.0439,  1.3230,
            -0.8258, -0.4774]) tensor([-16.0912,  -5.3877,   0.4597, -18.3853,  23.6599,  26.2087,  14.9652,
             24.6578,  -7.1817,  36.3544])
    tensor([[ 0.7659, -0.6032,  1.4541,  0.4097, -0.7326, -2.1740,  1.1578,  1.1869,
              1.1662,  0.4412],
            [ 1.6503,  0.8683,  0.1515, -1.1488,  0.7573, -0.2632, -1.6480, -0.0133,
             -1.0063, -0.5451],
            [-0.3146,  0.1335,  0.5604, -0.5692, -0.6393, -0.0273, -0.3589,  0.4974,
             -0.1273,  2.8326],
            [ 0.4240, -0.6971, -1.2689,  0.3608, -0.5695,  0.0989,  0.7067, -0.2029,
             -1.2607, -0.6261],
            [-0.3886, -1.6503, -0.5162, -0.0712, -0.2151,  0.0183, -0.6291,  1.5244,
              1.3728,  1.8771],
            [ 0.1266, -1.7873,  1.1623,  0.9243, -0.3172, -0.1621,  0.9333,  1.0202,
              0.4948,  1.8791],
            [ 0.8636,  0.5594,  0.1802, -0.0169,  0.7991, -0.2071,  0.4262, -1.6407,
             -1.2800, -0.1679],
            [-0.5284,  1.0226, -0.3843, -1.5020,  1.1407,  0.5655, -1.0194, -0.6008,
              0.4986,  0.3948],
            [ 0.3058,  1.2858,  0.4351, -0.8611,  0.4246,  1.2619,  0.0985, -0.5502,
             -0.6089, -0.0840],
            [ 0.3834, -0.4817, -0.6979, -0.2905,  0.0724,  1.7977, -0.7941, -1.0788,
              0.0608,  1.1868]]) tensor([[  7.5766,  -1.7920,  -0.7824,   3.7556,   8.0353,  13.9615,   3.5272,
              -1.9939,   1.6172,   7.7123],
            [-36.0612, -12.7079,  16.3473, -26.5118,   0.3187,  15.9935, -21.1314,
               6.7352, -77.0333,  10.1393],
            [-11.0572,   1.5964,  -0.8284,  -4.7374, -12.8237, -23.2855,  -7.9007,
               0.1870,   1.1301, -20.8020],
            [ 21.2710,   3.8933,   5.3605,   2.0798, -15.4191, -12.2879,  21.6953,
               4.1554,  29.7980,  14.4218],
            [  2.5027,   3.3298,   3.5817,  -3.4061,  -8.4604,  -9.9753,   7.4299,
               5.5182,   5.5448,   7.7599],
            [ 16.1547,   1.5396,   8.6462,  -0.1318,   4.0718,  18.4321,  20.6972,
               8.2672,   0.3619,  42.3349],
            [-12.5473,  -3.6704,  -7.9349,   2.8988,  17.2533,  14.5992, -17.5950,
              -6.9602, -17.1615, -15.7046],
            [ 17.6046,   8.6894, -13.4356,  17.6852,  18.0107,   9.5163,   9.3331,
              -2.3880,  32.6950,   5.4293],
            [ -2.8262,  -9.2500,  12.7710, -10.1816,   8.6765,  31.5380,   1.9571,
               4.2002, -38.7423,  31.6238],
            [ 16.9811,   0.4265, -11.1255,  16.2553,  16.3621,  16.0048,   3.1468,
              -9.2575,  24.4851,  -5.2462]])
    tensor([[-0.5118, -0.0363, -0.8812, -1.6472, -0.9704,  1.1933,  0.0948,  1.6917,
              0.3951, -2.3613],
            [ 0.4615,  0.1123, -1.3421,  0.1927,  1.3251,  0.9111, -0.9625, -0.9805,
              0.5606,  0.2641],
            [ 0.2689,  1.5751, -0.5784,  0.7034,  0.6416, -0.1830,  1.1279,  0.0833,
              0.2451, -1.1957],
            [-1.0130, -0.4051, -0.8299,  2.2369,  2.5224,  1.2327, -1.2666, -0.1522,
             -1.1591,  0.6510],
            [-0.3400,  0.5997,  0.1495, -0.5691, -0.5932, -1.2420,  0.8552,  0.3211,
              1.0371,  0.1014],
            [-0.6011,  1.3451, -0.7221, -1.2359,  0.1460, -0.2542,  1.7373, -0.5110,
              0.2005,  0.6980],
            [ 0.7967,  0.8377,  0.0042,  1.3089,  1.3353, -0.5939,  0.8851, -0.3126,
              1.1658, -1.3139],
            [-0.0103,  0.0938, -0.0496, -0.5770, -0.5091, -2.1695, -0.2089,  0.4840,
              0.2649, -1.7252],
            [ 0.7476,  0.0231,  0.8599,  0.4152,  1.1355,  0.4670,  1.3999,  0.7380,
             -1.0001,  0.2932],
            [-0.9383, -0.6114, -0.6527, -0.8013, -0.4649, -0.3609,  0.2404, -2.2256,
             -0.0147, -0.1721]]) tensor([[  8.2294,  24.5073,   6.4271, -14.6786,  15.0026,   0.4848, -19.5847,
             -18.6994, -21.5060, -16.1986],
            [ -9.8564, -20.1095, -17.0757,  18.7361,   2.0264,  19.5591,  27.0276,
             -12.8126,  46.9697,  19.1254],
            [ -7.3917,  -3.4363,  -8.1937,   5.0177,  -2.8364,  -0.4559,   3.1269,
              -2.8275,  11.8362,   2.0431],
            [ 16.1430,  41.5957,  20.4346, -30.9927,  26.3274,  -4.6047, -38.8979,
             -18.1920, -45.6317, -24.0433],
            [ -2.4755,  11.7591,  -0.5420,  -8.0102,   7.4100,  -3.6562, -13.3075,
              -7.1390,  -4.1679,  -4.9217],
            [  2.3569, -12.7006,   8.6200,   0.7295,  -4.0611,  -1.4916,   6.4029,
              21.2462,   0.8934,  12.6945],
            [-20.4752, -12.6956, -26.6442,  19.2658,  -9.0120,   3.3713,  15.7318,
             -13.5728,  38.9623,   5.6863],
            [-12.3305,   1.7891,  -9.7183,   0.1774,  -9.9291, -15.1030,  -9.4897,
               5.3606,   0.3719,  -4.4774],
            [ 19.6139,   7.8991,  17.5034, -10.0934,  14.2624,  11.8686,  -1.7259,
              -4.3879, -17.8529,  -1.5426],
            [ 12.6557,  33.3793,  10.3740, -19.9486,  13.8973,  -4.7881, -28.1139,
             -19.6569, -38.9218, -26.5994]])



```python
# step 3c: review attention after adjust value
import matplotlib.pyplot as plt
%matplotlib inline

N = torch.zeros((n_word, n_word), dtype=torch.float)
for i in range(n_word):
    for j in range(n_word):
        w2, w1 = vocabs[i], vocabs[j]
        N[i,j] = attention(w1, w2).item()
        
ploat_heatmap(N)
```


    
![png](2023-01-17-the-simplest-version-of-attention_files/2023-01-17-the-simplest-version-of-attention_13_0.png)
    


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


```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```
