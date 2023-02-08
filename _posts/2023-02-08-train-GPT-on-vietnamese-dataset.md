# 2023-01-30 train GPT on vietnamese dataset


```python
# download data
```


```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

    cuda



```python
# read on review data
with open('../data/truyen_kieu.csv', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace(',', ' , ')
    text = [o for sent in text.split("\n") for o in sent.split(".") if not o.isnumeric()]
    text = ' \n '.join(text)
    text = text.lower()
    truyen_kieu = text.split(" ")
```


```python
text = truyen_kieu
```


```python
# Here is all unique character that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}

encode = lambda s: [ctoi[c] for c in s if c in ctoi]
decode = lambda l: ' '.join([itoc[i] for i in l])

print(encode("tôi là".split(" ")))

print(decode(encode("tôi là".split(" "))))
```

    [2760, 1248]
    tôi là



```python
# Let now encode the entire text dataset and store it into torch.Tensor
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

data[0:100]
```

    torch.Size([49748]) torch.int64





    tensor([2646, 1905, 2606,  498, 1685, 2364,    6,    0,    1,  449, 2715,  449,
            1554, 1096, 1248,  742, 1740,    1,    0,    1, 2664, 2055, 1575,  467,
             225,  632,    1, 1842, 3155, 2638, 2509, 1454, 3138, 3318, 1289,    1,
               0,    1, 1332,  838,  227, 2331, 2781, 1991,    6,    0,    1, 2689,
            3001, 2063, 2466, 1464, 1022, 3185,  738,    1,    0,    1,  535, 2485,
            1348,  821, 2653, 3202,    6,    0,    1, 1991, 2744,  564, 1406,  492,
            2610, 2359, 3001,    1,    0,    1, 2193, 1905,  749, 2777, 2602, 1433,
               6,    0,    1,  236, 2024, 2040, 1367,    6,    0,  882, 1156, 2992,
            2893,    1,    0,    1])




```python
# Let's now split up the data into train set and validation set
n = round(len(data) * 0.98);
train_data = data[:n]
val_data   = data[n:]
len(train_data), len(val_data)
```




    (48753, 995)




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
```


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from pathlib import Path

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
```


```python
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

def train(lr=0.001, model_name=None, only_load_model=False):
    optimizer = AdamW(model.parameters())
    out_dir = Path('../checkpoints')
    fn = out_dir/model_name
    if fn.is_file():
        checkpoint = torch.load(fn, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    if only_load_model == False:
        for i in range(max_iter + 1):
            if (i % eval_iters == 0) and (i > 0):
                out = estimate_loss()
                print(f"Train loss: {out['train']}. Val loss: {out['val']}. ")

                # save checkpoint
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, fn)
            xb, yb = get_batch()
            loss, logits = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


```


```python
B, T, C = 64, 64, 384 ; # batch, time, channel
n_embd = C
batch_size, block_size = B, T
max_iter = 5000
num_heads = 6
num_blocks = 6
eval_iters = 500
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

    torch.Size([64, 64]) torch.Size([64, 64])
    13.270579 M parameters
    tensor(8.3334, device='cuda:0', grad_fn=<NllLossBackward0>)



```python
train(lr=lr, model_name='combine.pt')

sent = model.generate(idx=torch.zeros((1,1), dtype=torch.long).to(device), max_new_token=500)
print(decode(sent[0].tolist()))
```

    Train loss: 0.16425329446792603. Val loss: 0.16477559506893158. 
    saving checkpoint to ../checkpoints
    Train loss: 0.13645370304584503. Val loss: 0.1356091946363449. 
    saving checkpoint to ../checkpoints
    Train loss: 0.1277822107076645. Val loss: 0.12738379836082458. 
    saving checkpoint to ../checkpoints
    Train loss: 0.12291053682565689. Val loss: 0.1230657771229744. 
    saving checkpoint to ../checkpoints
    Train loss: 0.12174422293901443. Val loss: 0.12226003408432007. 
    saving checkpoint to ../checkpoints
    Train loss: 0.12167149782180786. Val loss: 0.12158866226673126. 
    saving checkpoint to ../checkpoints
    Train loss: 0.11997853964567184. Val loss: 0.1197664737701416. 
    saving checkpoint to ../checkpoints
    Train loss: 0.11667455732822418. Val loss: 0.11683859676122665. 
    saving checkpoint to ../checkpoints
    Train loss: 0.11371710896492004. Val loss: 0.11407747864723206. 
    saving checkpoint to ../checkpoints
    Train loss: 0.11282995343208313. Val loss: 0.11278162151575089. 
    saving checkpoint to ../checkpoints
     
     ở người thấy ai người cũ cũng yêu ,  
     xôn xao oanh yến rập rìu trúc mai 
      
     tin nhạn vẩn lá thư bài ,  
     đưa người cửa trước rước người cửa sau 
      
     lạ tai nghe chửa biết đâu ,  
     xem tình ra cũng những màu dở dang 
      
     lễ xong hương hỏa cái đường ,  
     nổi loan một được khôn sương 
      
     nghĩ tình ghi nghĩa mẹ kiếp gia 
      
     sao ngay biện bạch một bề ,  
     dạy cho má phấn lại về quê 
      
     thấy lời nghiêm huấn rành rành ,  
     đánh liều sinh mới lấy tình nài kêu 
      
     rằng: con biết tội đã nhiều ,  
     dẫu rằng sấm sét búa rìu cũng cam 
      
     trót vì tay đã nhúng chàm ,  
     dại rồi còn biết khôn làm sao đây 
      
     cùng nhau vả tiếng một ngày ,  
     ôm cầm ai nỡ dứt dây cho đành 
      
     lượng trên quyết chẳng thương tình ,  
     bạc đen thôi có tiếc mình làm chi 
      
     thấy lời sắt đá tri tri ,  
     sốt gan ông mới cáo quì cửa công 
      
     đất bằng nổi sóng đùng đùng ,  
     phủ đường sai lá phiếu hồng thôi tra 
      
     cùng nhau theo gót sai nha ,  
     song song vào trước sân hoa lạy quì 
      
     trông lên mặt sắt đen sì ,  
     lập nghiêm trước đã ra uy nặng lời: 
     gã kia dại nết chơi bời ,  
     mà con người thế là người đong đưa 
      
     tuồng chi hoa thải hương thừa ,  
     mượn màu son phấn đánh lừa con đen 
      
     suy trong tình trạng nguyên đơn ,  
     bề nào thì cũng chưa yên bề nào 
      
     phép công chiếu án luận vào 
      
     có hai đường ấy muốn sao mặc mình 
      
     một là cứ phép gia hình ,  
     một là lại cứ lầu xanh phó về 
      
     nàng rằng: đã quyết một bề! 
     nhện này vương lấy tơ kia mấy lần 
      
     đục trong thân cũng là thân 
      
     yếu thơ vâng chịu trước sân lôi đình! 
     dạy rằng: cứ phép gia hình! 
     ba cây chập lại một cành mẫu đơn 
      
     phận đành chi dám kêu kia ,  
     phận sao phận bạc như vôi ,  
     đã đành nước chẩy hoa trôi lỡ làng 
      
     ôi kim lang! hỡi kim lang! 
     thôi thôi thiếp đã phụ chàng từ đây! 
     cạn lời hồn ngất máu say ,  
     một hơi lặng ngắt



```python
sent = model.generate(idx=torch.zeros((1,1), dtype=torch.long).to(device), max_new_token=500)
print(decode(sent[0].tolist()))
```

     
     nàng càng trời thẳm đất dày! 
     thân này đã bỏ những ngày ra đi 
      
     thôi thì thôi có tiếc gì! 
     sẵn dao tay áo tức thì giở ra 
      
     sợ gan nát ngọc liều hoa! 
     mụ còn trông mặt nàng đà quá tay 
      
     thương ôi tài sắc bậc này ,  
     một dao oan nghiệt đứt dây phong trần 
      
      
     nỗi oan vỡ lở xa gần ,  
     trong nhà người chật một lần như nêm 
      
     nàng thì bằn bặt giấc tiên ,  
     mụ thì cầm cập mặt nhìn hồn bay 
      
     vực nàng vào chốn hiên tây ,  
     cắt người coi sóc chạy thầy thuốc thang 
      
     nào hay chưa hết trần duyên ,  
     trong mê dường đã đứng bên một nàng 
      
     rỉ rằng: nhân quả dở dang ,  
     đã toan trốn nợ đoạn trường được sao? 
     số còn nặng nợ má đào ,  
     người dầu muốn quyết trời nào đã cho 
      
     hãy xin hết kiếp liễu bồ ,  
     sông tiền đường sẽ hẹn hò về sau 
      
     thuốc thang suốt một ngày thâu ,  
     giấc mê nghe đã dàu dàu vừa tan 
      
     tú bà chực sẵn bên màn ,  
     lựa lời khuyên giải mơn man băng tơ 
      
     trông vào một những ngày xưa 
      
     bẻ bai rủ rỉ tiếng tơ ,  
     trầm bay nhạt khói gió đưa lay rèm 
      
     dường như bên nóc trước thềm ,  
     tiếng kiều đồng vọng bóng xiêm mơ màng ,  
     bởi lòng tạc đá ghi vàng ,  
     tưởng nàng nên lại thấy nàng về đây 
      
      
     những là phiền muộn đêm ngày ,  
     xuân thu biết đã đổi thay mấy lần? 
     chế khoa gặp hội trường văn 
      
     vương ,  kim cùng chiếm bảng xuân một ngày 
      
     cửa trời rộng mở đường mây ,  
     hoa chào ngõ hạnh hương bay dặm phần 
      
     chàng vương nhớ đến xa gần ,  
     sang nhà chung lão tạ ân chu tuyền 
      
     tình xưa ân trả nghĩa đền ,  
     gia thân lại mới kết duyên châu trần 
      
     kim từ nhẹ bước thanh vân ,  
     nỗi nàng càng nghĩ xa gần càng thương 
      
     ấy ai dặn ngọc thề vàng ,  
     bây giờ kim mã ngọc đường với ai? 
     ngọn bèo chân sóng lạc loài ,  
     nghĩ mình vinh hiển thương người lưu ly 
      
     vâng ra ngoại nhậm lâm truy ,  
     quan san nghìn dặm thê nhi một



```python
sent = model.generate(torch.tensor([encode("nặng lời".split(" "))]).to(device), max_new_token=50)
print(decode(sent[0].tolist()))
```

    nặng lời ,  
     rẩy xin chén nước cho người thác oan 
      
     bây giờ trâm gẫy bình tan ,  
     kể làm sao xiết muôn vàn ái ân 
      
     trăm nghìn gửi lại tình quân ,  
     tơ duyên ngắn ngủi có ngần ấy


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

    Input In [18], in <cell line: 1>()
    ----> 1 this_notebook


    NameError: name 'this_notebook' is not defined



```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```
