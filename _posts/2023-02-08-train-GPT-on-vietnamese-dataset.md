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
print(text[0:330])
```

    ['trăm', 'năm', 'trong', 'cõi', 'người', 'ta', ',', '', '\n', 'chữ', 'tài', 'chữ', 'mệnh', 'khéo', 'là', 'ghét', 'nhau', '\n', '', '\n', 'trải', 'qua', 'một', 'cuộc', 'bể', 'dâu', '\n', 'những', 'điều', 'trông', 'thấy', 'mà', 'đau', 'đớn', 'lòng', '\n', '', '\n', 'lạ', 'gì', 'bỉ', 'sắc', 'tư', 'phong', ',', '', '\n', 'trời', 'xanh', 'quen', 'thói', 'má', 'hồng', 'đánh', 'ghen', '\n', '', '\n', 'cảo', 'thơm', 'lần', 'giở', 'trước', 'đèn', ',', '', '\n', 'phong', 'tình', 'cổ', 'lục', 'còn', 'truyền', 'sử', 'xanh', '\n', '', '\n', 'rằng', 'năm', 'gia', 'tĩnh', 'triều', 'minh', ',', '', '\n', 'bốn', 'phương', 'phẳng', 'lặng', ',', '', 'hai', 'kinh', 'vững', 'vàng', '\n', '', '\n', 'có', 'nhà', 'viên', 'ngoại', 'họ', 'vương', ',', '', '\n', 'gia', 'tư', 'nghĩ', 'cũng', 'thường', 'thường', 'bực', 'trung', '\n', '', '\n', 'một', 'trai', 'con', 'thứ', 'rốt', 'lòng', ',', '', '\n', 'vương', 'quan', 'là', 'chữ', ',', '', 'nối', 'dòng', 'nho', 'gia', '\n', '', '\n', 'đầu', 'lòng', 'hai', 'ả', 'tố', 'nga', ',', '', '\n', 'thúy', 'kiều', 'là', 'chị', ',', '', 'em', 'là', 'thúy', 'vân', '\n', '', '\n', 'mai', 'cốt', 'cách', ',', '', 'tuyết', 'tinh', 'thần', ',', '', '\n', 'mỗi', 'người', 'một', 'vẻ', ',', '', 'mười', 'phân', 'vẹn', 'mười', '\n', '', '\n', 'vân', 'xem', 'trang', 'trọng', 'khác', 'vời', ',', '', '\n', 'khuôn', 'trăng', 'đầy', 'đặn', ',', '', 'nét', 'ngài', 'nở', 'nang', '\n', '', '\n', 'hoa', 'cười', 'ngọc', 'thốt', 'đoan', 'trang', ',', '', '\n', 'mây', 'thua', 'nước', 'tóc', ',', '', 'tuyết', 'nhường', 'màu', 'da', '\n', '', '\n', 'kiều', 'càng', 'sắc', 'sảo', ',', '', 'mặn', 'mà', ',', '', '\n', 'so', 'bề', 'tài', ',', '', 'sắc', ',', '', 'lại', 'là', 'phần', 'hơn', '\n', '', '\n', 'làn', 'thu', 'thủy', ',', '', 'nét', 'xuân', 'sơn', ',', '', '\n', 'hoa', 'ghen', 'thua', 'thắm', ',', '', 'liễu', 'hờn', 'kém', 'xanh', '\n', '', '\n', 'một', ',', '', 'hai', 'nghiêng', 'nước', 'nghiêng', 'thành', ',', '', '\n', 'sắc', 'đành', 'đòi', 'một', ',', '', 'tài', 'đành', 'họa', 'hai', '\n', '', '\n', 'thông', 'minh', 'vốn', 'sẵn', 'tư', 'trời', ',', '', '\n', 'pha', 'nghề', 'thi', 'họa', ',', '', 'đủ', 'mùi', 'ca', 'ngâm', '\n', '', '\n', 'cung', 'thương']



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

    [2175, 936]
    tôi là



```python
# Let now encode the entire text dataset and store it into torch.Tensor
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

data[0:100]
```

    torch.Size([32186]) torch.int64





    tensor([2080, 1457, 2046,  330, 1291, 1827,    3,    0,    1,  291, 2134,  291,
            1183,  811,  936,  524, 1336,    1,    0,    1, 2095, 1579, 1201,  302,
             111,  432,    1, 1410, 2495, 2072, 1963, 1107, 2481, 2629,  971,    1,
               0,    1, 1008,  606,  112, 1798, 2192, 1528,    3,    0,    1, 2112,
            2380, 1586, 1921, 1116,  758, 2522,  520,    1,    0,    1,  359, 1939,
            1022,  592, 2085, 2537,    3,    0,    1, 1528, 2161,  378, 1067,  324,
            2050, 1823, 2380,    1,    0,    1, 1690, 1457,  529, 2188, 2042, 1091,
               3,    0,    1,  119, 1555, 1568, 1037,    3,    0,  644,  860, 2372,
            2289,    1,    0,    1])




```python
# Let's now split up the data into train set and validation set
n = round(len(data) * 0.98);
train_data = data[:n]
val_data   = data[n:]
len(train_data), len(val_data)
```




    (31542, 644)




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
B, T, C = 64, 64, 36 ; # batch, time, channel
n_embd = C
batch_size, block_size = B, T
max_iter = 25000
num_heads = 6
num_blocks = 6
eval_iters = 1000
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
    0.293041 M parameters
    tensor(7.9529, device='cuda:0', grad_fn=<NllLossBackward0>)



```python
train(lr=lr, model_name='combine.pt_v4')
```

    Train loss: 3.715602397918701. Val loss: 3.7170333862304688. 
    saving checkpoint to ../checkpoints
    Train loss: 2.451805353164673. Val loss: 2.4547319412231445. 
    saving checkpoint to ../checkpoints
    Train loss: 1.6461000442504883. Val loss: 1.6471295356750488. 
    saving checkpoint to ../checkpoints
    Train loss: 1.167556881904602. Val loss: 1.166641116142273. 
    saving checkpoint to ../checkpoints
    Train loss: 0.8847333192825317. Val loss: 0.8847111463546753. 
    saving checkpoint to ../checkpoints
    Train loss: 0.7004756927490234. Val loss: 0.6995299458503723. 
    saving checkpoint to ../checkpoints
    Train loss: 0.609959602355957. Val loss: 0.6089354157447815. 
    saving checkpoint to ../checkpoints
    Train loss: 0.5567530393600464. Val loss: 0.5564932227134705. 
    saving checkpoint to ../checkpoints
    Train loss: 0.4830600321292877. Val loss: 0.48131075501441956. 
    saving checkpoint to ../checkpoints
    Train loss: 0.45059171319007874. Val loss: 0.450531929731369. 
    saving checkpoint to ../checkpoints
    Train loss: 0.42473775148391724. Val loss: 0.42357054352760315. 
    saving checkpoint to ../checkpoints
    Train loss: 0.40449753403663635. Val loss: 0.4051833748817444. 
    saving checkpoint to ../checkpoints
    Train loss: 0.39978060126304626. Val loss: 0.4000585973262787. 
    saving checkpoint to ../checkpoints
    Train loss: 0.3728296756744385. Val loss: 0.3725215494632721. 
    saving checkpoint to ../checkpoints
    Train loss: 0.36801204085350037. Val loss: 0.36851534247398376. 
    saving checkpoint to ../checkpoints
    Train loss: 0.3432505428791046. Val loss: 0.34522923827171326. 
    saving checkpoint to ../checkpoints
    Train loss: 0.33629897236824036. Val loss: 0.33626800775527954. 
    saving checkpoint to ../checkpoints
    Train loss: 0.33941349387168884. Val loss: 0.34081289172172546. 
    saving checkpoint to ../checkpoints
    Train loss: 0.3141455352306366. Val loss: 0.31441447138786316. 
    saving checkpoint to ../checkpoints
    Train loss: 0.3109095096588135. Val loss: 0.31082460284233093. 
    saving checkpoint to ../checkpoints
    Train loss: 0.29889073967933655. Val loss: 0.2982620596885681. 
    saving checkpoint to ../checkpoints
    Train loss: 0.29737386107444763. Val loss: 0.29717889428138733. 
    saving checkpoint to ../checkpoints
    Train loss: 0.2884153425693512. Val loss: 0.2893764078617096. 
    saving checkpoint to ../checkpoints
    Train loss: 0.2856851816177368. Val loss: 0.28586092591285706. 
    saving checkpoint to ../checkpoints
    Train loss: 0.2757451832294464. Val loss: 0.27520686388015747. 
    saving checkpoint to ../checkpoints



```python
sent = model.generate(idx=torch.zeros((1,1), dtype=torch.long).to(device), max_new_token=500)
print(decode(sent[0].tolist()))
```

     
     mắc lừa lọc đã dành có nơi 
      
     rõ ràng mở mắt đầy xưa nay ,  
     thầm một chiều quyền xót xa 
      
     gặp từ hương lửa tàn ,  
     thật tin nghe hẳn nghìn sầu 
      chia vò hồng nhan ,  
     khách hồng rụng một lời gửi 
      
     vân trăng nọ hoa đào ,  
     lòng kia giữ giàng họ thúc một xa 
      
     dâng thư trước đã thẹn nàng ,  
     khóc người thấy bóng trăng hoa? 
     mặt nào ai có hôm ngồi ,  
     là nhiều sao nói cười như không 
      
     vỗ nay trong miệng những ngày đào ,  
     thương sao hẳn thành con sai quan thì 
      
     rằng sông cũng bớt vài trên một lời ,  
     ngẫm những gạt lệ ,  
     ngập ngừng lại gieo lấy mình xa 
      
     đàn khoan bắt quì ,  
     sinh đà gieo vàng chờ được nào! 
     tơ vì nước đến sau ,  
     cơ từ đã thu ngại công 
      
     non người quốc sắc nước non ,  
     tiếc thay huyên rõ ràng đó luồn đây 
      
     thầm lời đã sống đọa theo sau 
      
     thương nàng báo đáp ân tình ,  
     chút nàng ba sinh nàng ra xin đi! 
     từ rằng: nghề mọn nhà ,  
     lòng thề nọ ngẩn ngơ ngẩn ngơ ngẩn sầu 
      
     bóng tà tà dâu ,  
     bóng tà tà đã ra phụ phàng ,  
     tiểu thư đã áp thẳng tìm tòi ngẩn ngơ 
      
     tơi am mây tạnh là ,  
     thiết quân có mụ vì nhà thường ,  
     lửa phiền càng dập càng khêu mối phiền 
      
     một lòng trong tiền văn lão cũng chôn sương ,  
     giãi lòng: 
     nhớ nơi mang những gạt vững chiều thần ,  
     hoa trôi dần dần mà đến đây? 
     êm ả ghềnh sẵn hai bề vẹn hai! 
     thôi ta sẽ chớ tình máu không 
      
     ngẫm duyên ta có mọi đồ ,  
     đã buồn cả ruột ngàn đã lề 
      
     nghề chơi lại càng dào mạch tương 
      
     đòi phen đổi mai ao ,  
     đủ ngần quả kiếp người đây ,  
     mười phần oanh 
     hai văn hơi khéo vỡ hai ,  
     giắt mặt mà liễu cờ mấy khi! 
     hoa bèo nữa vàng sắm sửa muộn xưa 
      
     có điều ngang ngửa vì này bèo ,  
     lòng sâu nghĩa người nhỏ xôn xao 
      
     người nách thước còn ngần này ,  biết trong quân chầy 
      
     hoa truyền sửa lần mơ chay lòng ,  
     tìm



```python
sent = model.generate(idx=torch.zeros((1,1), dtype=torch.long).to(device), max_new_token=500)
print(decode(sent[0].tolist()))
```

     
     chàng về viện sách nàng dời lầu trang 
      
      
     từ phen đá biết tuổi vàng ,  
     tình càng vén vì hoa kề ,  
     mấy lòng vừa ghé xiêu xiêu xiêu xiêu xiêu 
      
     vài tuần bạc ác sầu cho phu 
      
     xuân nước dẫy sóng đủ đường ,  
     phép về xuân dù phường chia hai 
      
     những là một lần mới ra ,  
     chàng càng trông tỏ thức hồng ,  
     rành rành tích việt duyên ngồi 
      
      
     lấy điều trúc lục e lệ ,  
     khóc than ngọc cho nàng tình đầu ,  
     thẹn mình chén gặp nàng cần dịu dàng! 
     mụ tháng thật quẩn trà cây 
      
     mảnh người dưới nguyệt thân ,  
     chàng vương nghe tiếng vàng liêu bưng kín chẳng ưa? 
      
     gia hoa đào khuya khăn ,  
     đất bằng ăn ngày một đau 
      
     nhớ nơi hằng thủy mai sau! 
     những là nặng nắng mưa ,  
     buồng không thương chi cho khi về lầu xanh 
      
     rằng sông chẳng chút cũ tràng oanh ,  
     uốn lưng bút giá dày đã đành ,  
     gấp người còn có ai sở trác giữa trời 
      
     bắt về đến kim ,  
     mụ quản huyền đâu đã giục đành ,  
     chiều lòng biết có nợ chiều đời 
      
     gieo trời cạn ý đà sương được lời 
      
     tình nhân mới hạ công ,  
     còn nhiều đã có gương truyền hôm nay! 
     tinh trướng nghe nói chẳng lựa sẻ lửa nhân ,  
     vân rằng: ái khỏi kiến lửa ba 
      
     cửa đóng then nhặt gói về lầu 
      
     mặn bất ý rụt rè ,  
     hoa kia đã chắp áo dài ,  
     xót liễu nước chưa vẹn chữ đồng tự hôn 
      
     cải nhậm hương lân nồi trời 
      
     sự nhà hành viện xưa nay ,  
     cũng đâu đã về chia cao 
      
     bao nhiêu đoạn khổ ,  tình chẳng treo trên 
      
     sự tình chàng thúc một tỉnh say ,  
     tin sương gieo xuống bóng cờ mấy hồi khốn hay 
      
     đầy sông tiền riêng chưa nện cầu vắng đâu 
      
     sợ lần khân quá ra ,  
     đây vì hoa cuối xuân đường trần hình ,  
     một phen đá biết là lỡ sinh 
      
     giác duyên ngắn chén tài hoa tự tôi 
      
     hơn người ngồi dai ,  
     nhìn nàng ra nặng nào thôi đền tình 
      
     khuyển thơ ngây thơ ngây thơ ngây 
      
     thoạt gánh như nung gan



```python
sent = model.generate(torch.tensor([encode("mây trôi".split(" "))]).to(device), max_new_token=150)
print(decode(sent[0].tolist()))
```

    mây trôi bèo dạt đã đành ,  
     lại càng đứng lặng nhìn được điều 
      
     giọng kiều rền rĩ trướng loan ,  
     nhà huyên chợt sinh? 
     bàn ngần ngọn vì sự bất xưa 
     sá đá tài trong ,  
     giở đồ chuông khánh các lạ đời 
      
     tâm thu ,  
     khi vào điều của dây thường ,  
     lập năm bể mới hay không? 
     sâm thương bằng tiện ở bắc mặn 
      
     ghế quanh những khan giọng tình ,  
     dập dìu bỗng khuôn đó cầm! 
     thời làng đình nghe hiếu tâm ,  
     ba bề vẹn một nhà thì nên bay bất kỳ ,  
     xôn xao ngoài hoặc có xuân đường vân mới giãi chiều 
      
     thưa rằng: sắc lâm truy ,  
     sắc đành đã


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

    Input In [19], in <cell line: 1>()
    ----> 1 this_notebook


    NameError: name 'this_notebook' is not defined



```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```
