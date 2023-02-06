# review-data-size


```python
import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
input_file_path = 'input.txt'
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
```


```python
print(data[:200])
```

    First Citizen:
    Before we proceed any further, hear me speak.
    
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you



```python
n_word = len(data.replace("\n", " ").split(" "))
```


```python
print(f"{n} chars and {n_word} words")
```

    1115394 chars and 209893 words



```python
# shakespeare train.bin has 301,966 tokens
# openwebtext train has ~9B tokens (9,035,582,198)
```


```python

```

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




    '2023-02-06-tiktoken.ipynb'




```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```

    [NbConvertApp] Converting notebook 2023-02-06-tiktoken.ipynb to markdown
    [NbConvertApp] Writing 3522 bytes to ../_posts/2023-02-06-tiktoken.md

