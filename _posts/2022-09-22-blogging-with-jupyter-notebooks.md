# Blogging-with-Jupyter-Notebooks

I write this article and convert it to markdown file using nbconvert.


```python
import os
from IPython.core.display import Javascript
import time
```


```python
%%js
IPython.notebook.kernel.execute('this_notebook = "' + IPython.notebook.notebook_name + '"')
```


    <IPython.core.display.Javascript object>



```python
this_notebook
```




    '2022-09-22-blogging-with-jupyter-notebooks.ipynb'




```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```

    [NbConvertApp] Converting notebook 2022-09-22-blogging-with-jupyter-notebooks.ipynb to markdown
    [NbConvertApp] Writing 761 bytes to ../_posts/2022-09-22-blogging-with-jupyter-notebooks.md

