# Beautiful functions in python

### 1. partial


```python
from functools import partial

# when we have a general function like
def multiply(x, y):
    return x * y

# You can reuse a function and create a new simpler function and name it with one line of code
# create a new double function that reuses the function multiply
double = partial(multiply, y=2)
double(4), double(5)
```




    (8, 10)




```python
# This is another way to do it
def double(x): return multiply(x, 2)

double(4), double(5)
```




    (8, 10)



### 2. map


```python
# map(fun, iter)
# fun : It is a function to which map passes each element of given iterable.
# iter : It is a iterable which is to be mapped.
```


```python
xs = [3, 4, 7]
# double each element in xs
double_xs = map(double, xs)
list(double_xs)
```




    [6, 8, 14]




```python
# another way
double_xs = [double(x) for x in xs]
double_xs
```




    [2, 6, 8, 14]



### 3. filter


```python
# filter(fun, iter)
# fun : It is a function which runs for each item in the iterable
# iter : It is a iterable which is to be mapped.
```


```python
xs = [1, 3, 4, 7]

def less_than_three(x): return x < 3

xs_less_than_three = filter(less_than_three, xs)
list(xs_less_than_three)
```




    [1]




```python
# another way
xs_less_than_three = [x for x in xs if less_than_three(x)]
xs_less_than_three
```




    [1]



### 4. lambda


```python
# lambda arguments: expression

# define a function in a short way
double = lambda x: 2 * x
double(4), double(5)
```




    (8, 10)



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




    '2022-09-22-blogging-with-jupyter-notebooks.ipynb'




```python
!jupyter nbconvert --to markdown {this_notebook} --output-dir=../_posts
```

    [NbConvertApp] Converting notebook 2022-09-22-blogging-with-jupyter-notebooks.ipynb to markdown
    [NbConvertApp] Writing 725 bytes to ../_posts/2022-09-22-blogging-with-jupyter-notebooks.md

