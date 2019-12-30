# TensorFlow

## 1. TensorFlow란?

> TensorFlow is an open source library for numerical computation and large-scale machine learning

- 구글 브레인 팀이 2015년에 오픈 소스로 공개
- Python API 제공 & 실제 코드가 실행되는 환경은 C/C++
- GPU 활용 가능 (GPU에서 일반 연산을 수행하게 하는 CUDA 확장기능을 사용)



## 2. TensorFlow basic

TensorFlow는 1) Building a TensorFlow Graph, 2) Executing the TensorFlow Graph 두 단계를 통해 계산을 수행함



### Two steps to perform a computation in TF

**1) Building a TensorFlow Graph**

​	Thensor들 사이의 연산 관계를 계산 그래프로 정의&선언  == **function definition**

**2) Executing the Tensor Flow Graph**

​	계산 그래프에 정의된 연산(계산)을 tf.Session을 통해 실제로 실행 == **function execution**



**tensorflow 1) Building a TensorFow Graph **

``` python
import tensorflow as tf
import numpy as np
import pandas as pd
```

```python
a = tf.add(3, 5)
print(a)
```

`Tensor("Add:0", shape=(), dtype=int32)`

3+5 = '8' 이 결과값으로 도출되지 않는다. a는 실수형 변수가 아닌 tensor이기 때문



**tensorflow 2) Executing the TensorFlow Graph**

a의 값을 도출하기 위해서는 tf.Session() 함수를 사용해야 한다.

```python
a = tf.add(3, 5)
ssess = tf.Session()
print(sess.run(a))
```

`8`

한번 생성한 tf.Session() 함수는 반드시 닫아주어야한다.

```python
sess.close()
```

아래의 경우에는 tf.Session() 함수를 반드시 닫아줄 필요가 없다.

```python
a = tf.add(3, 5)
with tf.Session() as sess:
	print(sess.run(a))
```

`8`



**New Example**

```python
x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)

with tf.Session() as sess:
	op3 = sess.run(op3)
    #op1, op2 tf.Session() 함수로 안 써도 op3와 직결되기 때문에 값이 실행됨
	print(op3)
```

`7776`

```python
x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
useless = tf.multiply(x, op1)   # 사용되지 않을 부분을 빼줌으로써 save computation
op3 = tf.pow(op2, op1)

with tf.Session() as sess:
	op3, useless = sess.run([op3, useless]) 
    #여러 개의 Tensor를 동시에 실행하고 싶을 경우에는 list로 전달
	print(op3)
```

