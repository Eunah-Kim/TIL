# 2. TensorFlow Basic

## 2-1. Linear Regression

#### 1. Prepare the data

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skleearn import datasets
```

```python
x_data = datasets.load_boston().data[:, 12]
y_data = datasets.load_boston().target
df = pd.DataFrame([x_data, y_data]).transpose()
df.head()
```

#### 2. Build the model

w, b는 파라미터, y_predicted는 예측된 모델

```python
# tf.Variable(초기화 방법)
w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

y_predicted = w * x_data + b #model
```

#### 3. Set the Crierion : Cost function & Gradient Descent method

>  reduce_mean : 평균   tf.square : 제곱함수

y_data와 y_predicted 의 오차를 제곱한 후 평균계산 >> **MES**

GradientDescentOptimizer(Learning rate) Learning rate = **0.001**로 설정 (hyper parameter)

```python
# Cost function
loss = tf.reduce_mean(tf.square(y_predicted - y_data))
# Gradient Descent method
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
```

#### 4. Train the model

```python
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for step in range(10000):
		sess.run(train)
		if step % 1000 == 0:
			print('Step {}: w {} b {}' .format(step, sess.run(w), sess.run(b)))
			print('loss {}'.format(sess.run(loss)))
			print()
			
	w_out, b_out = sess.run([w, b])
```

#### 5. Visualize the result

```python
plt.figure(figsize = (10, 10))
plt.plot(x_data, y_data, 'bo', label = 'Real data')
plt.plot(x_data, x_data * w_out + b_out, 'ro', label = 'Prediction')
plt.legend()
plt.show()
```

![image-20191230202229221](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230202229221.png)





