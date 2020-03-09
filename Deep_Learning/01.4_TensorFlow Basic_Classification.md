# 2. TensorFlow Basic

## 2-2. Classification (MNIST, 2Layers baseline)

### MNIST Introduction

- MNIST (Mixed National Institute of Standards and Technology database)
- 손글씨 숫자(0~9) 이미지 데이터
- 각 이미지는 가로와 세로가 각각 28px, 흑백 이미지로 만들어져 있음
- Training data == 55,000장, Validation data ==5,000장, Test data = 10,000장



#### 1. Prepare the data

```python
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
```

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
```

```python
type(mnist.train.images)
```

`numpy.ndarray`

```python
mnist.train.images.shape
```

`(55000, 784)`

```python
mnist.train.labels.shape
```

`(55000, 10)`



#### 2. Build the model

```python
# 데이터가 흘러들어올 접시(placeholder) 만들기 
X = tf.placeholder(tf.float32, [None, 784]) 
# [# of batch data, # of features(columns) == 총 784개의 열]
Y = tf.placeholder(tf.float32, [None, 10]) # 0~9 == 총 10개의 열

# 모든 Parameter Theta는 Variable로 선언

#hidden layer 1
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
# 256 : 다음 hidden layer의 노드의 수 ; 임의로 정하는 수
L1 = tf.nn.relu(tf.matmul(X, W1))

#hidden layer 2
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

#output layer
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
# 마지막 weight의 열수는 정답 열의 수와 맞춰준다
model = tf.matmul(L2, W3) # 마지막 층도 행렬곱까지만 진행
```



#### 3. Set the criterion

````python
# cost = tf.losses.mean_squared_error(Y, model) # for Regression

cost = tf.losses.softmax_cross_entropy(Y, model) # for Classification, "cross-entropy" after "softmax"
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost) # Select optimizer & connect with cost function (recommended start : "Adam")
````



#### 4. Train the model

```python
init = tf.global_variables_initializer() # Initialize all global variables (Parameter Theta)
sess = tf.Session()
sess.run(init)

# Gradient descent를 적용하기 전까지 한번에 밀어넣는 데이터의 수 지정 (Batch size == 하나의 데이터 덩어리 내 데이터 수)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
print(total_batch)


for epoch in range(15):   # 전체 데이터(55000개)를 15번 돌린다
    total_cost = 0

    for i in range(total_batch): # iterate over # of batches # 550번 넣는다 = 100개(batch_size)씩 들어간다
        batch_xs, batch_ys = mnist.train.next_batch(100)  #랜덤하게 100개를 뽑는다

        _, cost_val = sess.run([optimizer, cost], 
                               feed_dict={X: batch_xs, Y: batch_ys}) # 먹여줄(feed) 딕셔너리(dict)  # placeholder X, Y 안에 원본데이터 넣기
        total_cost += cost_val

    test_cost = sess.run([cost], feed_dict={X: mnist.test.images, Y: mnist.test.labels}) # current test error
    
    print('Epoch: {}'.format(epoch+1), 
          '|| Avg. Training cost = {:.3f}'.format(total_cost / total_batch),
          '|| Current Test cost = {:.3f}'.format(test_cost[0]))

print('Learning process is completed!')
```

```
Epoch: 1 || Avg. Training cost = 0.008 || Current Test cost = 0.098
Epoch: 2 || Avg. Training cost = 0.008 || Current Test cost = 0.105
Epoch: 3 || Avg. Training cost = 0.012 || Current Test cost = 0.094
Epoch: 4 || Avg. Training cost = 0.009 || Current Test cost = 0.091
Epoch: 5 || Avg. Training cost = 0.004 || Current Test cost = 0.094
Epoch: 6 || Avg. Training cost = 0.011 || Current Test cost = 0.113
Epoch: 7 || Avg. Training cost = 0.010 || Current Test cost = 0.122
Epoch: 8 || Avg. Training cost = 0.005 || Current Test cost = 0.110
Epoch: 9 || Avg. Training cost = 0.004 || Current Test cost = 0.115
Epoch: 10 || Avg. Training cost = 0.009 || Current Test cost = 0.101
Epoch: 11 || Avg. Training cost = 0.007 || Current Test cost = 0.118
Epoch: 12 || Avg. Training cost = 0.005 || Current Test cost = 0.142
Epoch: 13 || Avg. Training cost = 0.007 || Current Test cost = 0.143
Epoch: 14 || Avg. Training cost = 0.008 || Current Test cost = 0.115
Epoch: 15 || Avg. Training cost = 0.006 || Current Test cost = 0.133

Learning process is completed!
```



#### 5. Test the model

```python
# tf.argmax([0.1 0 0 0.7 0 0.2 0 0 0 0]) -> 3 (가장 큰 값의 index를 return)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)) # model : 예측값, Y : 실제 정답
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # 자료형 변환(type-"cast") 후, 차원을 줄이면서(reduce) 평균(mean) 계산

print('정확도 :', sess.run(accuracy,
                        feed_dict={X: mnist.test.images,
                                   Y: mnist.test.labels}))
```

`정확도 : 0.9804`

```python
# 모델 예측 결과값

predicted_labels = sess.run(tf.argmax(model, 1), feed_dict={X: mnist.test.images, Y: mnist.test.labels})
print(list(predicted_labels)[:10])
```

`[7, 2, 1, 0, 4, 1, 4, 9, 5, 9]`

```python
# 실제 정답 

import numpy as np
print(np.argmax(mnist.test.labels, 1)[:10])
```

`[7 2 1 0 4 1 4 9 5 9]`

