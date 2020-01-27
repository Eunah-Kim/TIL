# 2. Tensorflow basic

## 2-3. MNIST_Dropout

> Dropout 이란 네트웍의 일부를 생략하는 것
>
> 랜덤하게 Neuron을 꺼뜨려 학습을 방해함으로써 모델의 학습이 Training data에 편향되는 것을 막아주는 것이 핵심
>
> 동일한 데이터에 대해 매번 다른 모델을 학습시키는 것과 마찬가지의 효과를 발생시켜 일종의 **Model ensemble 효과**를 얻을 수 있음

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

```python
import os
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
```



#### 1. Prepare the data

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
```



#### 2. Build the model

```python
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Dropout을 적용하며 layer마다 살려줄 node의 비율을 지정합니다.
# 이 때에도 placeholder를 사용
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob) # (Dropout을 적용할 layer, 살릴 비율)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob) # Dropout을 적용할 layer & 살릴 비율

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)
```



#### 3. Set the criterion

```python
cost = tf.losses.softmax_cross_entropy(Y, model)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
```



#### 4. Train the model

```python
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#batch 단위를 100으로 설정
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
print(total_batch)

#15번 실행
for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.8}) # 살릴 비율 지정, node 중 80%만 유지하고 20%를 train 시마다 off
        total_cost += cost_val

    print('Epoch: {}'.format(epoch+1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('Learning process is completed!')
```

```
Epoch: 1 Avg. cost = 0.425
Epoch: 2 Avg. cost = 0.160
Epoch: 3 Avg. cost = 0.112
Epoch: 4 Avg. cost = 0.087
Epoch: 5 Avg. cost = 0.071
Epoch: 6 Avg. cost = 0.060
Epoch: 7 Avg. cost = 0.049
Epoch: 8 Avg. cost = 0.045
Epoch: 9 Avg. cost = 0.040
Epoch: 10 Avg. cost = 0.037
Epoch: 11 Avg. cost = 0.032
Epoch: 12 Avg. cost = 0.029
Epoch: 13 Avg. cost = 0.030
Epoch: 14 Avg. cost = 0.026
Epoch: 15 Avg. cost = 0.025
Learning process is completed!
```



#### 5. Test the model

```python
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('정확도:', sess.run(accuracy,
                        feed_dict={X: mnist.test.images,
                                   Y: mnist.test.labels,
                                   keep_prob: 1})) # 살릴 비율 지정, 정확도를 측정하는 Test 단계에서는 전체 Node를 살려줘야 함. -> keep_prob = 1
```

`정확도 : 0.9827`

