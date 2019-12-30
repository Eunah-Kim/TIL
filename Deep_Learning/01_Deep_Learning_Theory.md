# 딥러닝(Deep Learning) 이론

## 1. 머신러닝 vs 딥러닝

- **전통적인 머신러닝** :  사림이 직접 데이터의 중요한 부분들을 찾아 Feature로 정해주어야 함
- **딥러닝** :  신경망 내부에서 자체적으로 데이터의 중요한 Feature를 찾거나 구성함



**아이의 손을 찍은 X-ray 사진을 보고, 아이의 나이를 알아맞추시오.**

- **전통적인 머신러닝**

1. 길이를 재기 위한 뼈들의 위치를 찾는다.
2. 뼈의 길이에 따라 나이를 예측하도록 모델을 훈력

- **딥러닝 (End-to-end learning)**

1. (아주 많은) X-ray 이미지를 모델에게 준다.
2. 모델이 이미지를 보고 아이의 나이를 예측하도록 학습한다.
3. Pros
   - Let's the data speak. (우리가 알아채지 못한 관계를 찾아낼 수 있음)
   - Less hand-designing of components needed
4. Cons
   - Needs large amounts of labeled data
   - Excludes potentially useful hand-made components



### 2. 딥러닝 핵심 개념 이해

**Deep Learning**

 : Depp Neual Network(DNN)를 활용하여 학습하는 머신러닝 알고리즘 (Deep = Hidden layer가 2개 이상)

#### (Single-Layer) Perceptron (단층 퍼셉트론)

![image-20191230101028756](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230101028756.png)

- 뉴런을 본따 만든 알고리즘 하나의 단위

- 2가지 연산을 적용하여 출력값 계산

  **넘겨져 온 데이터와 theta들의 Linear combination** +  **Activation function**

  1) 넘겨져 온 데이터와 theta들의 **Linear combination** 을 계산 (합&곱)

    이전 Layer 혹은 외부로부터 넘겨져 온 데이터와 각 theta 사이의 Linear Combination

  ![image-20191230100921348](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230100921348.png)

  2) 선형결합의 결과에 **Activation function(활성화 함수)**을 적용

    Linear combination의 결과값이 Non-linear Function을 거치게 하여 최종 출력값을 계산

  ![image-20191230101018836](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230101018836.png)

### ★★ 가장 적합한 Theta들의 Set을 찾는 것이 목표!



#### Activation Functions

 : 이전 레이어의 모든 입력에 대한 가중합을 받아 출력값을 생성하여 다음 레이어로 전달하는 비선형함수    ![image-20191230120110273](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230120110273.png)

#### Multi-Layer Perceptron (MLP, 다층 퍼셉트론)

 : 복수의 Perceptron을 연결한 구조



#### (Artificial) Neural Network (인공 신경망)

 : Perceptron 을 모은 Layer를 깊이 방향으로 쌓아나가면서 복잡한 모델을 만들어내어 보다 더 어려운 문제를 풀어낼 수 있음

> MLP 역시도 일종의 ANN
>
> Hidden layer == Leamable kernels

- Input Layer : 외부로부터 데이터를 입력받는 신경망 입구의 layer

- Hidden Layer : Inpuy layer와 Output layer 사이의 모든 layer

- Output Layer : 모델의 최종 연산 결과를 내보내는 신경망 출구의 layer

  결과값을 그대로 받아 **Regression**,

  **Sigmoid**를 거쳐 **Binary Classification**

  **Softmax**를 거쳐 **K-Class Classification**

![image-20191230130810819](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230130810819.png)



#### Forward Propagation (Feedforward Neural Network)

 : Input Layer에서 시작하여 순반향으로 계산해 나아가며 Output Layer까지 값을 전파해가는 신경망 (노드들 간의 연결이 cycle/loop를 이루지 않음)

![image-20191230130742880](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230130742880.png)

- **# of Layers, # of Neurons, Activation func** 등은 **Hyper Parameter** 의 영역

1. Hidden Layer의 뉴런 수와 Hidden Layer의 개수는 신경망 설계자의 직관과 경험에 의존

2. Hidden Layer의 뉴런 수가 너무 많으면 Overfitting이 발생

   너무 적으면 데이터를 충분히 표현 못함(Underfitting)

Feedforward 신경망의 학습

 : 원하는 결과를 얻기 위해 뉴런 사이의 적당한 가중치 세타들을 알아내는 것.

  -> Model의 Output과 실제 정답의 차이를 바탕으로 Cost function을 구성하고, Cost를 낮추도록 Gradient Descent를 적용하여 최족의 가중치 Theta를 찾아감



#### Back Propagation Algorithm (오차 역전파 알고리즘) 

####     == 신경망의 효율적인 학습방법

 : 학습된 출력 값과 실제 값의 차이인 오차를 계산하여 Feedforward 반대인 역방향으로 전파(Propagation)하는 알고리즘

- Multi-Layer Perceptron으로 XOR 문제 해결!
- 그러나 Layer가 복잡해질수록 연산이 복잡해져서 현실적으로 매우 비효율적
- 이러한 문제를 해결하기 위해 Back propagation 알고리즘이 도입됨
- Forward 방향으로 한 번 연산을 한 다음 그 결과값(Cost)를 역방향(backward)으로 전달해가면서 Parameter를 Update!

**모델이 틀린 정도를 역방향으로 전달하며 '미분'하고 곱하고 더하는 것을 반복하여 Parameter를 갱신한다.(Reverse Feed-forward)**



### 3. 딥러닝 모델 최적화 이론

#### Neural Network Optimization

> Gradient descent를 활용한 신경망의 학습과정

1) 모든 **Parameter theta를 초기화**하고

2) **Cost Function 상의 가장 낮은 지점을 향해** 나아가며

3) 선택한 **Gradient descent method**를 적용해 **theta를 계속 update**



#### Neural Network Optimization 1) Weight Initialization

> Gradient descent 를 적용하기 위한 첫 단계는 모든 Parameter Theta를 초기화하는 것
>
> 초기화 시점의 작은 차이가 학습의 결과를 뒤바꿀 수 있으므로 보다 나은 초기화 방식을 모색하게 됨
>
> Perceptron의 Linear combination 결과 값(Activation function으로의 입력 값)이 너무 커지거나 작아지지 않게 만들어주려는 것이 핵심
>
> 발전된 초기화 방법들을 활용해 Vanishing gradient 혹은 Exploding gradient 문제를 줄일 수 있음

**1) Use Xavier Initialization**

- 활성화 함수로 Sigmoid 함수나 tanh 함수를 사용할 때 적용
- 다수의 딥러닝 라이브러리들에 Defalt 로 적용되어 있음
- 표준편차가 ![image-20191230132946073](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230132946073.png)인 정규분포를 따르도록 가중치 초기화

**2) He Initialization**

- 활성화 함수가 ReLU 함수일 때 적용
- 표준편차가 ![image-20191230133039073](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230133039073.png)인 정규분포를 따르도록 가중치 초기화

![image-20191230133059592](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230133059592.png)



#### Neural Network Optimization 2) Weight regularization

> 기존의 Gradient Descent 계산 시 y축에 위치해 있던 Cost function은 Training data에 대해 모델이 발생시키는 Error 값의 지표
>
> Training data만 고려된 이러한 Cost Function을 기준으로 하여 Gradient Descent 를 적용하면 Overfitting에 빠질 수 있음



모델이 복잡해질수록 모델 속에 숨어있는 **theta들은 그 개수가 많아지고 절대값이 커지는 경향**이 있음(숨겨져 있던 theta들이 값을 갖게 됨)

모델이 복잡해질수록 그 값이 커지는 theta에 대한 함수를 기존의 **Cost function에 더하여 Trade-off** 관계 속에서 최적값을 찾을 수 있음

![image-20191230165724827](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230165724827.png)

**L1 regularization (L1 정규화)** ![image-20191230165858194](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230165858194.png)

- 가중치의 절대값의 합에 비례하여 가중치에 페널티를 준다.
- 관련성이 없거나 매우 낮은 특성의 가중치를 정확히 0으로 유도하여, 모델에서 해당 특성을 배제하는 데 도움이 됨 == Feature selection 효과

**L2 regularization(L2 정규화) **![image-20191230165940959](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230165940959.png)

- 가중치의 제곱의 합에 비례하여 가중치에 페널티를 준다.
- 큰 값을 가진 가중치를 더욱 제약하는 효과가 있음

**Regularization Rate** : 정규화율(Lambda) ![image-20191230170024964](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230170024964.png)

- 스칼라 값
- 정규화 함수의 상대적 중요도를 지정
- 정규화율을 높이면 과적합이 감소하지만 모델의 정확성이 떨어질 수 있음
- theta의 수가 아주 많은 신경망은 정규화율을 아주 작게 주기도 함



#### Neural Network Optimization 3) Advanced gradient decent algorithms

**(Full-Batch) Gradient Descent**

>  **모든 Training data**에 대해 Cost를 구하고 Cost function 값을 구한 다음 이를 기반으로 Gradient descent를 적용

- Training data가 많으면 Cost function 등의 계산에 필요한 **연산의 양이 많아진다**. (학습이 오래 걸림)
- Weight initialization의 결과에 따라 Global  minimum이 아닌 local minimum으로 수렴할 수 있다.

**Stochastic Gradient Descent (SGD, 확률적 경사 하강법)**

> **하나의 Training data(Batch size = 1)**마다 Cost를 계산하고 바로 Gradient descent를 적용하여 weight를 빠르게 update

- 한 개의 Training data마다 매번 weight 를 갱신하기 때문에 **신경망의 성능이 들쑥날쑥 변함**(Cost 값이 안정적으로 줄어들지 않음)
- 최적의 Learning rate를 구하기 위해 일일이 튜닝하고 **수렴조건(early-Stop)을 조정**해야 함

**Mini-Batch Stochastic Gradient Descent**

> Training data 에서 **일정한 크기(== Batch size)의 데이터를 선택**하여 Cost function 계산 및 Gradient descent 적용

- 앞선 두 가지 Gradient descent 기법의 단점을 보완하고 장점을 취함.
- 설계자의 의도에 따라 속도와 안정성을 동시에 관리할 수 있으며, GPU 기반의 효율적인 병렬 연산이 가능해진다.



#### Avoid overfitting - 1) Dropout

> Training을 진행할 때 매 Batch 마다 **Layer 단위로 일정 비율 만큼의 Neuron을 꺼뜨리는** 방식으로 적용
>
> **Test / Inference 단계**에는 Dropout을 걷어내어 **전체 Neuron이 살아있는 채로** Inference를 진행해야 함

![image-20191230171115998](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230171115998.png)

- 랜덤하게 Neuron을 꺼뜨려 학습을 방해함으로써 모델의 학습이 Training data에 편향되는 것을 막아주는 것이 핵심
- 동일한 데이터에 대해 매번 다른 모델을 학습시키는 것과 마찬가지의 효과를 발생시켜 일종의 **Model ensemble 효과**를 얻을 수 있음
- 전반적으로 Overfittng을 줄여주므로 Test data에 대한 에러를 더욱 낮출 수 있게 해줌
- 가중치 값이 큰 특정 Neuron의 영향력이 커져 다른 Neuron들의 학습 속도에 문제를 발생시키는 Co-adaptation을 회피할 수 있게 함



#### Avoiding overfitting - 2) Batch Normalization

- Input data에 대해 Standardization**과 같은 Normalization을 적용하면 전반적으로 model의 성능이 높아짐  ![image-20191230171540606](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20191230171540606.png)

> 데이터 내 Column들의 Scale에 model이 너무 민감해지는 것을 막아주기 때문
>
> 신경망의 경우 **Normalization**이 제대로 적용되지 않으면 최적의 cost 지점으로 가는 길을 빠르게 찾지 못함

- 이러한 Normalization을 Input data 뿐만 아니라 신경망 내부의 중간에 있는 Hidden layer로의 input 에도 적용해주는 것 : BN
- **Actovation function을 적용하기 전에 Batch normalization을 먼저 적용**



**Process of Batch  Normalization**

1) 각 Hidden layer 로의 **Input data**에 대해 평균이 0, 분산이 1이 되도록 **Normalization**을 진행

2) Hidden layer의 출력값(Output)이 비선형을 유지할 수 있도록 Normalization의 결과에                  	**Scaling&Shifting** 적용

3) Scaling&Shifting을 적용한 결과를 **Activation function에게 전달** 후 Hidden layer의 최종 output 	계산



**Batch Normalization의 장점**

- **학습 속도 & 학습 결과**가 개선됨 (High learning rate 적용 가능)
- **가중치 초기값**에 크게 의존하지 않음 (매 layer마다 정규화를 진행하므로 초기화의 중요도 감소)
- **Overfitting을 억제** (Dropout, L1/L2 regularization 등의 필요성 감소)
- 핵심은 **학습속도의 향상** (Overfitting을 줄여주는 Regularization effect는 부수적 효과)