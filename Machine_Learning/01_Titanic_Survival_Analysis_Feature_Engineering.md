# 3. Titanic Survival Analysis

> Machine Learning을 이용한 Titanic Survival Analysis



###  1) 데이터 전처리 과정

**함수 import 하기**

```python
import mglearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import cluster
from sklearn import metrics
from sklearn import ensemble
```



**데이터 가져오기**

```python
data_df = pd.read_csv('titanic.csv')
y_data = data_df[['Survived']]
del data_df['Survived']
x_data = data_df.copy()
```



**데이터 내 null 값 확인하기**

```python
titanic_df.isnull().sum()
```

> ```python
> PassengerId      0
> Survived         0
> Pclass           0
> Name             0
> Sex              0
> Age            177       #age, cabin, embarked 열에 null값이 있음을 알 수 있음
> SibSp            0
> Parch            0
> Ticket           0
> Fare             0
> Cabin          687
> Embarked         2
> dtype: int64
> ```



**결측치 시각화해서 보여주기**

```python
msno.matrix(titanic_df, figsize=(12,5))
```

![titanic_결측치_image](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20200107185504181.png)



**Sex 분류하기**

> male: 0,  female: 1

```python
sex_mapping = {"male": 0, "female":1}
x_data['Sex'] = x_data['Sex'].map(sex_mapping)
```



**Fare 분류하기**

> fare를 분포로 표현하여 5개로 분류하여 0~4까지의 값으로 나타냄

```python
x_data.loc[x_data['Fare'] <= 7.854, 'Fare'] = 0
x_data.loc[(x_data['Fare'] > 7.854) & (x_data['Fare'] <= 10.5), 'Fare'] = 1
x_data.loc[(x_data['Fare'] > 10.5) & (x_data['Fare'] <= 21.679), 'Fare']   = 2
x_data.loc[(x_data['Fare'] > 21.679) & (x_data['Fare'] <= 39.688), 'Fare']   = 3
x_data.loc[ x_data['Fare'] > 39.688, 'Fare'] = 4
x_data['Fare'] = x_data['Fare'].astype(int)
```



**Age  분류하기**

Age 결측치 채우기

> 총 Age 열의 중위값으로 빈 값 채우기

```python
mean_age = x_data['Age'].median(skipna=True)
x_data['Age'] = x_data['Age'].fillna(mean_age)
```

Age 4개로 그룹화하여 0~3까지의 값으로 나타냄

```python
x_data.loc[ x_data['Age'] <=20, 'Age']=0,
x_data.loc[(x_data['Age'] > 20) & (x_data['Age'] <=40), 'Age'] = 1,
x_data.loc[(x_data['Age'] > 40) & (x_data['Age'] <=60), 'Age'] = 2,
x_data.loc[(x_data['Age'] > 60), 'Age'] = 3
```



**Embarked 분류하기**

Embarked 결측히 채우기

>Embarked의 결측치는 2개. Embarked 열에 가장 많이 있는 'S' 정박지로 채움
>
>S: 0, C: 1, Q: 2로 mapping

```python
embarked_mapping = {"S" : 0, "C" : 1, "Q" : 2}
x_data['Embarked'] = x_data['Embarked'].map(embarked_mapping)
x_data['Embarked'] = x_data['Embarked'].fillna(0)
```



**Cabin 분류하기** 

> Cabin 열의 데이터 경우 결측치가 매우 많고, Cabin 내에 값이 여러 개 등록된 리스트 형태임
>
> Cabin 데이터의 형태는 '영문'+'숫자' 형태로, 영문의 수를 count하여 cabin 값을 정규화함

```python
x_data['Cabin'] = x_data['Cabin'].fillna(0)
for i in range(len(x_data)):
    # Cabin 데이터 전처리 방수별로 숫자 카운팅
    count = 0
    if type(x_data.loc[i, ['Cabin']][0]) == type('str') :
        for j in list(x_data.loc[i, ['Cabin']][0]) :
            try :
                int(j)
            except:
                count = count + 1
                if j == ' ' :
                    count = count - 1
        x_data.loc[i, ['Cabin']] = count
```



**Name 분류하기**

Name 내에 Mr, Miss, Mrs, Master 만 추출하여 Title 열 만들기

```python
x_data['Title'] = x_data['Name'].str.extract('([A-za-z]+)\.', expand=False)
```

이름에서 Mr, Miss, Mrs, Master를 각각 0, 1, 2, 3으로 mapping. 나머지는 4로 지정.

```python
title_mapping = {"Mr" : 0, "Miss" : 1, "Mrs" : 2, "Master" : 3, "Dr" : 4, 
                 "Rev" : 4, "Mlle" : 4, "Col" : 4, "Major" : 4, "Sir" : 4, 
                 "Jonkheer" : 4, "Countess" : 4, "Mme" : 4, "Capt" : 4, "Lady" : 4,
                 "Ms" : 4, "Don" : 4}
x_data["Title"] = x_data["Title"].map(title_mapping)
```



**Parch & Sibsp (가구 단위로 합치고 Categorizing, 단독가구를 새로운 feature로 생성)**

```python
# parch, sibsp의 경우 0 or 1이 많음, family 전체 사이즈로 합쳐보기 (본인 포함 탑승한 가족 구성원 수)

x_data["FamilySize"] = x_data["SibSp"] + x_data["Parch"] +1
x_data[['FamilySize','Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived',ascending=False)
```

```python
# 1인 가구인 경우는 1, 아닌 경우는 0으로 'isAlone' column을 추가

titanic_df['isAlone'] = 0
titanic_df.loc[titanic_df['FamilySize'] == 1, 'isAlone'] = 1
```

```python
# 기존의 sibsp & parch 열을 지우고, Fare 의 구간과 유사하게 0~4로 구간화

del titanic_df['SibSp']
del titanic_df['Parch']

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
titanic_df['FamilySize'] = titanic_df['FamilySize'].map(family_mapping)
titanic_df.head(3)
```



**불필요한 열 삭제하기**

```python
del x_data["Name"]   # Title 열로 정규화함
del x_data['Ticket']  # 비정형적인 데이터
del x_data['PassengerId']  # 탑승자의 index
```



**숫자열 데이터타입 float로 지정**

```python
x_data['Age'] = x_data['Age'].astype('float32')
x_data['Fare'] = x_data['Fare'].astype('float32')
x_data['Embarked'] = x_data['Embarked'].astype('float32')
```



**데이터 Scaling**

```python
# 데이터 0~1사이로 정규화
sc = StandardScaler()
sc.fit(x_train)
X_train_scaled = sc.transform(x_train)
X_test_scaled = sc.transform(x_test)

df = pd.DataFrame(X_train_scaled)
df_y = pd.DataFrame(X_test_scaled)
df.head()
```

