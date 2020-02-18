# 3. Titanic Survival Analysis

> Machine Learning을 이용한 Titanic Survival Analysis



###  2) Train - Test split

```python
column_namelist = list(x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
```



### 3) Model Fitting & Create model instance variable

**함수 import하기**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
```



**KNN**

```python
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score

model_knn = neighbors.KNeighborsClassifier(6)
model_knn.fit(X_train_scaled, y_train)
y_predict = model_knn.predict(X_train_scaled)
print('KNN 모델 적용')
print('Accuracy on Training set: ',end='')
print(accuracy_score(y_predict, y_train))
y_predict = model_knn.predict(x_test)
print('Accuracy on Test set: ',end='')
print(accuracy_score(y_predict, y_test))
```

KNN 모델 적용
Accuracy on Training set: 0.841091492776886
Accuracy on Test set: 0.7425373134328358



**Linear SVM**

```python
# 가장 기본적인 서포트 벡터 머신
linear_svm = LinearSVC().fit(X_train_scaled, y_train)
print('linear SVC')
print("linear Accuracy on Training set: {:.3f}".format(linear_svm.score(x_train, y_train)))
print("linear Accuracy on Test set: {:.3f}".format(linear_svm.score(x_test, y_test)))
```

linear SVC
linear Accuracy on Training set: 0.799
linear Accuracy on Test set: 0.776



**Kernelized SVM**

```python
# Kernelized Support Vector Machine 
# Kernal = RBF
# Hyperparameter를 모두 default로 했을 때
svc = SVC()
print(svc)
svc.fit(X_train_scaled, y_train)
print('디폴트 값')
print("Accuracy on Training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on Test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
```

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
디폴트 값
Accuracy on Training set: 0.851
Accuracy on Test set: 0.817



**Hyper parameter C=100, gamma = 1.0**

```python
# Hyperparameter C=100
svc1 = SVC(C=100,gamma=1.0)
svc1.fit(X_train_scaled, y_train)
print('SVC setting : Kernal=rbf, C=100 , gamma=auto')
print("Accuracy on Training set: {:.3f}".format(svc1.score(X_train_scaled, y_train)))
print("Accuracy on Test set: {:.3f}".format(svc1.score(X_test_scaled, y_test)))
```

SVC setting : Kernal=rbf, C=100 , gamma=auto
Accuracy on Training set: 0.897
Accuracy on Test set: 0.813





**Hyper parameter  kernel= poly, C=1000, gamma= 0.1**

```python
# kernel = poly, C = 1000, gamma = 0.1을 적용한 SVC
svc2 = SVC(kernel='poly', C=1000, gamma=0.1).fit(X_train_scaled, y_train)
print('SVC setting : Kernal=Poly, C=1000, gamma=0.1')
print("Accuracy on Training set: {:.3f}".format(svc2.score(X_train_scaled, y_train)))
print("Accuracy on Test set: {:.3f}".format(svc2.score(X_test_scaled, y_test)))
```

SVC setting : Kernal=Poly, C=1000, gamma=0.1
Accuracy on Training set: 0.894
Accuracy on Test set: 0.795



**Hyper parameter  kernel= sigmoid, C=1000, gamma= 0.1**

```python
svc3 = SVC(kernel='sigmoid', C=1000, gamma=0.1).fit(X_train_scaled, y_train)
print('SVC setting : Kernal=sigmoid, C=1000, gamma=0.1')
print("Accuracy on Training set: {:.3f}".format(svc3.score(X_train_scaled, y_train)))
print("Accuracy on Test set: {:.3f}".format(svc3.score(X_test_scaled, y_test)))
```

SVC setting : Kernal=sigmoid, C=1000, gamma=0.1
Accuracy on Training set: 0.674
Accuracy on Test set: 0.683



**GridSearchCV for SVM**

>최적의 Hyper parameter 찾기

```python
embarked_mapping = {"S" : 0, "C" : 1, "Q" : 2}
x_data['Embarked'] = x_data['Embarked'].map(embarked_mapping)
x_data['Embarked'] = x_data['Emba# 최적의 Hyperparameter를 찾기 위해 GridSearchCV 적용!
from sklearn.model_selection import GridSearchCV
param_grid = {'C' : [0.1, 1, 10, 100, 1000, 10000], 
             'gamma' : [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
             'kernel' : ['rbf','poly','sigmoid']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
# refit : 찾아진 가장 좋은 params로 estimator를 setting할 지 여부 (setting해줘야 곧바로 predict가 가능)
# verbose : 설명의 자세한 정도 (verbose를 3과 같이 바꿔보시면 더 자세하게 매 param set 마다의 결과를 확인할 수 있습니다.)
grid.fit(X_train_scaled, y_train)
print('The best parameters are ', grid.best_params_)rked'].fillna(0)
```

Fitting 3 folds for each of 108 candidates, totalling 324 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
The best parameters are `{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}`
[Parallel(n_jobs=1)]: Done 324 out of 324 | elapsed:    3.1s finished



**GridSearchCV 적용된 SVC**

> GridSearchCV를 적용한 결과값이 임의로 설정한 C=100,gamma=1.0 값보다 높지 않았다

```python
svc_g = SVC(kernel='rbf', C=10, gamma=0.01)
svc_g.fit(X_train_scaled, y_train)

print('GridSearchCV 적용 결과')
print("Accuracy on Training set: {:.3f}".format(svc_g.score(X_train_scaled, y_train)))
print("Accuracy on Test set: {:.3f}".format(svc_g.score(X_test_scaled, y_test)))
```

GridSearchCV 적용 결과
Accuracy on Training set: 0.833
Accuracy on Test set: 0.802



**Gradient Boosting Classifier**

```python
params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.005}
model_GBC = ensemble.GradientBoostingClassifier(**params)
model_GBC.fit(X_train_scaled, y_train)
```

GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.005, loss='deviance', max_depth=4,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=1000,
              n_iter_no_change=None, presort='auto', random_state=None,
              subsample=1.0, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)



모델 평가 함수 정의

```python
# model_GBC / gradientBoostingClassifier
# svc_g / gridsearch
# svc
# linear_svm
# model_knn

def mostvalue(inp_model) :    
    aa = inp_model.score(X_train_scaled, y_train)
    print("Accuracy on Training set: {:.3f}".format(aa))
    bb = inp_model.score(X_test_scaled, y_test)
    print("Accuracy on Test set: {:.3f}".format(bb))

    # Plot feature importance
    feature_importance = inp_model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)
    sorted_name = []
    for val in sorted_idx :
        sorted_name.append(column_namelist[val])

    # print(sorted_name)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, sorted_name)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
```



**Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier

model_f = RandomForestClassifier(n_estimators=200,random_state=0)
model_f.fit(X_train_scaled, y_train)
# print("Accuracy on Training set: {:.3f}".format(model_f.score(X_train_scaled, y_train)))
# print("Accuracy on Test set: {:.3f}".format(model_f.score(X_test_scaled, y_test)))

mostvalue(model_GBC)
mostvalue(model_f)
```

![result of RandomForest](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20200107194403251.png)

**ROC curve 시각화**

```python
# ROC (RandomForest)
pred_test = model_f.predict_proba(X_test_scaled) # Predict 'probability'
pred_test[:,1]
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true=y_test, y_score=pred_test[:,1])
roc_auc = auc(fpr, tpr) # AUC 면적의 값 (수치)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title("ROC curve")

plt.show()
```

![ROC curve](C:\Users\student\AppData\Roaming\Typora\typora-user-images\image-20200107194506289.png)



