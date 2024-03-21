# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmGet the data and use label encoder to change all the values to numeric.
1. Drop the unwanted values,Check for NULL values, Duplicate values.     
2. Classify the training data and the test data.                
3. Calculate the accuracy score, confusion matrix and classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Dhiyaneshwar P
RegisterNumber: 212222110009
*/
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![image](https://github.com/Dhiyanesh24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118362288/28b0a1f0-dcbf-4934-aac3-231c7146940f)

![image](https://github.com/Dhiyanesh24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118362288/708156aa-7e86-4cc3-866c-002f88355b6b)

![image](https://github.com/Dhiyanesh24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118362288/14f03064-c650-49c9-871f-0fc9911a2963)

![image](https://github.com/Dhiyanesh24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118362288/46bb1458-081d-43f8-aa22-f29fff64aa96)

![image](https://github.com/Dhiyanesh24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118362288/864590c3-bdc8-489b-ad9f-dda8d0ed5ceb)

![image](https://github.com/Dhiyanesh24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118362288/e3daf3b9-b57e-49eb-9739-c48d3a51315b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
