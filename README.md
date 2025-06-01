# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DIVYA DHARSHINI R
RegisterNumber: 212223040042
```
```import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

accuracy=accuracy_score(y_test,y_pred)
accuracy

confusion=confusion_matrix(y_test,y_pred)
confusion

classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
input_data = pd.DataFrame(np.array([[1,80,1,90,1,1,90,1,0,85,1,85]]),columns=x.columns)
lr.predict(input_data)

```

## Output:
## TOP 5 ELEMENTS
![image](https://github.com/user-attachments/assets/ba44450a-4392-40ff-9e22-12a01815f531)
![image](https://github.com/user-attachments/assets/f55232f1-bd61-4cee-ab68-1dd539c29824)
## Data Duplicate:
![image](https://github.com/user-attachments/assets/526e9896-210e-4607-b087-2c5ab4a76c22)
## Print Data:
![image](https://github.com/user-attachments/assets/bf329859-ae8f-44b3-90ba-c95836b87732)
## Data-Status:
![image](https://github.com/user-attachments/assets/23870025-f22c-47b4-ada8-620f3192624e)
## y_prediction array:
![image](https://github.com/user-attachments/assets/08104893-d4d4-46b3-a895-f93ea0d14f1a)
## Accuracy Value:
![image](https://github.com/user-attachments/assets/7559a304-c4c3-4439-9955-0aebeb87850c)
## Confusion array:
![image](https://github.com/user-attachments/assets/2e95c950-415e-4f0d-b8a6-f458e2355495)
## Classification Report:
![image](https://github.com/user-attachments/assets/2bf91f61-9fe2-4a4e-8700-2213ed35f7fd)
## Prediction of LR:
![image](https://github.com/user-attachments/assets/9befd5aa-aed3-4d92-8c3d-3937124134f6)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
