# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
Program:

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Hiba Nasreen M 
Register Number: 212224040117

```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
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
## Output:
## Top 5 Elements
<img width="1221" height="226" alt="image" src="https://github.com/user-attachments/assets/7107bf20-048c-439f-a8e5-e0fabd1c09d6" />

<img width="1091" height="240" alt="image" src="https://github.com/user-attachments/assets/eafd70d7-804f-4ce7-aa37-ae1b88668864" />

<img width="982" height="497" alt="image" src="https://github.com/user-attachments/assets/d93dacb7-5d60-4f86-8817-cbd8adb4d063" />

## DATA DUPLICATE

<img width="61" height="48" alt="image" src="https://github.com/user-attachments/assets/f80164ea-5389-43e0-a4e4-2055f41af54f" />

## PRINT DATA

<img width="982" height="502" alt="image" src="https://github.com/user-attachments/assets/1ce605b7-f65e-49d5-a031-f310659cc67f" />

## DATA STATUS

<img width="922" height="510" alt="image" src="https://github.com/user-attachments/assets/a5dba796-e33c-4ba2-8270-a25588651d64" />

## Y_PREDICTION ARRAY

<img width="586" height="263" alt="image" src="https://github.com/user-attachments/assets/01d2e8c8-e7c3-4b00-90ce-50f67e4d17fa" />

## CONFUSION ARRAY

<img width="762" height="71" alt="image" src="https://github.com/user-attachments/assets/6500484b-2ccb-428a-8bf4-c215eb8a7067" />

## ACCURACY VALUE

<img width="210" height="51" alt="image" src="https://github.com/user-attachments/assets/d0b3e107-5d2a-4a7f-894a-c8e1b9b2f1bc" />

## CLASSFICATION REPORT

<img width="582" height="176" alt="image" src="https://github.com/user-attachments/assets/526a7b0a-6002-438f-82a3-297a17896087" />

## PREDICTION

<img width="303" height="33" alt="image" src="https://github.com/user-attachments/assets/1670142b-d706-4359-bf4f-3b3df2306d3c" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
