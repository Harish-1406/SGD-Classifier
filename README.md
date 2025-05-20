# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries

2.Load the dataset using sklearn.datasets()

3.Convert the dataset into a dataframe

4.Define the input and target variable

5.Split the dataset into training and testing data

6.Train the model using SGDClassifier(),.fit() and predict using .predict()

7.Measure the accuracy of the model using accuracy_score() and confusion_matrix()

## Program:
```

Program to implement the prediction of iris species using SGD Classifier.
Developed by: HARISH P K
RegisterNumber: 212224040104

```
```py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```
## Output:

![Screenshot 2025-05-04 135755](https://github.com/user-attachments/assets/5fbcdd25-9711-43eb-b719-d602c84dcef1)
![Screenshot 2025-05-04 135803](https://github.com/user-attachments/assets/ea4ee8a1-7c87-432f-844d-03446323b14f)
![Screenshot 2025-05-04 135808](https://github.com/user-attachments/assets/ed887db5-6e0c-439f-b31e-179e87cbda25)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
