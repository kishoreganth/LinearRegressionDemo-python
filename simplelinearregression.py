

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values

#splitting the dataset to train and test 
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y, test_size = 1/3 , random_state = 0)

#fitting into simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test results
y_pred = regressor.predict(X_test)

#prediction of the whole datset to add the prediction to the dataset 
y_whole_pred = regressor.predict(X)

dataset['Prediction'] = y_whole_pred


#visualising the training set results 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results 
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
