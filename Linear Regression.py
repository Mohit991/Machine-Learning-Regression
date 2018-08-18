import numpy as np
import pandas as pd
import seaborn as s
import matplotlib.pyplot as plt
df = pd.read_csv('USA_Housing.csv')
# print(df.head())

print(df.columns)
# this prints all the columns in the dataset
# s.jointplot(x = df['Price'],y = df['Avg. Area Number of Rooms'])
# plt.show()
# use jointplot to find the relationship between any two columns in the data set how one col varies with the values of the other col values.

# now to do the linear regression we should first split the whole data set into 2 parts
# one is the features which we want to use as predictors to predict the one variable
# and second the the variable whhich we want to predict.
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']
# now we will divide the whole 2 parts into training and test sets to do so we use scikit learn
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)
# test size is the percentage of the data that will be used to test and random state means how rows will be randomly
# become test or train rows

from sklearn.linear_model import LinearRegression
# we imported the linear regression
LR = LinearRegression()
LR.fit(X_train,y_train)
# this is the training of the Model. The model is trained for the training data and then we will predict the result for the test data.
# we pass the training preditors and also the results for all those rows ie prices

# print(LR.intercept_)
# print(LR.coef_) this will print all the coffecients for all the colummns

predictions = LR.predict(X_test)
# this will predict the results for the X_test which the model have not seen before
print(predictions)
# this array gives us the values for each row in the X_test
# this shows the price for every row in the X_test
# [2.15282755e+01 1.64883282e+05 1.22368678e+05 2.23380186e+03 ...  1.51504200e+01]
# but as we already know that y_test contains the actual prices of all these houses so we
# can compare these outputs with the actual results to see how much accurate the model has predicted
print(y_test)
# now we can actually visualise them


# s.jointplot(predictions,y_test)
# plt.show()

# or you can plot any other way to check the differnce between true and predicted values.

# now we find the errors in our model

from sklearn import metrics
print(metrics.mean_absolute_error(y_test,predictions))
# this will give us the mean absolute error
print(metrics.mean_squared_error(y_test,predictions))
# this will give us the mean squared errors
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
# this will give us the root mean square error
# all of them help to understand the accuracy of the model.