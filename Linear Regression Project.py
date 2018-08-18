import pandas as pd
import numpy as np
import seaborn as s
import matplotlib.pyplot as plt

# this project is given to us by a company to enhance their sales, we will analysis the dataset to find out trends and hpw
# can sales be boosted.

cus = pd.read_csv('Ecommerce Customers')
# print(cus.head())
print(cus.columns)
# first step is to do data analysis
# now we'll create a jointplot between the time on the website and the yearly amount spent columns to find out how  they vary wrt to each
# other

# s.jointplot(cus['Avg. Session Length'],cus['Yearly Amount Spent'])
# plt.show()
# it is clear by the plot that there is no clear trend between the two
# so now we'll do the same on the time on app and yeary amount spend.
#

# s.jointplot(cus['Time on App'],cus['Yearly Amount Spent'])
# plt.show()
# now lets create a hex plot between time on app and membership lenght
# s.jointplot(cus['Time on App'],cus['Length of Membership'],kind='hex')
# plt.show()


# now to clearly understand the relationship between all the cols
# we create a pairplot for th data set
# s.pairplot(cus)
# plt.show()


# since we want to predict for the yearly amount spent we try to find out which col is correleated with it he most
# we use the paiplot to find any trends wrt the yearly amount spent
# Length of Membership is the most correleated wrt the yearly amount spent
#  we verify this we create a linear plot
# s.lmplot(x = 'Length of Membership',y = 'Yearly Amount Spent',data=cus)
# plt.show()

# this plot clearly shows that there is a linear relationship between the 2
# now we use the Linear Regression to prediction

x = cus[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = cus['Yearly Amount Spent']

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=101)
lm = LinearRegression()
lm.fit(x_train,y_train)
pre = lm.predict(x_test)
print(pre)
# [456.44186104 402.72005312 409.2531539  591.4310343  590.01437275]

print(y_test)
# 18     452.315675
# 361    401.033135
# 104    410.069611
# 4      599.406092
# 156    586.155870


# we will now create a scatter plot to test the accuracy of the model
# plt.scatter(y_test,pre)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.show()

# now we will find out the errors
from sklearn import metrics
print(metrics.mean_absolute_error(y_test,pre))
print(metrics.mean_squared_error(y_test,pre))
print(np.sqrt(metrics.mean_squared_error(y_test,pre)))

