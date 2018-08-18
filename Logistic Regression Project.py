# it is done on a fake ad dataset
# it indicates wether or not an internet user clicked on an ad or not
# we will try to predict wether or not an internet user will click on an ad or not
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
ad = pd.read_csv('advertising.csv')
# print(ad.head())
#    Daily Time Spent on Site  Age  Area Income  Daily Internet Usage  \
# 0                     68.95   35     61833.90                256.09
# 1                     80.23   31     68441.85                193.77
# 2                     69.47   26     59785.94                236.50
# 3                     74.15   29     54806.18                245.89
# 4                     68.37   35     73889.99                225.58
#
#                            Ad Topic Line            City  Male     Country  \
# 0     Cloned 5thgeneration orchestration     Wrightburgh     0     Tunisia
# 1     Monitored national standardization       West Jodi     1       Nauru
# 2       Organic bottom-line service-desk        Davidton     0  San Marino
# 3  Triple-buffered reciprocal time-frame  West Terrifurt     1       Italy
# 4          Robust logistical utilization    South Manuel     0     Iceland
#
#              Timestamp  Clicked on Ad
# 0  2016-03-27 00:53:11              0
# 1  2016-04-04 01:39:02              0
# 2  2016-03-13 20:35:42              0
# 3  2016-01-10 02:31:19              0
# 4  2016-06-03 03:36:18              0
# these are the features
print(ad.columns)

# we will start with explainetory data analysis
# we'll build the histogram of the age
# ad['Age'].hist()
# plt.show()
# now a joint plot showing the relation between area income vs age
# s.jointplot(x='Area Income',y='Age',data=ad)
# plt.show()
# now a joint plot kde of time spent and age
# s.jointplot('Daily Time Spent on Site','Age',ad)
# plt.show()
# s.jointplot('Daily Time Spent on Site','Daily Internet Usage',ad)
# plt.show()
# s.pairplot(ad)
# plt.show()

from sklearn.cross_validation import train_test_split
x =ad[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad['Clicked on Ad']
x_t,x_test,y_t,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_t,y_t)
pre = lr.predict(x_test)


from sklearn.metrics import classification_report
print(classification_report(y_test,pre))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pre))
