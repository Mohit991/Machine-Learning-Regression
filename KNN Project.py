import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s

df = pd.read_csv('KNN_Project_Data')
print(df.head())
# s.pairplot(df,hue = 'TARGET CLASS')
# plt.show()
# now lets standardizze the variable
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
sd.fit(df.drop('TARGET CLASS',axis = 1))
sc = sd.transform(df.drop('TARGET CLASS',axis = 1))
dff = pd.DataFrame(sc,columns = df.columns[:-1])
print(dff.head(3))
  # scaling is done now
from sklearn.cross_validation import train_test_split
x = dff
y = df['TARGET CLASS']

x_t,x_test,y_t,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(x_t,y_t)
# pre = knn.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report
# print(confusion_matrix(y_test,pre))
# print(classification_report(y_test,pre))
# now to improve the accuracy of the model we wil try different values of k
err = []
# for i in range(1,40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(x_t, y_t)
#     pre = knn.predict(x_test)
#     err.append(np.mean(pre!=y_test))
# plt.plot(range(1,40),err,color = 'blue',linestyle = 'dashed')
# plt.show()
# n = 30 looks the best so we'll go with that
# so now

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(x_t,y_t)
pre = knn.predict(x_test)
# s.jointplot(y_test,pre)
# plt.show()

print(confusion_matrix(y_test,pre))
print(classification_report(y_test,pre))
# so accuracy has been improved.