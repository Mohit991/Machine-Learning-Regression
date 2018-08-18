import numpy as np
import pandas as pd
import seaborn as s
import matplotlib.pyplot as plt

df = pd.read_csv('Classified Data',index_col=0)
# print(df.head())
#    WTT       PTI       EQW       SBI       LQE       QWG       FDJ  \
# 0  0.913917  1.162073  0.567946  0.755464  0.780862  0.352608  0.759697
# 1  0.635632  1.003722  0.535342  0.825645  0.924109  0.648450  0.675334
# 2  0.721360  1.201493  0.921990  0.855595  1.526629  0.720781  1.626351
# 3  1.234204  1.386726  0.653046  0.825624  1.142504  0.875128  1.409708
# 4  1.279491  0.949750  0.627280  0.668976  1.232537  0.703727  1.115596
#
#         PJF       HQE       NXJ  TARGET CLASS
# 0  0.643798  0.879422  1.231409             1
# 1  1.013546  0.621552  1.492702             0
# 2  1.154483  0.957877  1.285597             0
# 3  1.380003  1.522692  1.153093             1
# 4  0.646691  1.463812  1.419167             1

# this is anonymous data and the column names are just some random letters
# this is given to us by some company and we will analyse and do some prediction without actaully knowing what a colunm is
# representing
# we will apply KNN for this and for KNN the scales of a feature matter the most and they must be in the same scale
# so we do feature scaling using sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis = 1))
# the scaler object will now fit to all the columns except for the target class

scale_feature = scaler.transform(df.drop('TARGET CLASS',axis = 1))
print(scale_feature)
# scale_feature is a dataframe here and we want to create a df from it now AS
df_feat = pd.DataFrame(scale_feature,columns = df.columns[:-1])
# this will not have the TARGET CLASS column
y = df['TARGET CLASS']

print(df_feat)
from sklearn.cross_validation import train_test_split
x_t,x_test,y_t,y_test  = train_test_split(df_feat,y,test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(x_t,y_t)
# pre =  knn.predict(x_test)
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test,pre))
# from sklearn.metrics import classification_report
# print(classification_report(y_test,pre))
# now we have set k = 1 but we may or may not get the highest accuracy at k = 1
# so we will try different values of k and then we will try to see how much accuarate model is at different values of k
er = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_t,y_t)
    pre = knn.predict(x_test)
    er.append(np.mean(pre != y_test))
# print(er)
# # [0.07666666666666666, 0.09, 0.05, 0.07, 0.056666666666666664, 0.05, 0.06666666666666667, 0.06, 0.05, 0.056666666666666664, 0.05333333333333334, 0.04666666666666667, 0.05, 0.056666666666666664, 0.056666666666666664, 0.05, 0.05, 0.04666666666666667, 0.05, 0.05333333333333334, 0.05, 0.05, 0.06333333333333334, 0.056666666666666664, 0.056666666666666664, 0.05, 0.05, 0.04666666666666667, 0.06, 0.05, 0.056666666666666664, 0.04666666666666667, 0.05333333333333334, 0.043333333333333335, 0.04666666666666667, 0.043333333333333335, 0.04666666666666667, 0.043333333333333335, 0.05]
# it is hard to read so we can plot this
plt.plot(range(1,40),er,color ='blue',linestyle = 'dashed',marker = 'o',markerfacecolor = 'red',markersize = 10)
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()
# from this k = 17 seems best so lets choose that
knn1 = KNeighborsClassifier(n_neighbors=17)
knn1.fit(x_t,y_t)
pre1 = knn.predict(x_test)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pre))
from sklearn.metrics import classification_report
print(classification_report(y_test,pre))
