import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  s

# we will first get the data set of the titanic survive challenge into a pandas dataframe
train = pd.read_csv('titanic_train.csv')
print(train.head())
#    PassengerId  Survived  Pclass
# 0            1         0       3
# 1            2         1       1
# 2            3         1       3
# 3            4         1       1
# 4            5         0       3
#
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1
# 2                             Heikkinen, Miss. Laina  female  26.0      0
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1
# 4                           Allen, Mr. William Henry    male  35.0      0
#
#    Parch            Ticket     Fare Cabin Embarked
# 0      0         A/5 21171   7.2500   NaN        S
# 1      0          PC 17599  71.2833   C85        C
# 2      0  STON/O2. 3101282   7.9250   NaN        S
# 3      0            113803  53.1000  C123        S
# 4      0            373450   8.0500   NaN        S

# most of the times youll get missing data in the data set
# to find out where data is missing you can use
# # train.isnull() this will show each entry at each col as true or false
# which means wether they have value or not

# you should create a heatmap to find out where you are missing the most data
# s.heatmap(train.isnull(),yticklabels=False,cmap='viridis',cbar=False)
# plt.show()
# It will show us in which columns we are missing the most data
# it clearly shows that we are missing most of our data in age and cabin cols

# NOW  lets do some data analysis at visual level
# s.countplot(train['Survived'])
# plt.show()
# this plot shows how many survived and how many did not
# now lets find out how many males and females survived
# s.countplot(train['Survived'],hue = train['Sex'])
# plt.show()
# this clearly shows that most of the survivers were females
# lets again crete this but this time wrt passengers class which is Pclass
# s.countplot(train['Survived'],hue = train['Pclass'])
# plt.show()
# this clearly shows that people that did not survive were mostly the 3rd class which is the lowest class

# now lets see the age of poeple
# s.distplot(train['Age'].dropna(),kde=False,bins=30)
# plt.show()
#   this shows the age groups of the people that were on the titanic
# lets explore other columns as well
# lets explore SibSp which is how many siblings or spouse on the titanic
# s.countplot(train['SibSp'])
# plt.show()

# lets explore the fare col as well
# train['Fare'].hist(bins = 40)
# plt.show()

# lets clean data and fix the missing data
# s.heatmap(train.isnull(),yticklabels=False,cbar=False)
# plt.show()
# we have many values in the age column that are missing so we will fill them with the mean of the column
# print(train['Age'].mean())
# this gave us the average
# rather than putting the mean age we will try to find out the mean age by the passangers class
# we can plot the variation
# s.boxplot(x='Pclass',data = train,y='Age')
# plt.show()
# the line at the middle of the box is the average of age for that Pclass
# we will fill the missing age values by the average ages of the people of that class
# ie if a person is missing age and he belonged to class 1 then we'll fill his age with the average of the pople of class 1
# so
def fill_age(cols):
    age = cols[0]
    pclass = cols[1]

    if pd.isnull(age):
        if pclass == 1:
            return 37
        if pclass == 2:
            return 29
        else:
            return 24
    else:
        return age
#     this function will get the desired results
train['Age'] = train[['Age','Pclass']].apply(fill_age,axis=1)
# after this we will again built the heatmap as
# s.heatmap(train.isnull(),yticklabels=False,cbar=False)
# plt.show()
# there is no missing age values
# col cabin has too many missing values so we have to simply drop the col
train.drop('Cabin',axis=1,inplace=True)
print(train.columns)
# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Embarked'],
# to remove all the missing values we simply apply as
train.dropna(inplace=True)
# nothing is now missing


# now something has be done about the cetagorical variables
# they must be converted into dummy variables
# machine learning algo cannot take a cetagorical var which contain string values such as Sex
# we'll have to create a column male and female each of which will contain a value 0 or 1
# where 1 will indicate true and 0 as false
# but it is clear that if male = 1 means female will definatly be 0
# so we will simply drop female col and will only have male either 1 or 0
sex = pd.get_dummies(train['Sex'],drop_first=True)
# drop first will drop the first column of the 2 created
# sex is this dataframe
# print(sex.head())
#    male
# 0     1
# 1     0
# 2     0
# 3     0
# 4     1

# same should be done to the Embarked column
em = pd.get_dummies(train['Embarked'],drop_first=True)
print(em.head())
#    Q  S
# 0  0  1
# 1  0  0
# 2  0  1
# 3  0  1
# 4  0  1

# there is no C colummn because it has been dropped
# now we need to added these newly created dataframes to the actual dataframe
train = pd.concat([train,sex,em],axis=1)
# train sex and em dataframes will be concatenated and will be concatenated into train
print(train.head(2))
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1
#
#    Parch     Ticket     Fare Embarked  male  Q  S
# 0      0  A/5 21171   7.2500        S     1  0  1
# 1      0   PC 17599  71.2833        C     0  0  0

# now we will drop the cols not needed
train.drop(['Sex','Name','Embarked','Ticket'],axis = 1,inplace = True)
print(train.columns)

# Index(['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
#        'male', 'Q', 'S'],

# Passenger_ID is also useless
train.drop('PassengerId',inplace = True,axis = 1)
# print(train.columns)Index(['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
#        'S'],

# now data cleaning has been done and our dataframe is ready to be applied a machine learning algo
# lets do the prediction

x = train.drop('Survived',axis = 1)
y = train['Survived']
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
pre = lr.predict(x_test)
# print(pre)
# print(y_test)
# now we need to test the classification we just did
from sklearn.metrics import classification_report
print(classification_report(y_test,pre))
# this will create the confusion matrix
#      precision    recall  f1-score   support
#
#           0       0.79      0.87      0.83       216
#           1       0.76      0.65      0.70       140
#
# avg / total       0.78      0.78      0.78       356
# if you want the actual confusion matrix then
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pre))
#
# [[187  29]
#  [ 49  91]]
#
