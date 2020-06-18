import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

train_df.Sex = train_df.Sex.astype('category').cat.codes
test_df.Sex = test_df.Sex.astype('category').cat.codes

train_df.info()
test_df.info()

train_df.isnull().sum()
test_df.isnull().sum()

train_df.drop(labels = ["Cabin","Name","Ticket"], axis=1, inplace=True)
test_df.drop(labels = ["Cabin","Name","Ticket"], axis=1, inplace=True)

train_df = train_df.dropna()
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

"""
Some Predictions:
Sex: Females are more likely to survive.
SibSp/Parch: People traveling alone are more likely to survive.
Age: Young children are more likely to survive.
Pclass: People of higher socioeconomic class are more likely to survive."""

#
# # Visualizations of Feature vs. Target
# fig=plt.figure()
# ax1=plt.subplot(321)
# sns.countplot(x = 'Survived', hue = 'Sex', data = train_df, ax=ax1)
#
# ax2=plt.subplot(322)
# sns.countplot(x = 'Survived', hue = 'Pclass', data = train_df, ax=ax2)
#
# ax3=plt.subplot(323)
# sns.countplot(x = 'Survived', hue = 'SibSp', data = train_df, ax=ax3)
# ax3.legend(loc=1, title='Sibling/Spouse Count', fontsize='x-small')
#
# ax4=plt.subplot(324, sharey=ax3)
# sns.countplot(x = 'Survived', hue = 'Parch', data = train_df, ax=ax4)
# ax4.legend(loc=1, title='Parent/Children Count', fontsize='x-small')
#
# ax5=plt.subplot(325)
# sns.countplot(x = 'Survived', hue = 'Embarked', data = train_df, ax=ax5)
# ax5.legend(loc=1, title='Embarked')
#
# fig.set_size_inches(8,12)
# fig.show()

#sort the ages into logical categories
bins = [0, 5, 12, 18, 24, 35, 60, 80]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)
train_df['AgeGroup'] = train_df['AgeGroup'].cat.codes

bins_t = [0, 5, 12, 18, 24, 35, 60, 80]
labels_t = ['1', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
test_df['AgeGroup'] = pd.cut(test_df["Age"], bins_t, labels = labels_t)
test_df['AgeGroup'] = test_df['AgeGroup'].cat.codes


#draw a bar plot of Age vs. survival
# fig1=plt.figure(figsize=(10,5))
# sns.barplot(x="AgeGroup", y="Survived", data=train_df)
# plt.show()

tab1=pd.crosstab(train_df.Pclass,train_df.Survived,margins=True)
print(tab1)
print("----------------------------------------------")
tab2=pd.crosstab(train_df.Sex,train_df.Survived,margins=True)
print(tab2)
# Encoding Catagorical Values
train_df.drop(labels = ["AgeGroup"], axis=1, inplace=True)
train_df= train_df.copy()
test_df.drop(labels = ["AgeGroup"], axis=1, inplace=True)
test_df= test_df.copy()

train_df = pd.get_dummies(train_df, columns=['Embarked', 'Pclass'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Embarked', 'Pclass'], drop_first=True)
# Correlations
# fig=plt.figure(figsize=(8,8))
# sns.heatmap(train_df.corr(), annot=True, cbar_kws={'label': 'Correlation coeff.'}, cmap="RdBu")
# fig.show()
# train_df.head()

X = train_df.drop('Survived', axis=1).values
y = train_df['Survived'].values
test_x = test_df

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(X)
# from sklearn.model_selection import train_test_split
#
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#
from sklearn.ensemble import RandomForestClassifier
#
# clf = RandomForestClassifier(max_features='auto',max_depth=6, min_samples_split=5,n_estimators=1750)
# clf.fit(x_train,y_train)
#
# train_pred = clf.predict(x_train)
# test_pred = clf.predict(x_test)
#
# from sklearn.metrics import classification_report
# print(classification_report(train_pred, y_train))
# print(classification_report(test_pred, y_test))

from sklearn.model_selection import cross_val_score

X=train_df.drop('Survived',axis=1)
y=train_df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

RFG=RandomForestClassifier()

RFG.fit(X_train,y_train)
prediction=RFG.predict(X_test)
score=cross_val_score(RFG,X_train,y_train,cv=5)
# print("Confusion_matrix:",confusion_matrix(y_test,prediction))
acc_log = RFG.score(X_train,y_train)
acc_log

test_pred = RFG.predict(test_df)

gender_submission_m = pd.DataFrame(columns=['Passengerid', 'Survived'])
gender_submission_m['Passengerid'] =test_df['PassengerId']
gender_submission_m['Survived'] = test_pred
gender_submission_m.to_csv('gender_submission.csv',index=None)

# test_df.head()