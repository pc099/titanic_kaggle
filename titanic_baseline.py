'''
Variable	Definition	            Key
survival	Survival	            0 = No, 1 = Yes
pclass	    Ticket class	        1 = 1st, 2 = 2nd, 3 = 3rd
sex	        Sex
Age	        Age in years
sibsp	    # of siblings /
            spouses aboard the Titanic
parch	    # of parents /
            children aboard the Titanic
ticket	    Ticket number
fare	    Passenger fare
cabin	    Cabin number
embarked	                           Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
'''
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score
train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
train_df.columns
test_df.columns
pred_col = 'Survived'
categorical_cols = ['Name', 'Sex', 'Ticket', 'Embarked', 'Cabin']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

print('male survived(1), dead(0) the crash :  {}'.format(train_df['Survived'] [train_df['Sex'] == 'male'].value_counts(normalize=True)))
print('female survived(1), dead(0) the crash :  {}'.format(train_df['Survived'] [train_df['Sex'] == 'female'].value_counts(normalize=True)))


title_list = []
for i in train_df['Name']:
    cmsp = i.split(', ')[1]
    t = cmsp.split('.')[0]
    title_list.append(t)
train_df['Title'] = title_list


for ix,i in enumerate(train_df['Title']):
    if i in ['Don','Major','Capt', 'Jonkheer','Rev', 'Col','Mr','Sir', 'Master']:
        train_df['Title'][ix] = 1
    elif i in ['the Countess', 'Mme','Mrs']:
        train_df['Title'][ix] = 2
    elif i in ['Mlle', 'Ms','Miss', 'Lady']:
        train_df['Title'][ix] = 3
    elif i == 'Dr':
        if i in train_df['Title'][train_df['Sex'] =='male']:
            train_df['Title'][ix] = 1
        else:
            train_df['Title'][ix] = 2

'''
0 - 20, 21 - 41, 41 - 61, 61 - 81,  81 - 91
'''
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Age'].unique()
train_df['Age_processed'] = float('NaN')
train_df['Age_processed'][train_df['Age'] <= 20] = 0
train_df['Age_processed'][train_df['Age'] > 20] = 1
train_df['Age_processed'][train_df['Age'] >=40] = 2
train_df['Age_processed'][train_df['Age'] >=60] = 3
train_df['Age_processed'][train_df['Age'] >=80] = 4


train_df['Cabin'] = train_df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
train_df['Cabin'] = train_df['Cabin'].replace(['A','B','C'], 1)
train_df['Cabin'] = train_df['Cabin'].replace(['D','E'], 2)
train_df['Cabin'] = train_df['Cabin'].replace(['F','G'], 3)
train_df['Cabin'] = train_df['Cabin'].replace(['M'], 4)
train_df['Cabin'] = train_df['Cabin'].replace(['T'], 5)

train_df['family_size'] = train_df['SibSp'] + train_df['Parch']
train_df['Age*Class']=train_df['Age']* train_df['Pclass']
train_df['Fare_Per_Person']= train_df['Fare']/(train_df['family_size']+1)
train_df['Sex'][train_df['Sex'] == 'male'] = 0
train_df['Sex'][train_df['Sex'] == 'female'] = 1
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df['Embarked'][train_df['Embarked'] == 'S'] = 0
train_df['Embarked'][train_df['Embarked'] == 'C'] = 1
train_df['Embarked'][train_df['Embarked'] == 'Q'] = 2
train_df['Sex'][train_df['Sex'] == 'male'] = 0
train_df['Sex'][train_df['Sex'] == 'female'] = 1
bins = [0, 5, 12, 18, 24, 35, 60, 80]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)
train_df['AgeGroup'] = train_df['AgeGroup'].astype('category').cat.codes

train_df.columns

# PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
#        'Ticket', 'Fare', 'Cabin', 'Embarked'
#
# title_list_t = []
# for i in test_df['Name']:
#     cmsp = i.split(', ')[1]
#     t = cmsp.split('.')[0]
#     title_list_t.append(t)
# test_df['Title'] = title_list_t
#
#
# for ix,i in enumerate(test_df['Title']):
#     if i in ['Don','Major','Capt', 'Jonkheer','Rev', 'Col','Mr','Sir', 'Master']:
#         test_df['Title'][ix] = 1
#     elif i in ['the Countess', 'Mme','Mrs']:
#         test_df['Title'][ix] = 2
#     elif i in ['Mlle', 'Ms','Miss', 'Lady','Dona']:
#         test_df['Title'][ix] = 3
#     elif i == 'Dr':
#         if i in test_df['Title'][test_df['Sex'] =='male']:
#             test_df['Title'][ix] = 1
#         else:
#             test_df['Title'][ix] = 2
#
# test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
# test_df['Age'].unique()
# test_df['Age_processed'] = float('NaN')
# test_df['Age_processed'][test_df['Age'] <= 20] = 0
# test_df['Age_processed'][test_df['Age'] > 20] = 1
# test_df['Age_processed'][test_df['Age'] >=40] = 2
# test_df['Age_processed'][test_df['Age'] >=60] = 3
# test_df['Age_processed'][test_df['Age'] >=80] = 4
#
# test_df['Cabin'] = test_df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
# test_df['Cabin'] = test_df['Cabin'].replace(['A','B','C'], 1)
# test_df['Cabin'] = test_df['Cabin'].replace(['D','E'], 2)
# test_df['Cabin'] = test_df['Cabin'].replace(['F','G'], 3)
# test_df['Cabin'] = test_df['Cabin'].replace(['M'], 4)
# test_df['Cabin'] = test_df['Cabin'].replace(['T'], 5)
#
# test_df['family_size'] = test_df['SibSp'] + test_df['Parch']
# test_df['Age*Class']=test_df['Age']* test_df['Pclass']
# test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
# test_df['Fare_Per_Person']= test_df['Fare']/(test_df['family_size']+1)
# test_df['Sex'][test_df['Sex'] == 'male'] = 0
# test_df['Sex'][test_df['Sex'] == 'female'] = 1
# test_df['Embarked'] = test_df['Embarked'].fillna('S')
# test_df['Embarked'][test_df['Embarked'] == 'S'] = 0
# test_df['Embarked'][test_df['Embarked'] == 'C'] = 1
# test_df['Embarked'][test_df['Embarked'] == 'Q'] = 2
# test_df['Sex'][test_df['Sex'] == 'male'] = 0
# test_df['Sex'][test_df['Sex'] == 'female'] = 1

target = train_df['Survived'].values
features = train_df[['Pclass','Title','Sex','Age_processed','Age*Class','Fare_Per_Person','Fare','Embarked','Cabin','family_size']].values
# features_test = test_df[['Pclass','Title','Sex','Age_processed','Age*Class','Fare_Per_Person','Fare','Embarked','Cabin','family_size']].values
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=39)
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=140)

# clf = RandomForestClassifier(max_features='auto',max_depth=6, min_samples_split=5,n_estimators=1750,random_state=39)

# clf.fit(x_train, y_train)
gb.fit(x_train,y_train)

train_pred = gb.predict(x_train)
test_pred = gb.predict(x_test)

# train_pred = clf.predict(x_train)
# test_pred = clf.predict(x_test)

# test_df_pred = clf.predict(features_test)
# len(test_pred)

gb.feature_importances_

print(classification_report(train_pred, y_train))
print(classification_report(test_pred, y_test))

# gender_submission = pd.DataFrame(columns=['Passengerid', 'Survived'])
# gender_submission['Passengerid'] =test_df['PassengerId']
# gender_submission['Survived'] = test_df_pred
# gender_submission.to_csv('gender_submission.csv',index=None)

# print("In sample Model Accuracy : {:.2f}%".format(roc_auc_score(train_pred,y_train)*100))
# print("Out sample Model Accuracy : {:.2f}%".format(roc_auc_score(test_pred,y_test)*100))


#
#             precision    recall  f1-score   support
#            0       0.93      0.88      0.90       457
#            1       0.80      0.88      0.84       255
#     accuracy                           0.88       712
#    macro avg       0.86      0.88      0.87       712
# weighted avg       0.88      0.88      0.88       712
#               precision    recall  f1-score   support
#            0       0.92      0.92      0.92       119
#            1       0.83      0.83      0.83        60
#     accuracy                           0.89       179
#    macro avg       0.87      0.87      0.87       179
# weighted avg       0.89      0.89      0.89       179

