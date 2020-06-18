import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)


dum_train_df = train_df.copy()

features_df = dum_train_df.drop(columns='Survived')

#'PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp' , 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

# Variable	Definition	        Key
# survival	Survival	        0 = No, 1 = Yes
# pclass	Ticket class	    1 = 1st, 2 = 2nd, 3 = 3rd
# sex	    Sex
# Age	    Age in years
# sibsp	    # of siblings /
#           spouses aboard the Titanic
# parch	    # of parents /
#           children aboard the Titanic
# ticket	Ticket number
# fare	    Passenger fare
# cabin	    Cabin number
# embarked	Port of Embarkation	    C = Cherbourg, Q = Queenstown, S = Southampton
# title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
#               'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
#               'Don', 'Jonkheer']

features_df['Age'].fillna(0, inplace=True)
features_df['Embarked'].fillna('not_mentioned', inplace=True)

features_df['family_size'] = sum(features_df['Parch'], features_df['SibSp'])

features_df['Pclass'].value_counts()

features_df['Cabin'].unique()

features_df['Deck'] = features_df['Cabin'].apply(lambda x:  str(x)[:1] if pd.notnull else 'M')
features_df['Deck'] = features_df['Deck'].replace(['A', 'B', 'C'], 'ABC')
features_df['Deck'] = features_df['Deck'].replace(['D', 'E'], 'DE')
features_df['Deck'] = features_df['Deck'].replace(['F', 'G',], 'FG')

features_df['Age'].value_counts()

features_df.info()
columns_To_be_encoded = ['Name', 'Sex', 'Ticket', 'Embarked', 'Deck']
cat_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'family_size']
# cat_columns = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'family_size']

# df = df.astype({"Name":'category', "Age":'int64'})
features_df = features_df.astype({'Age': 'category', 'Fare': 'category','Pclass': 'category'})

features_df.info()
features = features_df.drop(columns=['Cabin'])

encoder = OneHotEncoder()

for i in columns_To_be_encoded:
    features[i] = encoder.fit_transform(features[i].values.reshape(1,-1))
    encode_df = pd.DataFrame(features)

encoder.categories_

X = encode_df
y = train_df['Survived'].values

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)

x_train.shape
y_train.shape
