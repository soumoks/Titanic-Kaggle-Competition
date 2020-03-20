# %%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import tree

# %%
# Load data from csv
df = pd.read_csv('train.csv', index_col='PassengerId')

# %%
# Select relevant data from dataset
df2 = df.loc[:, ['SibSp', 'Parch', 'Survived']]

# %%
# Add a feature called relatives
for id, attributes in df2.iterrows():
    df2.loc[id, 'Relatives'] = attributes['SibSp'] + attributes['Parch']
    if df2.loc[id, 'Relatives'] == 0:
        df2.loc[id, 'Alone'] = 1
    else:
        df2.loc[id, 'Alone'] = 0

# %%
axes = sns.factorplot('Relatives','Survived', data=df2, aspect = 2.5, )

# %%
# Try Decision Tree
from sklearn.ensemble import RandomForestClassifier
df3 = df2.drop(['SibSp', 'Parch'], axis=1)
y = df3['Survived'].to_numpy()
X = df3.drop('Survived', axis=1).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# %%
