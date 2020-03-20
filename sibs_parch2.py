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
# Add a feature called Family Size
for id, attributes in df.iterrows():
    df.loc[id, 'Relatives'] = attributes['SibSp'] + attributes['Parch']
    if df.loc[id, 'Relatives'] == 0:
        df.loc[id, 'FamilySize'] = 0
    if df.loc[id, 'Relatives'] > 0:
        df.loc[id, 'FamilySize'] = 1
    if df.loc[id, 'Relatives'] > 2:
        df.loc[id, 'FamilySize'] = 2
    if df.loc[id, 'Relatives'] > 3:
        df.loc[id, 'FamilySize'] = 3
    if df.loc[id, 'Relatives'] > 5:
        df.loc[id, 'FamilySize'] = 4
    if df.loc[id, 'Relatives'] > 6:
        df.loc[id, 'FamilySize'] = 5

#df = df[df.Relatives != 0]
axes = sns.factorplot('Relatives','Survived', data=df, aspect = 2.5)

# %%
#Extract the passenger Title from Name
df2 = df
df2['Title'] = df2.Name
df2.Title = df2.Title.replace(regex={
    r'.*, Capt.*': 'Officer',
    r'.*, Col.*': 'Officer',
    r'.*, Major.*': 'Officer',
    r'.*, Jonkheer.*': 'Royalty',
    r'.*, Don.*': 'Royalty',
    r'.*, Sir.*': 'Royalty',
    r'.*, Dr.*': 'Officer',
    r'.*, Rev.*': 'Officer',
    r'.*, the Countess.*': 'Royalty',
    r'.*, Mme.*': 'Mrs',
    r'.*, Mlle.*': 'Miss',
    r'.*, Ms.*': 'Mrs',
    r'.*, Mr.*': 'Mr',
    r'.*, Mrs.*': 'Mrs',
    r'.*, Miss.*': 'Miss',
    r'.*, Master.*': 'Master',
    r'.*, Lady.*': 'Royalty'
})
axes = sns.factorplot('Title','Survived', data=df2, aspect = 2.5)

# %%
# Add a feature called AgeGroup
df3 = df2
bins = list(range(0, 71, 5))
labels = list(range(len(bins)-1))
df3['AgeGroup'] = pd.cut(df3.Age, bins, labels=labels, include_lowest=True)
axes = sns.factorplot('AgeGroup','Survived', data=df3, aspect = 2.5)
df3.AgeGroup = df3.AgeGroup = df.AgeGroup.replace({
    0:3, 
    1:1, 
    2:2, 
    3:2, 
    4:2, 
    5:1, 
    6:2, 
    7:2, 
    8:1, 
    9:3, 
    10:3, 
    11:3, 
    12:1, 
    13:0
})

# %%
# Replace Sex with values
df4 = df3
df4.Sex = df4.Sex.replace({'male': 1, 'female':0})
axes = sns.factorplot('Sex','Survived', data=df3, aspect = 2.5)


# %%
# Prepare data for training
df5=df4

df5 = df5.dropna(subset=['AgeGroup'])
#Replace titles with dummies
df5.Title = df5.Title.replace({'Officer': 1, 'Mrs': 2, 'Miss': 3, 'Mr': 4, 'Master': 5, 'Royalty': 6})
df5 = df5.loc[:, ['AgeGroup', 'Sex', 'Title', 'FamilySize', 'Survived']]



# %%
# Train a model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
df6 = df5
y = df6['Survived'].to_numpy()
X = df6.drop('Survived', axis=1).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))
print('RandomForst, Train score:', cross_val_score(clf, X_train, y_train, cv=20).mean())
print('RandomForst, Test score:', cross_val_score(clf, X_test, y_test, cv=20).mean())

# %%

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))
print('DecisionTree, Train score:', cross_val_score(clf, X_train, y_train, cv=20).mean())
print('DecisionTree, Test score:', cross_val_score(clf, X_test, y_test, cv=20).mean())




# %%
