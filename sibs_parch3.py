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
df_test = pd.read_csv('test.csv', index_col='PassengerId')

#vertical
df = df.append(df_test)

# %%
# Fill embarked
df.Embarked.replace({'S': 1, 'C': 2, 'Q': 3}, inplace=True)
df.Embarked = df.Embarked.fillna(1)
axes = sns.factorplot('Embarked','Survived', data=df, aspect = 2.5)

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
    r'.*, Mrs.*': 'Mrs',
    r'.*, Mr.*': 'Mr',
    r'.*, Miss.*': 'Miss',
    r'.*, Master.*': 'Master',
    r'.*, Lady.*': 'Royalty'
})
axes = sns.factorplot('Title','Survived', data=df2, aspect = 2.5)
df2.Title = df2.Title.replace({'Officer': 1, 'Mrs': 2, 'Miss': 3, 'Mr': 4, 'Master': 5, 'Royalty': 6})

# %%
# fill age based on sex, class and title
df3 = df2
grouped = df3.groupby(['Sex', 'Pclass', 'Title'])
grouped_median = grouped.median()
grouped_median = grouped_median.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

def fill_age(row):
    condition = (
        (grouped_median['Sex'] == row['Sex']) &
        (grouped_median['Title'] == row['Title']) &
        (grouped_median['Pclass'] == row['Pclass'])
    )
    if np.isnan(grouped_median[condition]['Age'].values[0]):
        condition = (
            (grouped_median['Sex'] == row['Sex']) &
            (grouped_median['Pclass'] == row['Pclass'])
        )
    return grouped_median[condition]['Age'].values[0]

df3['Age'] = df3.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

# %%
# Add a feature called AgeGroup
bins = list(range(0, 81, 5))
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
    13:0,
    14:0,
    15:4
})

# %%
# Replace Sex with values
df4 = df3
df4.Sex = df4.Sex.replace({'male': 1, 'female':0})
axes = sns.factorplot('Sex','Survived', data=df3, aspect = 2.5)

# %%
# Prepare data for training
df5 = df4
df5 = df5.loc[:, ['AgeGroup', 'Sex', 'Title', 'FamilySize', 'Survived', 'Embarked', 'Pclass']]
df5 = df5.loc[:, ['AgeGroup', 'Sex', 'Title', 'FamilySize', 'Survived', 'Pclass']]
df_test = df5[891:].drop('Survived', axis=1)
df5 = df5.dropna(subset=['Survived'])

# %%
# Train a RandomForst model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
df6 = df5
y = df6['Survived'].to_numpy()
X = df6.drop('Survived', axis=1).to_numpy()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=800, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', max_depth=50, bootstrap=False)
clf.fit(X, y)
#print(clf.score(X_test, y_test))
print('RandomForst, Train score:', cross_val_score(clf, X, y, cv=20).mean())
#print('RandomForst, Test score:', cross_val_score(clf, X_test, y_test, cv=20).mean())

# %%
# Train a Decision Tree model
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# #print(clf.score(X_test, y_test))
# print('DecisionTree, Train score:', cross_val_score(clf, X_train, y_train, cv=20).mean())
# print('DecisionTree, Test score:', cross_val_score(clf, X_test, y_test, cv=20).mean())

# %%
# Add result to test set
res = clf.predict(df_test.to_numpy()).astype(int)
df_test['Survived'] = res

# %%
# Prepare result file
df_test = df_test.loc[:,'Survived']
df_test.to_csv('results.csv', index = True, header=True)

#%%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
{'bootstrap': [True, False],
 'max_depth': [40, 45, 50, 55, 60, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [600, 650, 700, 750, 800, 850, 900, 950, 1000]}


rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)

# %%
