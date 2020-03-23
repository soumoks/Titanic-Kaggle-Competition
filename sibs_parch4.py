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
df = df.append(df_test)

# %%
# Fill embarked
# A quick inspect shows that here are 2 embarked values missing. Since this number is
# small, we fill the missing values with the most common class 'S'.
# As demonstrated in the plot, there is definitely a correlation between embarked and 
# the survival rate. Passengers from port 'Southampton' have only 34% of survival rate 
# while those from the port 'Cherbourg' have 55% chance.
df.Embarked = df.Embarked.fillna('S')
axes = sns.factorplot('Embarked','Survived', data=df, aspect = 2.5)

df.Fare = df.Fare.fillna(df.Fare.mean())

# %%
# Add features called Alone and Relatives
# Passengers traveled with families have higher survival rate than travel alone. Also, 
# passengers with 2 to 3 family members on board tends to have better chance of survival.
# As demonstrated in the plot, passenders travaled with 3 family members have 70% survival rate
# while passenger traveled with large family, 7-8 members have 0% survival rate.

df['Relatives'] = df['SibSp'] + df['Parch']
df.loc[df['Relatives'] > 0, 'Alone'] = 'No'
df.loc[df['Relatives'] == 0, 'Alone'] = 'Yes'
axes = sns.factorplot('Alone','Survived', data=df, aspect = 2.5)
axes = sns.factorplot('Relatives','Survived', data=df, aspect = 2.5)

# %%
# Extract the passenger's Title from Name
# We created an additonal features called Title to explore the correlation between survival rate
# based on the passenger's social status. The titles can be classified into 3 main categories,
# Royalty, Workers and Civilians. Class worker includes Master and Officers where Offers have
# only 30% survival rate while Masters have about 60%. Royalites also have a high survival rate
# at just above 60%. On the other hand, male civilians do not stand much chance to survive. Only
# 10% of male who are civilian survived. Guess Jack didn't make the cut :(
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

# %%
# fill age based on sex, class and title
# A quick inspect using .info() shows there are 263 passengers have the value Age missing. This
# is quite a large amount considering the dataset have only 1309 samples. Filling the age with
# mean would be an option, but may not be the best approach. Instead, we use the median based on
# three other features, Sex, Pclass and Title. As demonstrated in the plots, the passenger's age
# are highly influenced by these three features.
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
axes = sns.factorplot('Pclass','Age', data=df2, aspect = 2.5)
axes = sns.factorplot('Sex','Age', data=df2, aspect = 2.5)
axes = sns.factorplot('Title','Age', data=df2, aspect = 2.5)

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

# let's see how it's distributed train_df['Age'].value_counts()
axes = sns.factorplot('AgeGroup','Survived', data=df3, aspect = 2.5)
# %%
# Add a feature called Age Class

df3['Age_Class']= df3['AgeGroup']* df3['Pclass']
axes = sns.factorplot('Age_Class','Survived', data=df3, aspect = 2.5)

# %%
# Convert fare to categories
df3.loc[df3['Fare'] <= 7.91, 'Fare'] = 0
df3.loc[(df3['Fare'] > 7.91) & (df3['Fare'] <= 14.454), 'Fare'] = 1
df3.loc[(df3['Fare'] > 14.454) & (df3['Fare'] <= 31), 'Fare'] = 2
df3.loc[(df3['Fare'] > 31) & (df3['Fare'] <= 99), 'Fare'] = 3
df3.loc[(df3['Fare'] > 99) & (df3['Fare'] <= 250), 'Fare'] = 4
df3.loc[ df3['Fare'] > 250, 'Fare'] = 5
df3['Fare'] = df3['Fare'].astype(int)

# %%
df3['Fare_Per_Person'] = df3['Fare']/(df3['Relatives']+1)
df3['Fare_Per_Person'] = df3['Fare_Per_Person'].astype(int)


# %%
# Select features for training
df4 = df3
df4 = df4.loc[:, ['Sex', 'Title', 'AgeGroup', 'Fare', 'Relatives', 'Age_Class', 'Embarked', 'Pclass']]
#df4 = df4.loc[:, ['Sex', 'Title', 'AgeGroup', 'Fare', 'Relatives', 'Embarked', 'Pclass']]
#df4 = df4.loc[:, ['Sex', 'Title', 'SibSp', 'AgeGroup', 'Fare_Per_Person', 'Age_Class', 'Fare', 'Relatives', 'Embarked', 'Pclass']]
labels = df3['Survived'].dropna()

# %%
# Scaling the numerical data
from sklearn.preprocessing import StandardScaler

numeric_features = list(df4.select_dtypes(include=['int64', 'float64', 'int32']).columns)
scaler = StandardScaler()
df4[numeric_features] = scaler.fit_transform(df4[numeric_features])

# %%
# One-Hot encoding
encode_features = list(df4.select_dtypes(include=['object']).columns)
for feature in encode_features:
    df4 = pd.concat([df4, pd.get_dummies(df4[feature], prefix=feature)],axis=1)
    df4.drop(feature, axis = 1, inplace=True)


# %%
# Train a RandomForst model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
df_train = df4[:891]
y = labels.to_numpy()
X = df_train.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=1888, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', max_depth=11, bootstrap=False)
#clf = RandomForestClassifier(n_estimators=1200, min_samples_split=4, min_samples_leaf=1, max_features='auto', max_depth=5, bootstrap=True)

clf.fit(X, y)
print('RandomForst, Train score:', cross_val_score(clf, X_train, y_train, cv=10).mean())
print('RandomForst, Test score:', cross_val_score(clf, X_test, y_test, cv=10).mean())

# %%
importances = pd.DataFrame({'feature':df_train.columns,'importance':np.round(clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(25)

# %%
# Train a Decision Tree model
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# #print(clf.score(X_test, y_test))
# print('DecisionTree, Train score:', cross_val_score(clf, X_train, y_train, cv=20).mean())
# print('DecisionTree, Test score:', cross_val_score(clf, X_test, y_test, cv=20).mean())

# %%
# Add result to test set
res = clf.predict(df4[891:].to_numpy()).astype(int)
df_test['Survived'] = res

# %%
# Prepare result file
df_res = df_test.loc[:,'Survived']
df_res.to_csv('results.csv', index = True, header=True)

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
max_depth = [int(x) for x in np.linspace(40,60, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [3, 4, 5, 6, 7]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5]
# Method of selecting samples for training each tree
bootstrap = [True, False]

max_features = [2, 4, 6, 8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# {'bootstrap': [True, False],
#  'max_depth': [40, 45, 50, 55, 60, None],
#  'max_features': ['auto', 'sqrt'],
#  'min_samples_leaf': [1, 2, 4],
#  'min_samples_split': [2, 5, 10],
#  'n_estimators': [600, 650, 700, 750, 800, 850, 900, 950, 1000]}


rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
print('Score:', rf_random.score(X, y))
print('Best parameters:', rf_random.best_params_)

# %%
