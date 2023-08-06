import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import explained_variance_score as evs

from sklearn.tree import DecisionTreeRegressor


# Data Import
df = pd.read_csv('nuclear_explosions.csv')


# EDA & Data Preprocessing
print(df.info())

colums_to_drop = ['WEAPON DEPLOYMENT LOCATION', 'Data.Source', 'Data.Purpose', 'Data.Name', 'Date.Day',]
df = df.drop(colums_to_drop, axis= 1)

df['Data.Yeild'] = df['Data.Yeild.Upper']

colums_to_drop = ['Data.Yeild.Upper', 'Data.Yeild.Lower']
df = df.drop(colums_to_drop, axis= 1)

for c in df.columns:
    if df[c].dtype == 'object':
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
    df[c] = df[c].astype('float')

print(df, df.info())


# Train Test Split
features = df.drop('Data.Yeild', axis= 1)
target = df['Data.Yeild']

print(features, target)

X_train, X_test, Y_train, Y_test = train_test_split(features, target,
                                                    random_state= 42,
                                                    shuffle= True,
                                                    test_size= 0.20)


# Model Training
models = [DecisionTreeRegressor()]

for m in models:
    print(m)
    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy is : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy i : {evs(Y_test, pred_test)}\n')

# Bad Model Performance Due To The Lack Of Data