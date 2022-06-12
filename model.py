import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("oasis_longitudinal.csv")
categorical_cols = ['M/F']
numerical_cols = ['Age','Educ', 'SES', 'MMSE', 'eTIV', 'nWBV']
my_cols = categorical_cols + numerical_cols
X = df[my_cols]
y = df['CDR'].apply(str)

categorical_transformer = OneHotEncoder()
numerical_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer()),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
model = GradientBoostingClassifier(random_state=0)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])
clf.fit(X, y)

import joblib
joblib.dump(clf, "clf.pkl")