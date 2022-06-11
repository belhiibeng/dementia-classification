import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("oasis_longitudinal.csv")
categorical_cols = ['M/F']
numerical_cols = ['Age', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
my_cols = categorical_cols + numerical_cols
X = df[my_cols]
X = X.replace(["F", "M"], [0, 1])
y = df['CDR'].apply(str)

numerical_transformer = SimpleImputer(strategy='mean')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)
    ])
model = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])
clf.fit(X, y)
import joblib

joblib.dump(clf, "clf.pkl")
