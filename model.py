import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib

train_data = pd.read_csv("oasis_longitudinal.csv")

target_cols = ['CDR']
feature_cols = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV']
selected_cols = feature_cols + target_cols

train_data = train_data[selected_cols].copy()
train_data.dropna(axis=0, inplace=True)

X = train_data[feature_cols]
y = train_data['CDR'].apply(str)

encoder = LabelEncoder()
X['M/F'] = encoder.fit_transform(X['M/F'])

clf = GradientBoostingClassifier(random_state=0)
clf.fit(X, y)

joblib.dump(clf, "clf.pkl")