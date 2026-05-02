import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("Crop_recommendation.csv")

X = data.drop('label', axis=1)
y = data['label']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")