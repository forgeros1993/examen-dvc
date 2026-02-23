import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import yaml
import pickle
import os

# je charge param
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

grid_params = {
    'n_estimators': params['grid_search']['n_estimators'],
    'max_depth': params['grid_search']['max_depth']
}
cv = params['grid_search']['cv']
random_state = params['split']['random_state']

#  charge data
X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

#  gridsearch
model = GradientBoostingRegressor(random_state=random_state)
clf = GridSearchCV(model, grid_params, cv=cv, scoring='neg_mean_squared_error')
clf.fit(X_train_scaled, y_train.values.ravel())

#  save best params
os.makedirs("models", exist_ok=True)
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(clf.best_params_, f)

print("etape gridsearch terminee")