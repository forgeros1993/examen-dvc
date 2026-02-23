import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import yaml
import pickle

# charge param
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

random_state = params['split']['random_state']

#charge data
X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# charge best params
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# forme le modele final
model = GradientBoostingRegressor(**best_params, random_state=random_state)
model.fit(X_train_scaled, y_train.values.ravel())

# save modele
with open("models/gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("etape training terminee")