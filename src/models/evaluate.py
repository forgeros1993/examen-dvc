import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

# charge data test
X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# charge modele
with open("models/gbr_model.pkl", "rb") as f:
    model = pickle.load(f)

# predis
predictions = model.predict(X_test_scaled)

# save predictions
pd.DataFrame(predictions, columns=["predictions"]).to_csv("data/prediction.csv", index=False)

# calcul metriques
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# je save metriques
scores = {"mse": mse, "r2": r2}
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f)

print("etape evaluation terminee")