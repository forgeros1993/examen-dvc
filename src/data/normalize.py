import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# charge data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# normalise
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#  save data
X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)

print("etape normalize terminee")