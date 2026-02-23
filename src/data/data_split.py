import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

#  charge param
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]

#  charge data
df = pd.read_csv("data/raw/raw.csv")

#  suppr date si la
if "date" in df.columns:
    df = df.drop(columns=["date"])

#  separe feature et target
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

#  split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

#  save data
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("etape split terminee")