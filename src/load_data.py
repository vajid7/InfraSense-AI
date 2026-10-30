
from ucimlrepo import fetch_ucirepo
from pathlib import Path
import pandas as pd

print("Loading InfraSense dataset...")

dataset = fetch_ucirepo(id=601)

X = dataset.data.features
y = dataset.data.targets

# Features + target ko ek dataset mein combine karo
df = pd.concat([X, y], axis=1)

# Project root ka path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset folder
raw_dir = PROJECT_ROOT / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# Save dataset locally
file_path = raw_dir / "ai4i2020.csv"
df.to_csv(file_path, index=False)

print("\nDataset loaded and saved successfully!")
print("Dataset location:", file_path)
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFailure distribution:")
print(df["Machine failure"].value_counts())