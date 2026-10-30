
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw" / "ai4i2020.csv"
OUTPUT = ROOT / "data" / "processed" / "cleaned_dataset.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT)

print("Original shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove completely empty rows
df = df.dropna(how="all")

# Check required target
if "Machine failure" not in df.columns:
    raise ValueError("Target column 'Machine failure' not found.")

# Drop ID columns if present
df = df.drop(
    columns=["UDI", "Product ID"],
    errors="ignore"
)

# Save cleaned data
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Cleaned shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Saved to:", OUTPUT)
print("\nFailure distribution:")
print(df["Machine failure"].value_counts())