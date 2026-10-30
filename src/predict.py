
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "failure_model.joblib"

print("Loading InfraSense-AI prediction system...")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

print(f"Model loaded successfully!")
print(f"Failure threshold: {threshold:.4f}")

print("\nEnter machine readings")
print("Temperature values should use Kelvin.")
print("Rotational speed: RPM | Torque: Nm | Tool wear: minutes")

machine_type = input("Machine type (L/M/H): ").strip().upper()

if machine_type not in ["L", "M", "H"]:
    raise ValueError("Machine type must be L, M, or H.")

air_temp = float(input("Air temperature (K): "))
process_temp = float(input("Process temperature (K): "))
speed = float(input("Rotational speed (RPM): "))
torque = float(input("Torque (Nm): "))
tool_wear = float(input("Tool wear (minutes): "))

readings = {
    "Type": machine_type,
    "Air temperature": air_temp,
    "Process temperature": process_temp,
    "Rotational speed": speed,
    "Torque": torque,
    "Tool wear": tool_wear,
}

# Keep exactly the feature order used during training
input_df = pd.DataFrame([readings], columns=features)

probability = model.predict_proba(input_df)[0, 1]
prediction = int(probability >= threshold)

print("\n========== PREDICTION RESULT ==========")
print(f"Failure probability: {probability:.1%}")
print(f"Decision threshold: {threshold:.1%}")

if prediction == 1:
    print("Status: FAILURE RISK ALERT")
    print("Recommendation: Inspect the machine and review its condition.")
else:
    print("Status: NO FAILURE ALERT")
    print("Recommendation: Continue routine monitoring.")

print("\nNote: This is an ML prediction, not a guarantee.")
print("======================================")
