import pandas as pd
import numpy as np
import json
from evidently import Report
from evidently.metrics import ValueDrift

reference = pd.read_csv("reference_data.csv")
month1 = pd.read_csv("month1_data.csv")
month2 = pd.read_csv("month2_data.csv")
month3 = pd.read_csv("month3_data.csv")

# Track a single important feature across all three months
feature_to_track = "Study_Hours_per_Day"

print(f"Tracking drift for '{feature_to_track}' over time")
print("=" * 60)

timeline = []

for data, label in [(month1, "Month 1"), (month2, "Month 2"), (month3, "Month 3")]:
    report = Report(metrics=[ValueDrift(column=feature_to_track)])
    snapshot = report.run(reference_data=reference, current_data=data)
    result = snapshot.dict()

    metric = result["metrics"][0]
    drift_score = float(metric["value"])
    threshold = metric["config"]["threshold"]

    entry = {
        "period": label,
        "ref_mean": round(reference[feature_to_track].mean(), 3),
        "current_mean": round(data[feature_to_track].mean(), 3),
        "ref_std": round(reference[feature_to_track].std(), 3),
        "current_std": round(data[feature_to_track].std(), 3),
        "drift_score": round(drift_score, 6),
        "drift_detected": drift_score >= threshold,
    }
    timeline.append(entry)

    status = "DRIFT" if entry["drift_detected"] else "OK"
    print(f"\n{label}:")
    print(f"  Reference mean: {entry['ref_mean']}  |  Current mean: {entry['current_mean']}")
    print(f"  Reference std:  {entry['ref_std']}  |  Current std:  {entry['current_std']}")
    print(f"  Drift score:    {entry['drift_score']}")
    print(f"  Status:         {status}")

# Save timeline for potential dashboard use
with open("reports/drift_timeline.json", "w") as f:
    json.dump(timeline, f, indent=2)

print(f"\nTimeline saved to reports/drift_timeline.json")
print("\nNotice how the drift score increases over time.")
print("This is exactly the pattern monitoring systems watch for.")