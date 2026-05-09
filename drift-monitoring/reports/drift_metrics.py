import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset

reference = pd.read_csv("reference_data.csv")
month1 = pd.read_csv("month1_data.csv")
month2 = pd.read_csv("month2_data.csv")
month3 = pd.read_csv("month3_data.csv")

def get_drift_summary(reference, current, label):
    """Run drift detection and return a summary dictionary."""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    result = snapshot.dict()

    # First metric is DriftedColumnsCount with overall drift info
    drift_count_metric = result["metrics"][0]
    drifted_count = int(drift_count_metric["value"]["count"])
    drift_share = drift_count_metric["value"]["share"]

    # Remaining metrics are per-column ValueDrift
    feature_metrics = result["metrics"][1:]
    total_features = len(feature_metrics)

    summary = {
        "period": label,
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share": round(drift_share, 3),
        "dataset_drift": drift_share >= 0.5,
    }

    # Extract per-feature drift details
    feature_details = {}
    for metric in feature_metrics:
        column = metric["config"]["column"]
        threshold = metric["config"]["threshold"]
        drift_value = float(metric["value"])
        feature_details[column] = {
            "drifted": drift_value >= threshold,
            "threshold": threshold,
            "drift_score": round(drift_value, 4),
            "method": metric["config"]["method"],
        }

    summary["features"] = feature_details
    return summary

# Analyze all three months
for data, label in [(month1, "Month 1"), (month2, "Month 2"), (month3, "Month 3")]:
    summary = get_drift_summary(reference, data, label)

    print(f"\n{'=' * 60}")
    print(f"{label}: {summary['drifted_features']}/{summary['total_features']} features drifted "
          f"({summary['drift_share']*100:.1f}%)")
    print(f"Dataset drift detected: {summary['dataset_drift']}")
    print(f"{'=' * 60}")

    # Show which features drifted
    drifted = {k: v for k, v in summary["features"].items() if v["drifted"]}
    if drifted:
        print("Drifted features:")
        for feature, details in drifted.items():
            print(f"  {feature}: score = {details['drift_score']} (threshold = {details['threshold']})")
    else:
        print("No features showed significant drift.")