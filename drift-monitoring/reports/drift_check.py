import pandas as pd
import sys
import json
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

# Configuration
DRIFT_SHARE_WARNING = 0.20    # warn if more than 20% of features drift
DRIFT_SHARE_CRITICAL = 0.40   # fail if more than 40% of features drift

def check_drift(reference_path, current_path):
    """Run drift analysis and return status."""
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    result = report.as_dict()
    drift_data = result["metrics"][0]["result"]

    total = drift_data["number_of_columns"]
    drifted = drift_data["number_of_drifted_columns"]
    share = drift_data["share_of_drifted_columns"]

    # Build result
    check_result = {
        "total_features": total,
        "drifted_features": drifted,
        "drift_share": round(share, 3),
        "dataset_drift": drift_data["dataset_drift"],
        "status": "ok",
    }

    # Determine status
    if share >= DRIFT_SHARE_CRITICAL:
        check_result["status"] = "critical"
    elif share >= DRIFT_SHARE_WARNING:
        check_result["status"] = "warning"

    # List drifted features
    drifted_features = []
    for feature_name, feature_data in drift_data["drift_by_columns"].items():
        if feature_data["drift_detected"]:
            drifted_features.append(feature_name)
    check_result["drifted_feature_names"] = drifted_features

    return check_result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python drift_check.py <reference_data.csv> <current_data.csv>")
        sys.exit(1)

    reference_path = sys.argv[1]
    current_path = sys.argv[2]

    print(f"Checking drift: {current_path} vs {reference_path}")
    print("=" * 60)

    result = check_drift(reference_path, current_path)

    print(f"Features drifted: {result['drifted_features']}/{result['total_features']} "
          f"({result['drift_share']*100:.1f}%)")
    print(f"Dataset drift:    {result['dataset_drift']}")
    print(f"Status:           {result['status'].upper()}")

    if result["drifted_feature_names"]:
        print(f"\nDrifted features: {', '.join(result['drifted_feature_names'])}")

    # Save result for pipeline consumption
    output_path = Path("reports/drift_check_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to {output_path}")

    # Exit code based on status
    if result["status"] == "critical":
        print(f"\nCRITICAL: {result['drift_share']*100:.1f}% of features drifted "
              f"(threshold: {DRIFT_SHARE_CRITICAL*100:.0f}%)")
        print("Action required: investigate and consider retraining.")
        sys.exit(1)
    elif result["status"] == "warning":
        print(f"\nWARNING: {result['drift_share']*100:.1f}% of features drifted "
              f"(threshold: {DRIFT_SHARE_WARNING*100:.0f}%)")
        print("Monitor closely. Retraining may be needed soon.")
        sys.exit(0)  # warning but not failure
    else:
        print("\nAll clear. Feature distributions are stable.")
        sys.exit(0)