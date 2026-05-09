import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift

reference = pd.read_csv("reference_data.csv")
month3 = pd.read_csv("month3_data.csv")

# Only monitor the features we care most about
critical_features = ["GPA", "Attendance_Rate", "Study_Hours_per_Day", "Family_Income", "Age"]

# Build a report with individual column drift metrics
metrics = [ValueDrift(column=col) for col in critical_features]

report = Report(metrics=metrics)
snapshot = report.run(reference_data=reference, current_data=month3)

result = snapshot.dict()

print("Critical Feature Drift Analysis (Month 3)")
print("=" * 60)

for i, feature in enumerate(critical_features):
    metric = result["metrics"][i]
    score = float(metric["value"])
    threshold = metric["config"]["threshold"]
    method = metric["config"]["method"]
    drifted = score >= threshold

    status = "DRIFT DETECTED" if drifted else "stable"
    print(f"\n{feature}:")
    print(f"  Status:    {status}")
    print(f"  Score:     {score:.6f}")
    print(f"  Test used: {method}")

snapshot.save_html("reports/critical_features_month3.html")
print(f"\nDetailed report: reports/critical_features_month3.html")