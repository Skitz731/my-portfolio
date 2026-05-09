import pandas as pd
import numpy as np

def load_and_prepare():
    """Load the student dropout dataset and do basic cleaning."""
    url = "https://raw.githubusercontent.com/TripleTen-DS/Dataset/refs/heads/main/student_dropout_dataset.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=["Student_ID"])

    # Fill missing values so drift analysis is clean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def create_reference_and_production(df):
    """
    Split data into reference (training) and production batches.
    The reference set represents what the model was trained on.
    Production batches simulate data arriving over three months.
    """
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # First 60% is the reference (training) data
    split = int(len(df) * 0.6)
    reference = df.iloc[:split].copy()
    remaining = df.iloc[split:].copy()

    # Split remaining into three production "months"
    batch_size = len(remaining) // 3
    month1 = remaining.iloc[:batch_size].copy()
    month2 = remaining.iloc[batch_size:batch_size*2].copy()
    month3 = remaining.iloc[batch_size*2:].copy()

    return reference, month1, month2, month3

def introduce_drift(month2, month3):
    """
    Simulate realistic drift in months 2 and 3.
    Month 1 stays clean to show what 'no drift' looks like.
    """
    # Month 2: moderate drift
    # Study hours increase (new tutoring program makes students study more)
    month2["Study_Hours_per_Day"] = month2["Study_Hours_per_Day"] + np.random.normal(1.0, 0.3, len(month2))
    month2["Study_Hours_per_Day"] = month2["Study_Hours_per_Day"].clip(0, 12)

    # Stress index increases slightly
    month2["Stress_Index"] = month2["Stress_Index"] + np.random.normal(0.5, 0.2, len(month2))
    month2["Stress_Index"] = month2["Stress_Index"].clip(0, 10)

    # Month 3: significant drift
    # Family income shifts (new scholarship attracts wealthier students)
    month3["Family_Income"] = month3["Family_Income"] * np.random.uniform(1.3, 1.8, len(month3))

    # Study hours shift even more
    month3["Study_Hours_per_Day"] = month3["Study_Hours_per_Day"] + np.random.normal(2.0, 0.5, len(month3))
    month3["Study_Hours_per_Day"] = month3["Study_Hours_per_Day"].clip(0, 12)

    # Age distribution changes (adult learner program)
    adult_learners = np.random.uniform(28, 45, int(len(month3) * 0.3))
    indices = np.random.choice(month3.index, size=len(adult_learners), replace=False)
    month3.loc[indices, "Age"] = adult_learners

    # Department distribution shifts
    dept_shift_indices = np.random.choice(month3.index, size=int(len(month3) * 0.2), replace=False)
    month3.loc[dept_shift_indices, "Department"] = "CS"

    # Attendance drops across the board
    month3["Attendance_Rate"] = month3["Attendance_Rate"] - np.random.normal(8, 3, len(month3))
    month3["Attendance_Rate"] = month3["Attendance_Rate"].clip(0, 100)

    return month2, month3

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_and_prepare()
    print(f"Total rows: {len(df)}")

    print("\nSplitting into reference and production batches...")
    reference, month1, month2, month3 = create_reference_and_production(df)

    print("Introducing drift into months 2 and 3...")
    month2, month3 = introduce_drift(month2, month3)

    print(f"\nReference (training data): {len(reference)} rows")
    print(f"Month 1 (no drift):       {len(month1)} rows")
    print(f"Month 2 (moderate drift):  {len(month2)} rows")
    print(f"Month 3 (significant drift): {len(month3)} rows")

    # Save for use in other scripts
    reference.to_csv("reference_data.csv", index=False)
    month1.to_csv("month1_data.csv", index=False)
    month2.to_csv("month2_data.csv", index=False)
    month3.to_csv("month3_data.csv", index=False)

    print("\nData saved. Ready for drift analysis.")