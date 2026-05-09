"""
Drift Monitoring Script using Evidently
Compares reference data against simulated production data to detect drift.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import yaml

try:
    try:
        from evidently.report import Report # type: ignore
    except ImportError:
        from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    print("Evidently library is not installed. Please install it using 'pip install evidently'")
    sys.exit(1)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_data, load_config

def generate_production_data(reference_df: pd.DataFrame, drift_factor: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Simulate production data with some drift.
    
    Args:
        reference_df: The original training/reference dataframe.
        drift_factor: How much to shift the data (0.0 to 1.0).
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with simulated drift.
    """
    np.random.seed(seed)
    prod_df = reference_df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = prod_df.select_dtypes(include=[np.number]).columns
    categorical_cols = prod_df.select_dtypes(include=['object']).columns
    
    # Apply drift to numeric columns (shift mean)
    for col in numeric_cols:
        if col != 'Attrition': # Skip target if it's numeric
            # Shift the mean by a percentage of the standard deviation
            shift = np.random.normal(0, 1) * drift_factor * prod_df[col].std()
            prod_df[col] = prod_df[col] + shift
            
            # Clip to reasonable bounds if necessary (optional)
            # prod_df[col] = prod_df[col].clip(lower=prod_df[col].quantile(0.01), upper=prod_df[col].quantile(0.99))

    # Apply drift to categorical columns (change proportions)
    for col in categorical_cols:
        if col == 'Attrition':
            # Change the target distribution slightly (e.g., more attrition)
            current_yes_ratio = (prod_df[col] == 'Yes').mean()
            new_yes_ratio = min(current_yes_ratio + drift_factor, 0.8) # Cap at 80%
            
            # Resample to match new ratio
            n_yes = int(len(prod_df) * new_yes_ratio)
            n_no = len(prod_df) - n_yes
            
            new_values = ['Yes'] * n_yes + ['No'] * n_no
            np.random.shuffle(new_values)
            prod_df[col] = new_values
        else:
            # Randomly shuffle a portion of the column to introduce noise/drift
            if np.random.rand() < drift_factor:
                unique_vals = prod_df[col].unique()
                # Replace a subset of values with a random choice from unique values
                mask = np.random.rand(len(prod_df)) < drift_factor
                prod_df.loc[mask, col] = np.random.choice(unique_vals, size=mask.sum())


    return prod_df

def run_drift_monitoring(
    config_path: str = "configs/training_config.yaml",
    drift_threshold: float = 0.2,
    output_dir: str = "reports",
    generate_simulated: bool = True
):
    """
    Run drift monitoring and generate a report.
    
    Args:
        config_path: Path to config file.
        drift_threshold: Maximum allowed drift share before failing.
        output_dir: Directory to save the HTML report.
        generate_simulated: If True, generate simulated production data. If False, load from a file.
    """
    
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Load Reference Data (Training Data)
    ref_path = config['data']['raw_path']
    print(f"Loading reference data from {ref_path}...")
    reference_df = load_data(ref_path)
    
    # Prepare Production Data
    if generate_simulated:
        print("Generating simulated production data with drift...")
        # Simulate drift: 10% shift in numeric features, change in target distribution
        production_df = generate_production_data(reference_df, drift_factor=0.15)
    else:
        # Load from a specific file if provided (e.g., new incoming data)
        prod_path = config.get('data', {}).get('production_path')
        if not prod_path or not os.path.exists(prod_path):
            print("ERROR: Production path not found or generate_simulated is False.")
            sys.exit(1)
        print(f"Loading production data from {prod_path}...")
        production_df = load_data(prod_path)
    
    # Ensure target column is consistent
    target_col = config['data']['target_column']
    
    # Define the columns to monitor (exclude the target if you only care about feature drift)
    # Usually, we monitor features, not the target, for data drift.
    features_to_monitor = [col for col in reference_df.columns if col != target_col]
    
    print(f"Monitoring {len(features_to_monitor)} features for drift...")
    
    # Create the Evidently Report
    # We use the DataDriftPreset which includes many useful metrics
    report = Report(metrics=[
        DataDriftPreset(),
        # Optionally add specific column drift metrics if needed
        # ColumnDriftMetric(column_name=col) for col in features_to_monitor
    ])
    
    # Run the report
    print("Running Evidently drift detection...")
    report.run(
        reference_data=reference_df[features_to_monitor],
        current_data=production_df[features_to_monitor]
    )
    
    # Save the report
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
    
    print(f"Saving report to {report_path}...")
    report.save_html(report_path) # type: ignore
    
    # Extract actual drift results from the report
    report_result = report.as_dict() # type: ignore
    
    # DataDriftPreset contains DataDriftTable metric
    # We look for the 'share_of_drifted_columns' in the result
    actual_drift_share = 0.0
    for metric in report_result['metrics']:
        if metric['metric'] == 'DataDriftTable':
            actual_drift_share = metric['result']['share_of_drifted_columns']
            break

    print("\n--- Drift Analysis Summary ---")
    print(f"Reference rows: {len(reference_df)}")
    print(f"Production rows: {len(production_df)}")
    print(f"Report saved to: {report_path}")
    
    print(f"Estimated Drift Share: {actual_drift_share:.2%}")
    print(f"Threshold: {drift_threshold:.2%}")
    
    if actual_drift_share > drift_threshold:
        print(f"ALERT: Drift ({actual_drift_share:.2%}) exceeds threshold ({drift_threshold:.2%})!")
        print("Recommendation: Retrain model or investigate data source.")
        sys.exit(1)
    else:
        print(f"OK: Drift ({actual_drift_share:.2%}) is within acceptable limits.")
        print("Recommendation: Continue monitoring.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Monitor data drift using Evidently")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config file")
    parser.add_argument("--threshold", type=float, default=0.2, help="Drift threshold (0.0 to 1.0)")
    parser.add_argument("--output", type=str, default="reports", help="Output directory for reports")
    parser.add_argument("--simulate", action="store_true", default=True, help="Simulate production data (default: True)")
    parser.add_argument("--real-data", type=str, default=None, help="Path to real production data (overrides simulation)")
    
    args = parser.parse_args()
    
    try:
        run_drift_monitoring(
            config_path=args.config,
            drift_threshold=args.threshold,
            output_dir=args.output,
            generate_simulated=args.simulate and not args.real_data
        )
    except Exception as e:
        print(f"Drift monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()