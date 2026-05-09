"""
Compare Experiments Script
Queries MLflow to find the best performing run based on a specified metric.
"""

import mlflow
import pandas as pd
import argparse
import sys

def compare_experiments(
    experiment_name: str = "Employee Attrition Prediction",
    metric: str = "accuracy",
    top_n: int = 5
):
    """
    Query MLflow for the top N runs based on a specific metric.
    
    Args:
        experiment_name: Name of the MLflow experiment to query
        metric: The metric to sort by (e.g., 'accuracy', 'f1_score')
        top_n: Number of top runs to display
    """
    
    print(f"Connecting to MLflow...")
    
    # Set the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Error: Experiment '{experiment_name}' not found.")
        print("Make sure you have run the training script at least once.")
        sys.exit(1)
    
    experiment_id = experiment.experiment_id
    
    # Search for all runs in the experiment
    # We fetch all runs to sort them locally
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    if runs.empty:
        print(f"Error: No runs found in experiment '{experiment_name}'.")
        sys.exit(1)
    
    print(f"Found {len(runs)} runs in experiment '{experiment_name}'.")
    
    # Check if the requested metric exists in the runs
    if metric not in runs.columns:
        available_metrics = [col for col in runs.columns if col.startswith("metrics.")]
        print(f"Error: Metric '{metric}' not found in runs.")
        print(f"Available metrics: {available_metrics}")
        sys.exit(1)
    
    # Sort by the metric (descending order)
    runs_sorted = runs.sort_values(by=f"metrics.{metric}", ascending=False).head(top_n)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"TOP {top_n} RUNS BY '{metric}'")
    print(f"{'='*60}")
    
    # Prepare display columns
    display_cols = ["run_id", "status", f"metrics.{metric}"]
    
    # Add relevant parameters to the display
    param_cols = [col for col in runs.columns if col.startswith("params.")]
    # Filter for common model params
    relevant_params = [c for c in param_cols if any(k in c for k in ['n_estimators', 'max_depth', 'min_samples'])]
    display_cols.extend(relevant_params)
    
    # Create a clean dataframe for display
    display_df = runs_sorted[display_cols].copy()
    
    # Rename columns for readability
    display_df.columns = [col.replace("metrics.", "").replace("params.", "") for col in display_df.columns]
    
    # Reset index to make it cleaner
    display_df = display_df.reset_index(drop=True)
    
    # Print the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(display_df.to_string(index=False))
    
    # Identify the best run
    best_run = runs_sorted.iloc[0]
    best_run_id = best_run["run_id"]
    best_score = best_run[f"metrics.{metric}"]
    
    print(f"\n{'='*60}")
    print(f"BEST RUN DETAILS")
    print(f"{'='*60}")
    print(f"Run ID: {best_run_id}")
    print(f"Metric ({metric}): {best_score:.4f}")
    print(f"Status: {best_run['status']}")
    
    # Print all parameters of the best run
    print("\nParameters:")
    for col in param_cols:
        if not pd.isna(best_run[col]):
            param_name = col.replace("params.", "")
            print(f"  {param_name}: {best_run[col]}")
            
    print("\nMetrics:")
    for col in runs.columns:
        if col.startswith("metrics."):
            metric_name = col.replace("metrics.", "")
            if not pd.isna(best_run[col]):
                print(f"  {metric_name}: {best_run[col]}")