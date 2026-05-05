#!/bin/bash
source venv/bin/activate
export MLFLOW_EXPERIMENT_NAME="dropout-analysis-v1"
python dropout-experiments/src/src/experiment.py