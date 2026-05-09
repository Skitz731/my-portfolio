# MLOps Pipeline: Employee Attrition Prediction

A production-ready Machine Learning Operations (MLOps) pipeline demonstrating end-to-end lifecycle management for an Employee Attrition prediction model. This project integrates version control, experiment tracking, automated testing, CI/CD, and drift monitoring.

## Project Overview

This project takes a raw dataset of employee attributes and transforms it into a monitored, automated ML system. The core objectives are:
- **Reproducibility**: Using Git for code and DVC for data versioning.
- **Experiment Tracking**: Logging hyperparameters, metrics, and models with MLflow.
- **Quality Assurance**: Comprehensive testing with `pytest` (unit, data, and model validation).
- **Automation**: CI/CD pipelines via GitHub Actions for testing and training.
- **Monitoring**: Detecting data drift in production using Evidently.

**Dataset**: IBM HR Analytics Employee Attrition Dataset (~1,470 rows, 35 features).
**Task**: Binary Classification (Predicting if an employee will leave: Yes/No).

##  Project Structure
mlops-pipeline/ ├── .dvc/ # DVC configuration and cache pointers ├── .github/ │ └── workflows/ │ └── ci-cd.yml # GitHub Actions CI/CD pipeline (Test & Train) ├── configs/ │ └── training_config.yaml # Hyperparameters, paths, and thresholds ├── data/ │ ├── raw/ # Raw dataset (tracked by DVC, not Git) │ │ └── employee_attrition.csv.dvc │ └── processed/ # Preprocessed splits (optional DVC tracking) ├── reports/ # Generated Evidently drift reports (HTML) ├── src/ │ ├── init.py │ ├── data_preprocessing.py # Data loading, cleaning, encoding, splitting │ ├── model_training.py # Training script with MLflow integration │ └── monitor_drift.py # Drift detection script using Evidently ├── tests/ │ ├── init.py │ ├── test_preprocessing.py # Unit tests for data functions │ ├── test_data_validation.py # Tests for dataset integrity │ └── test_model_validation.py # Tests for model output and performance ├── compare_experiments.py # Script to query MLflow and find best run ├── requirements.txt # Python dependencies ├── .gitignore # Git ignore rules (data, venv, artifacts) ├── .dvcignore # DVC ignore rules ├── README.md # This file └── MONITORING.md # Drift analysis and recommendations

# Drift Monitoring Analysis

## 1. Which features showed drift and why?
In this simulation, we introduced drift by:
- **Numeric Features**: Shifting the mean of features like `Age`, `MonthlyIncome`, and `DistanceFromHome` by 15% of their standard deviation.
- **Target Variable**: Increasing the proportion of "Yes" (Attrition) from ~16% to ~31%.
Consequently, features with the highest drift share were those with the largest standard deviations and the target variable itself.

## 2. Would this drift likely affect model performance?
Yes. 
- **Feature Drift**: If the distribution of input features (e.g., Age) shifts significantly, the model's predictions may become biased because it was trained on the old distribution.
- **Target Drift**: A significant increase in the attrition rate implies the business environment has changed. The model, trained on a lower attrition rate, will likely under-predict attrition, leading to missed retention opportunities.

## 3. What action would you recommend?
- **If Drift < Threshold**: Continue monitoring. The model is still performing within acceptable bounds.
- **If Drift > Threshold**: 
  1. **Investigate**: Check if the data pipeline has changed or if there's a real business event (e.g., layoffs, policy change).
  2. **Retrain**: If the drift is real and persistent, retrain the model with the new production data to adapt to the new distribution.
  3. **Alert**: Notify stakeholders about the potential drop in model reliability.
