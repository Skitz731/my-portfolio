# MLOps Pipeline Project

## Overview

This project implements a complete MLOps pipeline for heart disease prediction, 
including version control with Git and DVC, experiment tracking with MLflow, 
automated testing with pytest, CI/CD with GitHub Actions, and drift monitoring with Evidently.

## Project Structure




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