# Prediction of Hospital Emergency Department Attendance

## Description:

This repository serves as the practical foundation of my Bachelor's Thesis. It includes the setup instructions for the virtual environment and a series of commits covering data cleaning and the implementation of various predictive models, such as: ARIMA, Baseline model, Gradient Boosting, HistGradientBoosting, Random Forest.
The repository documents the full workflow used to forecast emergency department attendance based on historical hospital data.

## Detailed Folder Description
**Img folder** 

This folder contains the plots generated during the exploratory data analysis. These visualizations include:

* Detection of outliers

* Identification of impossible or inconsistent values

* Analysis of missing data

* Exploration of temporal patterns in patient inflow (by day, shift, hospital, etc.)

* These plots help assess data quality and guide decisions during the preprocessing and modeling phases.

**scr folder**

This folder contains the Python scripts that define the complete data pipeline. It includes:

* Data cleaning and preprocessing

* Grouping by day, shift, and triage level

* Generation of additional features like the rolling mean (rolling_mean.py), which is a crucial step before modeling

* Implementation of different prediction algorithms: ARIMA, baseline model, Gradient Boosting, HistGradientBoosting, and Random Forest

Each .py file is named after the model or task it performs, making it easy to follow and maintain the codebase.
