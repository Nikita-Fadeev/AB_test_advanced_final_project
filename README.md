# AB-Test Experiment Evaluation Service

### Project Overview

This project implements a high-performance experiment evaluation service for A/B testing analysis. The service determines whether to implement changes based on statistical analysis of user purchase data across control and experimental groups.

### Project Goal
Evaluate 1000 A/B tests to identify statistically significant effects on average revenue per customer during week 6 (days 36-42).

### 1. **Stratification**
- Pre-stratifies users based on historical purchase patterns
- Reduces variance between experimental groups
- Improves test sensitivity

### 2. **CUPED with Machine Learning Covariates**
- Uses machine learning model to predict expected revenue based on historical behavior
- Employs model predictions as optimal covariates in CUPED framework
- Significantly improves detection of true treatment effects
- Automatically selects the most informative covariates from user history

### Project Structure
├── solution.py          # Main Flask application

├── Develop.ipynb        # Development and analysis notebook

├── df_sales_public.csv  # Sales dataset

└── README.md           # Project documentation
