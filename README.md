# AB-Test Experiment Evaluation Service

### Project Overview

This project implements a high-performance experiment evaluation service for A/B testing analysis. The service determines whether to implement changes based on statistical analysis of user purchase data across control and experimental groups.

### Project Goal
Evaluate 1000 A/B tests to identify statistically significant effects on average revenue per customer during week 6 (days 36-42).

### 1. **Stratification**
- Pre-stratifies users based on historical purchase patterns
- Reduces variance between experimental groups
- Improves test sensitivity

### 2. **CUPED (Controlled Experiment Using Pre-Experiment Data)**
- Uses pre-experiment covariates to reduce variance
- Correlates current outcomes with historical behavior
- Significantly increases statistical power

### 3. **Machine Learning Covariate Adjustment**
- Implements simple predictive models to estimate expected revenue
- Uses model predictions as covariates in statistical tests
- Enhances detection of true treatment effects

### Project Structure
├── solution.py          # Main Flask application
├── Develop.ipynb        # Development and analysis notebook
├── df_sales_public.csv  # Sales dataset
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
