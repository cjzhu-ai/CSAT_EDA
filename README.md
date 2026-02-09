# EDA & Regression Web App

A web application for exploratory data analysis (EDA), regression modeling, and report generation from CSV data. Use it to explore relationships in tabular data, run linear or logistic regression with optional subgroup filters and imputation, and export results to PDF.

## Features

- **Data ingestion**: Upload CSV files; view column types, null counts, and a data preview.
- **Variable binning**: Create binned (grouped) variables from numeric columns with custom edges and labels for use in analysis.
- **Exploratory data analysis**: Summary statistics for numeric columns; correlation matrix heatmap; customizable visualizations (bar, line, scatter, box, histogram) with optional aggregation.
- **Regression modeling**:
  - **Variable selection**: Choose a target variable, key predictors (covariates), and control variables.
  - **Subgroup filters**: Restrict the sample by one or more columns (e.g., segment or region); optional “exclude NA” per filter.
  - **Missing data**: Optional imputation for predictors (median or mean for numeric, most frequent for categorical); target missing still drops rows.
  - **Model type**: Linear regression for continuous targets; logistic regression for binary targets (auto-detected).
  - **Standardization**: Key predictors and (for linear) the target are z-scored so coefficients are comparable; controls stay in original units.
  - **Output**: Coefficients, standard errors, p-values, 95% CIs, R²/accuracy, and a text interpretation of key and control predictors.
- **VIF**: Variance inflation factor for the same predictor set (and filters) to assess multicollinearity.
- **Relative importance** (linear regression only): Decompose R² into each predictor’s contribution using:
  - **Shapley value regression** (default): average incremental R² over random orderings.
  - **Johnson’s relative weights**: orthogonalization-based relative weights.
  - **Dominance analysis**: general dominance (average incremental R²).
- **Report export**: Generate a PDF with data preview, binned variables, correlation matrix, regression method description, coefficient table, and interpretation.

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. Open http://localhost:5000 in your browser.

## Planned Features

- **Step 1** (current): Project setup & minimal app ✓
- **Step 2**: CSV upload and data ingestion
- **Step 3**: Exploratory data analysis with visualizations
- **Step 4**: Regression modeling (linear/logistic)
- **Step 5**: Results interpretation and summary
