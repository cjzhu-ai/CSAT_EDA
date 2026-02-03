# Employee Survey Impact Analysis

A web application to analyze how employee (claimant) survey scores impact:
- **Employer satisfaction score** – overall customer satisfaction with the insurance company
- **Employer persistency** – whether employers retain their insurance coverage

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
