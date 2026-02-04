"""
Employee Survey Impact Analysis - Web Application

Analyzes how claimant (employee) survey scores impact:
- Employer (customer) satisfaction score
- Employer persistency
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import io
from datetime import datetime
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
import plotly.express as px
import plotly.graph_objects as go
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Accept CSV upload, parse with pandas, return metadata and preview."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    filename = secure_filename(file.filename)

    try:
        df = pd.read_csv(file)
        # Save for subsequent analysis steps
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df.to_csv(save_path, index=False)
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400

    if df.empty:
        return jsonify({'error': 'CSV file is empty'}), 400

    # Build column info (name, dtype, sample values)
    columns = [
        {
            'name': col,
            'dtype': str(df[col].dtype),
            'non_null': int(df[col].notna().sum()),
            'null_count': int(df[col].isna().sum()),
        }
        for col in df.columns
    ]

    # Preview: first 10 rows as list of dicts
    preview = df.head(10).fillna('').to_dict(orient='records')
    for row in preview:
        for k, v in row.items():
            if isinstance(v, (int, float)) and pd.isna(v):
                row[k] = ''
            elif isinstance(v, float):
                row[k] = round(v, 4) if v == v else ''  # handle NaN

    return jsonify({
        'success': True,
        'filename': filename,
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': columns,
        'preview': preview,
    })


def _get_numeric_columns(df):
    """Return list of column names that are numeric."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _get_distribution_columns(df):
    """Columns suitable for distribution: numeric + object/category (e.g. binned)."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric + cats


@app.route('/api/bin', methods=['POST'])
def bin_variable():
    """Bin a numeric column by specified ranges and save as new column."""
    data = request.get_json() or {}
    filename = data.get('filename')
    column = data.get('column')
    bins = data.get('bins')  # list of edges e.g. [0, 25, 50, 75, 100]
    labels = data.get('labels')  # optional list of labels

    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Valid filename required'}), 400
    if not column:
        return jsonify({'error': 'Column required'}), 400
    if not bins or not isinstance(bins, list) or len(bins) < 2:
        return jsonify({'error': 'At least 2 bin edges required (e.g. [0, 25, 50, 100])'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.isfile(path):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': f'Failed to load CSV: {str(e)}'}), 400

    if column not in df.columns:
        return jsonify({'error': f'Column "{column}" not found'}), 400
    if not np.issubdtype(df[column].dtype, np.number):
        return jsonify({'error': f'Column "{column}" must be numeric'}), 400

    try:
        bins_sorted = sorted([float(b) for b in bins])
    except (TypeError, ValueError):
        return jsonify({'error': 'Bin edges must be numbers'}), 400

    if labels is not None:
        if not isinstance(labels, list) or len(labels) != len(bins_sorted) - 1:
            return jsonify({'error': f'Labels must be a list with {len(bins_sorted) - 1} items'}), 400
    else:
        labels = [f'{bins_sorted[i]}-{bins_sorted[i+1]}' for i in range(len(bins_sorted) - 1)]

    new_col = f'{column}_binned'
    df[new_col] = pd.cut(df[column], bins=bins_sorted, labels=labels, include_lowest=True)
    df.to_csv(path, index=False)

    columns = [
        {'name': col, 'dtype': str(df[col].dtype), 'non_null': int(df[col].notna().sum()), 'null_count': int(df[col].isna().sum())}
        for col in df.columns
    ]
    preview = df.head(10).fillna('').to_dict(orient='records')
    for row in preview:
        for k, v in row.items():
            if isinstance(v, (int, float)) and pd.isna(v):
                row[k] = ''
            elif isinstance(v, float):
                row[k] = round(v, 4) if v == v else ''
            elif hasattr(v, '__str__'):
                row[k] = str(v)

    return jsonify({
        'success': True,
        'new_column': new_col,
        'columns': columns,
        'preview': preview,
        'row_count': len(df),
    })


@app.route('/api/eda', methods=['POST'])
def exploratory_analysis():
    """Run exploratory data analysis and return stats + Plotly charts."""
    data = request.get_json() or {}
    filename = data.get('filename')
    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Valid filename required'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.isfile(path):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': f'Failed to load CSV: {str(e)}'}), 400

    if df.empty:
        return jsonify({'error': 'CSV is empty'}), 400

    numeric_cols = _get_numeric_columns(df)
    dist_cols = _get_distribution_columns(df)
    result = {'success': True, 'filename': filename, 'distribution_columns': dist_cols}

    # Descriptive statistics (numeric only)
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        desc_dict = desc.to_dict()
        stats = []
        for col in numeric_cols:
            d = desc_dict[col]
            stats.append({
                'column': col,
                'count': d.get('count', 0),
                'mean': d.get('mean'),
                'std': d.get('std'),
                'min': d.get('min'),
                '25%': d.get('25%'),
                '50%': d.get('50%'),
                '75%': d.get('75%'),
                'max': d.get('max'),
            })
        result['stats'] = stats

        # Correlation heatmap (if 2+ numeric columns)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values.tolist(),
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale='Blues',
                text=np.round(corr.values, 2).tolist(),
                texttemplate='%{text}',
                textfont={'size': 10},
            ))
            fig.update_layout(
                title='Correlation matrix',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#94a3b8'},
                margin=dict(l=120, r=40, t=50, b=120),
                height=300 + len(numeric_cols) * 25,
                xaxis={'tickangle': -45},
            )
            result['correlation_chart'] = fig.to_json()
        else:
            result['correlation_chart'] = None
    else:
        result['stats'] = []
        result['correlation_chart'] = None

    return jsonify(result)


@app.route('/api/visualize', methods=['POST'])
def visualize():
    """Build a chart from user-selected chart type, x, y, color, and aggregation options."""
    data = request.get_json() or {}
    filename = data.get('filename')
    chart_type = data.get('chart_type', 'bar')
    x_var = data.get('x')
    y_var = data.get('y')
    color_var = data.get('color') or None
    aggregate = data.get('aggregate', False)
    agg_func = data.get('agg_func', 'mean')

    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Valid filename required'}), 400
    if not x_var:
        return jsonify({'error': 'X variable is required'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.isfile(path):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': f'Failed to load CSV: {str(e)}'}), 400

    cols_needed = [x_var]
    if y_var:
        cols_needed.append(y_var)
    if color_var:
        cols_needed.append(color_var)
    for c in cols_needed:
        if c not in df.columns:
            return jsonify({'error': f'Column "{c}" not found'}), 400

    # Apply aggregation if requested
    if aggregate and y_var and y_var in df.select_dtypes(include=[np.number]).columns:
        group_cols = [x_var]
        if color_var:
            group_cols.append(color_var)
        agg_map = {y_var: agg_func}
        plot_df = df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    else:
        plot_df = df[cols_needed].copy()

    def _to_list(series_or_arr):
        """Convert pandas/numpy to plain Python list for reliable JSON serialization."""
        if series_or_arr is None:
            return []
        if hasattr(series_or_arr, 'tolist'):
            return series_or_arr.tolist()
        return list(series_or_arr)

    # Build Plotly figure using go.Figure with explicit lists (avoids Plotly's binary encoding)
    fig = None
    try:
        if chart_type == 'bar':
            if y_var:
                if color_var:
                    traces = []
                    for val in plot_df[color_var].unique():
                        sub = plot_df[plot_df[color_var] == val]
                        traces.append(go.Bar(
                            name=str(val),
                            x=_to_list(sub[x_var]),
                            y=_to_list(sub[y_var]),
                        ))
                    fig = go.Figure(data=traces, layout=go.Layout(barmode='group'))
                    fig.update_layout(title=f'{y_var} by {x_var}')
                else:
                    fig = go.Figure(data=[go.Bar(x=_to_list(plot_df[x_var]), y=_to_list(plot_df[y_var]))])
                    fig.update_layout(title=f'{y_var} by {x_var}')
            else:
                if color_var:
                    cnt = plot_df.groupby([x_var, color_var]).size().reset_index(name='count')
                    traces = []
                    for val in cnt[color_var].unique():
                        sub = cnt[cnt[color_var] == val]
                        traces.append(go.Bar(
                            name=str(val),
                            x=_to_list(sub[x_var]),
                            y=_to_list(sub['count']),
                        ))
                    fig = go.Figure(data=traces, layout=go.Layout(barmode='group'))
                    fig.update_layout(title=f'Count of {x_var}')
                else:
                    cnt = plot_df[x_var].value_counts().reset_index()
                    cnt.columns = [x_var, 'count']
                    fig = go.Figure(data=[go.Bar(x=_to_list(cnt[x_var]), y=_to_list(cnt['count']))])
                    fig.update_layout(title=f'Count of {x_var}')
        elif chart_type == 'line':
            if not y_var:
                return jsonify({'error': 'Y variable required for line chart'}), 400
            if color_var:
                traces = []
                for val in plot_df[color_var].unique():
                    sub = plot_df[plot_df[color_var] == val]
                    traces.append(go.Scatter(x=_to_list(sub[x_var]), y=_to_list(sub[y_var]), mode='lines+markers', name=str(val)))
                fig = go.Figure(data=traces)
            else:
                fig = go.Figure(data=[go.Scatter(x=_to_list(plot_df[x_var]), y=_to_list(plot_df[y_var]), mode='lines+markers')])
            fig.update_layout(title=f'{y_var} by {x_var}')
        elif chart_type == 'scatter':
            if not y_var:
                return jsonify({'error': 'Y variable required for scatter chart'}), 400
            if color_var:
                traces = []
                for val in plot_df[color_var].unique():
                    sub = plot_df[plot_df[color_var] == val]
                    traces.append(go.Scatter(x=_to_list(sub[x_var]), y=_to_list(sub[y_var]), mode='markers', name=str(val)))
                fig = go.Figure(data=traces)
            else:
                fig = go.Figure(data=[go.Scatter(x=_to_list(plot_df[x_var]), y=_to_list(plot_df[y_var]), mode='markers')])
            fig.update_layout(title=f'{y_var} vs {x_var}')
        elif chart_type == 'box':
            if not y_var:
                return jsonify({'error': 'Y variable required for box plot'}), 400
            if color_var:
                traces = []
                for val in plot_df[color_var].unique():
                    sub = plot_df[plot_df[color_var] == val]
                    traces.append(go.Box(x=_to_list(sub[x_var]), y=_to_list(sub[y_var]), name=str(val)))
                fig = go.Figure(data=traces)
            else:
                fig = go.Figure(data=[go.Box(x=_to_list(plot_df[x_var]), y=_to_list(plot_df[y_var]))])
            fig.update_layout(title=f'{y_var} by {x_var}')
        elif chart_type == 'histogram':
            if y_var:
                if color_var:
                    traces = []
                    for val in plot_df[color_var].unique():
                        sub = plot_df[plot_df[color_var] == val]
                        traces.append(go.Histogram(x=_to_list(sub[x_var]), y=_to_list(sub[y_var]), name=str(val)))
                    fig = go.Figure(data=traces)
                else:
                    fig = go.Figure(data=[go.Histogram(x=_to_list(plot_df[x_var]), y=_to_list(plot_df[y_var]))])
                fig.update_layout(title=f'Distribution of {x_var} by {y_var}')
            else:
                if color_var:
                    traces = []
                    for val in plot_df[color_var].unique():
                        sub = plot_df[plot_df[color_var] == val]
                        traces.append(go.Histogram(x=_to_list(sub[x_var]), name=str(val)))
                    fig = go.Figure(data=traces)
                else:
                    fig = go.Figure(data=[go.Histogram(x=_to_list(plot_df[x_var]))])
                fig.update_layout(title=f'Distribution of {x_var}')
        else:
            return jsonify({'error': f'Unknown chart type: {chart_type}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#94a3b8'},
        margin=dict(l=50, r=40, t=50, b=80),
        height=380,
        xaxis={'tickangle': -45},
    )

    chart_dict = fig.to_dict()
    return jsonify({'success': True, 'chart': chart_dict})


def _is_binary_target(series):
    """True if series has exactly 2 unique non-null values (suitable for logistic)."""
    uniq = series.dropna().unique()
    return len(uniq) == 2


def _prepare_regression_data(df, target_col, control_cols):
    """
    Build X (design matrix) and y from df. One-hot encode categoricals, drop rows with missing target.
    Returns (X_df, y_series, feature_names) with feature_names list for coefficient labels.
    """
    use_cols = [target_col] + [c for c in control_cols if c != target_col]
    work = df[use_cols].copy()
    work = work.dropna(subset=[target_col])
    y = work[target_col]
    X_cols = [c for c in control_cols if c != target_col]
    if not X_cols:
        return None, y, []

    X_parts = []
    feature_names = []
    for c in X_cols:
        col = work[c]
        if col.dtype in (np.int64, np.int32, np.float64, np.float32) or (hasattr(col.dtype, 'kind') and col.dtype.kind in 'iufc'):
            X_parts.append(pd.DataFrame({c: col}))
            feature_names.append(c)
        else:
            dummies = pd.get_dummies(col.astype(str), prefix=c, drop_first=True)
            X_parts.append(dummies)
            feature_names.extend(dummies.columns.tolist())
    X_df = pd.concat(X_parts, axis=1)
    return X_df, y, feature_names


@app.route('/api/regression', methods=['POST'])
def run_regression():
    """
    Run linear or logistic regression. Predictors = covariates (key) + controls.
    Model type: numeric with 2 unique values → logistic, else → linear.
    Uses statsmodels for p-values.
    """
    data = request.get_json() or {}
    filename = data.get('filename')
    target = data.get('target')
    covariates = data.get('covariates') or []
    controls = data.get('controls') or []

    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Valid filename required'}), 400
    if not target:
        return jsonify({'error': 'Target variable is required'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.isfile(path):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': f'Failed to load CSV: {str(e)}'}), 400

    if target not in df.columns:
        return jsonify({'error': f'Target column "{target}" not found'}), 400

    cov_cols = [c for c in covariates if c in df.columns and c != target]
    control_cols = [c for c in controls if c in df.columns and c != target]
    all_pred_cols = list(dict.fromkeys(cov_cols + control_cols))
    if not all_pred_cols:
        return jsonify({'error': 'Select at least one covariate or control variable'}), 400

    all_pred_cols = [target] + all_pred_cols
    X_df, y_series, feature_names = _prepare_regression_data(df, target, all_pred_cols)
    if X_df is None or len(X_df) < 2:
        return jsonify({'error': 'Need at least one predictor and 2 valid rows'}), 400

    y = y_series.values
    X = X_df.values.astype(float)
    n_obs = len(y)

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    if len(y) < 2:
        return jsonify({'error': 'Too few rows after removing missing values'}), 400

    # Normalize (z-score) key predictor columns so coefficients are comparable for relative importance
    def _is_covariate_feature(fname, cov_list):
        if fname in cov_list:
            return True
        return any(str(fname).startswith(str(c) + '_') for c in cov_list)

    for j in range(X.shape[1]):
        if j < len(feature_names) and _is_covariate_feature(feature_names[j], cov_cols):
            col = X[:, j]
            mu, sig = col.mean(), col.std()
            if sig > 1e-10:
                X[:, j] = (col - mu) / sig

    X_const = sm.add_constant(X, has_constant='add')
    const_names = ['const'] + feature_names

    model_type = 'linear'
    if df[target].dtype in (object, 'object') or (hasattr(df[target].dtype, 'kind') and df[target].dtype.kind not in 'iufc'):
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        if len(le.classes_) != 2:
            return jsonify({'error': 'For logistic regression target must have exactly 2 categories.'}), 400
        model_type = 'logistic'
        y = y_enc.astype(float)
    else:
        y = y.astype(float)
        if _is_binary_target(pd.Series(y)):
            model_type = 'logistic'
            y = y.astype(int)

    result = {
        'success': True,
        'model_type': model_type,
        'target': target,
        'n_obs': int(len(y)),
        'feature_names': feature_names,
        'covariate_names': cov_cols,
        'covariates_standardized': True,
    }

    def _param(p, i):
        return float(np.asarray(p)[i])

    try:
        if model_type == 'linear':
            model = sm.OLS(y, X_const).fit()
            result['r2'] = float(model.rsquared)
            result['rmse'] = float(np.sqrt(model.mse_resid))
            result['intercept'] = _param(model.params, 0)
            result['coefficients'] = [_param(model.params, i) for i in range(1, len(model.params))]
            result['coefficient_labels'] = feature_names
            result['pvalues'] = [_param(model.pvalues, i) for i in range(len(model.pvalues))]
            result['pvalue_labels'] = const_names
            result['bse'] = [_param(model.bse, i) for i in range(len(model.bse))]
            ci = np.asarray(model.conf_int(alpha=0.05))
            result['ci_lower'] = [float(ci[i, 0]) for i in range(len(const_names))]
            result['ci_upper'] = [float(ci[i, 1]) for i in range(len(const_names))]
        else:
            model = Logit(y, X_const).fit(disp=0)
            result['accuracy'] = float((model.predict(X_const).round() == y).mean())
            result['intercept'] = _param(model.params, 0)
            result['coefficients'] = [_param(model.params, i) for i in range(1, len(model.params))]
            result['coefficient_labels'] = feature_names
            result['pvalues'] = [_param(model.pvalues, i) for i in range(len(model.pvalues))]
            result['pvalue_labels'] = const_names
            result['bse'] = [_param(model.bse, i) for i in range(len(model.bse))]
            ci = np.asarray(model.conf_int(alpha=0.05))
            result['ci_lower'] = [float(ci[i, 0]) for i in range(len(const_names))]
            result['ci_upper'] = [float(ci[i, 1]) for i in range(len(const_names))]
    except Exception as e:
        return jsonify({'error': f'Model failed: {str(e)}'}), 400

    return jsonify(result)


@app.route('/api/vif', methods=['POST'])
def run_vif():
    """Compute VIF for the same predictor set as regression (no standardization)."""
    data = request.get_json() or {}
    filename = data.get('filename')
    target = data.get('target')
    covariates = data.get('covariates') or []
    controls = data.get('controls') or []

    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Valid filename required'}), 400
    if not target:
        return jsonify({'error': 'Target variable is required'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.isfile(path):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': f'Failed to load CSV: {str(e)}'}), 400

    if target not in df.columns:
        return jsonify({'error': f'Target column "{target}" not found'}), 400

    cov_cols = [c for c in covariates if c in df.columns and c != target]
    control_cols = [c for c in controls if c in df.columns and c != target]
    all_pred_cols = list(dict.fromkeys(cov_cols + control_cols))
    if not all_pred_cols:
        return jsonify({'error': 'Select at least one covariate or control variable'}), 400

    all_pred_cols = [target] + all_pred_cols
    X_df, y_series, feature_names = _prepare_regression_data(df, target, all_pred_cols)
    if X_df is None or len(X_df) < 2:
        return jsonify({'error': 'Need at least one predictor'}), 400

    X = X_df.values.astype(float)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    if len(X) < 2:
        return jsonify({'error': 'Too few rows after removing missing values'}), 400

    X_const = sm.add_constant(X, has_constant='add')
    vif_values = []
    for j in range(1, X_const.shape[1]):
        try:
            v = float(variance_inflation_factor(X_const, j))
            vif_values.append(v if np.isfinite(v) else None)
        except Exception:
            vif_values.append(None)

    return jsonify({
        'success': True,
        'feature_names': feature_names,
        'vif': vif_values,
    })


def _build_interpretation_text(reg):
    """Build interpretation paragraph text from regression result dict."""
    if not reg or not reg.get('coefficient_labels'):
        return ''
    labels = reg.get('coefficient_labels', [])
    coefs = reg.get('coefficients', [])
    pvalues = reg.get('pvalues', [])
    pvalue_labels = reg.get('pvalue_labels', [])
    cov_names = set(reg.get('covariate_names', []))
    target = reg.get('target', 'target')
    model_type = reg.get('model_type', 'linear')

    def is_cov(name):
        return name in cov_names or any(str(name).startswith(str(c) + '_') for c in cov_names)

    lines = []
    lines.append('Key predictors (standardized) were ranked by relative importance (|β|). ')
    covariate_features = [(labels[i], coefs[i], pvalues[i + 1] if i + 1 < len(pvalues) else None) for i in range(len(labels)) if is_cov(labels[i])]
    covariate_features.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, coef, p in covariate_features:
        sig = ' (p < 0.05)' if p is not None and p < 0.05 else ' (not significant)'
        lines.append('{}: β = {:.4f}{}. '.format(name, coef, sig))
    control_features = [(labels[i], coefs[i], pvalues[i + 1] if i + 1 < len(pvalues) else None) for i in range(len(labels)) if not is_cov(labels[i])]
    if control_features:
        sig_controls = [x[0] for x in control_features if x[2] is not None and x[2] < 0.05]
        if sig_controls:
            lines.append('Among control variables, {} were significant. '.format(', '.join(sig_controls)))
        else:
            lines.append('No control variables were significant at p < 0.05.')
    return ' '.join(lines)


@app.route('/api/export-report', methods=['POST', 'OPTIONS'])
def export_report():
    """Generate a PDF report: data preview, binned variables, correlation, method overview, regression output + interpretation."""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json(silent=True) or {}
    filename = data.get('filename')
    binned_columns = data.get('binned_columns') or []
    binned_metadata = data.get('binned_metadata') or {}  # { "col_binned": { "edges": [...], "labels": [...] } }
    regression_result = data.get('regression_result')

    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Valid filename required'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.isfile(path):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        story = []
        # Table width to fit within margins (letter width 8.5in, margins 1.5in total)
        table_width_pt = (8.5 - 1.5) * 72
        table_font_size = 6

        # Title and timestamp
        story.append(Paragraph('CSAT Regression – Analysis Report', styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        story.append(Paragraph('Analysis timestamp: {}'.format(ts), styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        # 1. Data preview – fit table to page, smaller font, truncate cells
        story.append(Paragraph('1. Data preview', styles['Heading2']))
        preview_df = df.head(20)
        ncols = len(preview_df.columns)
        col_width_pt = max(20, table_width_pt / ncols) if ncols else table_width_pt
        col_widths = [col_width_pt] * ncols
        cell_max = 10
        preview_data = [[str(c)[:cell_max] for c in preview_df.columns]]
        for _, row in preview_df.iterrows():
            preview_data.append([str(row[c])[:cell_max] if pd.notna(row[c]) else '' for c in preview_df.columns])
        t = Table(preview_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), table_font_size),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.25*inch))

        # 2. Variables created / binned (include bin edges and labels when available)
        story.append(Paragraph('2. Variables created / binned', styles['Heading2']))
        if binned_columns:
            for col in binned_columns:
                meta = binned_metadata.get(col) if isinstance(binned_metadata, dict) else None
                if meta and isinstance(meta, dict):
                    edges = meta.get('edges', [])
                    labels = meta.get('labels', [])
                    edges_str = ', '.join(str(x) for x in edges) if edges else '—'
                    labels_str = ', '.join(str(x) for x in labels) if labels else '—'
                    story.append(Paragraph('• <b>{}</b>: edges [{}]; labels [{}].'.format(col, edges_str, labels_str), styles['Normal']))
                else:
                    story.append(Paragraph('• {}'.format(col), styles['Normal']))
        else:
            story.append(Paragraph('None.', styles['Normal']))
        story.append(Spacer(1, 0.25*inch))

        # 3. Correlation matrix – fit to page, smaller font
        story.append(Paragraph('3. Correlation matrix', styles['Heading2']))
        numeric_cols = _get_numeric_columns(df)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            corr_arr = np.asarray(corr)
            ncorr = len(corr.columns) + 1
            cw = max(18, table_width_pt / ncorr)
            col_widths_corr = [cw] * ncorr
            corr_data = [[''] + [str(c)[:12] for c in corr.columns]]
            for i, r in enumerate(corr.index):
                row_vals = [str(round(float(corr_arr[i, j]), 2)) if np.isfinite(corr_arr[i, j]) else '' for j in range(len(corr.columns))]
                corr_data.append([str(r)[:12]] + row_vals)
            t = Table(corr_data, colWidths=col_widths_corr, repeatRows=1)
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), table_font_size),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)
        else:
            story.append(Paragraph('Fewer than 2 numeric columns; correlation matrix omitted.', styles['Normal']))
        story.append(Spacer(1, 0.25*inch))

        # 4. Regression method overview
        story.append(Paragraph('4. Regression analysis method', styles['Heading2']))
        method_text = 'Regression was run with target and predictor variables. Key predictors (covariates) were standardized (z-score) so coefficients are per 1 SD increase and comparable for relative importance. Control variables were included in original units. Model type was chosen from the target: linear regression for continuous targets, logistic regression for binary targets.'
        if regression_result:
            model_type = regression_result.get('model_type', 'linear')
            target = regression_result.get('target', '')
            n_obs = regression_result.get('n_obs', 0)
            method_text = '{} regression of {} (N = {}). Key predictors were standardized (1 SD unit); control variables in original units.'.format(model_type.capitalize(), target, n_obs) + ' ' + method_text
        story.append(Paragraph(method_text, styles['Normal']))
        story.append(Spacer(1, 0.25*inch))

        # 5. Regression output + interpretation
        story.append(Paragraph('5. Regression output and interpretation', styles['Heading2']))
        if regression_result:
            labels = regression_result.get('coefficient_labels', [])
            coefs = regression_result.get('coefficients', [])
            pvalues = regression_result.get('pvalues', [])
            bse = regression_result.get('bse', [])
            ci_lower = regression_result.get('ci_lower', [])
            ci_upper = regression_result.get('ci_upper', [])
            const_names = regression_result.get('pvalue_labels', ['const'] + labels)

            coef_data = [['Variable', 'Coef', 'SE', 'p-value', '95% CI']]
            coef_data.append(['(Intercept)', _fmt(regression_result.get('intercept')), _fmt(bse[0] if len(bse) > 0 else None), _fmt_p(pvalues[0] if len(pvalues) > 0 else None), _fmt_ci(ci_lower[0] if len(ci_lower) > 0 else None, ci_upper[0] if len(ci_upper) > 0 else None)])
            for i in range(len(labels)):
                p = pvalues[i + 1] if i + 1 < len(pvalues) else None
                se = bse[i + 1] if i + 1 < len(bse) else None
                lo = ci_lower[i + 1] if i + 1 < len(ci_lower) else None
                hi = ci_upper[i + 1] if i + 1 < len(ci_upper) else None
                coef_data.append([str(labels[i])[:25], _fmt(coefs[i]), _fmt(se), _fmt_p(p), _fmt_ci(lo, hi)])
            cw5 = table_width_pt / 5
            col_widths_coef = [cw5 * 1.8, cw5 * 0.9, cw5 * 0.9, cw5 * 0.9, cw5 * 1.3]
            t = Table(coef_data, colWidths=col_widths_coef, repeatRows=1)
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), table_font_size),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.15*inch))
            if regression_result.get('model_type') == 'linear':
                story.append(Paragraph('R² = {}; RMSE = {}.'.format(_fmt(regression_result.get('r2')), _fmt(regression_result.get('rmse'))), styles['Normal']))
            else:
                story.append(Paragraph('Accuracy = {}.'.format(_fmt(regression_result.get('accuracy'))), styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
            interp = _build_interpretation_text(regression_result)
            if interp:
                story.append(Paragraph('<b>Interpretation:</b> {}'.format(interp), styles['Normal']))
        else:
            story.append(Paragraph('No regression was run for this report. Run regression in the app and export again to include results.', styles['Normal']))

        doc.build(story)
        buf.seek(0)
        fname = 'CSAT_Regression_Report_{}.pdf'.format(datetime.utcnow().strftime('%Y%m%d_%H%M'))
        return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name=fname)
    except Exception as e:
        return jsonify({'error': 'PDF generation failed: {}'.format(str(e))}), 500


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    if isinstance(v, float):
        return '{:.4f}'.format(v)
    return str(v)


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '—'
    if p < 0.001:
        return '<0.001'
    return '{:.4f}'.format(p)


def _fmt_ci(lo, hi):
    if lo is None or hi is None:
        return '—'
    return '[{:.3f}, {:.3f}]'.format(lo, hi)


if __name__ == '__main__':
    # Use port 8000: macOS Monterey+ reserves 5000/7000 for AirPlay (causes 403 if Flask uses 5000)
    app.run(debug=True, host='127.0.0.1', port=8000)
