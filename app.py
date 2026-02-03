"""
Employee Survey Impact Analysis - Web Application

Analyzes how claimant (employee) survey scores impact:
- Employer (customer) satisfaction score
- Employer persistency
"""

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates', static_folder='static')
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
    if new_col in df.columns:
        return jsonify({'error': f'Column "{new_col}" already exists. Use a different source column.'}), 400

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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
