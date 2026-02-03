/**
 * Employee Survey Impact Analysis - Client-side logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const contentSection = document.getElementById('contentSection');
    let currentFilename = null;
    let currentData = null;

    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());

    // File selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (file) handleFile(file);
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const file = e.dataTransfer?.files?.[0];
        if (file && file.name.endsWith('.csv')) {
            handleFile(file);
        }
    });

    async function handleFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        uploadZone.querySelector('p').textContent = 'Uploading...';

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            currentFilename = data.filename;
            currentData = data;
            renderDataPreview(data);
            contentSection.style.display = 'block';
            contentSection.scrollIntoView({ behavior: 'smooth' });
            loadEDA(data.filename);
        } catch (err) {
            alert(err.message || 'Upload failed');
        } finally {
            uploadZone.querySelector('p').textContent =
                'Drag and drop your CSV file here, or click to browse';
        }
    }

    function getNumericColumns() {
        if (!currentData?.columns) return [];
        return currentData.columns
            .filter(c => ['int64', 'float64', 'int32', 'float32'].includes(c.dtype))
            .map(c => c.name);
    }

    function renderDataPreview(data) {
        const numericCols = data.columns.filter(c =>
            ['int64', 'float64', 'int32', 'float32'].includes(c.dtype)
        ).map(c => c.name);

        const binSectionHtml = numericCols.length > 0 ? `
            <div id="binSection" class="bin-section">
                <h3>Bin variables</h3>
                <p>Create grouped columns by specifying bin edges. The new column is saved for future analysis.</p>
                <div class="bin-form">
                    <label>Column <select id="binColumn">${numericCols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')}</select></label>
                    <label>Bin edges (comma-separated, e.g. 0,25,50,75,100) <input type="text" id="binEdges" placeholder="0,25,50,75,100"></label>
                    <label>Labels (optional, comma-separated) <input type="text" id="binLabels" placeholder="Low,Medium-Low,Medium-High,High"></label>
                    <button type="button" id="binApply">Apply binning</button>
                </div>
                <p id="binMessage" class="bin-message"></p>
            </div>
        ` : '';

        contentSection.innerHTML = `
            <h2>Data loaded: ${escapeHtml(data.filename)}</h2>
            <p>${data.row_count} rows × ${data.column_count} columns</p>
            ${binSectionHtml}
            <div class="data-info">
                <h3>Column summary</h3>
                <table class="column-table">
                    <thead><tr><th>Column</th><th>Type</th><th>Non-null</th><th>Null</th></tr></thead>
                    <tbody>
                        ${data.columns.map(c =>
                            `<tr><td>${escapeHtml(c.name)}</td><td>${escapeHtml(c.dtype)}</td>
                             <td>${c.non_null}</td><td>${c.null_count}</td></tr>`
                        ).join('')}
                    </tbody>
                </table>
                <h3>Preview (first 10 rows)</h3>
                <div class="preview-table-wrapper">
                    <table class="preview-table">
                        <thead><tr>${data.columns.map(c => `<th>${escapeHtml(c.name)}</th>`).join('')}</tr></thead>
                        <tbody>
                            ${data.preview.map(row =>
                                `<tr>${data.columns.map(c => `<td>${escapeHtml(String(row[c.name] ?? ''))}</td>`).join('')}</tr>`
                            ).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
            <div id="edaSection" class="eda-section">
                <h3>Exploratory Data Analysis</h3>
                <p class="eda-loading">Loading EDA...</p>
            </div>
        `;

        const binApply = document.getElementById('binApply');
        if (binApply) binApply.addEventListener('click', applyBinning);
    }

    async function applyBinning() {
        const col = document.getElementById('binColumn')?.value;
        const edgesStr = document.getElementById('binEdges')?.value?.trim();
        const labelsStr = document.getElementById('binLabels')?.value?.trim();
        const msgEl = document.getElementById('binMessage');
        if (!msgEl) return;

        if (!currentFilename || !col || !edgesStr) {
            msgEl.textContent = 'Select a column and enter bin edges.';
            msgEl.className = 'bin-message bin-error';
            return;
        }

        const bins = edgesStr.split(',').map(s => s.trim()).filter(Boolean).map(Number);
        if (bins.length < 2 || bins.some(isNaN)) {
            msgEl.textContent = 'Bin edges must be at least 2 numbers (e.g. 0,25,50,100).';
            msgEl.className = 'bin-message bin-error';
            return;
        }

        const payload = { filename: currentFilename, column: col, bins };
        if (labelsStr) {
            payload.labels = labelsStr.split(',').map(s => s.trim()).filter(Boolean);
            if (payload.labels.length !== bins.length - 1) {
                msgEl.textContent = `Labels must be ${bins.length - 1} items (one per bin).`;
                msgEl.className = 'bin-message bin-error';
                return;
            }
        }

        msgEl.textContent = 'Applying...';
        msgEl.className = 'bin-message';

        try {
            const res = await fetch('/api/bin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Binning failed');
            }

            currentData = { ...currentData, columns: data.columns, preview: data.preview, row_count: data.row_count };
            msgEl.textContent = `Created column "${data.new_column}".`;
            msgEl.className = 'bin-message bin-success';

            // Refresh column summary and preview
            const infoEl = contentSection.querySelector('.data-info');
            if (infoEl) {
                const tbody = infoEl.querySelector('.column-table tbody');
                const previewWrapper = infoEl.querySelector('.preview-table-wrapper');
                if (tbody) tbody.innerHTML = data.columns.map(c =>
                    `<tr><td>${escapeHtml(c.name)}</td><td>${escapeHtml(c.dtype)}</td><td>${c.non_null}</td><td>${c.null_count}</td></tr>`
                ).join('');
                if (previewWrapper) {
                    previewWrapper.innerHTML = `
                        <table class="preview-table">
                            <thead><tr>${data.columns.map(c => `<th>${escapeHtml(c.name)}</th>`).join('')}</tr></thead>
                            <tbody>${data.preview.map(row =>
                                `<tr>${data.columns.map(c => `<td>${escapeHtml(String(row[c.name] ?? ''))}</td>`).join('')}</tr>`
                            ).join('')}</tbody>
                        </table>
                    `;
                }
            }

            // Refresh bin dropdown with new numeric columns
            const numericCols = data.columns.filter(c =>
                ['int64', 'float64', 'int32', 'float32'].includes(c.dtype)
            ).map(c => c.name);
            const binColSelect = document.getElementById('binColumn');
            if (binColSelect) {
                binColSelect.innerHTML = numericCols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
            }

            // Refresh EDA (includes distribution columns) and distribution dropdown
            loadEDA(currentFilename);
        } catch (err) {
            msgEl.textContent = err.message || 'Binning failed';
            msgEl.className = 'bin-message bin-error';
        }
    }

    async function loadEDA(filename) {
        const edaEl = document.getElementById('edaSection');
        if (!edaEl) return;

        try {
            const res = await fetch('/api/eda', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename }),
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'EDA failed');
            }

            renderEDA(data, edaEl);
        } catch (err) {
            edaEl.querySelector('.eda-loading').textContent = 'EDA failed: ' + err.message;
        }
    }

    async function loadVisualization() {
        const container = document.getElementById('vizChartContainer');
        if (!container || !currentFilename) return;

        const chartType = document.getElementById('vizChartType')?.value || 'bar';
        const xVar = document.getElementById('vizX')?.value;
        const yVar = document.getElementById('vizY')?.value || '';
        const colorVar = document.getElementById('vizColor')?.value || '';
        const aggregate = document.getElementById('vizAggregate')?.checked || false;
        const aggFunc = document.getElementById('vizAggFunc')?.value || 'mean';

        if (!xVar) {
            container.innerHTML = '<p class="dist-placeholder">Select X variable to create chart.</p>';
            return;
        }

        container.innerHTML = '<p class="eda-loading">Loading chart...</p>';

        try {
            const payload = {
                filename: currentFilename,
                chart_type: chartType,
                x: xVar,
                y: yVar || null,
                color: colorVar || null,
                aggregate,
                agg_func: aggregate ? aggFunc : null,
            };
            const res = await fetch('/api/visualize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Failed to create chart');
            }

            if (data.chart && data.chart.data && data.chart.layout) {
                container.innerHTML = '<div id="vizPlotlyDiv" class="chart-cell"></div>';
                const chartEl = document.getElementById('vizPlotlyDiv');
                if (chartEl && typeof Plotly !== 'undefined') {
                    Plotly.newPlot(chartEl, data.chart.data, data.chart.layout, { responsive: true });
                } else {
                    container.innerHTML = '<p class="bin-error">Plotly not loaded.</p>';
                }
            } else {
                container.innerHTML = '<p class="bin-error">No chart data returned.</p>';
            }
        } catch (err) {
            container.innerHTML = '<p class="bin-error">' + escapeHtml(err.message) + '</p>';
        }
    }

    function renderEDA(data, container) {
        let html = '<h3>Exploratory Data Analysis</h3>';

        if (data.stats && data.stats.length > 0) {
            html += `
                <h4>Descriptive statistics</h4>
                <div class="preview-table-wrapper">
                    <table class="column-table">
                        <thead><tr><th>Column</th><th>count</th><th>mean</th><th>std</th><th>min</th><th>25%</th><th>50%</th><th>75%</th><th>max</th></tr></thead>
                        <tbody>
                            ${data.stats.map(s => `
                                <tr>
                                    <td>${escapeHtml(s.column)}</td>
                                    <td>${s.count}</td>
                                    <td>${num(s.mean)}</td>
                                    <td>${num(s.std)}</td>
                                    <td>${num(s.min)}</td>
                                    <td>${num(s['25%'])}</td>
                                    <td>${num(s['50%'])}</td>
                                    <td>${num(s['75%'])}</td>
                                    <td>${num(s.max)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }

        const allCols = data.distribution_columns || [];
        const numericCols = (data.stats || []).map(s => s.column);

        if (allCols.length > 0) {
            html += `
                <h4>Visualization</h4>
                <p>Select chart type, X and Y variables, and optional color/group. Enable aggregation to summarize Y by X (and color) groups.</p>
                <div class="viz-form">
                    <label>Chart type
                        <select id="vizChartType">
                            <option value="bar">Bar</option>
                            <option value="line">Line</option>
                            <option value="scatter">Scatter</option>
                            <option value="box">Box</option>
                            <option value="histogram">Histogram</option>
                        </select>
                    </label>
                    <label>X variable <select id="vizX"><option value="">-- Select --</option>${allCols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')}</select></label>
                    <label>Y variable (optional for bar count) <select id="vizY"><option value="">-- None --</option>${numericCols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')}</select></label>
                    <label>Color / Group (optional) <select id="vizColor"><option value="">-- None --</option>${allCols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')}</select></label>
                    <label class="viz-check">
                        <input type="checkbox" id="vizAggregate"> Aggregate data
                    </label>
                    <div id="vizAggOptions" class="viz-agg-options" style="display:none">
                        <label>Aggregation <select id="vizAggFunc"><option value="mean">Mean</option><option value="sum">Sum</option><option value="count">Count</option><option value="min">Min</option><option value="max">Max</option></select></label>
                    </div>
                    <button type="button" id="vizShow">Create chart</button>
                </div>
                <div id="vizChartContainer" class="dist-chart-container"></div>
            `;
        }

        if (data.correlation_chart) {
            html += '<h4>Correlation matrix</h4><div id="corrChartContainer" class="chart-cell"></div>';
        }

        if (!data.stats?.length && !allCols.length && !data.correlation_chart) {
            html += '<p>No numeric columns found for EDA.</p>';
        }

        container.innerHTML = html;

        // Correlation matrix: render with Plotly from JSON
        const corrContainer = document.getElementById('corrChartContainer');
        if (corrContainer && data.correlation_chart && typeof Plotly !== 'undefined') {
            try {
                const spec = typeof data.correlation_chart === 'string' ? JSON.parse(data.correlation_chart) : data.correlation_chart;
                Plotly.newPlot(corrContainer, spec.data, spec.layout, { responsive: true });
            } catch (e) {
                corrContainer.innerHTML = '<p class="bin-error">Failed to render correlation chart.</p>';
            }
        }

        // Aggregate checkbox toggle
        const aggCheck = document.getElementById('vizAggregate');
        const aggOptions = document.getElementById('vizAggOptions');
        if (aggCheck && aggOptions) {
            aggCheck.addEventListener('change', () => {
                aggOptions.style.display = aggCheck.checked ? 'block' : 'none';
            });
        }

        const vizShow = document.getElementById('vizShow');
        if (vizShow) vizShow.addEventListener('click', loadVisualization);
    }

    function num(v) {
        if (v == null || (typeof v === 'number' && isNaN(v))) return '';
        return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : escapeHtml(String(v));
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
});
