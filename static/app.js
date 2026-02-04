/**
 * Employee Survey Impact Analysis - Client-side logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const contentSection = document.getElementById('contentSection');
    let currentFilename = null;
    let currentData = null;
    let lastRegressionResult = null;
    let binnedMetadata = {};  // { "col_binned": { edges: [...], labels: [...] } } for PDF report

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
            binnedMetadata = {};
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
            <div class="content-tabs">
                <button type="button" class="tab-btn active" data-tab="data-eda">Data & EDA</button>
                <button type="button" class="tab-btn" data-tab="regression">Regression</button>
                <button type="button" class="tab-btn" data-tab="report">Report</button>
            </div>
            <div id="tabDataEda" class="tab-panel active">
                <div class="data-loaded-row">
                    <h2>Data loaded: ${escapeHtml(data.filename)}</h2>
                </div>
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
            </div>
            <div id="tabRegression" class="tab-panel">
                <h2>Regression</h2>
                <p>Specify target, covariates (key predictors to focus on), and control variables. Model type is chosen automatically: linear for continuous targets, logistic for binary.</p>
                <div class="regression-form">
                    <label>Target variable <select id="regTarget"><option value="">-- Select target --</option>${data.columns.map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('')}</select></label>
                    <label>Covariates (key predictors) <select id="regCovariates" multiple size="5">${data.columns.map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('')}</select></label>
                    <label>Control variables <select id="regControls" multiple size="5">${data.columns.map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('')}</select></label>
                    <p class="form-hint">Hold Ctrl/Cmd to select multiple. Covariates + controls form the full predictor set.</p>
                    <button type="button" id="regRun">Run regression</button>
                </div>
                <div id="regressionResults" class="regression-results"></div>
            </div>
            <div id="tabReport" class="tab-panel">
                <h2>Export report</h2>
                <p>Generate a PDF report with data preview, variables created/binned, correlation matrix, regression method, and regression output with interpretation.</p>
                <button type="button" id="exportPdfBtn" class="export-pdf-btn">Export report (PDF)</button>
            </div>
        `;

        document.querySelectorAll('.content-tabs .tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.getAttribute('data-tab');
                document.querySelectorAll('.content-tabs .tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                const panelId = tab === 'data-eda' ? 'tabDataEda' : (tab === 'regression' ? 'tabRegression' : 'tabReport');
                const panel = document.getElementById(panelId);
                if (panel) panel.classList.add('active');
            });
        });

        const regRun = document.getElementById('regRun');
        if (regRun) regRun.addEventListener('click', loadRegression);

        const exportPdfBtn = document.getElementById('exportPdfBtn');
        if (exportPdfBtn) exportPdfBtn.addEventListener('click', exportReport);

        const binApply = document.getElementById('binApply');
        if (binApply) binApply.addEventListener('click', applyBinning);
    }

    async function exportReport() {
        if (!currentFilename || !currentData) {
            alert('Upload data first.');
            return;
        }
        const binned_columns = (currentData.columns || []).filter(c => (c.name || '').endsWith('_binned')).map(c => c.name);
        const payload = {
            filename: currentFilename,
            binned_columns,
            binned_metadata: binnedMetadata || {},
            regression_result: lastRegressionResult || null,
        };
        try {
            const res = await fetch('/api/export-report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || 'Export failed');
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const disp = res.headers.get('Content-Disposition') || '';
            a.download = (disp.split('filename=')[1] || '').replace(/"/g, '').trim() || 'CSAT_Regression_Report.pdf';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            alert(err.message || 'Export failed');
        }
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

            const labelsUsed = labelsStr
                ? labelsStr.split(',').map(s => s.trim()).filter(Boolean)
                : bins.slice(0, -1).map((_, i) => String(bins[i]) + '-' + String(bins[i + 1]));
            binnedMetadata[data.new_column] = { edges: [...bins], labels: labelsUsed };

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

            // Refresh regression tab dropdowns so binned (and all) columns are available
            const allColOptions = data.columns.map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('');
            const regTarget = document.getElementById('regTarget');
            const regCovariates = document.getElementById('regCovariates');
            const regControls = document.getElementById('regControls');
            if (regTarget) {
                const cur = regTarget.value;
                regTarget.innerHTML = '<option value="">-- Select target --</option>' + allColOptions;
                if (cur) regTarget.value = cur;
            }
            if (regCovariates) {
                const cur = new Set(Array.from(regCovariates.selectedOptions).map(o => o.value));
                regCovariates.innerHTML = allColOptions;
                Array.from(regCovariates.options).forEach(o => { if (cur.has(o.value)) o.selected = true; });
            }
            if (regControls) {
                const cur = new Set(Array.from(regControls.selectedOptions).map(o => o.value));
                regControls.innerHTML = allColOptions;
                Array.from(regControls.options).forEach(o => { if (cur.has(o.value)) o.selected = true; });
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

    async function loadRegression() {
        const container = document.getElementById('regressionResults');
        if (!container || !currentFilename) return;

        const target = document.getElementById('regTarget')?.value;
        const covariatesEl = document.getElementById('regCovariates');
        const controlsEl = document.getElementById('regControls');
        const covariates = covariatesEl ? Array.from(covariatesEl.selectedOptions).map(o => o.value).filter(Boolean) : [];
        const controls = controlsEl ? Array.from(controlsEl.selectedOptions).map(o => o.value).filter(Boolean) : [];

        if (!target) {
            container.innerHTML = '<p class="bin-error">Select a target variable.</p>';
            return;
        }
        if (!covariates.length && !controls.length) {
            container.innerHTML = '<p class="bin-error">Select at least one covariate or control variable.</p>';
            return;
        }

        container.innerHTML = '<p class="eda-loading">Running regression...</p>';

        try {
            const res = await fetch('/api/regression', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: currentFilename,
                    target,
                    covariates,
                    controls,
                }),
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Regression failed');
            }

            lastRegressionResult = data;
            renderRegressionResults(data, container);
        } catch (err) {
            container.innerHTML = '<p class="bin-error">' + escapeHtml(err.message) + '</p>';
        }
    }

    function renderRegressionResults(data, container) {
        const labels = data.coefficient_labels || [];
        const coefs = data.coefficients || [];
        const pvalues = data.pvalues || [];
        const pvalueLabels = data.pvalue_labels || [];
        const bse = data.bse || [];
        const ciLower = data.ci_lower || [];
        const ciUpper = data.ci_upper || [];
        const covariateNames = (data.covariate_names || []).map(String);

        const getIdx = (label) => pvalueLabels.indexOf(label);
        const getPvalue = (label) => { const idx = getIdx(label); return idx >= 0 && pvalues[idx] != null ? pvalues[idx] : null; };
        const getBse = (label) => { const idx = getIdx(label); return idx >= 0 && bse[idx] != null ? bse[idx] : null; };
        const getCi = (label) => {
            const idx = getIdx(label);
            return idx >= 0 && ciLower[idx] != null && ciUpper[idx] != null ? [ciLower[idx], ciUpper[idx]] : null;
        };

        const interceptP = getPvalue('const');
        const interceptBse = getBse('const');
        const interceptCi = getCi('const');
        let tableRows = `<tr><td>(Intercept)</td><td>${num(data.intercept)}</td><td>${num(interceptBse)}</td><td>${pvalueFormat(interceptP)}</td><td>${ciFormat(interceptCi)}</td><td>${sigStar(interceptP)}</td></tr>`;
        labels.forEach((name, i) => {
            const isCov = covariateNames.some(c => name === c || name.startsWith(c + '_'));
            const covMark = isCov ? ' <span class="covariate-tag">key predictor</span>' : '';
            const p = pvalues[i + 1];
            const se = bse[i + 1];
            const ci = (ciLower[i + 1] != null && ciUpper[i + 1] != null) ? [ciLower[i + 1], ciUpper[i + 1]] : null;
            tableRows += `<tr class="${isCov ? 'covariate-row' : ''}"><td>${escapeHtml(name)}${covMark}</td><td>${num(coefs[i])}</td><td>${num(se)}</td><td>${pvalueFormat(p)}</td><td>${ciFormat(ci)}</td><td>${sigStar(p)}</td></tr>`;
        });

        let statsHtml = '';
        if (data.model_type === 'linear') {
            statsHtml = `<p>R² = ${num(data.r2)} &nbsp; RMSE = ${num(data.rmse)}</p>`;
        } else {
            statsHtml = `<p>Accuracy = ${num(data.accuracy)}</p>`;
        }
        if (data.covariates_standardized && covariateNames.length > 0) {
            statsHtml += `<p class="standardized-note">Key predictors are standardized (z-score); coefficients are per 1 SD increase and can be compared for relative importance.</p>`;
        }

        const interpretationHtml = buildInterpretation(data, labels, coefs, pvalues, covariateNames);

        container.innerHTML = `
            <h4>Model: ${data.model_type === 'linear' ? 'Linear' : 'Logistic'} regression</h4>
            <p>Target: <strong>${escapeHtml(data.target)}</strong> &nbsp; N = ${data.n_obs}</p>
            ${statsHtml}
            <h4>Coefficients</h4>
            <div class="preview-table-wrapper">
                <table class="column-table">
                    <thead><tr><th>Variable</th><th>Coefficient</th><th>SE</th><th>p-value</th><th>95% CI</th><th>Sig.</th></tr></thead>
                    <tbody>${tableRows}</tbody>
                </table>
            </div>
            <p class="sig-legend">Sig.: * p&lt;0.05</p>
            <div class="vif-section">
                <button type="button" id="vifRun" class="vif-btn">Run VIF test (multicollinearity)</button>
                <div id="vifResults" class="vif-results"></div>
            </div>
            <div class="interpretation-section">${interpretationHtml}</div>
        `;

        const vifRun = document.getElementById('vifRun');
        if (vifRun) vifRun.addEventListener('click', () => runVIF(container));
    }

    async function runVIF(container) {
        const vifResults = document.getElementById('vifResults');
        if (!vifResults || !currentFilename) return;

        const target = document.getElementById('regTarget')?.value;
        const covariatesEl = document.getElementById('regCovariates');
        const controlsEl = document.getElementById('regControls');
        const covariates = covariatesEl ? Array.from(covariatesEl.selectedOptions).map(o => o.value).filter(Boolean) : [];
        const controls = controlsEl ? Array.from(controlsEl.selectedOptions).map(o => o.value).filter(Boolean) : [];

        if (!target || (!covariates.length && !controls.length)) {
            vifResults.innerHTML = '<p class="bin-error">Select target and at least one covariate or control, then run regression first.</p>';
            return;
        }

        vifResults.innerHTML = '<p class="eda-loading">Computing VIF...</p>';

        try {
            const res = await fetch('/api/vif', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: currentFilename, target, covariates, controls }),
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'VIF failed');
            }

            const rows = (data.feature_names || []).map((name, i) => {
                const v = data.vif && data.vif[i];
                const vStr = v != null && Number.isFinite(v) ? Number(v).toFixed(2) : '—';
                const flag = v != null && v > 10 ? ' <span class="vif-high">High</span>' : (v != null && v > 5 ? ' <span class="vif-moderate">Moderate</span>' : '');
                return `<tr><td>${escapeHtml(name)}</td><td>${vStr}${flag}</td></tr>`;
            }).join('');

            vifResults.innerHTML = `
                <h4>Variance Inflation Factor (VIF)</h4>
                <p class="vif-legend">VIF &gt; 5 suggests moderate multicollinearity; VIF &gt; 10 suggests high multicollinearity.</p>
                <div class="preview-table-wrapper">
                    <table class="column-table">
                        <thead><tr><th>Variable</th><th>VIF</th></tr></thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>
            `;
        } catch (err) {
            vifResults.innerHTML = '<p class="bin-error">' + escapeHtml(err.message) + '</p>';
        }
    }

    function ciFormat(ci) {
        if (!ci || !Array.isArray(ci) || ci.length < 2) return '—';
        return `[${num(ci[0])}, ${num(ci[1])}]`;
    }

    function buildInterpretation(data, labels, coefs, pvalues, covariateNames) {
        const modelLabel = data.model_type === 'linear' ? 'Linear' : 'Logistic';
        const targetEsc = escapeHtml(data.target);

        let html = '<h4>Interpretation</h4>';

        // Tutorial paragraph
        html += `<p class="interpretation-tutorial"><strong>How to read the model:</strong> Key predictors are standardized (1 SD unit), so coefficients can be compared for relative importance. A 1 SD increase in a key predictor is associated with a ${data.model_type === 'linear' ? 'β unit change' : 'change in log-odds'} in ${targetEsc}, while accounting for control variables. Control variables are in their original units.</p>`;

        // Key predictors: rank by |β|, then explain each
        const covariateFeatures = [];
        labels.forEach((name, i) => {
            const isCov = covariateNames.some(c => name === c || name.startsWith(c + '_'));
            if (isCov) covariateFeatures.push({ name, coef: coefs[i], p: pvalues[i + 1], i });
        });

        covariateFeatures.sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));

        if (covariateFeatures.length > 0) {
            html += '<h5>Key predictors (relative importance)</h5>';
            html += '<p class="interpretation-summary">Ranking by relative importance (|standardized β|, largest first):</p><ol class="interpretation-bullets">';
            covariateFeatures.forEach((f, rank) => {
                const dir = f.coef > 0 ? 'higher' : 'lower';
                const outcomeText = data.model_type === 'linear'
                    ? `${dir} ${targetEsc}`
                    : (f.coef > 0 ? 'higher odds of positive outcome' : 'lower odds of positive outcome');
                const sigText = f.p != null && f.p < 0.05
                    ? ` (p ${f.p < 0.001 ? '&lt; 0.001' : '= ' + Number(f.p).toFixed(3)})`
                    : ' (not significant at p &lt; 0.05)';
                html += `<li><strong>${escapeHtml(f.name)}</strong>: A 1 SD increase is associated with ${outcomeText} (β = ${num(f.coef)})${sigText}.</li>`;
            });
            html += '</ol>';

            const sigCovs = covariateFeatures.filter(f => f.p != null && f.p < 0.05);
            if (sigCovs.length > 0) {
                html += '<p class="interpretation-summary">Among key predictors, ' + sigCovs.map(f => escapeHtml(f.name)).join(', ') + ' ' + (sigCovs.length === 1 ? 'is' : 'are') + ' statistically significant (p &lt; 0.05).</p>';
            }
        }

        // Control variables
        const controlFeatures = [];
        labels.forEach((name, i) => {
            const isCov = covariateNames.some(c => name === c || name.startsWith(c + '_'));
            if (!isCov) controlFeatures.push({ name, coef: coefs[i], p: pvalues[i + 1] });
        });

        if (controlFeatures.length > 0) {
            html += '<h5>Control variables</h5>';
            const sigControls = controlFeatures.filter(f => f.p != null && f.p < 0.05);
            if (sigControls.length > 0) {
                html += '<p class="interpretation-summary">Among control variables, ' + sigControls.map(f => `<strong>${escapeHtml(f.name)}</strong> (β = ${num(f.coef)}, p ${f.p < 0.001 ? '&lt; 0.001' : '= ' + Number(f.p).toFixed(3)})`).join('; ') + ' ' + (sigControls.length === 1 ? 'is' : 'are') + ' statistically significant.</p>';
            } else {
                html += '<p class="interpretation-summary">No control variables were statistically significant at p &lt; 0.05.</p>';
            }
        }

        return html;
    }

    function pvalueFormat(p) {
        if (p == null || (typeof p === 'number' && isNaN(p))) return '—';
        if (p < 0.001) return '&lt;0.001';
        return Number(p).toFixed(4);
    }

    function sigStar(p) {
        if (p == null || (typeof p === 'number' && isNaN(p))) return '';
        if (p < 0.05) return '*';
        return '';
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
