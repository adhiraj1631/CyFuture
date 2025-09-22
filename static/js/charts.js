// Chart rendering functions using Plotly.js

// Chart rendering functions
function renderPieChart(elementId, data, title) {
    if (!data || data.length === 0) return;

    const trace = {
        type: 'pie',
        labels: data.map(d => d.name || d.department || d.position),
        values: data.map(d => d.value || d.count || d.performance_score),
        hovertemplate: '%{label}<br>%{value}<br>%{percent}<extra></extra>',
        marker: {
            colors: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
            line: {
                color: '#ffffff',
                width: 2
            }
        },
        textinfo: 'label+percent',
        textposition: 'auto'
    };

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.1,
            x: 0.5,
            xanchor: 'center'
        },
        margin: { t: 60, b: 80, l: 50, r: 50 },
        font: { family: 'Inter, sans-serif' }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, [trace], layout, config);
}

function renderScatterPlot(elementId, data, xLabel, yLabel, title) {
    if (!data || data.length === 0) return;

    const trace = {
        x: data.map(d => d.x),
        y: data.map(d => d.y),
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: data.map(d => Math.max(8, (d.z || 50) / 5)),
            color: data.map(d => d.z || d.productivity_score || 50),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
                title: 'Value',
                titleside: 'right'
            },
            line: {
                color: '#ffffff',
                width: 1
            }
        },
        text: data.map(d => d.name || d.employee_name || ''),
        hovertemplate: '%{text}<br>' + xLabel + ': %{x}<br>' + yLabel + ': %{y}<extra></extra>'
    };

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        xaxis: {
            title: xLabel,
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        yaxis: {
            title: yLabel,
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        margin: { t: 60, b: 50, l: 60, r: 50 },
        font: { family: 'Inter, sans-serif' },
        plot_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, [trace], layout, config);
}

function renderHistogram(elementId, data, title, xLabel) {
    if (!data || data.length === 0) return;

    const trace = {
        x: data,
        type: 'histogram',
        marker: {
            color: '#667eea',
            opacity: 0.7,
            line: {
                color: '#ffffff',
                width: 1
            }
        },
        nbinsx: 20
    };

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        xaxis: {
            title: xLabel,
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        yaxis: {
            title: 'Frequency',
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        margin: { t: 60, b: 50, l: 50, r: 50 },
        font: { family: 'Inter, sans-serif' },
        plot_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, [trace], layout, config);
}

function renderBarChart(elementId, data, title, xLabel, yLabel) {
    if (!data || data.length === 0) return;

    const trace = {
        x: data.map(d => d.name || d.department || d.position),
        y: data.map(d => d.value || d.performance_score),
        type: 'bar',
        marker: {
            color: data.map((d, i) => {
                const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'];
                return colors[i % colors.length];
            }),
            line: { width: 1, color: '#ffffff' }
        },
        hovertemplate: '%{x}<br>%{y:.1f}<extra></extra>'
    };

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        xaxis: {
            title: xLabel,
            tickangle: -45,
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        yaxis: {
            title: yLabel,
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        margin: { t: 60, b: 100, l: 60, r: 50 },
        font: { family: 'Inter, sans-serif' },
        plot_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, [trace], layout, config);
}

function renderHeatmap(elementId, data, title) {
    if (!data || !data.matrix) return;

    const trace = {
        z: data.matrix,
        x: data.labels,
        y: data.labels,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        text: data.matrix.map(row => row.map(val => val.toFixed(2))),
        texttemplate: '%{text}',
        textfont: { size: 10 },
        hoverongaps: false,
        hovertemplate: '%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    };

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        xaxis: {
            side: 'bottom',
            tickangle: -45
        },
        yaxis: {
            side: 'left'
        },
        margin: { t: 60, b: 100, l: 100, r: 50 },
        font: { family: 'Inter, sans-serif' }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, [trace], layout, config);
}

function renderBoxPlot(elementId, data, title) {
    if (!data || data.length === 0) return;

    const traces = data.map(group => ({
        y: group.values,
        type: 'box',
        name: group.name,
        boxpoints: 'outliers',
        marker: {
            color: '#667eea',
            outliercolor: '#f5576c'
        },
        line: {
            color: '#667eea'
        }
    }));

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        yaxis: {
            title: 'Performance Score',
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        xaxis: {
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        margin: { t: 60, b: 50, l: 60, r: 50 },
        font: { family: 'Inter, sans-serif' },
        plot_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, traces, layout, config);
}

function renderRadarChart(elementId, data, title) {
    if (!data || data.length === 0) return;

    const traces = data.map(employee => ({
        type: 'scatterpolar',
        r: employee.values,
        theta: employee.categories,
        fill: 'toself',
        name: employee.name,
        line: {
            color: employee.color || '#667eea',
            width: 2
        },
        marker: {
            size: 6,
            color: employee.color || '#667eea'
        },
        fillcolor: `${employee.color || '#667eea'}20`, // 20% opacity
        hovertemplate: '%{theta}<br>%{r:.1f}<extra></extra>'
    }));

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 100],
                tickmode: 'linear',
                tick0: 0,
                dtick: 20,
                tickfont: { size: 10 },
                gridcolor: '#e5e7eb',
                linecolor: '#d1d5db'
            },
            angularaxis: {
                tickfont: { size: 12 },
                linecolor: '#d1d5db',
                gridcolor: '#e5e7eb'
            },
            bgcolor: '#ffffff'
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.1,
            x: 0.5,
            xanchor: 'center'
        },
        margin: { t: 60, b: 80, l: 50, r: 50 },
        font: { family: 'Inter, sans-serif' }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, traces, layout, config);
}

function renderLineChart(elementId, data, title, xLabel, yLabel) {
    if (!data || data.length === 0) return;

    const traces = data.map((series, index) => {
        const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'];
        return {
            x: series.x,
            y: series.y,
            type: 'scatter',
            mode: 'lines+markers',
            name: series.name,
            line: {
                width: 3,
                color: colors[index % colors.length]
            },
            marker: {
                size: 6,
                color: colors[index % colors.length]
            }
        };
    });

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        xaxis: {
            title: xLabel,
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        yaxis: {
            title: yLabel,
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        margin: { t: 60, b: 50, l: 60, r: 50 },
        font: { family: 'Inter, sans-serif' },
        plot_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, traces, layout, config);
}

function renderViolinPlot(elementId, data, title) {
    if (!data || data.length === 0) return;

    const traces = data.map((group, index) => {
        const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'];
        return {
            y: group.values,
            type: 'violin',
            name: group.name,
            box: { visible: true },
            meanline: { visible: true },
            fillcolor: colors[index % colors.length] + '40',
            line: { color: colors[index % colors.length] }
        };
    });

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        yaxis: {
            title: 'Distribution',
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        xaxis: {
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        margin: { t: 60, b: 50, l: 60, r: 50 },
        font: { family: 'Inter, sans-serif' },
        plot_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, traces, layout, config);
}

function renderAdvancedChart(elementId, data, title) {
    if (!data || !data.predictions) return;

    const actualTrace = {
        x: data.actual,
        y: data.actual,
        mode: 'markers',
        type: 'scatter',
        name: 'Actual vs Actual',
        marker: {
            color: '#667eea',
            size: 8,
            line: { color: '#ffffff', width: 1 }
        },
        hovertemplate: 'Actual: %{x}<br>Actual: %{y}<extra></extra>'
    };

    const predictedTrace = {
        x: data.actual,
        y: data.predictions,
        mode: 'markers',
        type: 'scatter',
        name: 'Actual vs Predicted',
        marker: {
            color: '#f5576c',
            size: 8,
            line: { color: '#ffffff', width: 1 }
        },
        hovertemplate: 'Actual: %{x}<br>Predicted: %{y}<extra></extra>'
    };

    const perfectLine = {
        x: [Math.min(...data.actual), Math.max(...data.actual)],
        y: [Math.min(...data.actual), Math.max(...data.actual)],
        mode: 'lines',
        type: 'scatter',
        name: 'Perfect Prediction',
        line: {
            color: '#9ca3af',
            dash: 'dash',
            width: 2
        },
        hovertemplate: 'Perfect Prediction Line<extra></extra>'
    };

    const layout = {
        title: {
            text: title,
            font: { size: 16, color: '#374151' }
        },
        xaxis: {
            title: 'Actual Performance',
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        yaxis: {
            title: 'Predicted Performance',
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db'
        },
        margin: { t: 60, b: 50, l: 60, r: 50 },
        font: { family: 'Inter, sans-serif' },
        plot_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, [actualTrace, predictedTrace, perfectLine], layout, config);
}