// Global variables
let currentData = null;
let currentFilters = {
    department: 'all',
    position: 'all',
    minPerformance: null,
    maxPerformance: null,
    minExperience: null,
    maxExperience: null
};

// Initialize AOS animations
AOS.init({
    duration: 800,
    easing: 'ease-in-out',
    once: true
});

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="flex items-center justify-between">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-4">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 5000);
}

function showLoading(show = true) {
    document.getElementById('loadingOverlay').classList.toggle('hidden', !show);
}

function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

// API functions
async function fetchData(endpoint, params = {}) {
    try {
        const url = new URL(`/api/${endpoint}`, window.location.origin);
        Object.keys(params).forEach(key => {
            if (params[key] !== null && params[key] !== undefined && params[key] !== '') {
                url.searchParams.append(key, params[key]);
            }
        });

        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error('Fetch error:', error);
        showNotification('Error loading data. Please try again.', 'error');
        return null;
    }
}

// Data loading and processing functions
async function loadInitialData() {
    showLoading(true);
    try {
        const [filtersData, kpiData] = await Promise.all([
            fetchData('filters'),
            fetchData('kpis', currentFilters)
        ]);

        if (filtersData) {
            populateFilters(filtersData);
        }

        if (kpiData) {
            updateKPIs(kpiData);
        }

        await loadActiveTabData();
        showNotification('Dashboard loaded successfully!', 'success');
    } catch (error) {
        console.error('Error loading initial data:', error);
        showNotification('Error loading dashboard data', 'error');
    } finally {
        showLoading(false);
    }
}

function populateFilters(data) {
    // Populate department filter
    const deptSelect = document.getElementById('departmentFilter');
    deptSelect.innerHTML = '<option value="all">All Departments</option>';
    data.departments.forEach(dept => {
        deptSelect.innerHTML += `<option value="${dept}">${dept}</option>`;
    });

    // Populate position filter
    const posSelect = document.getElementById('positionFilter');
    posSelect.innerHTML = '<option value="all">All Positions</option>';
    data.positions.forEach(pos => {
        posSelect.innerHTML += `<option value="${pos}">${pos}</option>`;
    });

    // Populate individual employee select
    const empSelect = document.getElementById('individualEmployeeSelect');
    empSelect.innerHTML = '<option value="">Select an Employee...</option>';
    data.employees.forEach(emp => {
        empSelect.innerHTML += `<option value="${emp.employee_id}">${emp.employee_name} (${emp.department})</option>`;
    });

    // Populate quick stats
    updateQuickStats(data.quick_stats || {});
}

function updateKPIs(data) {
    document.getElementById('totalEmployees').textContent = formatNumber(data.total_employees || 0);
    document.getElementById('avgPerformance').textContent = (data.avg_performance || 0).toFixed(1);
    document.getElementById('highPerformers').textContent = formatNumber(data.high_performers || 0);
    document.getElementById('atRiskEmployees').textContent = formatNumber(data.at_risk || 0);
}

function updateQuickStats(stats) {
    const quickStatsDiv = document.getElementById('quickStats');
    quickStatsDiv.innerHTML = `
        <div class="flex justify-between">
            <span>Departments:</span>
            <span class="font-semibold">${stats.departments || 0}</span>
        </div>
        <div class="flex justify-between">
            <span>Avg Training Hours:</span>
            <span class="font-semibold">${(stats.avg_training_hours || 0).toFixed(1)}</span>
        </div>
        <div class="flex justify-between">
            <span>Top Department:</span>
            <span class="font-semibold">${stats.top_department || 'N/A'}</span>
        </div>
        <div class="flex justify-between">
            <span>Productivity Score:</span>
            <span class="font-semibold">${(stats.avg_productivity || 0).toFixed(1)}</span>
        </div>
    `;
}

async function loadActiveTabData() {
    const activeTab = document.querySelector('.tab-button.active').dataset.tab;

    switch (activeTab) {
        case 'overview':
            await loadOverviewData();
            break;
        case 'performance':
            await loadPerformanceData();
            break;
        case 'trends':
            await loadTrendsData();
            break;
        case 'individual':
            await loadIndividualData();
            break;
        case 'advanced':
            await loadAdvancedData();
            break;
    }
}

async function loadOverviewData() {
    const [deptData, scatterData, trainingData, positionData] = await Promise.all([
        fetchData('department-performance', currentFilters),
        fetchData('performance-productivity-scatter', currentFilters),
        fetchData('training-distribution', currentFilters),
        fetchData('position-performance', currentFilters)
    ]);

    if (deptData) {
        renderPieChart('deptPerformancePie', deptData, 'Performance by Department');
    }

    if (scatterData) {
        renderScatterPlot('performanceProductivityScatter', scatterData,
            'Performance Score', 'Productivity Score', 'Performance vs Productivity');
    }

    if (trainingData) {
        renderHistogram('trainingHistogram', trainingData, 'Training Hours Distribution', 'Training Hours');
    }

    if (positionData) {
        renderBarChart('positionPerformanceBar', positionData,
            'Performance by Position', 'Position', 'Avg Performance');
    }
}

async function loadPerformanceData() {
    const [correlationData, experienceData, radarData] = await Promise.all([
        fetchData('correlation-matrix', currentFilters),
        fetchData('experience-performance', currentFilters),
        fetchData('department-radar', currentFilters)
    ]);

    if (correlationData) {
        renderHeatmap('correlationHeatmap', correlationData, 'Performance Correlation Matrix');
    }

    if (experienceData) {
        renderBoxPlot('experiencePerformanceBox', experienceData, 'Performance by Experience Level');
    }

    if (radarData) {
        renderRadarChart('deptComparisonRadar', radarData, 'Department Performance Comparison');
    }
}

async function loadTrendsData() {
    const [trendData, trainingEffData, productivityData] = await Promise.all([
        fetchData('performance-trends', currentFilters),
        fetchData('training-effectiveness', currentFilters),
        fetchData('productivity-distribution', currentFilters)
    ]);

    if (trendData) {
        renderLineChart('performanceTrendLine', trendData,
            'Performance Trends Over Time', 'Time Period', 'Performance Score');
    }

    if (trainingEffData) {
        renderScatterPlot('trainingEffectivenessScatter', trainingEffData,
            'Training Hours', 'Performance Improvement', 'Training Effectiveness');
    }

    if (productivityData) {
        renderViolinPlot('productivityViolin', productivityData, 'Productivity Distribution by Department');
    }
}

async function loadIndividualData() {
    // Individual data is loaded on demand when employee is selected
}

async function loadAdvancedData() {
    const [predictiveData, clusterData, riskData] = await Promise.all([
        fetchData('predictive-model', currentFilters),
        fetchData('cluster-analysis', currentFilters),
        fetchData('risk-assessment', currentFilters)
    ]);

    if (predictiveData) {
        renderAdvancedChart('predictiveModel', predictiveData, 'Predictive Performance Model');
    }

    if (clusterData) {
        renderScatterPlot('clusterAnalysis', clusterData,
            'Performance Score', 'Productivity Score', 'Employee Clusters');
    }

    if (riskData) {
        renderBarChart('riskAssessment', riskData,
            'Risk Assessment by Department', 'Department', 'Risk Score');
    }
}

// Event handlers
function setupEventHandlers() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', async (e) => {
            // Update active tab
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
                btn.classList.add('text-white');
            });
            e.target.classList.add('active');
            e.target.classList.remove('text-white');

            // Show/hide content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.add('hidden');
            });
            document.getElementById(`${e.target.dataset.tab}Content`).classList.remove('hidden');

            // Load tab data
            await loadActiveTabData();
        });
    });

    // Filter controls
    document.getElementById('applyFilters').addEventListener('click', async () => {
        await applyFilters();
    });

    document.getElementById('resetFilters').addEventListener('click', async () => {
        resetFilters();
        await applyFilters();
    });

    // Employee search
    document.getElementById('employeeSearch').addEventListener('input', handleEmployeeSearch);

    // Individual employee analysis
    document.getElementById('analyzeEmployee').addEventListener('click', analyzeIndividualEmployee);

    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', async () => {
        await loadInitialData();
    });

    // Download button
    document.getElementById('downloadBtn').addEventListener('click', downloadReport);

    // Floating button
    document.getElementById('floatingBtn').addEventListener('click', () => {
        showNotification('Quick actions menu coming soon!', 'info');
    });
}

async function applyFilters() {
    currentFilters = {
        department: document.getElementById('departmentFilter').value,
        position: document.getElementById('positionFilter').value,
        minPerformance: document.getElementById('minPerformance').value || null,
        maxPerformance: document.getElementById('maxPerformance').value || null,
        minExperience: document.getElementById('minExperience').value || null,
        maxExperience: document.getElementById('maxExperience').value || null
    };

    showLoading(true);
    try {
        // Update KPIs
        const kpiData = await fetchData('kpis', currentFilters);
        if (kpiData) {
            updateKPIs(kpiData);
        }

        // Reload active tab data
        await loadActiveTabData();

        showNotification('Filters applied successfully!', 'success');
    } catch (error) {
        console.error('Error applying filters:', error);
        showNotification('Error applying filters', 'error');
    } finally {
        showLoading(false);
    }
}

function resetFilters() {
    document.getElementById('departmentFilter').value = 'all';
    document.getElementById('positionFilter').value = 'all';
    document.getElementById('minPerformance').value = '';
    document.getElementById('maxPerformance').value = '';
    document.getElementById('minExperience').value = '';
    document.getElementById('maxExperience').value = '';

    currentFilters = {
        department: 'all',
        position: 'all',
        minPerformance: null,
        maxPerformance: null,
        minExperience: null,
        maxExperience: null
    };
}

function handleEmployeeSearch(e) {
    const searchTerm = e.target.value.toLowerCase();
    const resultsDiv = document.getElementById('employeeSearchResults');

    if (searchTerm.length < 2) {
        resultsDiv.innerHTML = '';
        return;
    }

    // This would normally search against the employee data
    resultsDiv.innerHTML = `
        <div class="bg-white bg-opacity-20 rounded p-2 mt-1 text-white text-sm">
            Search results for "${searchTerm}" will appear here
        </div>
    `;
}

async function analyzeIndividualEmployee() {
    const employeeId = document.getElementById('individualEmployeeSelect').value;
    if (!employeeId) {
        showNotification('Please select an employee first', 'error');
        return;
    }

    showLoading(true);
    try {
        const data = await fetchData('individual-analysis', { employee_id: employeeId });
        if (data) {
            displayIndividualAnalysis(data);
            document.getElementById('individualAnalysisResult').classList.remove('hidden');
        }
    } catch (error) {
        console.error('Error analyzing employee:', error);
        showNotification('Error analyzing employee data', 'error');
    } finally {
        showLoading(false);
    }
}

function displayIndividualAnalysis(data) {
    // Display metrics
    const metricsDiv = document.getElementById('individualMetrics');
    metricsDiv.innerHTML = `
        <div class="bg-gray-50 rounded-lg p-4">
            <h4 class="font-semibold text-lg mb-4">${data.employee.name}</h4>
            <div class="grid grid-cols-2 gap-4">
                <div class="text-center p-3 bg-blue-100 rounded">
                    <div class="text-2xl font-bold text-blue-700">${data.employee.performance_score}</div>
                    <div class="text-sm text-blue-600">Performance</div>
                </div>
                <div class="text-center p-3 bg-green-100 rounded">
                    <div class="text-2xl font-bold text-green-700">${data.employee.productivity_score}</div>
                    <div class="text-sm text-green-600">Productivity</div>
                </div>
                <div class="text-center p-3 bg-purple-100 rounded">
                    <div class="text-2xl font-bold text-purple-700">${data.employee.experience_years}</div>
                    <div class="text-sm text-purple-600">Years Experience</div>
                </div>
                <div class="text-center p-3 bg-yellow-100 rounded">
                    <div class="text-2xl font-bold text-yellow-700">${data.employee.training_hours}</div>
                    <div class="text-sm text-yellow-600">Training Hours</div>
                </div>
            </div>
        </div>
    `;

    // Display radar chart
    if (data.radar_data) {
        renderRadarChart('individualRadarChart', data.radar_data, 'Employee Performance Profile');
    }

    // Display recommendations
    const recsDiv = document.getElementById('individualRecommendations');
    recsDiv.innerHTML = `
        <div class="bg-blue-50 rounded-lg p-4">
            <h5 class="font-semibold text-lg mb-3">Recommendations</h5>
            <ul class="space-y-2">
                ${data.recommendations.map(rec => `<li class="flex items-start"><i class="fas fa-lightbulb text-yellow-500 mr-2 mt-1"></i>${rec}</li>`).join('')}
            </ul>
        </div>
    `;
}

async function downloadReport() {
    showLoading(true);
    try {
        const response = await fetch('/api/download-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentFilters)
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `employee_analytics_report_${new Date().toISOString().split('T')[0]}.pdf`;
            a.click();
            window.URL.revokeObjectURL(url);
            showNotification('Report downloaded successfully!', 'success');
        } else {
            throw new Error('Download failed');
        }
    } catch (error) {
        console.error('Download error:', error);
        showNotification('Error downloading report', 'error');
    } finally {
        showLoading(false);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    setupEventHandlers();
    await loadInitialData();
});