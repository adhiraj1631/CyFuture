// Load recruitment data on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadRecruitmentData();
});

async function loadRecruitmentData() {
    try {
        // Load KPIs
        const kpis = await fetch('/api/recruitment/kpis').then(r => r.json());
        updateKPIs(kpis);

        // Load chart data
        const [locations, grades, practices, skills] = await Promise.all([
            fetch('/api/recruitment/location-distribution').then(r => r.json()),
            fetch('/api/recruitment/grade-distribution').then(r => r.json()),
            fetch('/api/recruitment/practice-distribution').then(r => r.json()),
            fetch('/api/recruitment/skills-analysis').then(r => r.json())
        ]);

        // Render charts
        renderPieChart('locationChart', locations, 'Recruitment by Location');
        renderBarChart('gradeChart', grades, 'Grade Distribution');
        renderPieChart('practiceChart', practices, 'Practice Areas');
        renderBarChart('skillsChart', skills.slice(0, 10), 'Top 10 Skills');

        // Load table data
        const tableData = await fetch('/api/recruitment/table-data').then(r => r.json());
        renderTable(tableData);

    } catch (error) {
        console.error('Error loading recruitment data:', error);
        showNotification('Error loading data', 'error');
    }
}

function updateKPIs(kpis) {
    document.getElementById('totalRecruits').textContent = kpis.total_recruits;
    document.getElementById('totalLocations').textContent = kpis.total_locations;
    document.getElementById('totalPractices').textContent = kpis.total_practices;
    document.getElementById('topGrade').textContent = kpis.top_grade;
}

function renderTable(data) {
    const tableDiv = document.getElementById('dataTable');

    if (!data || data.length === 0) {
        tableDiv.innerHTML = '<p class="text-gray-500">No data available</p>';
        return;
    }

    const columns = Object.keys(data[0]);

    let tableHTML = `
        <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    ${columns.map(col => `<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">${col.replace('_', ' ')}</th>`).join('')}
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
    `;

    data.slice(0, 50).forEach((row, index) => {
        tableHTML += `
            <tr class="${index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}">
                ${columns.map(col => `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${row[col] || '-'}</td>`).join('')}
            </tr>
        `;
    });

    tableHTML += '</tbody></table>';

    if (data.length > 50) {
        tableHTML += `<p class="text-gray-500 text-sm mt-2">Showing first 50 of ${data.length} records</p>`;
    }

    tableDiv.innerHTML = tableHTML;
}

async function exportReport() {
    try {
        const response = await fetch('/api/recruitment/export-report');
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `recruitment_report_${new Date().toISOString().split('T')[0]}.xlsx`;
            a.click();
            window.URL.revokeObjectURL(url);
            showNotification('Report exported successfully!', 'success');
        }
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Error exporting report', 'error');
    }
}

function downloadTemplate() {
    const templateData = [
        ['Employee Name', 'Offered Grade', 'Joining Location', 'Practice', 'Skills'],
        ['John Doe', 'L2', 'New York', 'Technology', 'Python, React, Node.js'],
        ['Jane Smith', 'L3', 'London', 'Consulting', 'Project Management, Analysis'],
        ['Mike Johnson', 'L1', 'Mumbai', 'Digital', 'UI/UX, Figma, Design'],
    ];

    const csvContent = templateData.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'recruitment_template.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

async function refreshData() {
    await loadRecruitmentData();
    showNotification('Data refreshed!', 'success');
}

function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="flex items-center justify-between">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-4">×</button>
        </div>
    `;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 3000);
}