# Employee Analytics Dashboard

A comprehensive, professional-grade employee analytics platform built with Flask, featuring advanced data visualization, predictive modeling, and interactive dashboards for HR analytics.

## 🚀 Features

### 📊 Advanced Visualizations
- **Multiple Chart Types**: Pie charts, scatter plots, histograms, bar charts, heatmaps, box plots, radar charts, line charts, violin plots
- **Interactive Charts**: Built with Plotly.js for professional, interactive visualizations
- **Real-time Updates**: Dynamic chart updates based on applied filters

### 🔍 Comprehensive Analytics
- **Performance Analysis**: Individual and departmental performance metrics
- **Predictive Modeling**: Performance prediction and risk assessment
- **Correlation Analysis**: Statistical relationships between employee metrics
- **Trend Analysis**: Time-based performance trends and patterns

### 🎛️ Dynamic Filtering System
- **Multi-level Filters**: Department, position, performance range, experience range
- **Real-time Application**: Instant chart updates when filters are applied
- **Search Functionality**: Employee search with autocomplete

### 👤 Individual Employee Analysis
- **Detailed Profiles**: Comprehensive individual performance analysis
- **Radar Charts**: Multi-dimensional employee performance visualization
- **AI Recommendations**: Personalized development suggestions
- **Comparative Analysis**: Performance vs department/position averages

### 📱 Modern UI/UX
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Glass Morphism**: Modern design with glassmorphism effects
- **Smooth Animations**: AOS-powered animations and transitions
- **Professional Styling**: Tailwind CSS with custom components

### 📋 Data Management
- **Multiple Formats**: Excel (.xlsx, .xls) and CSV file support
- **Data Validation**: Comprehensive data cleaning and validation
- **Sample Data**: Built-in sample data generator for testing
- **Export Options**: Report generation and download functionality

## 🏗️ Project Structure

```
employee_analytics_dashboard/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── config.py                      # Configuration settings
├── run.py                         # Application runner
│
├── static/                        # Static files
│   ├── css/
│   │   ├── dashboard.css          # Custom CSS styles
│   │   └── animations.css         # Animation styles
│   ├── js/
│   │   ├── dashboard.js           # Main dashboard JavaScript
│   │   ├── charts.js              # Chart rendering functions
│   │   ├── filters.js             # Filter handling
│   │   └── utils.js               # Utility functions
│   └── images/
│       └── logo.png               # Application logo
│
├── templates/                     # HTML templates
│   ├── base.html                  # Base template
│   ├── upload.html                # File upload page
│   └── dashboard.html             # Main dashboard
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── analytics.py               # Analytics engine
│   ├── data_processor.py          # Data processing
│   └── report_generator.py        # Report generation
│
├── uploads/                       # File upload directory
├── reports/                       # Generated reports
├── tests/                         # Test files
└── docs/                          # Documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Quick Start

1. **Clone or create the project directory:**
```bash
mkdir employee_analytics_dashboard
cd employee_analytics_dashboard
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python run.py
```

5. **Access the dashboard:**
   - Open your browser and navigate to `http://127.0.0.1:5000`
   - Upload your employee data or use sample data to explore features

### Alternative Running Methods

**Development Mode:**
```bash
python run.py --debug
```

**Production Mode:**
```bash
python run.py --prod --host 0.0.0.0 --port 8080
```

**Custom Configuration:**
```bash
python run.py --host 192.168.1.100 --port 3000
```

## 📊 Dashboard Overview

### 1. Overview Tab
- **KPI Cards**: Total employees, average performance, high performers, at-risk employees
- **Department Performance**: Pie chart showing performance distribution
- **Performance vs Productivity**: Scatter plot with employee details
- **Training Distribution**: Histogram of training hours
- **Position Performance**: Bar chart by job position

### 2. Performance Analysis Tab
- **Correlation Matrix**: Heatmap showing relationships between metrics
- **Experience Analysis**: Box plots showing performance by experience level
- **Department Comparison**: Radar chart comparing departments

### 3. Trends & Patterns Tab
- **Performance Trends**: Line charts showing trends over time
- **Training Effectiveness**: Scatter plot of training impact
- **Productivity Distribution**: Violin plots by department

### 4. Individual Analysis Tab
- **Employee Selection**: Dropdown to choose specific employees
- **Performance Profile**: Individual radar chart
- **Recommendations**: AI-generated development suggestions
- **Comparative Metrics**: Performance vs averages

### 5. Advanced Analytics Tab
- **Predictive Modeling**: Performance prediction visualization
- **Cluster Analysis**: Employee clustering based on performance
- **Risk Assessment**: Department-level risk scoring

## 🎯 Usage Guide

### Data Upload
1. Click "Upload New File" or navigate to the upload page
2. Select an Excel (.xlsx, .xls) or CSV file containing employee data
3. The system will automatically validate and process the data
4. Alternatively, use "Use Sample Data" to explore with generated data

### Required Data Columns
- `employee_id`: Unique identifier
- `employee_name`: Employee name
- `department`: Department name
- `position`: Job position
- `performance_score`: Performance rating (0-100)
- `productivity_score`: Productivity rating (0-100)
- `training_hours`: Hours of training completed
- `experience_years`: Years of experience

### Filtering Data
1. Use the sidebar filters to narrow down data:
   - **Department**: Filter by specific department
   - **Position**: Filter by job position
   - **Performance Range**: Set min/max performance scores
   - **Experience Range**: Set min/max experience years
2. Click "Apply Filters" to update all visualizations
3. Use "Reset" to clear all filters

### Individual Analysis
1. Navigate to the "Individual Analysis" tab
2. Select an employee from the dropdown
3. Click "Analyze Employee" to generate detailed insights
4. View performance radar chart and personalized recommendations

### Exporting Reports
1. Click the "Export Report" button in the top navigation
2. The system generates a comprehensive PDF report
3. Report includes current filter settings and key insights

## 🔧 Configuration

### Environment Variables
```bash
FLASK_ENV=development          # development/production
FLASK_HOST=127.0.0.1          # Host to bind to
FLASK_PORT=5000               # Port to bind to
FLASK_DEBUG=True              # Enable debug mode
SECRET_KEY=your-secret-key    # Flask secret key
```

### Configuration Files
- `config.py`: Main configuration settings
- Supports development, production, and testing configurations
- Customizable file upload limits, security settings, and more

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Test individual modules:
```bash
python -m pytest tests/test_analytics.py
python -m pytest tests/test_api.py
```

## 🚀 Deployment

### Production Deployment

1. **Set production environment:**
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secure-secret-key
```

2. **Use production server:**
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

3. **With Docker:**
```bash
# Create Dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "run.py", "--prod", "--host", "0.0.0.0"]
```

### Cloud Deployment Options
- **Heroku**: Use provided Procfile
- **AWS**: Deploy with Elastic Beanstalk or EC2
- **Google Cloud**: Use App Engine or Cloud Run
- **Digital Ocean**: Deploy on droplets or App Platform

## 🔐 Security Considerations

- File upload validation and size limits
- CSRF protection enabled
- Secure session handling
- Input validation and sanitization
- Production security headers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support, please contact:
- Email: support@yourcompany.com
- Documentation: /docs
- Issues: GitHub Issues page

## 🙏 Acknowledgments

- Flask framework for the web application
- Plotly.js for interactive visualizations
- Tailwind CSS for responsive styling
- AOS library for smooth animations
- Font Awesome for icons

---

**Built with ❤️ for modern HR analytics**