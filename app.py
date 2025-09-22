from flask import Flask, request, render_template, jsonify, redirect, url_for, send_file
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import warnings
from faker import Faker
import json
import io
from datetime import datetime, timedelta
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import plotly.graph_objs as go
import plotly.express as px

# Setup
warnings.filterwarnings('ignore')


class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'xlsx', 'csv', 'xls'}
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'


class AdvancedEmployeeAnalytics:
    """Advanced analytics engine for employee performance data"""

    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()

    def prepare_data(self):
        """Prepare and enhance data for advanced analytics"""
        # Ensure required columns exist
        required_columns = ['employee_id', 'employee_name', 'department', 'position',
                            'performance_score', 'productivity_score', 'training_hours', 'experience_years']

        for col in required_columns:
            if col not in self.df.columns:
                if col == 'employee_id':
                    self.df[col] = [f"EMP{i:04d}" for i in range(1, len(self.df) + 1)]
                elif col == 'employee_name':
                    fake = Faker()
                    self.df[col] = [fake.name() for _ in range(len(self.df))]
                elif col in ['performance_score', 'productivity_score']:
                    self.df[col] = np.random.randint(50, 100, len(self.df))
                elif col == 'training_hours':
                    self.df[col] = np.random.randint(10, 120, len(self.df))
                elif col == 'experience_years':
                    self.df[col] = np.random.randint(1, 20, len(self.df))
                elif col == 'department':
                    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
                    self.df[col] = np.random.choice(departments, len(self.df))
                elif col == 'position':
                    positions = ['Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager']
                    self.df[col] = np.random.choice(positions, len(self.df))

        # Create derived metrics
        self.df['performance_category'] = pd.cut(
            self.df['performance_score'],
            bins=[0, 60, 80, 100],
            labels=['Needs Improvement', 'Good', 'Excellent']
        )

        self.df['productivity_category'] = pd.cut(
            self.df['productivity_score'],
            bins=[0, 70, 85, 100],
            labels=['Low', 'Medium', 'High']
        )

        self.df['experience_category'] = pd.cut(
            self.df['experience_years'],
            bins=[0, 2, 5, 10, 50],
            labels=['Entry', 'Junior', 'Mid', 'Senior']
        )

        # Risk and potential indicators
        self.df['at_risk'] = (
                (self.df['performance_score'] < 70) |
                (self.df['productivity_score'] < 70) |
                (self.df['training_hours'] < 20)
        ).astype(int)

        self.df['high_potential'] = (
                (self.df['performance_score'] > 85) &
                (self.df['productivity_score'] > 85) &
                (self.df['training_hours'] > 40)
        ).astype(int)

        # Training efficiency
        self.df['training_efficiency'] = (
                self.df['performance_score'] / (self.df['training_hours'] + 1) * 100
        ).clip(0, 200)

        # Overall rating
        self.df['overall_rating'] = (
                self.df['performance_score'] * 0.4 +
                self.df['productivity_score'] * 0.3 +
                (self.df['training_hours'] / self.df['training_hours'].max() * 100) * 0.3
        )

    def apply_filters(self, filters):
        """Apply filters to the dataset"""
        df = self.df.copy()

        if filters.get('department') and filters['department'] != 'all':
            df = df[df['department'] == filters['department']]

        if filters.get('position') and filters['position'] != 'all':
            df = df[df['position'] == filters['position']]

        if filters.get('minPerformance'):
            try:
                df = df[df['performance_score'] >= float(filters['minPerformance'])]
            except (ValueError, TypeError):
                pass

        if filters.get('maxPerformance'):
            try:
                df = df[df['performance_score'] <= float(filters['maxPerformance'])]
            except (ValueError, TypeError):
                pass

        if filters.get('minExperience'):
            try:
                df = df[df['experience_years'] >= float(filters['minExperience'])]
            except (ValueError, TypeError):
                pass

        if filters.get('maxExperience'):
            try:
                df = df[df['experience_years'] <= float(filters['maxExperience'])]
            except (ValueError, TypeError):
                pass

        return df

    def get_kpis(self, filters=None):
        """Calculate key performance indicators"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return {
                'total_employees': 0,
                'avg_performance': 0,
                'avg_productivity': 0,
                'high_performers': 0,
                'at_risk': 0
            }

        return {
            'total_employees': len(df),
            'avg_performance': round(df['performance_score'].mean(), 1),
            'avg_productivity': round(df['productivity_score'].mean(), 1),
            'high_performers': int(df['high_potential'].sum()),
            'at_risk': int(df['at_risk'].sum())
        }

    def get_department_performance(self, filters=None):
        """Get performance metrics by department"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        dept_stats = df.groupby('department').agg({
            'performance_score': 'mean',
            'employee_id': 'count'
        }).round(1)

        return [
            {
                'name': dept,
                'value': float(row['performance_score']),
                'count': int(row['employee_id'])
            }
            for dept, row in dept_stats.iterrows()
        ]

    def get_performance_productivity_scatter(self, filters=None):
        """Get scatter plot data for performance vs productivity"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        return [
            {
                'x': float(row['performance_score']),
                'y': float(row['productivity_score']),
                'z': float(row['training_hours']),
                'name': row['employee_name'],
                'department': row['department']
            }
            for _, row in df.iterrows()
        ]

    def get_training_distribution(self, filters=None):
        """Get training hours distribution"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        return df['training_hours'].tolist()

    def get_position_performance(self, filters=None):
        """Get performance metrics by position"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        pos_stats = df.groupby('position')['performance_score'].mean().round(1)

        return [
            {
                'name': pos,
                'value': float(score)
            }
            for pos, score in pos_stats.items()
        ]

    def get_correlation_matrix(self, filters=None):
        """Get correlation matrix for numerical columns"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return {'matrix': [], 'labels': []}

        numeric_cols = ['performance_score', 'productivity_score', 'training_hours',
                        'experience_years', 'training_efficiency', 'overall_rating']

        available_cols = [col for col in numeric_cols if col in df.columns]
        if len(available_cols) < 2:
            return {'matrix': [], 'labels': []}

        corr_matrix = df[available_cols].corr()

        return {
            'matrix': corr_matrix.values.tolist(),
            'labels': corr_matrix.columns.tolist()
        }

    def get_experience_performance(self, filters=None):
        """Get performance distribution by experience level"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        result = []
        for category in df['experience_category'].unique():
            if pd.notna(category):
                values = df[df['experience_category'] == category]['performance_score'].tolist()
                if values:
                    result.append({
                        'name': str(category),
                        'values': values
                    })

        return result

    def get_department_radar(self, filters=None):
        """Get radar chart data for department comparison"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        metrics = ['performance_score', 'productivity_score', 'training_efficiency']
        categories = ['Performance', 'Productivity', 'Training Efficiency']

        result = []
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']

        for i, dept in enumerate(df['department'].unique()):
            dept_data = df[df['department'] == dept]
            values = [
                float(dept_data['performance_score'].mean()),
                float(dept_data['productivity_score'].mean()),
                float(dept_data['training_efficiency'].mean())
            ]

            result.append({
                'name': dept,
                'values': values,
                'categories': categories,
                'color': colors[i % len(colors)]
            })

        return result

    def get_filters_data(self):
        """Get available filter options"""
        quick_stats = {
            'departments': len(self.df['department'].unique()),
            'avg_training_hours': float(self.df['training_hours'].mean()),
            'top_department': self.df.groupby('department')['performance_score'].mean().idxmax(),
            'avg_productivity': float(self.df['productivity_score'].mean())
        }

        return {
            'departments': sorted(self.df['department'].unique().tolist()),
            'positions': sorted(self.df['position'].unique().tolist()),
            'employees': [
                {
                    'employee_id': row['employee_id'],
                    'employee_name': row['employee_name'],
                    'department': row['department']
                }
                for _, row in self.df[['employee_id', 'employee_name', 'department']].iterrows()
            ],
            'quick_stats': quick_stats
        }

    def get_individual_analysis(self, employee_id):
        """Get detailed analysis for individual employee"""
        employee_data = self.df[self.df['employee_id'] == employee_id]

        if employee_data.empty:
            return None

        emp = employee_data.iloc[0]

        # Radar chart data
        radar_data = [{
            'name': emp['employee_name'],
            'values': [
                float(emp['performance_score']),
                float(emp['productivity_score']),
                float(emp['training_efficiency']),
                float(emp['overall_rating'])
            ],
            'categories': ['Performance', 'Productivity', 'Training Efficiency', 'Overall Rating'],
            'color': '#667eea'
        }]

        # Generate recommendations
        recommendations = []
        if emp['performance_score'] < 70:
            recommendations.append("Focus on core skill development and performance improvement")
        if emp['training_hours'] < self.df['training_hours'].mean():
            recommendations.append("Increase training participation to enhance skills")
        if emp['productivity_score'] > emp['performance_score']:
            recommendations.append("Work on improving quality and effectiveness of output")
        if not recommendations:
            recommendations.append("Continue excellent performance and consider leadership opportunities")

        return {
            'employee': {
                'name': emp['employee_name'],
                'performance_score': float(emp['performance_score']),
                'productivity_score': float(emp['productivity_score']),
                'experience_years': int(emp['experience_years']),
                'training_hours': int(emp['training_hours'])
            },
            'radar_data': radar_data,
            'recommendations': recommendations
        }

    def get_performance_trends(self, filters=None):
        """Get performance trends over time (simulated)"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        # Simulate time-based data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        result = []

        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]
            base_performance = dept_data['performance_score'].mean()

            # Simulate trend with some variation
            trend_data = []
            for i, month in enumerate(months):
                variation = np.random.normal(0, 3)
                performance = max(0, min(100, base_performance + variation + (i * 0.5)))
                trend_data.append(performance)

            result.append({
                'name': dept,
                'x': months,
                'y': trend_data
            })

        return result

    def get_training_effectiveness(self, filters=None):
        """Get training effectiveness data"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        # Calculate performance improvement based on training hours
        result = []
        for _, row in df.iterrows():
            # Simulate training effectiveness
            baseline_performance = max(40, row['performance_score'] - (row['training_hours'] * 0.3))
            improvement = row['performance_score'] - baseline_performance

            result.append({
                'x': float(row['training_hours']),
                'y': float(improvement),
                'name': row['employee_name']
            })

        return result

    def get_productivity_distribution(self, filters=None):
        """Get productivity distribution by department"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        result = []
        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]
            values = dept_data['productivity_score'].tolist()
            if values:
                result.append({
                    'name': dept,
                    'values': values
                })

        return result

    def get_predictive_model_data(self, filters=None):
        """Generate predictive model visualization data"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return {'actual': [], 'predictions': []}

        # Simulate predictions based on features
        actual_performance = df['performance_score'].tolist()

        # Simple prediction model simulation
        predictions = []
        for _, row in df.iterrows():
            # Predict based on training hours, experience, and productivity
            predicted = (row['training_hours'] * 0.3 +
                         row['experience_years'] * 2 +
                         row['productivity_score'] * 0.4 +
                         np.random.normal(0, 5))
            predictions.append(max(0, min(100, predicted)))

        return {
            'actual': actual_performance,
            'predictions': predictions
        }

    def get_cluster_analysis(self, filters=None):
        """Get cluster analysis data"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        # Simple clustering based on performance and productivity
        result = []
        for _, row in df.iterrows():
            # Assign cluster based on performance/productivity quadrants
            if row['performance_score'] >= 75 and row['productivity_score'] >= 75:
                cluster = 'High Performers'
                color = 'green'
            elif row['performance_score'] >= 75:
                cluster = 'Quality Focused'
                color = 'blue'
            elif row['productivity_score'] >= 75:
                cluster = 'Quantity Focused'
                color = 'orange'
            else:
                cluster = 'Needs Development'
                color = 'red'

            result.append({
                'x': float(row['performance_score']),
                'y': float(row['productivity_score']),
                'name': row['employee_name'],
                'cluster': cluster,
                'color': color
            })

        return result

    def get_risk_assessment(self, filters=None):
        """Get risk assessment by department"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return []

        result = []
        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]

            # Calculate risk score based on multiple factors
            avg_performance = dept_data['performance_score'].mean()
            avg_training = dept_data['training_hours'].mean()
            at_risk_count = dept_data['at_risk'].sum()
            total_employees = len(dept_data)

            # Risk score (0-100, higher is more risky)
            risk_score = (
                    (100 - avg_performance) * 0.4 +
                    (100 - min(100, avg_training * 2)) * 0.3 +
                    (at_risk_count / total_employees * 100) * 0.3
            )

            result.append({
                'name': dept,
                'value': float(risk_score)
            })

        return result

    def generate_ai_insights(self, filters=None):
        """Generate AI-powered insights and recommendations"""
        df = self.apply_filters(filters or {})

        if df.empty:
            return {'insights': [], 'recommendations': []}

        insights = []
        recommendations = []

        # Performance Analysis Insights
        high_performers = df[df['performance_score'] > 90]
        low_performers = df[df['performance_score'] < 60]

        if len(high_performers) > 0:
            insights.append(f"Identified {len(high_performers)} exceptional performers with scores above 90")
            # Specific promotion recommendations
            for _, emp in high_performers.head(3).iterrows():
                recommendations.append({
                    'type': 'promotion',
                    'employee': emp['employee_name'],
                    'employee_id': emp['employee_id'],
                    'current_performance': emp['performance_score'],
                    'recommendation': f"Consider promoting {emp['employee_name']} (Performance: {emp['performance_score']:.1f}) - consistently high performer in {emp['department']}",
                    'priority': 'High'
                })

        if len(low_performers) > 0:
            insights.append(f"Found {len(low_performers)} employees requiring immediate attention with scores below 60")
            # Specific improvement recommendations
            for _, emp in low_performers.head(3).iterrows():
                recommendations.append({
                    'type': 'improvement',
                    'employee': emp['employee_name'],
                    'employee_id': emp['employee_id'],
                    'current_performance': emp['performance_score'],
                    'recommendation': f"Urgent: Develop improvement plan for {emp['employee_name']} (Performance: {emp['performance_score']:.1f}) - Consider additional training or mentoring",
                    'priority': 'Critical'
                })

        # Training Effectiveness Analysis
        low_training = df[df['training_hours'] < 20]
        if len(low_training) > 0:
            insights.append(f"{len(low_training)} employees have insufficient training hours (<20 hours)")
            recommendations.append({
                'type': 'training',
                'recommendation': f"Increase training allocation for {len(low_training)} employees to improve overall performance",
                'priority': 'Medium'
            })

        # Department Analysis
        dept_performance = df.groupby('department')['performance_score'].mean()
        best_dept = dept_performance.idxmax()
        worst_dept = dept_performance.idxmin()

        insights.append(f"{best_dept} department leads with {dept_performance[best_dept]:.1f} average performance")
        insights.append(
            f"{worst_dept} department needs attention with {dept_performance[worst_dept]:.1f} average performance")

        if dept_performance[worst_dept] < 70:
            recommendations.append({
                'type': 'department',
                'recommendation': f"Implement department-wide improvement initiative in {worst_dept} - performance below acceptable threshold",
                'priority': 'High'
            })

        # Experience vs Performance Insights
        experienced_low_performers = df[(df['experience_years'] > 5) & (df['performance_score'] < 70)]
        if len(experienced_low_performers) > 0:
            insights.append(
                f"{len(experienced_low_performers)} experienced employees are underperforming - potential skill mismatch")
            recommendations.append({
                'type': 'reassignment',
                'recommendation': f"Review role alignment for {len(experienced_low_performers)} experienced underperformers - consider role reassignment or specialized training",
                'priority': 'Medium'
            })

        # High Potential Identification
        high_potential = df[
            (df['performance_score'] > 80) & (df['productivity_score'] > 80) & (df['experience_years'] < 3)]
        if len(high_potential) > 0:
            insights.append(
                f"Discovered {len(high_potential)} high-potential junior employees for accelerated development")
            for _, emp in high_potential.head(2).iterrows():
                recommendations.append({
                    'type': 'development',
                    'employee': emp['employee_name'],
                    'employee_id': emp['employee_id'],
                    'recommendation': f"Fast-track development for {emp['employee_name']} - shows exceptional potential with {emp['performance_score']:.1f} performance despite limited experience",
                    'priority': 'High'
                })


class NewJoinersAnalytics:
    """Analytics module for tracking new employee joiners"""

    def __init__(self, df):
        self.df = df.copy()
        self.prepare_joiners_data()

    def prepare_joiners_data(self):
        """Prepare new joiners data with joining dates and details"""
        # Generate future joining dates (next 60 days)
        from datetime import datetime, timedelta
        current_date = datetime.now()

        # Create joining dates for the next 60 days
        joining_dates = []
        for i in range(len(self.df)):
            # Random date in next 60 days
            days_ahead = np.random.randint(1, 61)
            join_date = current_date + timedelta(days=days_ahead)
            joining_dates.append(join_date)

        self.df['joining_date'] = joining_dates

        # Add additional joining details
        locations = ['New York', 'London', 'Singapore', 'Mumbai', 'San Francisco', 'Berlin', 'Tokyo', 'Sydney']
        grades = ['L1', 'L2', 'L3', 'L4', 'L5', 'M1', 'M2', 'S1']
        skills = ['Python', 'Java', 'React', 'Data Science', 'DevOps', 'UI/UX', 'Product Management', 'Sales',
                  'Marketing', 'Finance']
        practices = ['Technology', 'Consulting', 'Digital', 'Analytics', 'Cloud', 'Cybersecurity', 'AI/ML']

        self.df['joining_location'] = np.random.choice(locations, len(self.df))
        self.df['offered_grade'] = np.random.choice(grades, len(self.df))
        self.df['primary_skill'] = np.random.choice(skills, len(self.df))
        self.df['practice'] = np.random.choice(practices, len(self.df))
        self.df['joining_status'] = np.random.choice(['Confirmed', 'Pending', 'Documentation'], len(self.df),
                                                     p=[0.7, 0.2, 0.1])

        # Calculate days until joining
        self.df['days_until_joining'] = (self.df['joining_date'] - current_date).dt.days

    def get_joiners_by_timeframe(self):
        """Get joiners grouped by different timeframes"""
        current_date = datetime.now()

        timeframes = {
            '1_day': self.df[self.df['days_until_joining'] <= 1],
            '7_days': self.df[self.df['days_until_joining'] <= 7],
            '14_days': self.df[self.df['days_until_joining'] <= 14],
            '28_days': self.df[self.df['days_until_joining'] <= 28],
            '30_days': self.df[self.df['days_until_joining'] <= 30],
            '60_days': self.df[self.df['days_until_joining'] <= 60]
        }

        result = {}
        for timeframe, data in timeframes.items():
            result[timeframe] = {
                'count': len(data),
                'employees': data[['employee_name', 'department', 'position', 'joining_location',
                                   'offered_grade', 'primary_skill', 'practice', 'joining_date',
                                   'joining_status', 'days_until_joining']].to_dict('records')
            }

        return result

    def get_joiners_summary(self):
        """Get summary statistics for new joiners"""
        total_joiners = len(self.df)

        # Count by timeframes
        next_7_days = len(self.df[self.df['days_until_joining'] <= 7])
        next_14_days = len(self.df[self.df['days_until_joining'] <= 14])
        next_30_days = len(self.df[self.df['days_until_joining'] <= 30])

        # Department distribution
        dept_distribution = self.df['department'].value_counts().to_dict()

        # Location distribution
        location_distribution = self.df['joining_location'].value_counts().to_dict()

        # Grade distribution
        grade_distribution = self.df['offered_grade'].value_counts().to_dict()

        # Status distribution
        status_distribution = self.df['joining_status'].value_counts().to_dict()

        return {
            'total_joiners': total_joiners,
            'next_7_days': next_7_days,
            'next_14_days': next_14_days,
            'next_30_days': next_30_days,
            'department_distribution': dept_distribution,
            'location_distribution': location_distribution,
            'grade_distribution': grade_distribution,
            'status_distribution': status_distribution
        }

    def get_joiners_timeline_chart(self):
        """Get data for joiners timeline visualization"""
        # Group by week
        self.df['week'] = self.df['joining_date'].dt.isocalendar().week
        weekly_counts = self.df.groupby('week').size().reset_index(name='count')

        return {
            'weeks': weekly_counts['week'].tolist(),
            'counts': weekly_counts['count'].tolist()
        }

    def get_location_distribution_chart(self):
        """Get data for location distribution pie chart"""
        location_counts = self.df['joining_location'].value_counts()

        return [
            {
                'name': location,
                'value': int(count)
            }
            for location, count in location_counts.items()
        ]

    def get_grade_analysis(self):
        """Get analysis by offered grades"""
        grade_stats = self.df.groupby('offered_grade').agg({
            'employee_name': 'count',
            'department': lambda x: x.value_counts().index[0],  # Most common department
            'joining_location': lambda x: x.value_counts().index[0]  # Most common location
        }).reset_index()

        grade_stats.columns = ['grade', 'count', 'top_department', 'top_location']

        return grade_stats.to_dict('records')


class DataProcessor:
    """Data processing and validation module"""

    def __init__(self):
        self.required_columns = [
            'employee_id', 'employee_name', 'department', 'position',
            'performance_score', 'productivity_score', 'training_hours', 'experience_years'
        ]

    def load_file(self, filepath):
        """Load data from Excel or CSV file"""
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format")

            # Validate and clean data
            df = self.validate_data(df)
            df = self.clean_data(df)

            return df

        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")

    def validate_data(self, df):
        """Validate data structure and content"""
        if df.empty:
            raise ValueError("Empty dataset")

        return df

    def clean_data(self, df):
        """Clean and standardize data"""
        # Remove duplicates
        df = df.drop_duplicates()

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Validate numeric columns
        numeric_columns = ['performance_score', 'productivity_score', 'training_hours', 'experience_years']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 50)

        return df

    def generate_sample_data(self, n_employees=300):
        """Generate comprehensive sample employee data"""
        fake = Faker()

        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'IT', 'Research']
        positions = ['Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager', 'Director']

        # Generate base employee data
        data = {
            'employee_id': [f"EMP{i:04d}" for i in range(1, n_employees + 1)],
            'employee_name': [fake.name() for _ in range(n_employees)],
            'department': np.random.choice(departments, n_employees),
            'position': np.random.choice(positions, n_employees),
            'experience_years': np.random.randint(1, 20, n_employees),
            'performance_score': np.clip(np.random.normal(75, 15, n_employees), 0, 100).round(1),
            'productivity_score': np.clip(np.random.normal(78, 18, n_employees), 0, 100).round(1),
            'training_hours': np.random.randint(5, 120, n_employees),
            'projects_completed': np.random.randint(1, 15, n_employees),
            'training_completion_rate': np.random.randint(60, 101, n_employees),
            'customer_satisfaction': np.random.randint(1, 11, n_employees),
            'teamwork_rating': np.round(np.random.uniform(1, 5, n_employees), 1),
            'salary': np.random.randint(40000, 150000, n_employees),
            'hire_date': [fake.date_between(start_date='-10y', end_date='today') for _ in range(n_employees)],
            'job_satisfaction': np.random.randint(1, 11, n_employees),
            'attendance_rate': np.clip(np.random.normal(95, 5, n_employees), 70, 100).round(1)
        }

        df = pd.DataFrame(data)
        return df


def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Create directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Global analytics instances
    analytics_instance = None
    joiners_instance = None

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    def get_filters_from_request():
        return {
            'department': request.args.get('department'),
            'position': request.args.get('position'),
            'minPerformance': request.args.get('minPerformance'),
            'maxPerformance': request.args.get('maxPerformance'),
            'minExperience': request.args.get('minExperience'),
            'maxExperience': request.args.get('maxExperience')
        }

    @app.route('/')
    def upload_page():
        return render_template('upload.html')

    @app.route('/', methods=['POST'])
    def upload_file():
        nonlocal analytics_instance, joiners_instance

        if 'file' not in request.files:
            return redirect('/')

        file = request.files['file']
        if file.filename == '':
            return redirect('/')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                processor = DataProcessor()
                df = processor.load_file(filepath)

                # Determine file type and route accordingly
                file_type = request.form.get('file_type', 'analytics')

                if file_type == 'joiners':
                    joiners_instance = NewJoinersAnalytics(df)
                    return redirect('/joiners-dashboard')
                else:
                    analytics_instance = AdvancedEmployeeAnalytics(df)
                    return redirect('/dashboard')

            except Exception as e:
                print(f"Error loading file: {e}")
                return redirect('/')

        return redirect('/')

    @app.route('/sample')
    def use_sample_data():
        nonlocal analytics_instance

        processor = DataProcessor()
        df = processor.generate_sample_data(300)
        analytics_instance = AdvancedEmployeeAnalytics(df)

        return redirect('/dashboard')

    @app.route('/sample-joiners')
    def use_sample_joiners_data():
        nonlocal joiners_instance

        processor = DataProcessor()
        df = processor.generate_sample_data(150)  # Smaller dataset for joiners
        joiners_instance = NewJoinersAnalytics(df)

        return redirect('/joiners-dashboard')

    @app.route('/dashboard')
    def dashboard():
        if not analytics_instance:
            return redirect('/')

        return render_template('dashboard.html')

    @app.route('/joiners-dashboard')
    def joiners_dashboard():
        if not joiners_instance:
            return redirect('/')

        return render_template('joiners_dashboard.html')

    # API Routes
    @app.route('/api/filters')
    def api_filters():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        return jsonify(analytics_instance.get_filters_data())

    @app.route('/api/kpis')
    def api_kpis():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_kpis(filters))

    @app.route('/api/department-performance')
    def api_department_performance():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_department_performance(filters))

    @app.route('/api/performance-productivity-scatter')
    def api_performance_productivity_scatter():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_performance_productivity_scatter(filters))

    @app.route('/api/training-distribution')
    def api_training_distribution():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_training_distribution(filters))

    @app.route('/api/position-performance')
    def api_position_performance():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_position_performance(filters))

    @app.route('/api/correlation-matrix')
    def api_correlation_matrix():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_correlation_matrix(filters))

    @app.route('/api/experience-performance')
    def api_experience_performance():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_experience_performance(filters))

    @app.route('/api/department-radar')
    def api_department_radar():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_department_radar(filters))

    @app.route('/api/performance-trends')
    def api_performance_trends():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_performance_trends(filters))

    @app.route('/api/training-effectiveness')
    def api_training_effectiveness():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_training_effectiveness(filters))

    @app.route('/api/productivity-distribution')
    def api_productivity_distribution():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_productivity_distribution(filters))

    @app.route('/api/individual-analysis')
    def api_individual_analysis():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        employee_id = request.args.get('employee_id')
        if not employee_id:
            return jsonify({'error': 'Employee ID required'}), 400

        result = analytics_instance.get_individual_analysis(employee_id)
        if not result:
            return jsonify({'error': 'Employee not found'}), 404

        return jsonify(result)

    @app.route('/api/predictive-model')
    def api_predictive_model():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_predictive_model_data(filters))

    @app.route('/api/cluster-analysis')
    def api_cluster_analysis():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_cluster_analysis(filters))

    @app.route('/api/risk-assessment')
    def api_risk_assessment():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.get_risk_assessment(filters))

    # AI Insights API
    @app.route('/api/ai-insights')
    def api_ai_insights():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        filters = get_filters_from_request()
        return jsonify(analytics_instance.generate_ai_insights(filters))

    @app.route('/api/ai-chat', methods=['POST'])
    def api_ai_chat():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        try:
            data = request.json
            query = data.get('query', '').lower()

            # Get the dataframe for analysis
            df = analytics_instance.df
            filters = data.get('filters', {})
            filtered_df = analytics_instance.apply_filters(filters)

            response = ""
            employee_list = []

            # Enhanced AI query processing
            if any(word in query for word in ['name', 'list', 'who', 'which employees']):

                if any(word in query for word in ['underperform', 'poor', 'low', 'bad', 'worst', 'below']):
                    # Get underperforming employees
                    threshold = 65
                    if 'below' in query:
                        # Try to extract number after "below"
                        import re
                        numbers = re.findall(r'below\s+(\d+)', query)
                        if numbers:
                            threshold = int(numbers[0])

                    low_performers = filtered_df[filtered_df['performance_score'] < threshold].sort_values(
                        'performance_score')

                    if len(low_performers) > 0:
                        response = f"Found {len(low_performers)} employees with performance scores below {threshold}:"
                        employee_list = [
                            {
                                'name': row['employee_name'],
                                'id': row['employee_id'],
                                'department': row['department'],
                                'position': row['position'],
                                'performance_score': round(row['performance_score'], 1),
                                'productivity_score': round(row['productivity_score'], 1),
                                'training_hours': int(row['training_hours']),
                                'status': 'Needs Improvement'
                            }
                            for _, row in low_performers.head(10).iterrows()
                        ]
                    else:
                        response = f"No employees found with performance scores below {threshold}."

                elif any(word in query for word in ['promote', 'promotion', 'best', 'top', 'high', 'excellent']):
                    # Get top performers
                    threshold = 85
                    if 'above' in query:
                        import re
                        numbers = re.findall(r'above\s+(\d+)', query)
                        if numbers:
                            threshold = int(numbers[0])

                    high_performers = filtered_df[filtered_df['performance_score'] > threshold].sort_values(
                        'performance_score', ascending=False)

                    if len(high_performers) > 0:
                        response = f"Found {len(high_performers)} employees eligible for promotion (performance > {threshold}):"
                        employee_list = [
                            {
                                'name': row['employee_name'],
                                'id': row['employee_id'],
                                'department': row['department'],
                                'position': row['position'],
                                'performance_score': round(row['performance_score'], 1),
                                'productivity_score': round(row['productivity_score'], 1),
                                'training_hours': int(row['training_hours']),
                                'status': 'Promotion Ready'
                            }
                            for _, row in high_performers.head(10).iterrows()
                        ]
                    else:
                        response = f"No employees found with performance scores above {threshold}."

                elif any(word in query for word in ['risk', 'at risk', 'attention', 'concern']):
                    # Get at-risk employees
                    at_risk_employees = filtered_df[filtered_df['at_risk'] == 1].sort_values('performance_score')

                    if len(at_risk_employees) > 0:
                        response = f"Found {len(at_risk_employees)} employees requiring immediate attention:"
                        employee_list = [
                            {
                                'name': row['employee_name'],
                                'id': row['employee_id'],
                                'department': row['department'],
                                'position': row['position'],
                                'performance_score': round(row['performance_score'], 1),
                                'productivity_score': round(row['productivity_score'], 1),
                                'training_hours': int(row['training_hours']),
                                'status': 'At Risk'
                            }
                            for _, row in at_risk_employees.head(10).iterrows()
                        ]
                    else:
                        response = "No employees currently identified as at-risk."

                elif any(word in query for word in ['training', 'skill', 'development']):
                    # Get employees needing training
                    avg_training = filtered_df['training_hours'].mean()
                    low_training = filtered_df[filtered_df['training_hours'] < avg_training * 0.7].sort_values(
                        'training_hours')

                    if len(low_training) > 0:
                        response = f"Found {len(low_training)} employees with below-average training hours (< {avg_training * 0.7:.1f}):"
                        employee_list = [
                            {
                                'name': row['employee_name'],
                                'id': row['employee_id'],
                                'department': row['department'],
                                'position': row['position'],
                                'performance_score': round(row['performance_score'], 1),
                                'productivity_score': round(row['productivity_score'], 1),
                                'training_hours': int(row['training_hours']),
                                'status': 'Needs Training'
                            }
                            for _, row in low_training.head(10).iterrows()
                        ]
                    else:
                        response = "All employees have adequate training hours."

                elif any(word in query for word in ['department', 'team']):
                    # Get department-specific employees
                    dept_mentioned = None
                    for dept in filtered_df['department'].unique():
                        if dept.lower() in query:
                            dept_mentioned = dept
                            break

                    if dept_mentioned:
                        dept_employees = filtered_df[filtered_df['department'] == dept_mentioned].sort_values(
                            'performance_score', ascending=False)
                        response = f"Found {len(dept_employees)} employees in {dept_mentioned} department:"
                        employee_list = [
                            {
                                'name': row['employee_name'],
                                'id': row['employee_id'],
                                'department': row['department'],
                                'position': row['position'],
                                'performance_score': round(row['performance_score'], 1),
                                'productivity_score': round(row['productivity_score'], 1),
                                'training_hours': int(row['training_hours']),
                                'status': 'Department Member'
                            }
                            for _, row in dept_employees.head(10).iterrows()
                        ]
                    else:
                        response = "Please specify which department you'd like to see employees from."

                else:
                    # Default: show all employees
                    all_employees = filtered_df.sort_values('performance_score', ascending=False)
                    response = f"Here are the employees (showing top 10 by performance):"
                    employee_list = [
                        {
                            'name': row['employee_name'],
                            'id': row['employee_id'],
                            'department': row['department'],
                            'position': row['position'],
                            'performance_score': round(row['performance_score'], 1),
                            'productivity_score': round(row['productivity_score'], 1),
                            'training_hours': int(row['training_hours']),
                            'status': 'Employee'
                        }
                        for _, row in all_employees.head(10).iterrows()
                    ]

            elif any(word in query for word in ['average', 'mean', 'statistics', 'stats']):
                # Statistical queries
                avg_performance = filtered_df['performance_score'].mean()
                avg_productivity = filtered_df['productivity_score'].mean()
                avg_training = filtered_df['training_hours'].mean()

                response = f"Current statistics: Average Performance: {avg_performance:.1f}, Average Productivity: {avg_productivity:.1f}, Average Training Hours: {avg_training:.1f}"

            elif any(word in query for word in ['count', 'how many', 'number']):
                # Count queries
                total = len(filtered_df)
                high_perf = len(filtered_df[filtered_df['performance_score'] > 85])
                low_perf = len(filtered_df[filtered_df['performance_score'] < 65])
                at_risk = len(filtered_df[filtered_df['at_risk'] == 1])

                response = f"Employee counts: Total: {total}, High Performers (>85): {high_perf}, Low Performers (<65): {low_perf}, At Risk: {at_risk}"

            elif any(word in query for word in ['compare', 'comparison', 'versus', 'vs']):
                # Department comparison
                dept_performance = filtered_df.groupby('department')['performance_score'].mean().sort_values(
                    ascending=False)
                comparison_text = "Department Performance Comparison: "
                for dept, score in dept_performance.head(5).items():
                    comparison_text += f"{dept}: {score:.1f}, "
                response = comparison_text.rstrip(', ')

            else:
                # General query response
                total_employees = len(filtered_df)
                avg_performance = filtered_df['performance_score'].mean()
                high_performers = len(filtered_df[filtered_df['performance_score'] > 85])
                departments = len(filtered_df['department'].unique())

                response = f"I have data on {total_employees} employees across {departments} departments. Average performance score is {avg_performance:.1f} with {high_performers} high performers. What specific information would you like to know? You can ask about specific employees, departments, performance levels, or request lists of employees by criteria."

            return jsonify({
                'response': response,
                'employee_list': employee_list,
                'has_employees': len(employee_list) > 0
            })

        except Exception as e:
            print(f"AI Chat error: {e}")
            return jsonify({'error': 'Error processing query',
                            'response': 'I encountered an error processing your request. Please try rephrasing your question.'}), 500

    # Chart Download API
    @app.route('/api/download-chart/<chart_type>')
    def api_download_chart(chart_type):
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        try:
            filters = get_filters_from_request()

            # Generate the specific chart data
            if chart_type == 'department-performance':
                data = analytics_instance.get_department_performance(filters)
                chart_title = 'Department Performance Distribution'
            elif chart_type == 'performance-productivity':
                data = analytics_instance.get_performance_productivity_scatter(filters)
                chart_title = 'Performance vs Productivity Analysis'
            elif chart_type == 'correlation-matrix':
                data = analytics_instance.get_correlation_matrix(filters)
                chart_title = 'Performance Correlation Matrix'
            elif chart_type == 'risk-assessment':
                data = analytics_instance.get_risk_assessment(filters)
                chart_title = 'Risk Assessment by Department'
            else:
                return jsonify({'error': 'Invalid chart type'}), 400

            # Create matplotlib figure
            import matplotlib.pyplot as plt
            import seaborn as sns
            from io import BytesIO

            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(12, 8))

            if chart_type == 'department-performance' and data:
                departments = [d['name'] for d in data]
                scores = [d['value'] for d in data]
                colors = plt.cm.Set3(range(len(departments)))
                ax.pie(scores, labels=departments, autopct='%1.1f%%', colors=colors)
                ax.set_title(chart_title, fontsize=16, fontweight='bold')

            elif chart_type == 'performance-productivity' and data:
                x_vals = [d['x'] for d in data]
                y_vals = [d['y'] for d in data]
                z_vals = [d['z'] for d in data]
                scatter = ax.scatter(x_vals, y_vals, c=z_vals, cmap='viridis', s=60, alpha=0.7)
                ax.set_xlabel('Performance Score')
                ax.set_ylabel('Productivity Score')
                ax.set_title(chart_title, fontsize=16, fontweight='bold')
                plt.colorbar(scatter, label='Training Hours')

            elif chart_type == 'correlation-matrix' and data and data['matrix']:
                sns.heatmap(data['matrix'], annot=True, xticklabels=data['labels'],
                            yticklabels=data['labels'], cmap='RdBu_r', center=0, ax=ax)
                ax.set_title(chart_title, fontsize=16, fontweight='bold')

            elif chart_type == 'risk-assessment' and data:
                departments = [d['name'] for d in data]
                risk_scores = [d['value'] for d in data]
                bars = ax.bar(departments, risk_scores, color='coral', alpha=0.7)
                ax.set_xlabel('Department')
                ax.set_ylabel('Risk Score')
                ax.set_title(chart_title, fontsize=16, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.1f}', ha='center', va='bottom')

            plt.tight_layout()

            # Save to bytes
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()

            return send_file(
                img_buffer,
                as_attachment=True,
                download_name=f'{chart_type}_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                mimetype='image/png'
            )

        except Exception as e:
            print(f"Error generating chart: {e}")
            return jsonify({'error': 'Error generating chart'}), 500

    @app.route('/api/joiners/summary')
    def api_joiners_summary():
        if not joiners_instance:
            return jsonify({'error': 'No joiners data loaded'}), 400

        return jsonify(joiners_instance.get_joiners_summary())

    @app.route('/api/joiners/timeframe')
    def api_joiners_timeframe():
        if not joiners_instance:
            return jsonify({'error': 'No joiners data loaded'}), 400

        return jsonify(joiners_instance.get_joiners_by_timeframe())

    @app.route('/api/joiners/timeline')
    def api_joiners_timeline():
        if not joiners_instance:
            return jsonify({'error': 'No joiners data loaded'}), 400

        return jsonify(joiners_instance.get_joiners_timeline_chart())

    @app.route('/api/joiners/locations')
    def api_joiners_locations():
        if not joiners_instance:
            return jsonify({'error': 'No joiners data loaded'}), 400

        return jsonify(joiners_instance.get_location_distribution_chart())

    @app.route('/api/joiners/grades')
    def api_joiners_grades():
        if not joiners_instance:
            return jsonify({'error': 'No joiners data loaded'}), 400

        return jsonify(joiners_instance.get_grade_analysis())

    @app.route('/api/download-report', methods=['POST'])
    def api_download_report():
        if not analytics_instance:
            return jsonify({'error': 'No data loaded'}), 400

        try:
            filters = request.json or {}

            # Generate a comprehensive text report
            kpis = analytics_instance.get_kpis(filters)
            dept_performance = analytics_instance.get_department_performance(filters)

            report_content = f"""
ADVANCED EMPLOYEE ANALYTICS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Total Employees: {kpis['total_employees']}
Average Performance Score: {kpis['avg_performance']}
Average Productivity Score: {kpis['avg_productivity']}
High Performers: {kpis['high_performers']}
At-Risk Employees: {kpis['at_risk']}

DEPARTMENT PERFORMANCE
======================
"""

            for dept in dept_performance:
                report_content += f"{dept['name']}: {dept['value']:.1f} (Count: {dept['count']})\n"

            report_content += """

RECOMMENDATIONS
===============
1. Focus on improving performance in departments with scores below 70
2. Implement targeted training programs for at-risk employees
3. Recognize and retain high performers
4. Monitor productivity trends across all departments

This report provides a comprehensive overview of employee performance metrics
and recommendations for organizational improvement.
"""

            # Create a file-like object
            report_bytes = report_content.encode('utf-8')

            return send_file(
                io.BytesIO(report_bytes),
                as_attachment=True,
                download_name=f'employee_analytics_report_{datetime.now().strftime("%Y%m%d")}.txt',
                mimetype='text/plain'
            )

        except Exception as e:
            print(f"Error generating report: {e}")
            return jsonify({'error': 'Error generating report'}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    print("🚀 Starting Advanced Employee Analytics Dashboard...")
    print("✅ Navigate to: http://127.0.0.1:5000")
    print("📊 Features: Dynamic filtering, multiple chart types, individual analysis")
    print("💾 Supports: Excel (.xlsx) and CSV files")
    app.run(debug=True, host='0.0.0.0', port=5000)