import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


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
                    self.df[col] = [f"Employee {i}" for i in range(1, len(self.df) + 1)]
                else:
                    self.df[col] = np.random.randint(50, 100, len(self.df))

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

        if filters.get('minPerformance') is not None:
            df = df[df['performance_score'] >= float(filters['minPerformance'])]

        if filters.get('maxPerformance') is not None:
            df = df[df['performance_score'] <= float(filters['maxPerformance'])]

        if filters.get('minExperience') is not None:
            df = df[df['experience_years'] >= float(filters['minExperience'])]

        if filters.get('maxExperience') is not None:
            df = df[df['experience_years'] <= float(filters['maxExperience'])]

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