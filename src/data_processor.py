import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


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

        # Check for minimum required columns
        missing_cols = []
        for col in self.required_columns:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Will be generated automatically.")

        return df

    def clean_data(self, df):
        """Clean and standardize data"""
        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = self.handle_missing_values(df)

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Validate numeric columns
        numeric_columns = ['performance_score', 'productivity_score', 'training_hours', 'experience_years']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())

        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill missing employee IDs
        if 'employee_id' not in df.columns or df['employee_id'].isna().any():
            df['employee_id'] = [f"EMP{i:04d}" for i in range(1, len(df) + 1)]

        # Fill missing employee names
        if 'employee_name' not in df.columns or df['employee_name'].isna().any():
            fake = Faker()
            df['employee_name'] = df['employee_name'].fillna(fake.name())

        # Fill missing departments
        if 'department' not in df.columns:
            departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
            df['department'] = np.random.choice(departments, len(df))

        # Fill missing positions
        if 'position' not in df.columns:
            positions = ['Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager']
            df['position'] = np.random.choice(positions, len(df))

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
            'last_promotion': [fake.date_between(start_date='-3y', end_date='today') for _ in range(n_employees)],
            'job_satisfaction': np.random.randint(1, 11, n_employees),
            'attendance_rate': np.clip(np.random.normal(95, 5, n_employees), 70, 100).round(1),
            'avg_delivery_delay': np.clip(np.random.normal(2, 3, n_employees), 0, 10).round(2),
            'project_success_rate': np.clip(np.random.normal(85, 10, n_employees), 50, 100).round(1)
        }

        df = pd.DataFrame(data)

        # Add some realistic correlations
        df = self.add_realistic_correlations(df)

        return df

    def add_realistic_correlations(self, df):
        """Add realistic correlations between variables"""
        # Performance should correlate with experience and training
        for i, row in df.iterrows():
            # Adjust performance based on experience and training
            experience_boost = min(10, row['experience_years'] * 0.5)
            training_boost = min(15, row['training_hours'] * 0.1)

            # Add some randomness but maintain correlation
            adjustment = experience_boost + training_boost + np.random.normal(0, 5)
            new_performance = np.clip(row['performance_score'] + adjustment, 0, 100)
            df.at[i, 'performance_score'] = round(new_performance, 1)

        # Productivity should correlate somewhat with performance
        for i, row in df.iterrows():
            correlation_factor = 0.3  # 30% correlation
            random_factor = 0.7  # 70% random

            performance_influence = row['performance_score'] * correlation_factor
            random_influence = np.random.normal(75, 15) * random_factor

            new_productivity = np.clip(performance_influence + random_influence, 0, 100)
            df.at[i, 'productivity_score'] = round(new_productivity, 1)

        # Job satisfaction should correlate with performance and salary
        for i, row in df.iterrows():
            perf_influence = (row['performance_score'] - 50) * 0.05  # Scale to satisfaction range
            salary_influence = (row['salary'] - 70000) / 20000  # Salary influence

            base_satisfaction = 5 + perf_influence + salary_influence + np.random.normal(0, 1.5)
            df.at[i, 'job_satisfaction'] = np.clip(round(base_satisfaction, 1), 1, 10)

        return df

    def export_data(self, df, filepath, format='xlsx'):
        """Export processed data to file"""
        try:
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif format.lower() in ['xlsx', 'excel']:
                df.to_excel(filepath, index=False)
            else:
                raise ValueError("Unsupported export format")

            return True

        except Exception as e:
            raise Exception(f"Error exporting data: {str(e)}")

    def get_data_summary(self, df):
        """Get summary statistics of the dataset"""
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'categorical_summary': {}
        }

        # Get categorical column summaries
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].empty else None
            }

        return summary