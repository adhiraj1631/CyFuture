import pandas as pd
import numpy as np
from collections import Counter


class RecruitmentAnalytics:
    """Analytics module for recruitment data analysis"""

    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()

    def prepare_data(self):
        """Prepare and enhance recruitment data"""
        # Normalize column names
        column_mapping = {}
        for col in self.df.columns:
            lower_col = col.lower()
            if any(keyword in lower_col for keyword in ['name', 'employee']):
                column_mapping[col] = 'employee_name'
            elif 'grade' in lower_col:
                column_mapping[col] = 'offered_grade'
            elif 'location' in lower_col:
                column_mapping[col] = 'joining_location'
            elif 'practice' in lower_col:
                column_mapping[col] = 'practice'
            elif 'skill' in lower_col:
                column_mapping[col] = 'skills'

        self.df = self.df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = ['employee_name', 'offered_grade', 'joining_location', 'practice', 'skills']
        for col in required_columns:
            if col not in self.df.columns:
                self.df[col] = 'Unknown'

    def get_recruitment_kpis(self):
        """Calculate recruitment KPIs"""
        return {
            'total_recruits': len(self.df),
            'total_locations': self.df['joining_location'].nunique(),
            'total_practices': self.df['practice'].nunique(),
            'top_grade': self.df['offered_grade'].mode().iloc[0] if not self.df['offered_grade'].empty else 'N/A'
        }

    def get_location_distribution(self):
        """Get recruitment distribution by location"""
        location_counts = self.df['joining_location'].value_counts()
        return [
            {'name': location, 'count': int(count)}
            for location, count in location_counts.items()
        ]

    def get_grade_distribution(self):
        """Get distribution by offered grades"""
        grade_counts = self.df['offered_grade'].value_counts()
        return [
            {'name': grade, 'count': int(count)}
            for grade, count in grade_counts.items()
        ]

    def get_practice_distribution(self):
        """Get distribution by practice areas"""
        practice_counts = self.df['practice'].value_counts()
        return [
            {'name': practice, 'count': int(count)}
            for practice, count in practice_counts.items()
        ]

    def get_skills_analysis(self):
        """Analyze skills distribution"""
        all_skills = []
        for skills_str in self.df['skills'].dropna():
            if isinstance(skills_str, str):
                skills = [skill.strip() for skill in skills_str.split(',')]
                all_skills.extend(skills)

        skill_counts = Counter(all_skills)
        return [
            {'name': skill, 'count': count}
            for skill, count in skill_counts.most_common(20)
        ]

    def get_recruitment_table_data(self):
        """Get formatted table data for display"""
        return self.df.to_dict('records')

    def generate_recruitment_summary_report(self):
        """Generate comprehensive recruitment summary"""
        kpis = self.get_recruitment_kpis()
        location_data = self.get_location_distribution()
        grade_data = self.get_grade_distribution()
        practice_data = self.get_practice_distribution()

        return {
            'kpis': kpis,
            'distributions': {
                'locations': location_data,
                'grades': grade_data,
                'practices': practice_data
            }
        }