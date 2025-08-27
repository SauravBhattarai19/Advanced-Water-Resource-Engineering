"""
Utilities for Water Resources Learning Modules
Data management, analysis tools, and UI components

Author: TA Saurav Bhattarai
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from typing import List, Dict, Optional, Tuple
from io import BytesIO

class DataManager:
    """Manages datasets for learning modules"""
    
    @staticmethod
    @st.cache_data
    def get_precipitation_data() -> pd.DataFrame:
        """Annual maximum precipitation data (1980-2023)"""
        np.random.seed(42)
        data = {
            'Year': list(range(1980, 2024)),
            'Annual_Max_Precip': [
                45, 52, 38, 67, 41, 59, 73, 48, 55, 62, 39, 71, 58, 64, 49,
                77, 43, 68, 56, 72, 51, 44, 66, 53, 75, 47, 61, 69, 42, 74,
                50, 57, 63, 46, 70, 54, 65, 48, 76, 52, 60, 67, 45, 73
            ]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    @st.cache_data  
    def get_flood_data() -> pd.DataFrame:
        """Peak flow data (1974-2023)"""
        data = {
            'Year': list(range(1974, 2024)),
            'Peak_Flow': [334, 203, 197, 183, 567, 255, 217, 335, 285, 292, 173, 229, 
                         288, 349, 386, 348, 229, 367, 309, 160, 277, 257, 244, 238, 
                         238, 356, 345, 117, 179, 409, 251, 251, 328, 204, 248, 230, 
                         220, 304, 189, 634, 331, 275, 475, 284, 190, 469, 222, 327, 251, 147]
        }
        return pd.DataFrame(data)

class AnalysisTools:
    """Statistical analysis functions"""
    
    @staticmethod
    def weibull_positions(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Weibull plotting positions"""
        n = len(data)
        sorted_data = np.sort(data)[::-1]  # Descending order
        ranks = np.arange(1, n + 1)
        plotting_positions = ranks / (n + 1)
        return_periods = 1 / plotting_positions
        return sorted_data, return_periods, plotting_positions
    
    @staticmethod
    def fit_distributions(data: np.ndarray) -> Dict[str, Dict]:
        """Fit common distributions to data"""
        results = {}
        
        # Normal
        mu, sigma = stats.norm.fit(data)
        results['normal'] = {'params': (mu, sigma), 'dist': stats.norm}
        
        # Log-normal
        s, loc, scale = stats.lognorm.fit(data, floc=0)
        results['lognormal'] = {'params': (s, loc, scale), 'dist': stats.lognorm}
        
        # Exponential
        loc, scale = stats.expon.fit(data)
        results['exponential'] = {'params': (loc, scale), 'dist': stats.expon}
        
        # Gumbel
        loc, scale = stats.gumbel_r.fit(data)
        results['gumbel'] = {'params': (loc, scale), 'dist': stats.gumbel_r}
        
        return results
    
    @staticmethod
    def calculate_risk(return_period: float, design_life: int) -> Tuple[float, float]:
        """Calculate lifetime risk and reliability"""
        annual_prob = 1 / return_period
        risk = 1 - (1 - annual_prob) ** design_life
        reliability = (1 - annual_prob) ** design_life
        return risk, reliability

class PlotTools:
    """Plotting utilities with consistent styling"""
    
    @staticmethod
    def apply_theme(fig):
        """Apply consistent plot theme"""
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'size': 14, 'color': '#2c3e50'},
            title_font={'size': 18, 'color': '#2c3e50'},
            margin=dict(t=60, l=60, r=30, b=60),
            showlegend=True
        )
        fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1', gridwidth=1)
        fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1', gridwidth=1)
        return fig
    
    @staticmethod
    def frequency_plot(data: np.ndarray, title: str = "Frequency Analysis") -> go.Figure:
        """Create frequency analysis plot"""
        sorted_data, return_periods, _ = AnalysisTools.weibull_positions(data)
        
        fig = go.Figure()
        
        # Data points
        fig.add_trace(go.Scatter(
            x=return_periods,
            y=sorted_data,
            mode='markers',
            name='Observed Data',
            marker=dict(size=10, color='#3498db')
        ))
        
        # Trend line
        log_T = np.log(return_periods)
        coeffs = np.polyfit(log_T, sorted_data, 1)
        trend_T = np.logspace(0, 2, 50)
        trend_y = coeffs[0] * np.log(trend_T) + coeffs[1]
        
        fig.add_trace(go.Scatter(
            x=trend_T,
            y=trend_y,
            mode='lines',
            name='Trend Line',
            line=dict(color='#e74c3c', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Return Period (years)',
            yaxis_title='Value',
            xaxis_type='log'
        )
        
        return PlotTools.apply_theme(fig)
    
    @staticmethod
    def risk_vs_life_plot(return_period: float) -> go.Figure:
        """Create risk vs design life plot"""
        life_years = np.arange(1, 101)
        risks = [AnalysisTools.calculate_risk(return_period, life)[0] for life in life_years]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=life_years,
            y=risks,
            mode='lines',
            name=f'{return_period}-Year Event',
            line=dict(color='#e74c3c', width=3)
        ))
        
        # Risk level lines
        for level, color in [(0.1, '#27ae60'), (0.2, '#f39c12'), (0.5, '#e74c3c')]:
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                         annotation_text=f"{level*100:.0f}% Risk")
        
        fig.update_layout(
            title=f'Lifetime Risk for {return_period}-Year Design Event',
            xaxis_title='Design Life (years)',
            yaxis_title='Lifetime Risk'
        )
        
        return PlotTools.apply_theme(fig)

class UIComponents:
    """UI components and styling"""
    
    @staticmethod
    def load_presentation_css():
        """Load CSS for presentation-style interface"""
        st.markdown("""
        <style>
        /* Main styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            margin: 0;
            font-weight: 700;
        }
        
        .course-info {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1rem;
            font-size: 1rem;
        }
        
        .module-header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .module-header h1 {
            margin: 0;
            font-size: 2rem;
        }
        
        .module-meta {
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }
        
        .difficulty {
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-weight: bold;
            margin-right: 1rem;
        }
        
        .difficulty-beginner { background: #27ae60; }
        .difficulty-intermediate { background: #f39c12; }
        .difficulty-advanced { background: #e74c3c; }
        
        /* Slide styling */
        .slide-content {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            min-height: 500px;
        }
        
        .theory-slide {
            background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
            border-left: 5px solid #3498db;
        }
        
        .interactive-slide {
            background: linear-gradient(135deg, #fff8f0 0%, #ffe6cc 100%);
            border-left: 5px solid #f39c12;
        }
        
        .quiz-slide {
            background: linear-gradient(135deg, #f0fff4 0%, #c6f7d6 100%);
            border-left: 5px solid #27ae60;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
            border: 2px solid #e74c3c;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Compact text for presentations */
        .big-number {
            font-size: 3rem;
            font-weight: bold;
            color: #3498db;
            text-align: center;
        }
        
        .formula-box {
            background: #2c3e50;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2rem;
            font-family: 'Courier New', monospace;
            margin: 1rem 0;
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%);
            border: 2px solid #f1c40f;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        /* Navigation styling */
        .stButton > button {
            border-radius: 20px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def slide_container(content_type: str = "default"):
        """Create slide container with appropriate styling"""
        class_name = f"slide-content {content_type}-slide" if content_type != "default" else "slide-content"
        return st.container()
    
    @staticmethod
    def formula_display(formula: str, description: str = ""):
        """Display mathematical formula"""
        st.markdown(f"""
        <div class="formula-box">
            {formula}
        </div>
        """, unsafe_allow_html=True)
        
        if description:
            st.markdown(f"*{description}*")
    
    @staticmethod
    def big_number_display(number: str, label: str):
        """Display large number with label"""
        st.markdown(f"""
        <div class="big-number">{number}</div>
        <div style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">{label}</div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def highlight_box(content: str):
        """Create highlighted content box"""
        st.markdown(f"""
        <div class="highlight-box">
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def two_column_layout():
        """Create two-column layout for slides"""
        return st.columns(2)
    
    @staticmethod
    def three_column_layout():
        """Create three-column layout for slides"""
        return st.columns(3)

class ExcelExporter:
    """Excel template generation for hands-on exercises"""
    
    @staticmethod
    def create_weibull_template(data: pd.DataFrame) -> bytes:
        """Create Excel template for Weibull analysis"""
        template_data = {
            'Year': data['Year'].values if 'Year' in data.columns else range(len(data)),
            'Data': data.iloc[:, -1].values,  # Last column assumed to be data
            'Rank': ['=RANK(B2,$B$2:$B$' + str(len(data)+1) + ',0)'] + [''] * (len(data)-1),
            'Plotting_Position': ['=C2/(COUNT($B$2:$B$' + str(len(data)+1) + ')+1)'] + [''] * (len(data)-1),
            'Return_Period': ['=1/D2'] + [''] * (len(data)-1)
        }
        
        df_template = pd.DataFrame(template_data)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_template.to_excel(writer, sheet_name='Analysis', index=False)
            
            # Instructions sheet
            instructions = pd.DataFrame({
                'Instructions': [
                    '1. Copy rank formula down column C',
                    '2. Copy plotting position formula down column D', 
                    '3. Copy return period formula down column E',
                    '4. Create scatter plot: Return Period (X) vs Data (Y)',
                    '5. Set X-axis to logarithmic scale',
                    '6. Add trendline for extrapolation'
                ]
            })
            instructions.to_excel(writer, sheet_name='Instructions', index=False)
        
        return output.getvalue()