"""
Utilities for Water Resources Learning Modules
Data management, analysis tools, and UI components
Updated with GEV analysis and IDF curve support

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
        """Peak flow data (1974-2023) - Enhanced for GEV analysis"""
        data = {
            'Year': list(range(1974, 2024)),
            'Peak_Flow': [334, 203, 197, 183, 567, 255, 217, 335, 285, 292, 173, 229, 
                         288, 349, 386, 348, 229, 367, 309, 160, 277, 257, 244, 238, 
                         238, 356, 345, 117, 179, 409, 251, 251, 328, 204, 248, 230, 
                         220, 304, 189, 634, 331, 275, 475, 284, 190, 469, 222, 327, 251, 147]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    @st.cache_data
    def get_extended_flood_data() -> pd.DataFrame:
        """Extended flood dataset for advanced GEV analysis"""
        np.random.seed(123)
        # Generate more realistic flood data with some extreme events
        base_data = np.random.lognormal(mean=5.5, sigma=0.6, size=60)
        # Add some extreme events
        extreme_indices = [15, 32, 45, 58]
        for idx in extreme_indices:
            if idx < len(base_data):
                base_data[idx] *= np.random.uniform(1.5, 2.5)
        
        data = {
            'Year': list(range(1964, 2024)),
            'Peak_Flow': base_data
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
        
        # GEV (Generalized Extreme Value)
        try:
            gev_params = stats.genextreme.fit(data)
            results['gev'] = {'params': gev_params, 'dist': stats.genextreme}
        except:
            results['gev'] = None
        
        return results
    
    @staticmethod
    def calculate_risk(return_period: float, design_life: int) -> Tuple[float, float]:
        """Calculate lifetime risk and reliability"""
        annual_prob = 1 / return_period
        risk = 1 - (1 - annual_prob) ** design_life
        reliability = (1 - annual_prob) ** design_life
        return risk, reliability
    
    @staticmethod
    def gev_parameter_interpretation(xi: float, mu: float, sigma: float) -> Dict[str, str]:
        """Interpret GEV parameters for educational purposes"""
        interpretation = {}
        
        # Shape parameter interpretation
        if xi > 0.1:
            interpretation['type'] = "Fréchet (Type II)"
            interpretation['tail'] = "Heavy tail - higher probability of extreme events"
            interpretation['implications'] = "More conservative design needed, potential for very large events"
        elif xi < -0.1:
            interpretation['type'] = "Weibull (Type III)"
            interpretation['tail'] = "Light tail - bounded extreme events"
            interpretation['implications'] = "Upper limit on possible events, less extreme extrapolation"
        else:
            interpretation['type'] = "Gumbel (Type I)"
            interpretation['tail'] = "Medium tail - exponential decay"
            interpretation['implications'] = "Most common in hydrology, standard extrapolation behavior"
        
        # Parameter meanings
        interpretation['location'] = f"μ = {mu:.2f} - Central location parameter"
        interpretation['scale'] = f"σ = {sigma:.2f} - Variability parameter"
        interpretation['shape'] = f"ξ = {xi:.4f} - Tail behavior parameter"
        
        return interpretation
    
    @staticmethod
    def gev_design_values(params: Tuple, return_periods: List[int]) -> Dict[int, float]:
        """Calculate design values for given return periods"""
        design_values = {}
        
        for T in return_periods:
            try:
                prob = 1 - 1/T
                value = stats.genextreme.ppf(prob, *params)
                design_values[T] = value
            except:
                design_values[T] = None
        
        return design_values
    
    @staticmethod
    def goodness_of_fit_measures(data: np.ndarray, dist, params: Tuple) -> Dict[str, float]:
        """Calculate goodness of fit measures"""
        n = len(data)
        measures = {}
        
        try:
            # Log-likelihood
            log_likelihood = np.sum(dist.logpdf(data, *params))
            measures['log_likelihood'] = log_likelihood
            
            # AIC and BIC
            k = len(params)  # number of parameters
            measures['aic'] = 2*k - 2*log_likelihood
            measures['bic'] = k*np.log(n) - 2*log_likelihood
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
            measures['ks_statistic'] = ks_stat
            measures['ks_p_value'] = ks_p_value
            
        except Exception as e:
            measures['error'] = str(e)
        
        return measures
    
    @staticmethod
    def idf_disaggregation(rainfall_30min: np.ndarray, durations: List[int]) -> Dict[int, np.ndarray]:
        """Apply standard disaggregation ratios to convert 30-min data to other durations"""
        # Standard disaggregation ratios based on research
        standard_ratios = {
            5: 0.29,
            10: 0.54, 
            15: 0.70,
            30: 1.00,  # Base data
            60: 0.81,
            120: 0.66
        }
        
        disaggregated_data = {}
        
        for duration in durations:
            if duration in standard_ratios:
                ratio = standard_ratios[duration]
                disaggregated_data[duration] = rainfall_30min * ratio
            else:
                # Linear interpolation for other durations (if needed)
                # For now, just return None for unsupported durations
                disaggregated_data[duration] = None
        
        return disaggregated_data
    
    @staticmethod
    def rainfall_to_intensity(rainfall_depth: np.ndarray, duration_minutes: int) -> np.ndarray:
        """Convert rainfall depth to intensity"""
        return (rainfall_depth / duration_minutes) * 60  # Convert to mm/hr

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
    def idf_curves_plot(idf_data: Dict[int, Dict[int, float]], title: str = "IDF Curves") -> go.Figure:
        """Create IDF curves plot"""
        fig = go.Figure()
        
        durations = sorted(list(idf_data.keys()))
        return_periods = sorted(list(idf_data[durations[0]].keys()))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, T in enumerate(return_periods):
            intensities = [idf_data[dur][T] for dur in durations]
            
            fig.add_trace(go.Scatter(
                x=durations,
                y=intensities,
                mode='lines+markers',
                name=f'{T}-year',
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Duration (minutes)",
            yaxis_title="Intensity (mm/hr)",
            xaxis_type='log',
            yaxis_type='log',
            legend=dict(x=0.7, y=0.95)
        )
        
        return PlotTools.apply_theme(fig)
    
    @staticmethod
    def gev_diagnostic_plots(data: np.ndarray, gev_params: Tuple) -> Dict[str, go.Figure]:
        """Create comprehensive GEV diagnostic plots"""
        plots = {}
        
        # 1. Q-Q Plot
        fig_qq = go.Figure()
        n = len(data)
        p_values = (np.arange(1, n+1) - 0.5) / n
        theoretical_quantiles = stats.genextreme.ppf(p_values, *gev_params)
        empirical_quantiles = np.sort(data)
        
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles, y=empirical_quantiles,
            mode='markers', name='Data Points', marker=dict(size=8)
        ))
        
        # Perfect fit line
        min_val = min(min(theoretical_quantiles), min(empirical_quantiles))
        max_val = max(max(theoretical_quantiles), max(empirical_quantiles))
        fig_qq.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Fit', line=dict(dash='dash', color='red')
        ))
        
        fig_qq.update_layout(
            title='Q-Q Plot: GEV vs Empirical',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Empirical Quantiles'
        )
        plots['qq'] = PlotTools.apply_theme(fig_qq)
        
        # 2. P-P Plot
        fig_pp = go.Figure()
        empirical_cdf = np.arange(1, n+1) / n
        theoretical_cdf = stats.genextreme.cdf(empirical_quantiles, *gev_params)
        
        fig_pp.add_trace(go.Scatter(
            x=theoretical_cdf, y=empirical_cdf,
            mode='markers', name='Data Points', marker=dict(size=8)
        ))
        
        fig_pp.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Perfect Fit', line=dict(dash='dash', color='red')
        ))
        
        fig_pp.update_layout(
            title='P-P Plot: GEV vs Empirical',
            xaxis_title='Theoretical Cumulative Probability',
            yaxis_title='Empirical Cumulative Probability'
        )
        plots['pp'] = PlotTools.apply_theme(fig_pp)
        
        # 3. Return Period Plot
        fig_rp = PlotTools.frequency_plot(data, "Return Period Analysis with GEV Fit")
        
        # Add GEV fitted line
        return_periods_extended = np.logspace(0, 3, 100)
        prob_extended = 1 - 1/return_periods_extended
        gev_values = stats.genextreme.ppf(prob_extended, *gev_params)
        
        fig_rp.add_trace(go.Scatter(
            x=return_periods_extended,
            y=gev_values,
            mode='lines',
            name='GEV Fit',
            line=dict(color='purple', width=3)
        ))
        
        plots['return_period'] = fig_rp
        
        return plots
    
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
    
    @staticmethod
    def parameter_sensitivity_plot(data: np.ndarray, base_params: Tuple) -> go.Figure:
        """Show how GEV parameters affect distribution shape"""
        fig = go.Figure()
        
        xi_base, mu_base, sigma_base = base_params
        x_range = np.linspace(min(data)*0.8, max(data)*1.3, 200)
        
        # Base distribution
        pdf_base = stats.genextreme.pdf(x_range, *base_params)
        fig.add_trace(go.Scatter(
            x=x_range, y=pdf_base, mode='lines',
            name=f'Base (ξ={-xi_base:.3f})', line=dict(width=3)
        ))
        
        # Vary shape parameter
        for delta in [-0.1, 0.1]:
            if xi_base + delta != 0:  # Avoid exactly zero
                new_params = (xi_base + delta, mu_base, sigma_base)
                try:
                    pdf_var = stats.genextreme.pdf(x_range, *new_params)
                    fig.add_trace(go.Scatter(
                        x=x_range, y=pdf_var, mode='lines',
                        name=f'ξ={-(xi_base + delta):.3f}',
                        line=dict(dash='dash', width=2)
                    ))
                except:
                    pass
        
        fig.update_layout(
            title='GEV Shape Parameter Sensitivity',
            xaxis_title='Value',
            yaxis_title='Probability Density'
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
        
        /* GEV specific styling */
        .gev-parameter-box {
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            border: 2px solid #28a745;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .distribution-type-box {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            text-align: center;
        }
        
        /* IDF specific styling */
        .idf-calculator-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 2px solid #2196f3;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .disaggregation-box {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border: 2px solid #9c27b0;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
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
    def gev_parameter_box(xi: float, mu: float, sigma: float):
        """Special box for displaying GEV parameters with interpretation"""
        interpretation = AnalysisTools.gev_parameter_interpretation(xi, mu, sigma)
        
        st.markdown(f"""
        <div class="gev-parameter-box">
            <h4>GEV Parameters</h4>
            <p><strong>{interpretation['location']}</strong></p>
            <p><strong>{interpretation['scale']}</strong></p>  
            <p><strong>{interpretation['shape']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="distribution-type-box">
            <h4>{interpretation['type']}</h4>
            <p>{interpretation['tail']}</p>
            <p><em>{interpretation['implications']}</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def idf_calculator_box(title: str, content: str):
        """Special box for IDF calculations"""
        st.markdown(f"""
        <div class="idf-calculator-box">
            <h4>{title}</h4>
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def disaggregation_box(content: str):
        """Special box for disaggregation ratios and explanations"""
        st.markdown(f"""
        <div class="disaggregation-box">
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
    
    @staticmethod
    def create_idf_base_data(base_data: pd.DataFrame) -> bytes:
        """Create Excel file with base 30-minute rainfall data"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Base data sheet
            base_data.to_excel(writer, sheet_name='Base_Data', index=False)
            
            # Information sheet
            info_data = pd.DataFrame({
                'Parameter': ['Data Type', 'Duration', 'Years of Record', 'Period', 'Source'],
                'Value': ['Annual Maximum Rainfall', '30 minutes', f'{len(base_data)} years', 
                         f'{base_data["Year"].min()}-{base_data["Year"].max()}', 'Generated for Educational Use']
            })
            info_data.to_excel(writer, sheet_name='Information', index=False)
        
        return output.getvalue()
    
    @staticmethod
    def create_idf_workshop_template(base_data: pd.DataFrame) -> bytes:
        """Create Excel template for IDF workshop"""
        n_years = len(base_data)
        
        # Create template with formulas
        template_data = {
            'Year': base_data['Year'].values,
            'Rain_30min': base_data['Rainfall_30min'].values,
            'Rain_5min': [f'=B{i+2}*0.29' for i in range(n_years)],
            'Rain_10min': [f'=B{i+2}*0.54' for i in range(n_years)], 
            'Rain_15min': [f'=B{i+2}*0.70' for i in range(n_years)],
            'Rain_60min': [f'=B{i+2}*0.81' for i in range(n_years)],
            'Rain_120min': [f'=B{i+2}*0.66' for i in range(n_years)],
            'Intensity_5min': [f'=(C{i+2}/5)*60' for i in range(n_years)],
            'Intensity_10min': [f'=(D{i+2}/10)*60' for i in range(n_years)],
            'Intensity_15min': [f'=(E{i+2}/15)*60' for i in range(n_years)],
            'Intensity_30min': [f'=(B{i+2}/30)*60' for i in range(n_years)],
            'Intensity_60min': [f'=(F{i+2}/60)*60' for i in range(n_years)],
            'Intensity_120min': [f'=(G{i+2}/120)*60' for i in range(n_years)]
        }
        
        template_df = pd.DataFrame(template_data)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main calculation sheet
            template_df.to_excel(writer, sheet_name='IDF_Calculations', index=False)
            
            # Frequency analysis template
            freq_template = pd.DataFrame({
                'Duration': ['5min', '10min', '15min', '30min', '60min', '120min'],
                'Data_Range': ['H2:H35', 'I2:I35', 'J2:J35', 'K2:K35', 'L2:L35', 'M2:M35'],
                'Rank_Formula': ['=RANK(H2,$H$2:$H$35,0)', '=RANK(I2,$I$2:$I$35,0)', 
                               '=RANK(J2,$J$2:$J$35,0)', '=RANK(K2,$K$2:$K$35,0)',
                               '=RANK(L2,$L$2:$L$35,0)', '=RANK(M2,$M$2:$M$35,0)'],
                'Notes': ['Copy formula down', 'Copy formula down', 'Copy formula down', 
                         'Copy formula down', 'Copy formula down', 'Copy formula down']
            })
            freq_template.to_excel(writer, sheet_name='Frequency_Analysis', index=False)
            
            # Disaggregation ratios reference
            ratios_ref = pd.DataFrame({
                'Duration': ['5 minutes', '10 minutes', '15 minutes', '30 minutes', '60 minutes', '120 minutes'],
                'Ratio': [0.29, 0.54, 0.70, 1.00, 0.81, 0.66],
                'Source': ['Research', 'Research', 'Research', 'Base Data', 'Extension', 'Extension'],
                'Formula': ['=30min_rain * 0.29', '=30min_rain * 0.54', '=30min_rain * 0.70', 
                           'Base data', '=30min_rain * 0.81', '=30min_rain * 0.66']
            })
            ratios_ref.to_excel(writer, sheet_name='Ratios_Reference', index=False)
            
            # Instructions
            instructions = pd.DataFrame({
                'Step': range(1, 8),
                'Task': [
                    'Load base 30-minute rainfall data',
                    'Apply disaggregation ratios (columns C-G)',
                    'Calculate intensities (columns H-M)', 
                    'Sort each intensity column by descending order',
                    'Calculate ranks and plotting positions',
                    'Determine return periods for design events',
                    'Create IDF curve chart'
                ],
                'Formula_Example': [
                    'Copy from Base_Data sheet',
                    '=B2*0.29 (for 5-min)',
                    '=(C2/5)*60 (for 5-min intensity)',
                    'Use Excel Sort function',
                    '=RANK(H2,$H$2:$H$35,0)',
                    'Interpolate for 2,5,10,25,50,100 years',
                    'Select data and Insert > Chart'
                ]
            })
            instructions.to_excel(writer, sheet_name='Instructions', index=False)
        
        return output.getvalue()
    
    @staticmethod
    def create_idf_template(idf_table: pd.DataFrame, durations: List[int]) -> bytes:
        """Create Excel file with completed IDF analysis"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # IDF table
            idf_table.to_excel(writer, sheet_name='IDF_Table')
            
            # Chart data preparation
            chart_data = []
            for T in idf_table.index:
                for dur in durations:
                    chart_data.append({
                        'Return_Period': T,
                        'Duration': dur,
                        'Intensity': idf_table.loc[T, f'{dur} min']
                    })
            
            chart_df = pd.DataFrame(chart_data)
            chart_df.to_excel(writer, sheet_name='Chart_Data', index=False)
            
            # Design examples
            examples = pd.DataFrame({
                'Project_Type': ['Residential Area', 'Shopping Center', 'Industrial Site', 'Hospital'],
                'Design_Return_Period': ['10-year', '25-year', '50-year', '100-year'],
                'Typical_Duration': ['15 min', '20 min', '30 min', '60 min'],
                'Sample_Intensity': [
                    f"{idf_table.loc[10, '15 min']:.1f} mm/hr",
                    f"{idf_table.loc[25, '15 min']:.1f} mm/hr",
                    f"{idf_table.loc[50, '30 min']:.1f} mm/hr", 
                    f"{idf_table.loc[100, '60 min']:.1f} mm/hr"
                ]
            })
            examples.to_excel(writer, sheet_name='Design_Examples', index=False)
        
        return output.getvalue()
    
    @staticmethod 
    def create_idf_instructions() -> bytes:
        """Create PDF instructions for IDF workshop"""
        # For now, return a simple text file as bytes
        instructions_text = """
IDF CURVE WORKSHOP INSTRUCTIONS

OBJECTIVE: Create Intensity-Duration-Frequency curves from 30-minute rainfall data

MATERIALS NEEDED:
- Excel software
- Base rainfall data (30-minute annual maximums)
- Calculator

STEP-BY-STEP PROCEDURE:

1. DATA PREPARATION
   - Load 30-minute annual maximum rainfall data
   - Verify data completeness (no missing values)
   - Check data range (realistic values)

2. DISAGGREGATION
   - Apply standard ratios to create shorter duration data:
     * 5-min = 30-min × 0.29
     * 10-min = 30-min × 0.54  
     * 15-min = 30-min × 0.70
   - For longer durations:
     * 60-min = 30-min × 0.81
     * 120-min = 30-min × 0.66

3. INTENSITY CALCULATION
   - Convert rainfall depths to intensities
   - Formula: Intensity = (Depth/Duration) × 60
   - Units: mm/hr

4. FREQUENCY ANALYSIS
   - Sort intensity data (highest to lowest)
   - Assign ranks (1, 2, 3, ...)
   - Calculate plotting positions: P = rank/(n+1)
   - Calculate return periods: T = 1/P

5. DESIGN VALUE INTERPOLATION
   - Extract design intensities for standard return periods:
     * 2, 5, 10, 25, 50, 100 years
   - Use interpolation between data points

6. IDF CURVE CREATION
   - Create chart with Duration (X-axis) vs Intensity (Y-axis)
   - Plot separate lines for each return period
   - Use logarithmic scales for better visualization
   - Add proper labels and legend

7. VERIFICATION
   - Check that intensities decrease with longer duration
   - Verify that intensities increase with longer return period
   - Compare with published IDF curves for region

COMMON MISTAKES TO AVOID:
- Using wrong units (check mm/hr vs mm)
- Incorrect disaggregation ratios
- Wrong ranking order (should be descending)
- Missing logarithmic scales on charts

EXPECTED RESULTS:
- Family of curves showing I-D relationship
- Higher return periods = higher intensities
- Shorter durations = higher intensities
- Professional-quality charts suitable for engineering use

FOR QUESTIONS:
Contact TA Saurav Bhattarai
Advanced Water Resources Engineering
Jackson State University
        """
        
        return instructions_text.encode('utf-8')
    
    @staticmethod
    def create_gev_template(data: pd.DataFrame) -> bytes:
        """Create Excel template for GEV analysis"""
        template_data = {
            'Year': data['Year'].values if 'Year' in data.columns else range(len(data)),
            'Data': data.iloc[:, -1].values,
            'Rank': ['=RANK(B2,$B$2:$B$' + str(len(data)+1) + ',0)'] + [''] * (len(data)-1),
            'Plotting_Position': ['=C2/(COUNT($B$2:$B$' + str(len(data)+1) + ')+1)'] + [''] * (len(data)-1),
            'Return_Period': ['=1/D2'] + [''] * (len(data)-1),
            'GEV_CDF': ['Fitted using Python/R'] + [''] * (len(data)-1),
            'Design_Values': ['From GEV parameters'] + [''] * (len(data)-1)
        }
        
        df_template = pd.DataFrame(template_data)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_template.to_excel(writer, sheet_name='GEV_Analysis', index=False)
            
            # GEV Instructions sheet
            instructions = pd.DataFrame({
                'GEV_Analysis_Steps': [
                    '1. Load data and check for stationarity',
                    '2. Fit GEV distribution using Maximum Likelihood',
                    '3. Estimate shape (ξ), location (μ), scale (σ) parameters',
                    '4. Assess goodness of fit (Q-Q plots, AIC/BIC)',
                    '5. Calculate design values for required return periods',
                    '6. Compare with other distributions (Gumbel, etc.)',
                    '7. Estimate confidence intervals',
                    '8. Prepare engineering report'
                ]
            })
            instructions.to_excel(writer, sheet_name='Instructions', index=False)
            
            # Parameter interpretation sheet
            param_guide = pd.DataFrame({
                'Parameter': ['Shape (ξ)', 'Shape (ξ)', 'Shape (ξ)', 'Location (μ)', 'Scale (σ)'],
                'Value_Range': ['ξ > 0', 'ξ = 0', 'ξ < 0', 'Any real number', 'σ > 0'],
                'Distribution_Type': ['Fréchet', 'Gumbel', 'Weibull', 'N/A', 'N/A'],
                'Interpretation': [
                    'Heavy tail - extreme events likely',
                    'Medium tail - exponential decay', 
                    'Light tail - bounded extremes',
                    'Central location of distribution',
                    'Spread/variability of distribution'
                ]
            })
            param_guide.to_excel(writer, sheet_name='Parameter_Guide', index=False)
        
        return output.getvalue()