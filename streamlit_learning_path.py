"""
Advanced Water Resources Engineering: Frequency Analysis
From Data to Design - A Comprehensive Learning Journey

This application provides a structured learning path for understanding frequency analysis
in water resources engineering, progressing from basic data analysis to advanced
probability theory and practical design applications.

Course: Advanced Water Resources Engineering
Level: Senior Undergraduate
Author: Dr. [Instructor Name]
Date: August 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import math
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Water Resources Frequency Analysis - Learning Path",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 1rem;
    }
    .module-header {
        font-size: 1.8rem;
        color: #0f172a;
        margin: 1.5rem 0 1rem 0;
        padding: 1rem;
        background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 100%);
        border-left: 5px solid #3b82f6;
        border-radius: 0.5rem;
        font-weight: 500;
    }
    .theory-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 0.75rem;
        border: 1px solid #cbd5e1;
        margin: 1.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .data-exploration {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #93c5fd;
        margin: 1rem 0;
    }
    .quiz-section {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #f59e0b;
        margin: 1.5rem 0;
    }
    .excel-section {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #10b981;
        margin: 1.5rem 0;
    }
    .equation-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        text-align: center;
    }
    .key-concept {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 4px solid #059669;
        margin: 1rem 0;
    }
    .warning-note {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 4px solid #d97706;
        margin: 1rem 0;
    }
    .progress-tracker {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_precipitation_data():
    """Load sample precipitation data for analysis"""
    # Sample annual maximum precipitation data (mm)
    np.random.seed(42)  # For reproducible results
    precip_data = {
        'Year': list(range(1980, 2024)),
        'Annual_Max_Precip': [
            45, 52, 38, 67, 41, 59, 73, 48, 55, 62, 39, 71, 58, 64, 49,
            77, 43, 68, 56, 72, 51, 44, 66, 53, 75, 47, 61, 69, 42, 74,
            50, 57, 63, 46, 70, 54, 65, 48, 76, 52, 60, 67, 45, 73
        ]
    }
    return pd.DataFrame(precip_data)

@st.cache_data
def load_flood_data():
    """Load flood data for comparison"""
    data = {
        'Year': list(range(1974, 2024)),
        'Peak_Flow': [334, 203, 197, 183, 567, 255, 217, 335, 285, 292, 173, 229, 288, 349, 
                     386, 348, 229, 367, 309, 160, 277, 257, 244, 238, 238, 356, 345, 117, 
                     179, 409, 251, 251, 328, 204, 248, 230, 220, 304, 189, 634, 331, 275, 
                     475, 284, 190, 469, 222, 327, 251, 147]
    }
    return pd.DataFrame(data)

def weibull_plotting_position(data):
    """Calculate Weibull plotting positions"""
    n = len(data)
    sorted_data = np.sort(data)[::-1]  # Sort in descending order
    ranks = np.arange(1, n + 1)
    plotting_positions = ranks / (n + 1)
    return_periods = 1 / plotting_positions
    return sorted_data, return_periods, plotting_positions

def create_quiz_question(question, options, correct_idx, explanation):
    """Create an interactive quiz question"""
    st.markdown('<div class="quiz-section">', unsafe_allow_html=True)
    st.markdown(f"### 🎯 Knowledge Check")
    st.markdown(f"**{question}**")
    
    user_answer = st.radio("Select your answer:", options, key=f"quiz_{hash(question)}")
    
    if st.button("Submit Answer", key=f"submit_{hash(question)}"):
        if user_answer == options[correct_idx]:
            st.success(f"✅ **Correct!** {explanation['correct']}")
            return True
        else:
            st.error(f"❌ **Incorrect.** {explanation['incorrect']}")
            return False
    
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def generate_excel_template(data, analysis_type):
    """Generate Excel template for hands-on practice"""
    if analysis_type == "weibull":
        template_data = {
            'Year': data['Year'].values,
            'Precipitation (mm)': data['Annual_Max_Precip'].values,
            'Rank': '',
            'Plotting_Position': '',
            'Return_Period': ''
        }
    else:
        template_data = {
            'Year': data['Year'].values,
            'Flow (m³/s)': data['Peak_Flow'].values,
            'Rank': '',
            'Plotting_Position': '',
            'Return_Period': ''
        }
    
    df_template = pd.DataFrame(template_data)
    
    # Convert to Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_template.to_excel(writer, sheet_name='Data', index=False)
        
        # Add instructions sheet
        instructions = pd.DataFrame({
            'Instructions': [
                '1. Sort data in descending order',
                '2. Assign ranks (1 to n)',
                '3. Calculate plotting position: P = m/(n+1)',
                '4. Calculate return period: T = 1/P',
                '5. Plot T vs Data on semi-log paper',
                '6. Fit trend line and extrapolate'
            ]
        })
        instructions.to_excel(writer, sheet_name='Instructions', index=False)
    
    return output.getvalue()

def apply_theme(fig):
    """Apply consistent theme to plots"""
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': '#374151', 'size': 12},
        title_font={'size': 16, 'color': '#1f2937'},
        showlegend=True,
        margin=dict(t=60, l=60, r=20, b=60)
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#e5e7eb')
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Water Resources Frequency Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #64748b; margin-bottom: 1rem;"><em>From Data Exploration to Engineering Design</em></p>', unsafe_allow_html=True)
    
    # Attribution
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                 padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem; border: 1px solid #0ea5e9;">
        <p style="margin: 0; font-size: 0.95rem; color: #0c4a6e;">
            <strong>🎓 Made by TA Saurav Bhattarai</strong> for the class of <strong>Advanced Water Resource Engineering</strong><br>
            taught by <strong>Dr. Rocky Talchabhadel</strong> at <strong>Jackson State University</strong> - Fall 2025<br>
            <span style="color: #0369a1;">🐍 Built with Python & Streamlit</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress tracking
    if 'completed_modules' not in st.session_state:
        st.session_state.completed_modules = set()
    
    # Sidebar navigation
    st.sidebar.title("📚 Learning Modules")
    
    modules = [
        "1. Data Exploration & Weibull Analysis",
        "2. Understanding Probability Concepts", 
        "3. Risk, Reliability & Return Periods",
        "4. Probability Distribution Functions",
        "5. CDF vs PDF - Key Differences",
        "6. Distribution Selection & Fitting",
        "7. Distribution Shapes & Equations",
        "8. Engineering Applications",
        "9. Excel Workshop & Practice",
        "10. Comprehensive Design Project"
    ]
    
    # Progress indicator
    completed_count = len(st.session_state.completed_modules)
    st.sidebar.progress(completed_count / len(modules))
    st.sidebar.write(f"Progress: {completed_count}/{len(modules)} modules completed")
    
    selected_module = st.sidebar.selectbox("Select Module:", modules)
    
    # Module content
    if selected_module == "1. Data Exploration & Weibull Analysis":
        st.markdown('<div class="module-header">Module 1: Data Exploration & Weibull Analysis</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="theory-section">', unsafe_allow_html=True)
        st.markdown("""
        ### Learning Objectives
        - Understand the importance of data visualization in engineering analysis
        - Learn the Weibull plotting position method
        - Apply frequency analysis to precipitation data
        - Develop skills in data ranking and probability estimation
        
        ### Why Start with Precipitation?
        Precipitation frequency analysis is fundamental to:
        - **Stormwater design** (drainage systems, detention ponds)
        - **Infrastructure planning** (culverts, storm sewers)
        - **Flood risk assessment** (urban hydrology)
        - **Climate analysis** (extreme event characterization)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load and display data
        precip_df = load_precipitation_data()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Annual Maximum Precipitation Data")
            st.markdown('<div class="data-exploration">', unsafe_allow_html=True)
            
            # Interactive data exploration
            st.dataframe(precip_df.tail(10), use_container_width=True)
            
            # Basic statistics
            precip_data = precip_df['Annual_Max_Precip'].values
            st.markdown(f"""
            **Dataset Summary (1980-2023):**
            - Sample size (n): {len(precip_data)}
            - Mean: {np.mean(precip_data):.1f} mm
            - Standard deviation: {np.std(precip_data, ddof=1):.1f} mm
            - Minimum: {np.min(precip_data):.1f} mm
            - Maximum: {np.max(precip_data):.1f} mm
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Time series plot
            fig = px.line(precip_df, x='Year', y='Annual_Max_Precip', 
                         title='Annual Maximum Precipitation Time Series',
                         markers=True)
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Annual Maximum Precipitation (mm)',
                height=400
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Interactive Data Analysis")
            
            # Let users select year range
            year_range = st.slider(
                "Select year range for analysis:",
                1980, 2023, (1980, 2023), key="year_range_1"
            )
            
            filtered_data = precip_df[
                (precip_df['Year'] >= year_range[0]) & 
                (precip_df['Year'] <= year_range[1])
            ]['Annual_Max_Precip'].values
            
            st.info(f"""
            **Selected Period Analysis:**
            - Years: {year_range[0]}-{year_range[1]}
            - Sample size: {len(filtered_data)}
            - Mean: {np.mean(filtered_data):.1f} mm
            - Max: {np.max(filtered_data):.1f} mm
            """)
            
            # Histogram
            fig_hist = px.histogram(
                x=filtered_data, 
                nbins=8,
                title=f'Precipitation Distribution ({year_range[0]}-{year_range[1]})'
            )
            fig_hist.update_layout(
                xaxis_title='Precipitation (mm)',
                yaxis_title='Frequency',
                height=300
            )
            fig_hist = apply_theme(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Weibull Analysis Section
        st.markdown("### 📈 Weibull Plotting Position Analysis")
        
        st.markdown('<div class="theory-section">', unsafe_allow_html=True)
        st.markdown("""
        ### The Weibull Method
        
        The Weibull plotting position is a non-parametric method for estimating probabilities:
        
        **Step-by-Step Procedure:**
        1. **Rank the data** in descending order (largest = rank 1)
        2. **Calculate plotting position**: P = m/(n+1)
        3. **Calculate return period**: T = 1/P
        4. **Plot** data vs return period on appropriate scale
        
        Where:
        - m = rank of the observation
        - n = total number of observations
        - P = plotting position (probability of exceedance)
        - T = return period (years)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Perform Weibull analysis
        sorted_precip, return_periods, plotting_positions = weibull_plotting_position(precip_data)
        
        # Create analysis table
        weibull_df = pd.DataFrame({
            'Rank': range(1, len(sorted_precip) + 1),
            'Precipitation (mm)': sorted_precip,
            'Plotting Position': plotting_positions,
            'Return Period (years)': return_periods
        })
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📋 Weibull Analysis Results")
            st.dataframe(weibull_df.head(15), use_container_width=True)
            
            st.markdown('<div class="key-concept">', unsafe_allow_html=True)
            st.markdown(f"""
            ### Key Insights:
            - **Most frequent event** (T≈1.1 years): {sorted_precip[-1]:.1f} mm
            - **10-year event**: {np.interp(10, return_periods[::-1], sorted_precip[::-1]):.1f} mm
            - **25-year event**: {np.interp(25, return_periods[::-1], sorted_precip[::-1]):.1f} mm
            - **50-year event**: {np.interp(50, return_periods[::-1], sorted_precip[::-1]):.1f} mm
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Frequency Analysis Plot")
            
            # Create frequency plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=return_periods,
                y=sorted_precip,
                mode='markers',
                name='Observed Data',
                marker=dict(size=8, color='#2563eb')
            ))
            
            # Add trend line
            log_T = np.log(return_periods)
            coeffs = np.polyfit(log_T, sorted_precip, 1)
            trend_T = np.logspace(0, 2, 100)  # 1 to 100 years on log scale
            trend_precip = coeffs[0] * np.log(trend_T) + coeffs[1]

            fig.add_trace(go.Scatter(
                x=trend_T,
                y=trend_precip,
                mode='lines',
                name='Trend Line',
                line=dict(color='#dc2626', dash='dash')
            ))
            
            fig.update_layout(
                title='Precipitation Frequency Analysis (Weibull Method)',
                xaxis_title='Return Period (years)',
                yaxis_title='Annual Maximum Precipitation (mm)',
                xaxis_type='log',
                xaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                    ticktext=['1', '2', '5', '10', '20', '50', '100', '200', '500', '1000'],
                    range=[0, 3]  # Limit X-axis to ~1000 (log10(1000) ≈ 3)
                ),
                height=500
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Knowledge check
        result = create_quiz_question(
            "In the Weibull plotting position method, if an event has a plotting position P = 0.04, what is its return period?",
            ["4 years", "25 years", "0.04 years", "96 years"],
            1,
            {
                "correct": "Exactly! Return period T = 1/P = 1/0.04 = 25 years. This means the event has a 4% chance of being exceeded in any given year.",
                "incorrect": "Remember: Return period T = 1/P. So if P = 0.04, then T = 1/0.04 = 25 years."
            }
        )
        
        if result == True:
            st.session_state.completed_modules.add("1. Data Exploration & Weibull Analysis")
        
        # Excel section
        st.markdown("### 📊 Excel Practice")
        st.markdown('<div class="excel-section">', unsafe_allow_html=True)
        st.markdown("""
        ### Hands-on Excel Exercise
        
        **Download the template and practice the Weibull method:**
        1. Download the Excel template below
        2. Complete the Weibull analysis calculations
        3. Create the frequency plot
        4. Compare your results with the analysis above
        
        **Excel Functions You'll Use:**
        - `RANK()` for ranking data
        - `=(rank)/(count+1)` for plotting position
        - `=1/plotting_position` for return period
        - Chart wizard for plotting
        """)
        
        # Generate Excel template
        excel_data = generate_excel_template(precip_df, "weibull")
        
        st.download_button(
            label="📥 Download Excel Template",
            data=excel_data,
            file_name="precipitation_weibull_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_module == "2. Understanding Probability Concepts":
        st.markdown('<div class="module-header">Module 2: Understanding Probability Concepts</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="theory-section">', unsafe_allow_html=True)
        st.markdown("""
        ### Learning Objectives
        - Define probability in engineering context
        - Distinguish between frequency and probability
        - Understand exceedance vs non-exceedance probability
        - Connect probability to engineering risk assessment
        
        ### What is Probability?
        
        **Probability** is a measure of uncertainty, expressed as a number between 0 and 1:
        - **P = 0**: Event will never occur
        - **P = 1**: Event will certainly occur
        - **P = 0.5**: Event has equal chance of occurring or not occurring
        
        **In Engineering Context:**
        Probability quantifies the likelihood of hydrologic events for design and risk assessment.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive probability concepts
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 Probability Fundamentals")
            
            # Interactive probability calculator
            event_count = st.slider("Number of times event occurred:", 0, 50, 10)
            total_years = st.slider("Total years of record:", 10, 100, 44)
            
            if total_years > 0:
                probability = event_count / total_years
                percentage = probability * 100
                
                st.markdown('<div class="key-concept">', unsafe_allow_html=True)
                st.markdown(f"""
                ### Calculation Results:
                - **Frequency**: {event_count} out of {total_years} years
                - **Probability**: {probability:.3f}
                - **Percentage**: {percentage:.1f}%
                - **Return Period**: {1/probability:.1f} years (if P > 0)
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 📊 Probability Types")
            prob_type = st.selectbox(
                "Select probability type:",
                ["Exceedance (P ≥ x)", "Non-exceedance (P < x)"]
            )
            
            if prob_type == "Exceedance (P ≥ x)":
                st.info("**Exceedance Probability**: The chance that a value will be **greater than or equal to** a specified level. Used for flood/storm design.")
            else:
                st.info("**Non-exceedance Probability**: The chance that a value will be **less than** a specified level. Used for drought/low-flow analysis.")
        
        with col2:
            st.markdown("### 📈 Probability from Weibull Analysis")
            
            # Use precipitation data from Module 1
            precip_df = load_precipitation_data()
            precip_data = precip_df['Annual_Max_Precip'].values
            sorted_precip, return_periods, plotting_positions = weibull_plotting_position(precip_data)
            
            # Interactive threshold selection
            threshold = st.slider(
                "Select precipitation threshold (mm):",
                float(min(precip_data)), float(max(precip_data)), 60.0
            )
            
            # Calculate probabilities
            exceedance_count = sum(1 for p in precip_data if p >= threshold)
            exceedance_prob = exceedance_count / len(precip_data)
            non_exceedance_prob = 1 - exceedance_prob
            
            # Visualization
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=precip_data,
                nbinsx=12,
                name='Precipitation Data',
                opacity=0.7,
                marker_color='lightblue'
            ))
            
            # Add threshold line
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold:.1f} mm"
            )
            
            # Shade exceedance area
            above_threshold = [p for p in precip_data if p >= threshold]
            if above_threshold:
                fig.add_trace(go.Histogram(
                    x=above_threshold,
                    nbinsx=12,
                    name='Exceedance Events',
                    marker_color='red',
                    opacity=0.8
                ))
            
            fig.update_layout(
                title='Probability Visualization',
                xaxis_title='Annual Maximum Precipitation (mm)',
                yaxis_title='Frequency',
                height=400
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # Results
            st.markdown('<div class="data-exploration">', unsafe_allow_html=True)
            st.markdown(f"""
            ### Probability Analysis Results:
            
            **For threshold = {threshold:.1f} mm:**
            - **Exceedance events**: {exceedance_count} out of {len(precip_data)}
            - **Exceedance probability**: P(X ≥ {threshold:.1f}) = {exceedance_prob:.3f}
            - **Non-exceedance probability**: P(X < {threshold:.1f}) = {non_exceedance_prob:.3f}
            - **Return period**: T = {1/exceedance_prob:.1f} years (if exceeded)
            
            **Engineering Interpretation:**
            There is a {exceedance_prob*100:.1f}% chance that annual precipitation will exceed {threshold:.1f} mm in any given year.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Probability vs Frequency
        st.markdown("### 🔍 Probability vs Frequency: Key Distinction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="key-concept">', unsafe_allow_html=True)
            st.markdown("""
            ### Frequency (Observed)
            - **Historical count** of how often something happened
            - **Sample-specific** (depends on data period)
            - **Changes** as you get more data
            - **Example**: "Floods >500 m³/s occurred 3 times in 50 years"
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="warning-note">', unsafe_allow_html=True)
            st.markdown("""
            ### Probability (Theoretical)
            - **Long-term expectation** of how often something will happen
            - **Population parameter** (true underlying value)
            - **Estimated** from frequency data
            - **Example**: "Probability of flood >500 m³/s is 0.06 per year"
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quiz
        result = create_quiz_question(
            "If a 25-year flood has occurred twice in a 50-year record, what can you conclude?",
            [
                "The probability estimate is wrong - it should have occurred only twice",
                "This is normal variation - the true probability is still 1/25 = 0.04",
                "The flood is no longer a 25-year event - it's now a 12.5-year event",
                "The data must be incorrect"
            ],
            1,
            {
                "correct": "Correct! Probability represents long-term expectation. Short-term variations are normal due to randomness. The 25-year classification refers to the long-term average return period.",
                "incorrect": "Remember: Probability is a long-term concept. Short-term deviations from expected values are normal due to natural variability in hydrologic processes."
            }
        )
        
        if result == True:
            st.session_state.completed_modules.add("2. Understanding Probability Concepts")
    
    elif selected_module == "3. Risk, Reliability & Return Periods":
        st.markdown('<div class="module-header">Module 3: Risk, Reliability & Return Periods</div>', unsafe_allow_html=True)

        # Efficient summary table and quick reference
        st.markdown('<div class="theory-section">', unsafe_allow_html=True)
        st.markdown("""
        <div style='display: flex; flex-wrap: wrap; gap: 1.5rem;'>
        <div style='flex: 1; min-width: 220px;'>
        <strong>Definitions</strong><br>
        <ul style='margin:0; padding-left:1.2em;'>
        <li><b>Return Period (T):</b> Avg. years between events. <br><code>T = 1/P</code></li>
        <li><b>Risk (R):</b> Probability event is exceeded in n years. <br><code>R = 1 - (1-P)<sup>n</sup></code></li>
        <li><b>Reliability (Rel):</b> Probability event is <i>not</i> exceeded. <br><code>Rel = (1-P)<sup>n</sup></code></li>
        </ul>
        </div>
        <div style='flex: 1; min-width: 220px;'>
        <strong>Quick Reference</strong><br>
        <ul style='margin:0; padding-left:1.2em;'>
        <li>Annual Probability: <code>P = 1/T</code></li>
        <li>Design Life: <code>n</code> (years)</li>
        <li>Low Risk: <10%</li>
        <li>High Risk: >50%</li>
        </ul>
        </div>
        <div style='flex: 1; min-width: 220px;'>
        <strong>Common Pitfalls</strong><br>
        <ul style='margin:0; padding-left:1.2em;'>
        <li>Short records may <b>underestimate risk</b></li>
        <li>Return period ≠ time until next event</li>
        <li>Risk increases with design life</li>
        </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Interactive calculator and plots
        st.markdown("### 🧮 Risk & Reliability Calculator")
        col1, col2 = st.columns([1,2])
        with col1:
            design_return_period = st.selectbox("Design return period:",[2,5,10,25,50,100,200,500,1000],index=4)
            design_life = st.slider("Design life (years):", 1, 100, 50)
            annual_prob = 1/design_return_period
            lifetime_risk = 1-(1-annual_prob)**design_life
            reliability = (1-annual_prob)**design_life
            st.markdown('<div class="key-concept">', unsafe_allow_html=True)
            st.markdown(f"""
            <b>Annual Probability:</b> {annual_prob:.4f}<br>
            <b>{design_life}-Year Risk:</b> {lifetime_risk:.3f} ({lifetime_risk*100:.1f}%)<br>
            <b>Reliability:</b> {reliability:.3f} ({reliability*100:.1f}%)
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            life_range = np.arange(1,101)
            risks = 1-(1-annual_prob)**life_range
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=life_range, y=risks, mode='lines', name='Lifetime Risk', line=dict(color='#dc2626', width=3)))
            fig.add_scatter(x=[design_life], y=[lifetime_risk], mode='markers', name='Current Design', marker=dict(size=12, color='blue', symbol='star'))
            for risk_level,label in [(0.1,'10%'),(0.2,'20%'),(0.5,'50%')]:
                fig.add_hline(y=risk_level, line_dash="dash", line_color="gray", annotation_text=f"{label} Risk")
            fig.update_layout(title=f'Lifetime Risk vs Design Life ({design_return_period}-Year Event)',xaxis_title='Design Life (years)',yaxis_title='Lifetime Risk',height=350)
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Standards summary
        st.markdown('<div class="excel-section">', unsafe_allow_html=True)
        st.markdown("**Design Standards (50-year risk):**")
        standards = {'Residential':(10,0.4),'Commercial':(25,0.2),'Critical Infra':(100,0.1),'High-Risk':(500,0.05)}
        rows = []
        for name,(T,acc_risk) in standards.items():
            risk_50 = 1-(1-1/T)**50
            status = "✅" if risk_50<=acc_risk else "⚠️"
            rows.append(f"<tr><td>{name}</td><td>{T}</td><td>{risk_50:.2f}</td><td>{status}</td></tr>")
        st.markdown(f"""
        <table style='width:100%;font-size:0.95em;'>
        <tr><th>Facility</th><th>Design T</th><th>50yr Risk</th><th>Status</th></tr>
        {''.join(rows)}
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Comparison plots
        st.markdown("### 📊 Return Period Comparison")
        precip_df = load_precipitation_data()
        flood_df = load_flood_data()
        col1, col2 = st.columns(2)
        with col1:
            precip_data = precip_df['Annual_Max_Precip'].values
            sorted_precip, T_precip, _ = weibull_plotting_position(precip_data)
            fig_precip = go.Figure()
            fig_precip.add_trace(go.Scatter(x=T_precip, y=sorted_precip, mode='markers+lines', name='Precipitation', marker=dict(size=8, color='#2563eb')))
            fig_precip.update_layout(title='Precipitation Frequency Curve',xaxis_title='Return Period (years)',yaxis_title='Max Precipitation (mm)',xaxis_type='log',height=300)
            fig_precip = apply_theme(fig_precip)
            st.plotly_chart(fig_precip, use_container_width=True)
        with col2:
            flood_data = flood_df['Peak_Flow'].values
            sorted_flood, T_flood, _ = weibull_plotting_position(flood_data)
            fig_flood = go.Figure()
            fig_flood.add_trace(go.Scatter(x=T_flood, y=sorted_flood, mode='markers+lines', name='Flood', marker=dict(size=8, color='#dc2626')))
            fig_flood.update_layout(title='Flood Frequency Curve',xaxis_title='Return Period (years)',yaxis_title='Peak Flow (m³/s)',xaxis_type='log',height=300)
            fig_flood = apply_theme(fig_flood)
            st.plotly_chart(fig_flood, use_container_width=True)

        # Compact comparison table
        st.markdown("**Design Event Comparison**")
        return_periods = [2,5,10,25,50,100]
        rows = []
        for T in return_periods:
            precip_val = np.interp(T, T_precip[::-1], sorted_precip[::-1])
            flood_val = np.interp(T, T_flood[::-1], sorted_flood[::-1])
            annual_risk = 1/T
            risk_50 = 1-(1-annual_risk)**50
            rows.append(f"<tr><td>{T}</td><td>{annual_risk:.4f}</td><td>{risk_50:.3f}</td><td>{precip_val:.1f}</td><td>{flood_val:.0f}</td></tr>")
        st.markdown(f"""
        <table style='width:100%;font-size:0.95em;'>
        <tr><th>T (yrs)</th><th>P</th><th>50yr Risk</th><th>Precip (mm)</th><th>Flood (m³/s)</th></tr>
        {''.join(rows)}
        </table>
        """, unsafe_allow_html=True)

        # Quiz
        result = create_quiz_question(
            "A storm drain is designed for a 10-year storm with a 25-year design life. What is the risk that the design storm will be exceeded during the structure's lifetime?",
            ["10%", "36%", "25%", "40%"],
            1,
            {
                "correct": "Correct! Risk = 1 - (1-P)^n = 1 - (1-0.1)^25 = 0.36 or 36%.",
                "incorrect": "Use: Risk = 1 - (1-P)^n, P = 1/10 = 0.1, n = 25. Risk = 1 - (0.9)^25 = 0.36 or 36%."
            }
        )
        if result == True:
            st.session_state.completed_modules.add("3. Risk, Reliability & Return Periods")
    
    elif selected_module == "4. Probability Distribution Functions":
        st.markdown('<div class="module-header">Module 4: Probability Distribution Functions</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="theory-section">', unsafe_allow_html=True)
        st.markdown("""
        ### Learning Objectives
        - Understand what probability distribution functions represent
        - Distinguish between discrete and continuous distributions
        - Learn the mathematical foundation of PDFs
        - Connect distributions to real hydrologic data
        
        ### What is a Probability Distribution Function?
        
        A **Probability Distribution Function (PDF)** describes how probability is distributed over all possible values of a random variable.
        
        **Key Properties:**
        - Area under the curve = 1 (total probability)
        - f(x) ≥ 0 for all x (probabilities are non-negative)
        - Height represents probability density (not probability itself)
        
        **Engineering Application:**
        PDFs allow us to model and predict hydrologic variables mathematically, enabling:
        - Design event estimation
        - Risk assessment
        - Statistical inference
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive PDF exploration
        st.markdown("### 🔍 Interactive PDF Explorer")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Distribution Parameters")
            
            dist_type = st.selectbox(
                "Select distribution:",
                ["Normal", "Log-Normal", "Exponential", "Uniform"]
            )
            
            if dist_type == "Normal":
                mean = st.slider("Mean (μ):", -5.0, 5.0, 0.0, 0.1)
                std = st.slider("Standard Deviation (σ):", 0.1, 3.0, 1.0, 0.1)
                x_range = np.linspace(mean - 4*std, mean + 4*std, 200)
                pdf_values = stats.norm.pdf(x_range, mean, std)
                
                st.markdown('<div class="equation-box">', unsafe_allow_html=True)
                st.markdown(r"""
                **Normal Distribution PDF:**
                
                $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif dist_type == "Log-Normal":
                mu_log = st.slider("μ (log scale):", -1.0, 2.0, 0.0, 0.1)
                sigma_log = st.slider("σ (log scale):", 0.1, 1.5, 0.5, 0.1)
                x_range = np.linspace(0.01, 10, 200)
                pdf_values = stats.lognorm.pdf(x_range, sigma_log, scale=np.exp(mu_log))
                
                st.markdown('<div class="equation-box">', unsafe_allow_html=True)
                st.markdown(r"""
                **Log-Normal Distribution PDF:**
                
                $$f(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}$$
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif dist_type == "Exponential":
                lam = st.slider("Rate parameter (λ):", 0.1, 3.0, 1.0, 0.1)
                x_range = np.linspace(0, 5/lam, 200)
                pdf_values = stats.expon.pdf(x_range, scale=1/lam)
                
                st.markdown('<div class="equation-box">', unsafe_allow_html=True)
                st.markdown(r"""
                **Exponential Distribution PDF:**
                
                $$f(x) = \lambda e^{-\lambda x}$$
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:  # Uniform
                a = st.slider("Lower bound (a):", -3.0, 0.0, -1.0, 0.1)
                b = st.slider("Upper bound (b):", 0.0, 3.0, 1.0, 0.1)
                if b <= a:
                    b = a + 0.1
                x_range = np.linspace(a - 0.5, b + 0.5, 200)
                pdf_values = stats.uniform.pdf(x_range, a, b-a)
                
                st.markdown('<div class="equation-box">', unsafe_allow_html=True)
                st.markdown(r"""
                **Uniform Distribution PDF:**
                
                $$f(x) = \frac{1}{b-a}$$ for a ≤ x ≤ b
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Plot PDF
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=pdf_values,
                mode='lines',
                fill='tonexty',
                name=f'{dist_type} PDF',
                line=dict(color='#2563eb', width=3),
                fillcolor='rgba(37, 99, 235, 0.3)'
            ))
            
            fig.update_layout(
                title=f'{dist_type} Distribution - Probability Density Function',
                xaxis_title='x',
                yaxis_title='Probability Density f(x)',
                height=400
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # PDF properties
            st.markdown('<div class="data-exploration">', unsafe_allow_html=True)
            
            # Calculate properties
            total_area = np.trapezoid(pdf_values, x_range)
            max_density = np.max(pdf_values)
            
            if dist_type == "Normal":
                theoretical_mean = mean
                theoretical_var = std**2
            elif dist_type == "Log-Normal":
                theoretical_mean = np.exp(mu_log + sigma_log**2/2)
                theoretical_var = (np.exp(sigma_log**2) - 1) * np.exp(2*mu_log + sigma_log**2)
            elif dist_type == "Exponential":
                theoretical_mean = 1/lam
                theoretical_var = 1/lam**2
            else:  # Uniform
                theoretical_mean = (a + b)/2
                theoretical_var = (b - a)**2/12
            
            st.markdown(f"""
            ### PDF Properties:
            - **Total area under curve**: {total_area:.3f} ≈ 1.0
            - **Maximum density**: {max_density:.3f}
            - **Theoretical mean**: {theoretical_mean:.3f}
            - **Theoretical variance**: {theoretical_var:.3f}
            - **Standard deviation**: {np.sqrt(theoretical_var):.3f}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Real data fitting example
        st.markdown("### 📊 Fitting PDFs to Real Data")
        
        # Use precipitation data
        precip_df = load_precipitation_data()
        precip_data = precip_df['Annual_Max_Precip'].values
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Data vs Normal Distribution")
            
            # Fit normal distribution
            mu_fit, sigma_fit = stats.norm.fit(precip_data)
            
            fig = go.Figure()
            
            # Histogram of data
            fig.add_trace(go.Histogram(
                x=precip_data,
                nbinsx=12,
                histnorm='probability density',
                name='Observed Data',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Fitted normal PDF
            x_fit = np.linspace(min(precip_data), max(precip_data), 100)
            pdf_fit = stats.norm.pdf(x_fit, mu_fit, sigma_fit)
            
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=pdf_fit,
                mode='lines',
                name=f'Normal PDF (μ={mu_fit:.1f}, σ={sigma_fit:.1f})',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title='Precipitation Data with Fitted Normal Distribution',
                xaxis_title='Annual Max Precipitation (mm)',
                yaxis_title='Probability Density',
                height=400
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Data vs Log-Normal Distribution")
            
            # Fit log-normal distribution
            sigma_lognorm, loc, scale = stats.lognorm.fit(precip_data, floc=0)
            
            fig = go.Figure()
            
            # Histogram of data
            fig.add_trace(go.Histogram(
                x=precip_data,
                nbinsx=12,
                histnorm='probability density',
                name='Observed Data',
                marker_color='lightgreen',
                opacity=0.7
            ))
            
            # Fitted log-normal PDF
            pdf_lognorm = stats.lognorm.pdf(x_fit, sigma_lognorm, loc=loc, scale=scale)
            
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=pdf_lognorm,
                mode='lines',
                name=f'Log-Normal PDF',
                line=dict(color='darkgreen', width=3)
            ))
            
            fig.update_layout(
                title='Precipitation Data with Fitted Log-Normal Distribution',
                xaxis_title='Annual Max Precipitation (mm)',
                yaxis_title='Probability Density',
                height=400
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Goodness of fit comparison
        st.markdown("### 🎯 Goodness of Fit Comparison")
        
        # Calculate goodness of fit statistics
        ks_normal = stats.kstest(precip_data, lambda x: stats.norm.cdf(x, mu_fit, sigma_fit))[0]
        ks_lognorm = stats.kstest(precip_data, lambda x: stats.lognorm.cdf(x, sigma_lognorm, loc=loc, scale=scale))[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="key-concept">', unsafe_allow_html=True)
            st.markdown(f"""
            ### Normal Distribution Fit:
            - **Parameters**: μ = {mu_fit:.2f}, σ = {sigma_fit:.2f}
            - **KS Statistic**: {ks_normal:.4f}
            - **Fit Quality**: {'Good' if ks_normal < 0.15 else 'Poor'}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="warning-note">', unsafe_allow_html=True)
            st.markdown(f"""
            ### Log-Normal Distribution Fit:
            - **Parameters**: σ = {sigma_lognorm:.2f}, scale = {scale:.2f}
            - **KS Statistic**: {ks_lognorm:.4f}
            - **Fit Quality**: {'Good' if ks_lognorm < 0.15 else 'Poor'}
            - **Better fit**: {'Yes' if ks_lognorm < ks_normal else 'No'}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quiz
        result = create_quiz_question(
            "What does the height of a probability density function (PDF) represent?",
            [
                "The probability that the variable equals exactly that value",
                "The probability density at that point (rate of probability per unit)",
                "The cumulative probability up to that point",
                "The return period for that value"
            ],
            1,
            {
                "correct": "Exactly! PDF height represents probability density (probability per unit), not probability itself. For continuous variables, probability is the area under the curve, not the height.",
                "incorrect": "Remember: For continuous distributions, the height of the PDF is probability density (probability per unit), while actual probabilities are represented by areas under the curve."
            }
        )
        
        if result == True:
            st.session_state.completed_modules.add("4. Probability Distribution Functions")
    
    # For remaining modules, show coming soon message
    else:
        st.markdown(f'<div class="module-header">{selected_module}</div>', unsafe_allow_html=True)
        st.markdown('<div class="theory-section">', unsafe_allow_html=True)
        st.markdown(f"""
        ### {selected_module.split('. ')[1]} - Coming Soon!
        
        You've made excellent progress! This module is under development.
        
        **What you've learned so far:**
        {chr(10).join([f"✅ {module}" for module in st.session_state.completed_modules])}
        
        **Coming in this module:**
        - Advanced theoretical concepts
        - Hands-on calculations
        - Excel practice exercises  
        - Real engineering applications
        - Knowledge assessments
        
        Continue with the available modules to build your foundation!
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show progress
        progress_data = {
            'Module': [f"Module {i+1}" for i in range(len(modules))],
            'Status': ['✅ Completed' if module in st.session_state.completed_modules else '🚧 Available' if i < 4 else '⏳ Coming Soon' for i, module in enumerate(modules)]
        }
        
        st.dataframe(pd.DataFrame(progress_data), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1.5rem;">
        <p><strong>Advanced Water Resources Engineering</strong> | Frequency Analysis Learning Path</p>
        <p><em>Building engineering expertise through hands-on data analysis and theoretical understanding</em></p>
        <br>
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
                     padding: 1rem; border-radius: 0.5rem; border: 1px solid #cbd5e1;">
            <p style="margin: 0; font-size: 0.9rem; color: #475569;">
                <strong>🎓 Developed by TA Saurav Bhattarai</strong><br>
                Advanced Water Resource Engineering Course<br>
                <strong>Dr. Rocky Talchabhadel</strong> | <strong>Jackson State University</strong> | Fall 2025<br>
                <span style="color: #0f766e;">🐍 Powered by Python & Streamlit</span>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
