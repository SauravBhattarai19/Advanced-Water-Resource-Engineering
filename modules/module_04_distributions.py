"""
Module 4: Probability Distribution Functions
Based on original comprehensive learning path

Author: TA Saurav Bhattarai
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from typing import List, Optional

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents

class Module04_Distributions(LearningModule):
    """Module 4: Probability Distribution Functions"""
    
    def __init__(self):
        objectives = [
            LearningObjective("Define probability distribution functions", "understand"),
            LearningObjective("Identify common hydrologic distributions", "remember"),
            LearningObjective("Fit distributions to data", "apply"),
            LearningObjective("Select appropriate distributions", "evaluate")
        ]
        
        info = ModuleInfo(
            id="module_04",
            title="Probability Distribution Functions",
            description="Learn to fit and use probability distributions for hydrologic analysis",
            duration_minutes=60,
            prerequisites=["module_01", "module_02", "module_03"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=10
        )
        
        super().__init__(info)
        self.data = DataManager.get_precipitation_data()
    
    def get_slide_titles(self) -> List[str]:
        return [
            "What are PDFs?",
            "PDF vs CDF Explained",
            "Getting Probabilities from Distributions",
            "Common Distributions", 
            "Normal Distribution",
            "Log-Normal Distribution",
            "Interactive Distribution Explorer",
            "Distribution Selection Guide",
            "Hands-On Python Tutorial",
            "Knowledge Check"
        ]
    
    def render_slide(self, slide_num: int) -> Optional[bool]:
        if slide_num == 0:
            return self._slide_introduction()
        elif slide_num == 1:
            return self._slide_pdf_vs_cdf()
        elif slide_num == 2:
            return self._slide_getting_probabilities()
        elif slide_num == 3:
            return self._slide_common_distributions()
        elif slide_num == 4:
            return self._slide_normal()
        elif slide_num == 5:
            return self._slide_lognormal()
        elif slide_num == 6:
            return self._slide_explorer()
        elif slide_num == 7:
            return self._slide_selection()
        elif slide_num == 8:
            return self._slide_python_tutorial()
        elif slide_num == 9:
            return self._slide_quiz()
        
        return False
    
    def _slide_introduction(self) -> Optional[bool]:
        """Slide 1: What are PDFs?"""
        with UIComponents.slide_container("theory"):
            st.markdown("## What is a Probability Distribution Function (PDF)?")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Key Properties")
                UIComponents.highlight_box("""
                - Describes how probability is distributed over possible values
                - Area under curve = 1
                - f(x) ≥ 0 for all x
                - Height = probability density (not probability)
                """)
                
                UIComponents.formula_display("∫ f(x) dx = 1", "Total Probability")
                
            with col2:
                st.markdown("### Why Use Distributions?")
                UIComponents.highlight_box("""
                **Instead of calculating probabilities for every value:**
                - Fit a distribution to data
                - Use standard formulas/tables
                - Estimate probabilities for any value
                - Extrapolate beyond observed data
                """)
                
                UIComponents.big_number_display("Standard Tables", "Key Advantage")
        
        return None
    
    def _slide_python_fitting(self) -> Optional[bool]:
        """Slide 9: Hands-On Python Distribution Fitting"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Hands-On Python: Distribution Fitting")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### What You'll Learn")
                UIComponents.highlight_box("""
                **Complete Python Tutorial:**
                - Load and analyze real hydrologic data
                - Fit multiple probability distributions
                - Compare goodness of fit statistics
                - Calculate design return periods
                - Create professional publication-quality plots
                - Export results for engineering reports
                """)
                
                st.markdown("### Tutorial Contents")
                tutorial_topics = [
                    "Data loading and preprocessing",
                    "Exploratory data analysis", 
                    "Distribution fitting (Normal, Log-Normal, Gumbel, etc.)",
                    "Parameter estimation methods",
                    "Goodness of fit testing",
                    "Return period calculations",
                    "Confidence intervals",
                    "Professional plotting"
                ]
                
                for topic in tutorial_topics:
                    st.markdown(f"• {topic}")
                
            with col2:
                UIComponents.big_number_display("Complete", "Python Tutorial")
                UIComponents.big_number_display("Ready-to-Use", "Real Data")
                UIComponents.big_number_display("Step-by-Step", "Instructions")
                
                # Access buttons
                st.markdown("### 🚀 Access Resources")
                
                col2a, col2b = st.columns(2)
                
                with col2a:
                    if st.button("📊 Open Google Colab", key="open_colab", help="Complete Python tutorial", use_container_width=True):
                        st.markdown('[Click here to open the Google Colab notebook](https://colab.research.google.com/drive/1LkAEFUfspJRZKceOIW2StQ9UnhE_WVBE?usp=sharing)')
                
                with col2b:
                    if st.button("📈 Download Data", key="download_data", help="Excel file with hydrologic data", use_container_width=True):
                        st.markdown('[Click here to access the data file](https://jsums-my.sharepoint.com/:x:/g/personal/j01013381_students_jsums_edu/ETVE3UpUxsFAoC_5TLVfnG0BDRlAwWYIPs-Epxs8dd4WvA?e=15qB4x)')
            
            # Direct links section
            st.markdown("---")
            st.markdown("### 🔗 Direct Access Links")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                UIComponents.highlight_box("""
                **📓 Google Colab Notebook**
                
                Complete hands-on Python tutorial with:
                - Pre-written code cells
                - Detailed explanations
                - Real data examples
                - Interactive exercises
                
                [**➡️ Open Google Colab Tutorial**](https://colab.research.google.com/drive/1LkAEFUfspJRZKceOIW2StQ9UnhE_WVBE?usp=sharing)
                """)
            
            with col2:
                UIComponents.highlight_box("""
                **📊 Hydrologic Dataset**
                
                Excel file containing:
                - Historical precipitation data
                - Multiple stations/variables
                - Ready for analysis
                - Engineering applications
                
                [**⬇️ Download Data File**](https://jsums-my.sharepoint.com/:x:/g/personal/j01013381_students_jsums_edu/ETVE3UpUxsFAoC_5TLVfnG0BDRlAwWYIPs-Epxs8dd4WvA?e=15qB4x)
                """)
            
            st.markdown("### 📋 Instructions")
            instructions = [
                "1. **Click the Google Colab link** to open the tutorial",
                "2. **Make a copy** of the notebook to your Google Drive",
                "3. **Download the data file** from the SharePoint link",
                "4. **Upload the data** to your Colab session",
                "5. **Run the code cells** step by step",
                "6. **Modify parameters** to explore different scenarios"
            ]
            
            for instruction in instructions:
                st.markdown(instruction)
        
        return None
    
    def _slide_python_goodness_of_fit(self) -> Optional[bool]:
        """Slide 10: Python Goodness of Fit Tests"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Python: Statistical Testing & Model Selection")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### What the Tutorial Covers")
                
                testing_topics = [
                    "**Kolmogorov-Smirnov Test** - Overall distribution fit",
                    "**Anderson-Darling Test** - Emphasis on tail behavior", 
                    "**Chi-Square Test** - Goodness of fit for binned data",
                    "**AIC/BIC Criteria** - Model comparison metrics",
                    "**Visual Diagnostics** - Q-Q plots, P-P plots",
                    "**Cross-validation** - Out-of-sample performance"
                ]
                
                for topic in testing_topics:
                    st.markdown(f"• {topic}")
                
                st.markdown("### Statistical Outputs")
                UIComponents.highlight_box("""
                **The tutorial provides:**
                - Test statistics and p-values
                - Confidence intervals for parameters
                - Model ranking tables
                - Professional summary reports
                - Uncertainty quantification
                """)
                
            with col2:
                st.markdown("### Learning Outcomes")
                
                outcomes = [
                    "Understand different goodness-of-fit tests",
                    "Interpret statistical test results", 
                    "Select the best-fitting distribution",
                    "Quantify model uncertainty",
                    "Create publication-quality results",
                    "Apply industry best practices"
                ]
                
                for outcome in outcomes:
                    st.markdown(f"✓ {outcome}")
                
                UIComponents.big_number_display("Professional", "Statistical Analysis")
                UIComponents.big_number_display("Industry", "Best Practices")
            
            st.markdown("---")
            st.markdown("### 🎯 Ready to Practice?")
            
            # Large access button
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("🚀 Launch Complete Python Tutorial", key="launch_full_tutorial", use_container_width=True):
                    st.success("Opening Google Colab notebook...")
                    st.markdown("""
                    **Click this link to access the complete tutorial:**
                    
                    [**📓 Google Colab: Distribution Fitting & Statistical Testing**](https://colab.research.google.com/drive/1LkAEFUfspJRZKceOIW2StQ9UnhE_WVBE?usp=sharing)
                    
                    **And download the data here:**
                    
                    [**📊 Hydrologic Data (Excel File)**](https://jsums-my.sharepoint.com/:x:/g/personal/j01013381_students_jsums_edu/ETVE3UpUxsFAoC_5TLVfnG0BDRlAwWYIPs-Epxs8dd4WvA?e=15qB4x)
                    """)
            
            UIComponents.highlight_box("""
            **💡 Pro Tip:** The Colab notebook includes everything you need - code, explanations, 
            and sample data. You can modify the examples with your own data for real projects.
            """)
        
        return None
    
    def _slide_applications(self) -> Optional[bool]:
        """Slide 11: Practical Applications"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Real-World Applications")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Engineering Design Process")
                steps = [
                    "1. **Collect Data** - Historical records",
                    "2. **Analyze Data** - Basic statistics",
                    "3. **Fit Distributions** - Multiple candidates",
                    "4. **Test Goodness of Fit** - Statistical tests",
                    "5. **Select Best Distribution** - Engineering judgment",
                    "6. **Calculate Design Values** - For various return periods",
                    "7. **Apply Safety Factors** - Engineering conservatism"
                ]
                
                for step in steps:
                    st.markdown(step)
                
            with col2:
                st.markdown("### Case Study Results")
                
                # Calculate design values using best-fit distribution
                precip_values = self.data['Annual_Max_Precip'].values
                mu, sigma = stats.norm.fit(precip_values)  # Assuming normal is best fit
                
                design_data = []
                return_periods = [2, 5, 10, 25, 50, 100, 500]
                
                for T in return_periods:
                    prob = 1 - 1/T
                    design_value = stats.norm.ppf(prob, mu, sigma)
                    annual_risk = 1/T
                    risk_50yr = 1 - (1-annual_risk)**50
                    
                    design_data.append({
                        'Return Period': f'{T} years',
                        'Design Value': f'{design_value:.1f} mm',
                        '50-Year Risk': f'{risk_50yr:.1%}'
                    })
                
                design_df = pd.DataFrame(design_data)
                st.dataframe(design_df, use_container_width=True)
                
            UIComponents.highlight_box("""
            **Key Engineering Insights:**
            - Always fit multiple distributions and compare
            - Use statistical tests but also engineering judgment
            - Consider the physical basis of the distribution
            - Account for data limitations and uncertainty
            - Apply appropriate safety factors for critical infrastructure
            """)
            
            # Download link for data
            st.markdown("### 📊 Access Real Data")
            st.markdown("""
            **Practice with Real Dataset:**
            [Download Excel Data](https://jsums-my.sharepoint.com/:x:/g/personal/j01013381_students_jsums_edu/ETVE3UpUxsFAoC_5TLVfnG0BDRlAwWYIPs-Epxs8dd4WvA?e=15qB4x)
            
            **Complete Python Tutorial:**
            [Google Colab Notebook](https://colab.research.google.com/drive/1LkAEFUfspJRZKceOIW2StQ9UnhE_WVBE?usp=sharing)
            """)
        
        return None
    
    def _slide_pdf_vs_cdf(self) -> Optional[bool]:
        """Slide 2: PDF vs CDF Explained"""
        with UIComponents.slide_container("theory"):
            st.markdown("## PDF vs CDF: Key Differences")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### PDF (Probability Density Function)")
                UIComponents.highlight_box("""
                **f(x) = probability density**
                - Height at each point
                - NOT probability itself
                - Area under curve = probability
                - Can exceed 1.0
                - Used for: visualization, understanding shape
                """)
                
                UIComponents.formula_display("P(a ≤ X ≤ b) = ∫[a to b] f(x) dx", "Probability from PDF")
                
            with col2:
                st.markdown("### CDF (Cumulative Distribution Function)")
                UIComponents.highlight_box("""
                **F(x) = P(X ≤ x)**
                - Actual probability values
                - Always between 0 and 1
                - Non-decreasing function
                - F(∞) = 1, F(-∞) = 0
                - Used for: calculating probabilities
                """)
                
                UIComponents.formula_display("P(X ≤ x) = F(x)", "Probability from CDF")
            
            # Interactive demonstration
            st.markdown("### Interactive Demo")
            threshold = st.slider("Select threshold value:", -3.0, 3.0, 0.0, 0.1, key="pdf_cdf_threshold")
            
            # Create combined PDF/CDF plot
            x_range = np.linspace(-4, 4, 200)
            pdf_values = stats.norm.pdf(x_range, 0, 1)
            cdf_values = stats.norm.cdf(x_range, 0, 1)
            
            fig = go.Figure()
            
            # PDF
            fig.add_trace(go.Scatter(
                x=x_range, y=pdf_values, mode='lines', name='PDF f(x)',
                line=dict(color='blue', width=3), yaxis='y'
            ))
            
            # CDF
            fig.add_trace(go.Scatter(
                x=x_range, y=cdf_values, mode='lines', name='CDF F(x)',
                line=dict(color='red', width=3), yaxis='y2'
            ))
            
            # Threshold line
            fig.add_vline(x=threshold, line_dash="dash", line_color="green")
            
            # Shade area under PDF
            x_shade = x_range[x_range <= threshold]
            y_shade = stats.norm.pdf(x_shade, 0, 1)
            fig.add_trace(go.Scatter(
                x=x_shade, y=y_shade, fill='tonexty', mode='none',
                fillcolor='rgba(0,100,80,0.3)', name='P(X ≤ threshold)', yaxis='y'
            ))
            
            fig.update_layout(
                title=f'PDF vs CDF Demo (threshold = {threshold})',
                xaxis_title='x',
                yaxis=dict(title='PDF f(x)', side='left', color='blue'),
                yaxis2=dict(title='CDF F(x)', side='right', overlaying='y', color='red'),
                height=400
            )
            fig = PlotTools.apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show calculated probabilities
            prob_cdf = stats.norm.cdf(threshold, 0, 1)
            st.markdown(f"""
            **At threshold = {threshold}:**
            - **PDF height f({threshold:.1f}):** {stats.norm.pdf(threshold, 0, 1):.3f}
            - **CDF value F({threshold:.1f}):** {prob_cdf:.3f}
            - **P(X ≤ {threshold:.1f}):** {prob_cdf:.3f} ({prob_cdf*100:.1f}%)
            """)
        
        return None
    
    def _slide_getting_probabilities(self) -> Optional[bool]:
        """Slide 3: Getting Probabilities from Distributions"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## How to Calculate Probabilities")
            
            st.markdown("### Three Common Probability Questions:")
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            with col1:
                st.markdown("**1. P(X ≤ a)**")
                st.markdown("*'Less than or equal'*")
                UIComponents.formula_display("F(a)", "Use CDF directly")
                
            with col2:
                st.markdown("**2. P(X ≥ a)**")
                st.markdown("*'Greater than or equal'*")
                UIComponents.formula_display("1 - F(a)", "Complement of CDF")
                
            with col3:
                st.markdown("**3. P(a ≤ X ≤ b)**")
                st.markdown("*'Between two values'*")
                UIComponents.formula_display("F(b) - F(a)", "Difference of CDFs")
            
            st.markdown("### Interactive Probability Calculator")
            
            # Distribution selector
            dist_type = st.selectbox("Select distribution:", ["Normal", "Log-Normal"], key="prob_calc_dist")
            
            if dist_type == "Normal":
                mean = st.slider("Mean:", -2.0, 2.0, 0.0, 0.1, key="prob_mean")
                std = st.slider("Std Dev:", 0.1, 2.0, 1.0, 0.1, key="prob_std")
                
                # Probability questions
                col1, col2 = UIComponents.two_column_layout()
                
                with col1:
                    st.markdown("**Calculate Probabilities:**")
                    value_a = st.number_input("Value a:", value=0.0, key="prob_a")
                    value_b = st.number_input("Value b (if needed):", value=1.0, key="prob_b")
                    
                    # Calculate probabilities
                    prob_leq = stats.norm.cdf(value_a, mean, std)
                    prob_geq = 1 - stats.norm.cdf(value_a, mean, std)
                    prob_between = stats.norm.cdf(value_b, mean, std) - stats.norm.cdf(value_a, mean, std)
                    
                    UIComponents.big_number_display(f"{prob_leq:.3f}", f"P(X ≤ {value_a})")
                    UIComponents.big_number_display(f"{prob_geq:.3f}", f"P(X ≥ {value_a})")
                    UIComponents.big_number_display(f"{prob_between:.3f}", f"P({value_a} ≤ X ≤ {value_b})")
                
                with col2:
                    # Visual demonstration
                    x_range = np.linspace(mean-4*std, mean+4*std, 200)
                    pdf_vals = stats.norm.pdf(x_range, mean, std)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_range, y=pdf_vals, mode='lines', name='PDF', line=dict(color='blue', width=2)))
                    
                    # Shade regions
                    x_leq = x_range[x_range <= value_a]
                    y_leq = stats.norm.pdf(x_leq, mean, std)
                    fig.add_trace(go.Scatter(x=x_leq, y=y_leq, fill='tonexty', mode='none', fillcolor='rgba(255,0,0,0.3)', name=f'P(X ≤ {value_a})'))
                    
                    fig.add_vline(x=value_a, line_dash="dash", line_color="red")
                    if abs(value_b - value_a) > 0.1:
                        fig.add_vline(x=value_b, line_dash="dash", line_color="green")
                    
                    fig.update_layout(title='Probability Visualization', height=350)
                    fig = PlotTools.apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_common_distributions(self) -> Optional[bool]:
        """Slide 2: Common Distributions"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Common Distributions in Hydrology")
            
            dist_data = {
                'Distribution': ['Normal', 'Log-Normal', 'Exponential', 'Gumbel', 'Pearson Type III'],
                'Parameters': ['μ, σ', 'μ, σ (log scale)', 'λ', 'μ, β', 'μ, σ, γ'],
                'Shape': ['Symmetric', 'Right-skewed', 'Right-skewed', 'Right-skewed', 'Variable skew'],
                'Hydrologic Use': [
                    'General rainfall',
                    'Flood peaks, precipitation extremes',
                    'Time between events', 
                    'Extreme events (floods)',
                    'US standard for floods'
                ]
            }
            
            dist_df = pd.DataFrame(dist_data)
            st.dataframe(dist_df, use_container_width=True)
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            with col1:
                UIComponents.big_number_display("μ", "Mean")
                UIComponents.big_number_display("σ", "Std Dev")
                
            with col2:
                UIComponents.big_number_display("λ", "Rate")
                UIComponents.big_number_display("β", "Scale")
                
            with col3:
                UIComponents.big_number_display("γ", "Skewness")
                st.markdown("Controls distribution shape")
        
        return None
    
    def _slide_normal(self) -> Optional[bool]:
        """Slide 3: Normal Distribution"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Normal Distribution")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Interactive Parameters")
                mean = st.slider("Mean (μ):", -5.0, 5.0, 0.0, 0.2, key="normal_mean")
                std = st.slider("Standard Deviation (σ):", 0.1, 3.0, 1.0, 0.1, key="normal_std")
                
                UIComponents.formula_display(
                    "f(x) = (1/(σ√2π)) × e^(-(x-μ)²/2σ²)", 
                    "Normal PDF Formula"
                )
                
                st.markdown("### Properties")
                st.markdown(f"""
                - Mean = {mean}
                - Standard Deviation = {std}
                - Symmetric around mean
                - Bell-shaped curve
                """)
                
            with col2:
                # Plot normal distribution
                x_range = np.linspace(-10, 10, 200)
                pdf_values = stats.norm.pdf(x_range, mean, std)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_range, 
                    y=pdf_values,
                    mode='lines',
                    fill='tonexty',
                    name='Normal PDF',
                    line=dict(color='#3498db', width=3),
                    fillcolor='rgba(52,152,219,0.3)'
                ))
                
                fig.update_layout(
                    title=f'Normal Distribution (μ={mean}, σ={std})',
                    xaxis_title='x',
                    yaxis_title='Probability Density f(x)',
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_lognormal(self) -> Optional[bool]:
        """Slide 4: Log-Normal Distribution"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Log-Normal Distribution")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Interactive Parameters")
                mu_log = st.slider("μ (log scale):", -1.0, 2.0, 0.5, 0.1, key="lognorm_mu")
                sigma_log = st.slider("σ (log scale):", 0.1, 1.5, 0.5, 0.1, key="lognorm_sigma")
                
                UIComponents.formula_display(
                    "f(x) = (1/(xσ√2π)) × e^(-(ln(x)-μ)²/2σ²)", 
                    "Log-Normal PDF Formula"
                )
                
                st.markdown("### Properties")
                st.markdown("""
                - Only positive values (x > 0)
                - Right-skewed (long tail)
                - Common for flood data
                - If ln(X) ~ Normal, then X ~ Log-Normal
                """)
                
            with col2:
                # Plot log-normal distribution
                x_range = np.linspace(0.01, 15, 200)
                pdf_values = stats.lognorm.pdf(x_range, sigma_log, scale=np.exp(mu_log))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_range, 
                    y=pdf_values,
                    mode='lines',
                    fill='tonexty',
                    name='Log-Normal PDF',
                    line=dict(color='#e74c3c', width=3),
                    fillcolor='rgba(231,76,60,0.3)'
                ))
                
                fig.update_layout(
                    title=f'Log-Normal Distribution (μ={mu_log}, σ={sigma_log})',
                    xaxis_title='x',
                    yaxis_title='Probability Density f(x)',
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_explorer(self) -> Optional[bool]:
        """Slide 5: Interactive Distribution Explorer"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Distribution Comparison Tool")
            
            # Distribution selector
            selected_dists = st.multiselect(
                "Select distributions to compare:",
                ['Normal', 'Log-Normal', 'Exponential', 'Gumbel'],
                default=['Normal', 'Log-Normal'],
                key="dist_selector"
            )
            
            if selected_dists:
                fig = go.Figure()
                x_range = np.linspace(0.1, 10, 200)
                
                colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
                
                for i, dist_name in enumerate(selected_dists):
                    if dist_name == 'Normal':
                        pdf_vals = stats.norm.pdf(x_range, 3, 1)
                    elif dist_name == 'Log-Normal':
                        pdf_vals = stats.lognorm.pdf(x_range, 0.5, scale=2)
                    elif dist_name == 'Exponential':
                        pdf_vals = stats.expon.pdf(x_range, scale=2)
                    elif dist_name == 'Gumbel':
                        pdf_vals = stats.gumbel_r.pdf(x_range, loc=3, scale=1)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=pdf_vals,
                        mode='lines',
                        name=f'{dist_name} PDF',
                        line=dict(color=colors[i % len(colors)], width=3)
                    ))
                
                fig.update_layout(
                    title='Distribution Comparison',
                    xaxis_title='x',
                    yaxis_title='Probability Density f(x)',
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                UIComponents.highlight_box("""
                **Compare the shapes:** Notice how each distribution has different characteristics - 
                symmetric vs skewed, bounded vs unbounded, thick vs thin tails.
                """)
        
        return None
    
    def _slide_fitting(self) -> Optional[bool]:
        """Slide 6: Fitting to Real Data"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Fitting Distributions to Precipitation Data")
            
            # Get precipitation data
            precip_values = self.data['Annual_Max_Precip'].values
            
            # Fit distributions
            distributions = AnalysisTools.fit_distributions(precip_values)
            
            # Show toggle options
            show_normal = st.checkbox("Show Normal fit", value=True, key="show_normal_fit")
            show_lognorm = st.checkbox("Show Log-Normal fit", value=True, key="show_lognorm_fit")
            show_gumbel = st.checkbox("Show Gumbel fit", value=False, key="show_gumbel_fit")
            
            # Create plot
            fig = go.Figure()
            
            # Histogram of data
            fig.add_trace(go.Histogram(
                x=precip_values,
                nbinsx=12,
                histnorm='probability density',
                name='Observed Data',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Fitted distributions
            x_fit = np.linspace(min(precip_values), max(precip_values), 200)
            
            if show_normal:
                params = distributions['normal']['params']
                pdf_vals = stats.norm.pdf(x_fit, *params)
                fig.add_trace(go.Scatter(
                    x=x_fit, y=pdf_vals, mode='lines',
                    name='Normal fit', line=dict(color='red', width=3)
                ))
            
            if show_lognorm:
                params = distributions['lognormal']['params']
                pdf_vals = stats.lognorm.pdf(x_fit, *params)
                fig.add_trace(go.Scatter(
                    x=x_fit, y=pdf_vals, mode='lines',
                    name='Log-Normal fit', line=dict(color='green', width=3)
                ))
            
            if show_gumbel:
                params = distributions['gumbel']['params']
                pdf_vals = stats.gumbel_r.pdf(x_fit, *params)
                fig.add_trace(go.Scatter(
                    x=x_fit, y=pdf_vals, mode='lines',
                    name='Gumbel fit', line=dict(color='purple', width=3)
                ))
            
            fig.update_layout(
                title='Fitted Distributions to Precipitation Data',
                xaxis_title='Annual Max Precipitation (mm)',
                yaxis_title='Probability Density',
                height=400
            )
            fig = PlotTools.apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            UIComponents.highlight_box("""
            **Which fits best?** Compare how well each curve matches the histogram shape.
            """)
        
        return None
    
    def _slide_selection(self) -> Optional[bool]:
        """Slide 7: Distribution Selection"""
        with UIComponents.slide_container("theory"):
            st.markdown("## How to Select the Right Distribution")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Selection Criteria")
                selection_data = {
                    'Data Characteristic': ['Symmetric', 'Right-skewed', 'Left-skewed', 'Extreme values only'],
                    'Recommended Distribution': ['Normal', 'Log-Normal, Gumbel', 'Reflected distributions', 'Gumbel, Weibull']
                }
                
                selection_df = pd.DataFrame(selection_data)
                st.dataframe(selection_df, use_container_width=True)
                
            with col2:
                st.markdown("### Decision Process")
                steps = [
                    "1. **Plot histogram** - check shape",
                    "2. **Calculate skewness** - measure asymmetry", 
                    "3. **Consider physics** - what makes sense?",
                    "4. **Test multiple distributions**",
                    "5. **Use goodness-of-fit tests**",
                    "6. **Follow standards** (if applicable)"
                ]
                
                for step in steps:
                    st.markdown(step)
            
            UIComponents.highlight_box("""
            **Engineering Judgment:** Statistical tests are important, but physical reasoning 
            and engineering experience also matter in distribution selection.
            """)
        
        return None
    
    def _slide_goodness_of_fit(self) -> Optional[bool]:
        """Slide 8: Goodness of Fit"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Goodness of Fit Testing")
            
            precip_values = self.data['Annual_Max_Precip'].values
            distributions = AnalysisTools.fit_distributions(precip_values)
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Kolmogorov-Smirnov Test")
                st.markdown("Measures maximum difference between observed and theoretical CDFs")
                
                # Calculate KS statistics
                results = []
                for name, dist_info in distributions.items():
                    params = dist_info['params']
                    dist = dist_info['dist']
                    
                    # KS test
                    ks_stat, p_value = stats.kstest(
                        precip_values, 
                        lambda x: dist.cdf(x, *params)
                    )
                    
                    results.append({
                        'Distribution': name.title(),
                        'KS Statistic': f'{ks_stat:.4f}',
                        'P-Value': f'{p_value:.4f}',
                        'Fit Quality': 'Good' if ks_stat < 0.15 else 'Poor'
                    })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
            with col2:
                st.markdown("### Interpretation")
                UIComponents.highlight_box("""
                **KS Statistic:**
                - Lower values = better fit
                - < 0.10 = excellent
                - 0.10-0.15 = good  
                - > 0.15 = poor
                
                **P-Value:**
                - > 0.05 = don't reject (good fit)
                - < 0.05 = reject (poor fit)
                """)
                
                # Find best fit
                best_dist = min(results, key=lambda x: float(x['KS Statistic']))
                
                UIComponents.big_number_display(
                    best_dist['Distribution'], 
                    "Best Fit Distribution"
                )
        
        return None
    
    def _slide_quiz(self) -> Optional[bool]:
        """Slide 12: Knowledge Check"""
        with UIComponents.slide_container():
            st.markdown("## Knowledge Check")
            
            result = QuizEngine.create_multiple_choice(
                "To find P(X ≤ 75) for a normally distributed variable, which function should you use, and what does the PDF height at x=75 represent?",
                [
                    "Use CDF: F(75) gives the probability; PDF height is probability density, not probability itself",
                    "Use PDF: f(75) gives the probability directly; CDF shows cumulative density",
                    "Use both: PDF for probability, CDF for verification",
                    "Use PDF area: integrate from 0 to 75; CDF height shows probability rate"
                ],
                0,
                {
                    "correct": "Correct! Use CDF F(75) to get P(X ≤ 75) directly. The PDF height f(75) is probability density (probability per unit), not the actual probability. For continuous distributions, probabilities come from CDF values or PDF areas.",
                    "incorrect": "Remember: CDF F(x) gives P(X ≤ x) directly. PDF f(x) gives probability density (height), not probability itself. PDF height can exceed 1.0, but CDF values are always between 0 and 1."
                },
                f"{self.info.id}_final_quiz"
            )
            
            if result is True:
                st.success("🎉 Module 4 Complete! You've mastered probability distributions, PDF vs CDF, and Python implementation.")
                return True
            
            return None