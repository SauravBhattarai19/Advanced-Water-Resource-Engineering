"""
Module 4: Probability Distribution Functions
Fixed version without errors

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
            LearningObjective("Distinguish PDF from CDF", "analyze"),
            LearningObjective("Calculate probabilities from distributions", "apply"),
            LearningObjective("Select appropriate distributions", "evaluate")
        ]
        
        info = ModuleInfo(
            id="module_04",
            title="Probability Distribution Functions",
            description="Learn to fit and use probability distributions for hydrologic analysis",
            duration_minutes=50,
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
            "Getting Probabilities",
            "Common Distributions", 
            "Normal Distribution",
            "Log-Normal Distribution",
            "Distribution Explorer",
            "Selection Guide",
            "Python Tutorial",
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
        
        return None
    
    def _slide_getting_probabilities(self) -> Optional[bool]:
        """Slide 3: Getting Probabilities from Distributions"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## How to Calculate Probabilities")
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            with col1:
                st.markdown("**P(X ≤ a)**")
                UIComponents.formula_display("F(a)", "Use CDF directly")
                
            with col2:
                st.markdown("**P(X ≥ a)**")
                UIComponents.formula_display("1 - F(a)", "Complement of CDF")
                
            with col3:
                st.markdown("**P(a ≤ X ≤ b)**")
                UIComponents.formula_display("F(b) - F(a)", "Difference of CDFs")
            
            # Interactive demo
            st.markdown("### Interactive Calculator")
            threshold = st.slider("Select threshold:", -3.0, 3.0, 0.0, 0.1, key="prob_calc")
            
            prob_leq = stats.norm.cdf(threshold, 0, 1)
            prob_geq = 1 - prob_leq
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                UIComponents.big_number_display(f"{prob_leq:.3f}", f"P(X ≤ {threshold:.1f})")
                UIComponents.big_number_display(f"{prob_geq:.3f}", f"P(X ≥ {threshold:.1f})")
            
            with col2:
                x_range = np.linspace(-4, 4, 200)
                pdf_vals = stats.norm.pdf(x_range, 0, 1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_range, y=pdf_vals, mode='lines', name='PDF'))
                
                # Shade area
                x_shade = x_range[x_range <= threshold]
                y_shade = stats.norm.pdf(x_shade, 0, 1)
                fig.add_trace(go.Scatter(x=x_shade, y=y_shade, fill='tonexty', mode='none', name='P(X ≤ threshold)'))
                
                fig.add_vline(x=threshold, line_dash="dash", line_color="red")
                fig.update_layout(height=300)
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_common_distributions(self) -> Optional[bool]:
        """Slide 4: Common Distributions"""
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
        
        return None
    
    def _slide_normal(self) -> Optional[bool]:
        """Slide 5: Normal Distribution"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Normal Distribution")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                mean = st.slider("Mean (μ):", -3.0, 3.0, 0.0, 0.1, key="normal_mean")
                std = st.slider("Standard Deviation (σ):", 0.1, 2.0, 1.0, 0.1, key="normal_std")
                
                UIComponents.formula_display("f(x) = (1/(σ√2π)) × e^(-(x-μ)²/2σ²)", "Normal PDF")
                
                st.markdown(f"""
                - Mean = {mean}
                - Standard Deviation = {std}
                - Symmetric around mean
                - Bell-shaped curve
                """)
                
            with col2:
                x_range = np.linspace(-6, 6, 200)
                pdf_values = stats.norm.pdf(x_range, mean, std)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_range, 
                    y=pdf_values,
                    mode='lines',
                    fill='tonexty',
                    name='Normal PDF'
                ))
                
                fig.update_layout(
                    title=f'Normal Distribution (μ={mean}, σ={std})',
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_lognormal(self) -> Optional[bool]:
        """Slide 6: Log-Normal Distribution"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Log-Normal Distribution")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                mu_log = st.slider("μ (log scale):", 0.0, 2.0, 0.5, 0.1, key="lognorm_mu")
                sigma_log = st.slider("σ (log scale):", 0.1, 1.0, 0.5, 0.1, key="lognorm_sigma")
                
                UIComponents.formula_display("f(x) = (1/(xσ√2π)) × e^(-(ln(x)-μ)²/2σ²)", "Log-Normal PDF")
                
                st.markdown("""
                - Only positive values (x > 0)
                - Right-skewed (long tail)
                - Common for flood data
                - If ln(X) ~ Normal, then X ~ Log-Normal
                """)
                
            with col2:
                x_range = np.linspace(0.01, 10, 200)
                pdf_values = stats.lognorm.pdf(x_range, sigma_log, scale=np.exp(mu_log))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_range, 
                    y=pdf_values,
                    mode='lines',
                    fill='tonexty',
                    name='Log-Normal PDF'
                ))
                
                fig.update_layout(
                    title=f'Log-Normal Distribution (μ={mu_log}, σ={sigma_log})',
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_explorer(self) -> Optional[bool]:
        """Slide 7: Interactive Distribution Explorer"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Distribution Comparison Tool")
            
            selected_dists = st.multiselect(
                "Select distributions to compare:",
                ['Normal', 'Log-Normal', 'Exponential', 'Gumbel'],
                default=['Normal', 'Log-Normal'],
                key="dist_selector"
            )
            
            if selected_dists:
                fig = go.Figure()
                x_range = np.linspace(0.1, 8, 200)
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
                        line=dict(width=3)
                    ))
                
                fig.update_layout(
                    title='Distribution Comparison',
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_selection(self) -> Optional[bool]:
        """Slide 8: Distribution Selection Guide"""
        with UIComponents.slide_container("theory"):
            st.markdown("## How to Select the Right Distribution")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Selection Criteria")
                selection_data = {
                    'Data Characteristic': ['Symmetric', 'Right-skewed', 'Left-skewed', 'Extreme values'],
                    'Recommended': ['Normal', 'Log-Normal, Gumbel', 'Reflected distributions', 'Gumbel, Weibull']
                }
                
                selection_df = pd.DataFrame(selection_data)
                st.dataframe(selection_df, use_container_width=True)
                
            with col2:
                st.markdown("### Decision Process")
                steps = [
                    "1. Plot histogram - check shape",
                    "2. Calculate skewness", 
                    "3. Consider physical reasoning",
                    "4. Test multiple distributions",
                    "5. Use goodness-of-fit tests",
                    "6. Follow standards if applicable"
                ]
                
                for step in steps:
                    st.markdown(step)
        
        return None
    
    def _slide_python_tutorial(self) -> Optional[bool]:
        """Slide 9: Python Tutorial"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Hands-On Python Tutorial")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Complete Tutorial Notebook")
                UIComponents.highlight_box("""
                **Google Colab Tutorial**
                
                [Open Python Notebook](https://colab.research.google.com/drive/1LkAEFUfspJRZKceOIW2StQ9UnhE_WVBE?usp=sharing)
                """)
                
            with col2:
                st.markdown("### Dataset")
                UIComponents.highlight_box("""
                **Hydrologic Data**
                
                [Download Excel File](https://jsums-my.sharepoint.com/:x:/g/personal/j01013381_students_jsums_edu/ETVE3UpUxsFAoC_5TLVfnG0BDRlAwWYIPs-Epxs8dd4WvA?e=15qB4x)
                """)
            
            st.markdown("---")
            st.markdown("**Instructions:** Click the notebook link, make a copy, download the data, and follow the tutorial.")
        
        return None
    
    def _slide_quiz(self) -> Optional[bool]:
        """Slide 10: Knowledge Check"""
        with UIComponents.slide_container():
            st.markdown("## Knowledge Check")
            
            result = QuizEngine.create_multiple_choice(
                "To find P(X ≤ 75) for a normally distributed variable, which function should you use?",
                [
                    "Use CDF: F(75) gives the probability directly",
                    "Use PDF: f(75) gives the probability directly",
                    "Use both PDF and CDF together",
                    "Use the area under PDF from 0 to 75"
                ],
                0,
                {
                    "correct": "Correct! Use CDF F(75) to get P(X ≤ 75) directly. The PDF height f(75) is probability density, not probability itself.",
                    "incorrect": "Remember: CDF F(x) gives P(X ≤ x) directly. PDF f(x) gives probability density (height), not probability itself."
                },
                f"{self.info.id}_final_quiz"
            )
            
            if result is True:
                st.success("Module 4 Complete! You understand probability distributions and Python implementation.")
                return True
            
            return None