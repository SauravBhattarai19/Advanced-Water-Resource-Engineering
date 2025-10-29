"""
Module 9: Trend Detection in Hydrologic Time Series
Mann-Kendall Test and Sen's Slope Estimator

Author: TA Saurav Bhattarai
Course: Advanced Water Resources Engineering
Institution: Jackson State University
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
from scipy import stats

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents


class Module09_TrendDetection(LearningModule):
    """Module 9: Trend Detection in Hydrologic Time Series"""

    def __init__(self):
        objectives = [
            LearningObjective("Understand the concept of trends in hydrologic data", "understand"),
            LearningObjective("Apply Mann-Kendall test for trend detection", "apply"),
            LearningObjective("Calculate Sen's slope estimator", "analyze"),
            LearningObjective("Interpret trend analysis results for engineering decisions", "evaluate")
        ]

        info = ModuleInfo(
            id="module_09",
            title="Trend Detection in Hydrologic Time Series",
            description="Statistical methods for detecting monotonic trends in water resources data",
            duration_minutes=40,
            prerequisites=["module_01", "module_02"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=1  # Paper-like format, single continuous page
        )

        super().__init__(info)

    def get_slide_titles(self) -> List[str]:
        return ["Trend Detection: A Comprehensive Guide"]

    def render_slide(self, slide_num: int) -> Optional[bool]:
        """Render the complete paper-like module"""
        return self._render_complete_module()

    def _render_complete_module(self) -> Optional[bool]:
        """Render complete module in paper-like academic format"""
        
        # Module title and metadata
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;'>
            <h1 style='margin: 0; color: white;'>Module 7: Trend Detection in Hydrologic Time Series</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
                Statistical Methods for Detecting Long-Term Changes in Water Resources Data
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Abstract
        with st.expander("📄 **ABSTRACT**", expanded=True):
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; 
                        border-left: 4px solid #667eea;'>
            
            **Objective:** This module introduces systematic methods for detecting and quantifying trends 
            in hydrologic time series data, essential for understanding long-term changes in water 
            resources systems due to climate variability, land use changes, and human interventions.
            
            **Methods:** We present the Mann-Kendall test, a non-parametric statistical test widely used 
            in hydrology for trend detection, along with Sen's slope estimator for quantifying trend magnitude.
            
            **Applications:** These methods are fundamental for updating design standards, assessing climate 
            change impacts, and making informed decisions about water resources infrastructure.
            
            **Keywords:** Trend analysis, Mann-Kendall test, Sen's slope, non-parametric statistics, 
            hydrologic time series, climate change detection
            
            </div>
            """, unsafe_allow_html=True)

        # Section 1: Introduction
        with st.expander("## 1. INTRODUCTION", expanded=False):
            st.markdown("### 1.1 The Need for Trend Detection in Water Resources")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Engineering Motivation:**
                
                Water resources infrastructure is typically designed based on historical data with the 
                implicit assumption of stationarity - that statistical properties remain constant over time. 
                However, numerous factors can cause systematic changes:
                
                • **Climate variability and change** - Altered precipitation patterns
                • **Land use modifications** - Urbanization, deforestation
                • **Water management practices** - Dam operations, diversions
                • **Natural cycles** - Multi-decadal oscillations
                
                **Consequences of Ignoring Trends:**
                
                1. **Under-designed infrastructure** - If flows are increasing
                2. **Over-designed structures** - If flows are decreasing
                3. **Inaccurate risk assessment** - Flood/drought probabilities change
                4. **Economic inefficiency** - Wasted resources or inadequate capacity
                """)
            
            with col2:
                # Generate example data showing trend
                np.random.seed(42)
                years = np.arange(1970, 2025)
                n_years = len(years)
                
                # Create increasing trend with noise
                trend_component = 2.5 * np.arange(n_years)
                seasonal_component = 15 * np.sin(2 * np.pi * np.arange(n_years) / 10)
                noise = np.random.normal(0, 20, n_years)
                flow_data = 150 + trend_component + seasonal_component + noise
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years, y=flow_data,
                    mode='lines+markers',
                    name='Observed Flow',
                    line=dict(color='steelblue', width=2),
                    marker=dict(size=4)
                ))
                
                # Add trend line
                z = np.polyfit(years, flow_data, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=years, y=p(years),
                    mode='lines',
                    name=f'Trend Line (slope={z[0]:.2f} m³/s/year)',
                    line=dict(color='red', width=3, dash='dash')
                ))
                
                fig.update_layout(
                    title="Example: Increasing Streamflow Trend (1970-2024)",
                    xaxis_title="Year",
                    yaxis_title="Annual Peak Flow (m³/s)",
                    height=400,
                    hovermode='x unified'
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"""
                **Visual Observation:** Flow has increased by approximately 
                {z[0] * n_years:.1f} m³/s over {n_years} years. But is this change 
                statistically significant or just random variation?
                """)
            
            st.markdown("### 1.2 Types of Trends in Hydrologic Data")
            
            st.markdown("""
            **Monotonic Trends:**
            - Consistently increasing or decreasing over time
            - Most common focus in hydrologic analysis
            - Detected by Mann-Kendall test
            
            **Step Changes:**
            - Abrupt shifts in mean level
            - Addressed in Module 8 (Change Point Detection)
            
            **Cyclical Patterns:**
            - Periodic oscillations (e.g., ENSO, PDO)
            - Require different analytical approaches
            
            **This module focuses on monotonic trends**, which are most relevant for 
            long-term planning and design standard updates.
            """)

        # Section 2: Parametric vs Non-Parametric Methods
        with st.expander("## 2. PARAMETRIC VS NON-PARAMETRIC TREND TESTS", expanded=False):
            st.markdown("### 2.1 Why Non-Parametric Methods?")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Parametric Methods (e.g., Linear Regression t-test):**
                
                **Assumptions:**
                - Data follow normal distribution
                - Constant variance (homoscedasticity)
                - Independence of residuals
                - Linear relationship
                
                **Advantages:**
                - More powerful when assumptions are met
                - Familiar to most engineers
                - Direct interpretation
                
                **Limitations in Hydrology:**
                - ❌ Hydrologic data often skewed (not normal)
                - ❌ Extreme values common (outliers)
                - ❌ Variance often changes over time
                - ❌ Sensitive to violations of assumptions
                """)
            
            with col2:
                st.markdown("""
                **Non-Parametric Methods (Mann-Kendall Test):**
                
                **Assumptions:**
                - ✅ No distribution requirement
                - ✅ Works with any monotonic trend
                - ✅ Data independence (or can be corrected)
                
                **Advantages:**
                - Robust to outliers
                - Distribution-free
                - Handles missing data well
                - Works with small samples
                - Widely accepted in hydrology
                
                **Why Mann-Kendall for Hydrology:**
                - ✅ Recommended by WMO (World Meteorological Organization)
                - ✅ Standard in climate change studies
                - ✅ Handles non-normal hydrologic data
                - ✅ Less affected by extreme events
                """)
            
            # Visual comparison
            st.markdown("### 2.2 Visual Comparison: Parametric vs Non-Parametric Robustness")
            
            # Create comparison datasets
            np.random.seed(123)
            x = np.arange(1, 21)
            
            # Normal data with trend
            y_normal = 10 + 2*x + np.random.normal(0, 5, 20)
            
            # Skewed data with trend and outlier
            y_skewed = 10 + 2*x + np.random.gamma(2, 2, 20)
            y_skewed[15] = 80  # Add extreme outlier
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Normal Data: Both Methods Work Well',
                              'Skewed Data with Outlier: Non-Parametric More Robust')
            )
            
            # Normal data plot
            fig.add_trace(go.Scatter(x=x, y=y_normal, mode='markers',
                                    name='Normal Data', marker=dict(color='blue', size=8)),
                         row=1, col=1)
            z1 = np.polyfit(x, y_normal, 1)
            fig.add_trace(go.Scatter(x=x, y=np.poly1d(z1)(x), mode='lines',
                                    name='Parametric Fit', line=dict(color='red', dash='dash')),
                         row=1, col=1)
            
            # Skewed data plot
            fig.add_trace(go.Scatter(x=x, y=y_skewed, mode='markers',
                                    name='Skewed Data', marker=dict(color='green', size=8)),
                         row=1, col=2)
            z2 = np.polyfit(x, y_skewed, 1)
            fig.add_trace(go.Scatter(x=x, y=np.poly1d(z2)(x), mode='lines',
                                    name='Parametric (Biased)', line=dict(color='red', dash='dash')),
                         row=1, col=2)
            
            # Highlight outlier
            fig.add_trace(go.Scatter(x=[x[15]], y=[y_skewed[15]], mode='markers',
                                    marker=dict(color='red', size=15, symbol='x', line=dict(width=3)),
                                    name='Outlier', showlegend=True),
                         row=1, col=2)
            
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_yaxes(title_text="Flow (m³/s)", row=1, col=1)
            fig.update_yaxes(title_text="Flow (m³/s)", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            fig = PlotTools.apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("""
            **Key Insight:** The Mann-Kendall test uses ranks instead of actual values, making it 
            resistant to the outlier's influence. The parametric regression is heavily influenced 
            by the extreme value, potentially leading to incorrect conclusions.
            """)
            
            st.markdown("### 📝 Practice Questions")
            
            result_p1 = QuizEngine.create_multiple_choice(
                "A dataset of annual peak flows has a strongly skewed distribution with several extreme outliers. "
                "Which test is more appropriate for detecting trends?",
                [
                    "Linear regression with t-test because it's more powerful",
                    "Mann-Kendall test because it's robust to non-normal distributions and outliers",
                    "Neither - must transform data to normal distribution first",
                    "Both tests will give identical results for trend detection"
                ],
                1,
                {
                    "correct": "✅ Correct! The Mann-Kendall test is specifically designed for situations like this. "
                              "It uses ranks instead of actual values, making it robust to outliers and not requiring "
                              "normality assumptions. This is exactly why it's the standard method in hydrology where "
                              "extreme events and skewed distributions are common.",
                    "incorrect": "Consider the properties of non-parametric tests. They work with ranks, not actual "
                                "values, making them robust to outliers and distribution shape. Linear regression "
                                "assumes normality and is sensitive to outliers, which makes it problematic for "
                                "typical hydrologic data."
                },
                f"{self.info.id}_practice_1"
            )
            
            if result_p1:
                st.markdown("---")

        # Section 3: Mann-Kendall Test Theory
        with st.expander("## 3. MANN-KENDALL TEST: THEORETICAL FOUNDATION", expanded=False):
            st.markdown("### 3.1 Hypothesis Framework")
            
            st.markdown("""
            <div style='background-color: #e8f4f8; padding: 1.5rem; border-radius: 8px; 
                        border-left: 4px solid #2196F3; margin: 1rem 0;'>
            
            **Null Hypothesis (H₀):** There is no monotonic trend in the time series.  
            *Mathematically:* The data are independent and identically distributed.
            
            **Alternative Hypothesis (H₁):** There is a monotonic trend (increasing or decreasing).  
            *Mathematically:* There exists a tendency for values to increase or decrease over time.
            
            **Significance Level (α):** Typically 0.05 (5% chance of Type I error)
            
            **Decision Rule:**
            - If p-value < α → Reject H₀ → **Significant trend exists**
            - If p-value ≥ α → Fail to reject H₀ → **No significant trend detected**
            
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 3.2 The Mann-Kendall S Statistic")
            
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                st.markdown("""
                **Mathematical Formulation:**
                
                The Mann-Kendall test compares every data point with all subsequent points:
                """)
                
                st.latex(r"""
                S = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \text{sgn}(X_j - X_i)
                """)
                
                st.markdown("""
                Where the sign function is defined as:
                """)
                
                st.latex(r"""
                \text{sgn}(X_j - X_i) = \begin{cases}
                +1 & \text{if } X_j > X_i \text{ (increasing)}\\
                0 & \text{if } X_j = X_i \text{ (no change)}\\
                -1 & \text{if } X_j < X_i \text{ (decreasing)}
                \end{cases}
                """)
                
                st.markdown("""
                **Interpretation of S:**
                
                - **S > 0:** More pairs showing increase → Positive (upward) trend
                - **S < 0:** More pairs showing decrease → Negative (downward) trend
                - **S ≈ 0:** Roughly equal increases and decreases → No trend
                
                **Number of Comparisons:**
                
                For a dataset with n observations, the total number of pairwise comparisons is:
                """)
                
                st.latex(r"""
                \text{Number of pairs} = \frac{n(n-1)}{2}
                """)
                
                st.markdown("""
                This means:
                - n = 10 data points → 45 comparisons
                - n = 30 data points → 435 comparisons
                - n = 50 data points → 1,225 comparisons
                """)
            
            with col2:
                st.markdown("**Example Calculation:**")
                
                # Simple example dataset
                example_data = [23, 25, 22, 28, 30, 29, 32, 35]
                n_ex = len(example_data)
                
                st.markdown(f"""
                Consider 8 years of data:
                
                | Year | Value |
                |------|-------|
                {chr(10).join([f"| {i+1} | {val} |" for i, val in enumerate(example_data)])}
                
                **Sample Comparisons:**
                """)
                
                # Show some comparisons
                comparisons = []
                for i in range(min(3, n_ex-1)):
                    for j in range(i+1, min(i+4, n_ex)):
                        diff = example_data[j] - example_data[i]
                        sign = '+1' if diff > 0 else ('0' if diff == 0 else '-1')
                        comparisons.append(f"X[{j+1}] vs X[{i+1}]: {example_data[j]} - {example_data[i]} = {diff} → {sign}")
                
                for comp in comparisons[:8]:
                    st.markdown(f"• {comp}")
                
                # Calculate actual S
                S = 0
                for i in range(n_ex-1):
                    for j in range(i+1, n_ex):
                        S += np.sign(example_data[j] - example_data[i])
                
                total_comparisons = n_ex * (n_ex - 1) // 2
                
                st.markdown(f"""
                ... ({total_comparisons} total comparisons)
                
                **Result: S = {S}**
                
                {f"**Interpretation:** S > 0 indicates an upward trend" if S > 0 else 
                 f"**Interpretation:** S < 0 indicates a downward trend" if S < 0 else
                 "**Interpretation:** S ≈ 0 indicates no clear trend"}
                """)
            
            st.markdown("### 3.3 Standardization and P-Value Calculation")
            
            st.markdown("""
            **Variance of S (for large samples, n ≥ 10):**
            
            When there are no tied values:
            """)
            
            st.latex(r"""
            \text{Var}(S) = \frac{n(n-1)(2n+5)}{18}
            """)
            
            st.markdown("""
            **Standardized Test Statistic (Z):**
            """)
            
            st.latex(r"""
            Z = \begin{cases}
            \frac{S-1}{\sqrt{\text{Var}(S)}} & \text{if } S > 0\\
            0 & \text{if } S = 0\\
            \frac{S+1}{\sqrt{\text{Var}(S)}} & \text{if } S < 0
            \end{cases}
            """)
            
            st.markdown("""
            The continuity correction (±1) improves accuracy for discrete distributions.
            
            **P-Value Interpretation:**
            
            The p-value represents the probability of observing a test statistic as extreme as 
            Z (or more extreme) under the null hypothesis of no trend.
            
            - **Two-tailed test:** We're interested in trends in either direction
            - **P-value < 0.05:** Strong evidence against H₀ → Significant trend
            - **P-value < 0.01:** Very strong evidence → Highly significant trend
            - **P-value < 0.001:** Extremely strong evidence → Very highly significant trend
            """)
            
            # Example calculation
            st.markdown("### 3.4 Complete Numerical Example")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Given Data:** Annual rainfall (mm)
                
                | Year | Rainfall | Year | Rainfall |
                |------|----------|------|----------|
                | 1 | 850 | 6 | 920 |
                | 2 | 870 | 7 | 940 |
                | 3 | 860 | 8 | 930 |
                | 4 | 890 | 9 | 960 |
                | 5 | 900 | 10 | 970 |
                
                **Step 1: Calculate S**
                
                Perform all pairwise comparisons:
                - Total comparisons = 10×9/2 = 45
                - Count +1 for increases
                - Count -1 for decreases
                
                **Result: S = +38**
                """)
            
            with col2:
                st.markdown("""
                **Step 2: Calculate Variance**
                """)
                
                n = 10
                var_S = n * (n-1) * (2*n+5) / 18
                
                st.markdown(f"""
                ```
                Var(S) = {n}×{n-1}×{2*n+5} / 18
                       = {n}×{n-1}×{2*n+5} / 18
                       = {var_S:.2f}
                ```
                
                **Step 3: Calculate Z**
                
                Since S > 0:
                ```
                Z = (S - 1) / √Var(S)
                  = ({38} - 1) / √{var_S:.2f}
                  = {37} / {np.sqrt(var_S):.2f}
                  = {37 / np.sqrt(var_S):.3f}
                ```
                
                **Step 4: Find P-Value**
                
                For Z = {37 / np.sqrt(var_S):.3f} (two-tailed):
                ```
                p-value ≈ {2 * (1 - stats.norm.cdf(37 / np.sqrt(var_S))):.6f}
                ```
                
                **Conclusion: p < 0.001**  
                **Highly significant increasing trend!**
                """)
            
            st.markdown("### 📝 Practice Questions")
            
            # Numerical question
            st.markdown("**Numerical Problem:**")
            st.markdown("""
            Given the following 6 years of annual rainfall data (mm): 850, 870, 860, 890, 900, 920
            
            **Calculate:**
            1. The Mann-Kendall S statistic (count all pairwise comparisons)
            2. Determine if the trend is increasing or decreasing
            
            *Hint: There are C(6,2) = 15 pairwise comparisons*
            """)
            
            show_solution_1 = st.checkbox("💡 Show Solution", key="solution_mk_numerical")
            
            if show_solution_1:
                st.markdown("""
                **Solution:**
                
                **All pairwise comparisons:**
                
                | Pair | Comparison | Xⱼ - Xᵢ | sgn |
                |------|------------|---------|-----|
                | (1,2) | 870 - 850 | +20 | +1 |
                | (1,3) | 860 - 850 | +10 | +1 |
                | (1,4) | 890 - 850 | +40 | +1 |
                | (1,5) | 900 - 850 | +50 | +1 |
                | (1,6) | 920 - 850 | +70 | +1 |
                | (2,3) | 860 - 870 | -10 | -1 |
                | (2,4) | 890 - 870 | +20 | +1 |
                | (2,5) | 900 - 870 | +30 | +1 |
                | (2,6) | 920 - 870 | +50 | +1 |
                | (3,4) | 890 - 860 | +30 | +1 |
                | (3,5) | 900 - 860 | +40 | +1 |
                | (3,6) | 920 - 860 | +60 | +1 |
                | (4,5) | 900 - 890 | +10 | +1 |
                | (4,6) | 920 - 890 | +30 | +1 |
                | (5,6) | 920 - 900 | +20 | +1 |
                
                **S = Sum of all sgn values = 14 × (+1) + 1 × (-1) = +13**
                
                **Answer:**
                1. S = +13
                2. Positive S → **Increasing trend**
                """)
            
            result_p2 = QuizEngine.create_multiple_choice(
                "For a Mann-Kendall test with n=25 observations, you calculate S = +180 and p = 0.08. "
                "What should you conclude at α = 0.05 significance level?",
                [
                    "Reject H₀; significant increasing trend exists",
                    "Fail to reject H₀; no significant trend detected",
                    "The trend is decreasing because S is positive",
                    "Need more data before making any conclusion"
                ],
                1,
                {
                    "correct": "✅ Correct! With p = 0.08 > 0.05 (significance level α), we fail to reject the null "
                              "hypothesis. While S is positive suggesting an increasing pattern, the p-value indicates "
                              "this could reasonably occur by chance. The apparent trend is not statistically significant "
                              "at the 0.05 level. Note: It would be significant at α = 0.10 if that were chosen.",
                    "incorrect": "Look carefully at the p-value (0.08) and compare it to the significance level (0.05). "
                                "The decision rule is: reject H₀ if p < α. Here, p > α, so we cannot reject H₀. "
                                "The positive S does suggest an increasing pattern, but it's not statistically significant."
                },
                f"{self.info.id}_practice_2"
            )
            
            if result_p2:
                st.markdown("---")

        # Section 4: Sen's Slope Estimator
        with st.expander("## 4. SEN'S SLOPE ESTIMATOR: QUANTIFYING TREND MAGNITUDE", expanded=False):
            st.markdown("### 4.1 Why Do We Need Sen's Slope?")
            
            st.markdown("""
            The Mann-Kendall test tells us **IF** a trend exists, but not **HOW MUCH** change 
            is occurring per unit time. Sen's slope estimator provides:
            
            - **Magnitude of trend** (e.g., increase of 2.5 m³/s per year)
            - **Robust estimate** (not affected by outliers)
            - **Engineering utility** (for projecting future values)
            - **Complements Mann-Kendall** (used together in practice)
            """)
            
            st.markdown("### 4.2 Mathematical Formulation")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Step 1: Calculate Slopes for All Data Pairs**
                
                For each pair of observations (i, j) where j > i:
                """)
                
                st.latex(r"""
                Q_{ij} = \frac{X_j - X_i}{j - i}
                """)
                
                st.markdown("""
                Where:
                - X_j, X_i = data values at times j and i
                - j - i = time difference between observations
                
                **Step 2: Calculate Median Slope**
                
                Sen's slope estimator (β) is the **median** of all Q_ij values:
                """)
                
                st.latex(r"""
                \beta = \text{median}(Q_{ij})
                """)
                
                st.markdown("""
                **Why Median Instead of Mean?**
                
                - ✅ Robust to outliers (extreme values don't bias result)
                - ✅ More representative of typical rate of change
                - ✅ Consistent with non-parametric philosophy
                - ✅ Provides more stable estimates for hydrologic data
                """)
            
            with col2:
                st.markdown("""
                **Example: 5 Years of Data**
                
                | Year | Flow (m³/s) |
                |------|-------------|
                | 1 | 100 |
                | 2 | 105 |
                | 3 | 108 |
                | 4 | 112 |
                | 5 | 115 |
                
                **All Pairwise Slopes:**
                
                ```
                Q₁₂ = (105-100)/(2-1) = 5.0
                Q₁₃ = (108-100)/(3-1) = 4.0
                Q₁₄ = (112-100)/(4-1) = 4.0
                Q₁₅ = (115-100)/(5-1) = 3.75
                Q₂₃ = (108-105)/(3-2) = 3.0
                Q₂₄ = (112-105)/(4-2) = 3.5
                Q₂₅ = (115-105)/(5-2) = 3.33
                Q₃₄ = (112-108)/(4-3) = 4.0
                Q₃₅ = (115-108)/(5-3) = 3.5
                Q₄₅ = (115-112)/(5-4) = 3.0
                ```
                
                **Sorted Slopes:**  
                3.0, 3.0, 3.33, 3.5, 3.5, 3.75, 4.0, 4.0, 4.0, 5.0
                
                **Median (Sen's Slope):**  
                β = (3.75 + 3.5) / 2 = **3.625 m³/s/year**
                
                **Interpretation:**  
                Flow is increasing at approximately 3.6 m³/s per year
                """)
            
            st.markdown("### 4.3 Confidence Intervals for Sen's Slope")
            
            st.markdown("""
            To assess uncertainty in the slope estimate, we calculate confidence intervals:
            
            **For 95% Confidence Interval:**
            
            1. Calculate the confidence limit:
            """)
            
            st.latex(r"""
            C_\alpha = Z_{1-\alpha/2} \times \sqrt{\text{Var}(S)}
            """)
            
            st.markdown("""
            Where Z₁₋α/₂ = 1.96 for 95% confidence
            
            2. Find the positions in the ordered slopes:
            """)
            
            st.latex(r"""
            M_1 = \frac{N - C_\alpha}{2}, \quad M_2 = \frac{N + C_\alpha}{2}
            """)
            
            st.markdown("""
            Where N = total number of slopes
            
            3. The lower and upper confidence limits are:
            - **Lower limit:** Q_(M₁)
            - **Upper limit:** Q_(M₂)
            
            **Engineering Significance:**
            
            - Narrow CI → High confidence in trend magnitude
            - Wide CI → Large uncertainty, more data needed
            - If CI includes zero → Trend may not be significant
            """)
            
            st.markdown("### 📝 Practice Questions")
            
            # Numerical question for Sen's slope
            st.markdown("**Numerical Problem:**")
            st.markdown("""
            Given 4 years of annual flow data (m³/s): Year 1=100, Year 2=105, Year 3=108, Year 4=112
            
            **Calculate the Sen's slope estimator:**
            1. List all pairwise slopes Qᵢⱼ = (Xⱼ - Xᵢ)/(j - i)
            2. Find the median slope
            """)
            
            show_solution_2 = st.checkbox("💡 Show Solution", key="solution_sens_numerical")
            
            if show_solution_2:
                st.markdown("""
                **Solution:**
                
                **All pairwise slopes:**
                
                | Pair | Calculation | Slope (Qᵢⱼ) |
                |------|-------------|-------------|
                | Q₁₂ | (105-100)/(2-1) | 5.0 |
                | Q₁₃ | (108-100)/(3-1) | 4.0 |
                | Q₁₄ | (112-100)/(4-1) | 4.0 |
                | Q₂₃ | (108-105)/(3-2) | 3.0 |
                | Q₂₄ | (112-105)/(4-2) | 3.5 |
                | Q₃₄ | (112-108)/(4-3) | 4.0 |
                
                **Sorted slopes:** 3.0, 3.5, 4.0, 4.0, 4.0, 5.0
                
                **Median (n=6, even):**
                Median = (3rd value + 4th value) / 2 = (4.0 + 4.0) / 2 = **4.0 m³/s/year**
                
                **Interpretation:** Flow is increasing at a rate of 4.0 m³/s per year
                """)
            
            result_p3 = QuizEngine.create_multiple_choice(
                "You calculate Sen's slope = +2.5 mm/year with 95% CI = [+1.8, +3.2] for rainfall data. "
                "The Mann-Kendall test gives p = 0.04. What is the correct interpretation?",
                [
                    "Rainfall increasing 2.5 mm/year; trend is significant; high confidence in magnitude",
                    "Rainfall decreasing because confidence interval is positive",
                    "Trend is not significant because slope is small",
                    "Cannot interpret without knowing the mean rainfall"
                ],
                0,
                {
                    "correct": "✅ Excellent! All three pieces work together: (1) Sen's slope = +2.5 mm/year shows "
                              "magnitude and direction of change, (2) p = 0.04 < 0.05 confirms statistical significance, "
                              "(3) narrow CI [+1.8, +3.2] that doesn't include zero shows high confidence. The positive "
                              "slope and CI confirm increasing rainfall with well-constrained magnitude.",
                    "incorrect": "Consider all three components: Sen's slope shows the magnitude (+2.5 mm/year = increasing), "
                                "p-value shows significance (p=0.04 < 0.05 = significant), and CI shows confidence "
                                "(narrow range, doesn't include zero = high confidence). All three point to a real, "
                                "well-quantified increasing trend."
                },
                f"{self.info.id}_practice_3"
            )
            
            if result_p3:
                st.markdown("---")

        # Section 5: Practical Application
        with st.expander("## 5. INTERACTIVE DEMONSTRATION", expanded=False):
            st.markdown("### 5.1 Explore Different Trend Scenarios")
            
            # Interactive controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_magnitude = st.slider(
                    "Trend Strength (units/year):",
                    min_value=-5.0, max_value=5.0, value=2.0, step=0.5,
                    help="Positive = increasing trend, Negative = decreasing trend"
                )
            
            with col2:
                noise_level = st.slider(
                    "Data Variability:",
                    min_value=0.5, max_value=3.0, value=1.0, step=0.5,
                    help="Higher values add more random variation"
                )
            
            with col3:
                n_years = st.slider(
                    "Number of Years:",
                    min_value=10, max_value=50, value=30, step=5,
                    help="More years generally give clearer trend detection"
                )
            
            # Generate synthetic data
            np.random.seed(42)
            years = np.arange(1, n_years + 1)
            trend = trend_magnitude * years
            noise = np.random.normal(0, noise_level * 20, n_years)
            data_values = 100 + trend + noise
            
            # Calculate Mann-Kendall test
            S = 0
            n = len(data_values)
            for i in range(n-1):
                for j in range(i+1, n):
                    S += np.sign(data_values[j] - data_values[i])
            
            var_S = n * (n-1) * (2*n+5) / 18
            
            if S > 0:
                Z = (S - 1) / np.sqrt(var_S)
            elif S < 0:
                Z = (S + 1) / np.sqrt(var_S)
            else:
                Z = 0
            
            p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
            
            # Calculate Sen's slope
            slopes = []
            for i in range(n-1):
                for j in range(i+1, n):
                    slopes.append((data_values[j] - data_values[i]) / (j - i))
            sens_slope = np.median(slopes)
            
            # Visualization
            fig = go.Figure()
            
            # Data points
            fig.add_trace(go.Scatter(
                x=years, y=data_values,
                mode='lines+markers',
                name='Observed Data',
                line=dict(color='steelblue', width=2),
                marker=dict(size=6)
            ))
            
            # Sen's slope line
            intercept = np.median(data_values - sens_slope * years)
            sens_line = intercept + sens_slope * years
            fig.add_trace(go.Scatter(
                x=years, y=sens_line,
                mode='lines',
                name=f"Sen's Slope = {sens_slope:.3f} units/year",
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig.update_layout(
                title="Interactive Trend Analysis",
                xaxis_title="Year",
                yaxis_title="Value",
                height=400,
                hovermode='x unified'
            )
            fig = PlotTools.apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # Results display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mann-Kendall S", f"{S:.0f}")
                st.metric("Z-statistic", f"{Z:.3f}")
            
            with col2:
                st.metric("P-value", f"{p_value:.4f}")
                if p_value < 0.001:
                    st.success("✅ Highly Significant")
                elif p_value < 0.01:
                    st.success("✅ Very Significant")
                elif p_value < 0.05:
                    st.info("✓ Significant")
                else:
                    st.warning("○ Not Significant")
            
            with col3:
                st.metric("Sen's Slope", f"{sens_slope:.3f}")
                trend_direction = "Increasing" if sens_slope > 0 else "Decreasing" if sens_slope < 0 else "No trend"
                st.info(f"**{trend_direction}** trend")
            
            # Statistical summary
            st.markdown("### 5.2 Statistical Summary")
            
            st.markdown(f"""
            **Test Results:**
            
            - **Mann-Kendall Test Statistic (S):** {S:.0f}
            - **Standardized Z-score:** {Z:.3f}
            - **P-value (two-tailed):** {p_value:.6f}
            - **Significance at α=0.05:** {'✅ YES' if p_value < 0.05 else '❌ NO'}
            - **Sen's Slope Estimator (β):** {sens_slope:.4f} units/year
            - **Total Number of Comparisons:** {len(slopes)}
            
            **Interpretation:**
            
            {f"A statistically significant {'increasing' if sens_slope > 0 else 'decreasing'} trend was detected "
             f"(p = {p_value:.4f}). The rate of change is approximately {abs(sens_slope):.3f} units per year." 
             if p_value < 0.05 else 
             f"No statistically significant trend was detected (p = {p_value:.4f}). "
             f"The apparent slope of {sens_slope:.3f} units/year could be due to random variation."}
            
            **Engineering Implications:**
            
            {f"• Over the {n_years}-year period, values have changed by approximately {sens_slope * n_years:.1f} units" if p_value < 0.05 else
             f"• The data can be considered stationary for design purposes"}
            {f"• For future projections: Value in year {n_years+10} ≈ {data_values[-1] + sens_slope*10:.1f} units" if p_value < 0.05 else ""}
            {f"• Design standards should account for this trend" if p_value < 0.05 else
             f"• Historical statistics remain valid for design"}
            """)
            
            st.markdown("### 📝 Practice Question")
            
            result_p4 = QuizEngine.create_multiple_choice(
                "You analyze 30 years of data with the interactive tool and find: trend = +3.2 units/year, "
                "p = 0.001, but the data variability (noise) is very high. What should you consider for engineering design?",
                [
                    "Ignore the trend because high noise makes results unreliable",
                    "Use the trend but add extra safety factors due to high variability",
                    "Collect more data until noise decreases",
                    "The trend is invalid if noise is high"
                ],
                1,
                {
                    "correct": "✅ Correct! A significant trend (p=0.001) is real even with high variability. The trend "
                              "tells you about the central tendency change, while high variability tells you about "
                              "uncertainty. Engineering response: account for the trend in mean values AND increase "
                              "safety factors to handle the high variability. Both the trend and the variability are "
                              "real characteristics that affect design.",
                    "incorrect": "Separate the concepts: trend significance (p=0.001 is very strong) versus data "
                                "variability (scatter around the trend line). A significant trend exists even in noisy "
                                "data - that's the power of Mann-Kendall test! But high variability does increase "
                                "uncertainty, warranting conservative design approaches."
                },
                f"{self.info.id}_practice_4"
            )
            
            if result_p4:
                st.markdown("---")

        # Section 6: Engineering Applications
        with st.expander("## 6. ENGINEERING APPLICATIONS AND CASE STUDIES", expanded=False):
            st.markdown("### 6.1 Infrastructure Design Standards")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Case Study: Urban Drainage System**
                
                **Problem:**  
                A city's storm drainage system was designed in 1985 based on 50 years of rainfall data. 
                Recent flooding events suggest inadequacy.
                
                **Analysis:**  
                - Dataset: Annual maximum 24-hour rainfall (1970-2023)
                - Mann-Kendall test performed on data
                
                **Results:**
                - S = +342
                - Z = 4.23
                - p-value < 0.0001
                - Sen's slope = +1.8 mm/year
                
                **Findings:**
                - Highly significant increasing trend
                - Rainfall intensity increased by ~95 mm over 53 years
                - Current design storms underestimate by ~15%
                
                **Engineering Decision:**
                - Update IDF curves using recent 20-year data
                - Increase drainage capacity by 20% for new designs
                - Prioritize retrofits in flood-prone areas
                - Re-evaluate existing infrastructure
                """)
            
            with col2:
                # Create case study visualization
                np.random.seed(100)
                case_years = np.arange(1970, 2024)
                n_case = len(case_years)
                case_rainfall = 120 + 1.8 * np.arange(n_case) + np.random.normal(0, 15, n_case)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=case_years, y=case_rainfall,
                    mode='lines+markers',
                    name='Observed Rainfall',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                # Design level from 1985
                design_level = 120 + 1.8 * 15  # Value at 1985
                fig.add_hline(
                    y=design_level,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="1985 Design Level",
                    annotation_position="right"
                )
                
                # Current typical level
                current_level = 120 + 1.8 * (n_case - 1)
                fig.add_hline(
                    y=current_level,
                    line_dash="dot",
                    line_color="green",
                    annotation_text="Current Level",
                    annotation_position="right"
                )
                
                fig.update_layout(
                    title="Case Study: Increasing Rainfall Trend",
                    xaxis_title="Year",
                    yaxis_title="24-Hour Rainfall (mm)",
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                st.warning("""
                **Gap between design and current conditions = 15%**  
                This explains recent flooding issues!
                """)
            
            st.markdown("### 6.2 Water Supply Planning")
            
            st.markdown("""
            **Application Areas:**
            
            1. **Reservoir Yield Analysis**
               - Detect trends in inflow patterns
               - Adjust operating rules
               - Update firm yield calculations
            
            2. **Drought Management**
               - Identify long-term drying trends
               - Trigger conservation measures
               - Plan augmentation projects
            
            3. **Climate Change Adaptation**
               - Quantify historical changes
               - Project future conditions
               - Develop adaptation strategies
            
            4. **Agricultural Water Demand**
               - Assess irrigation requirement trends
               - Plan water allocation
               - Optimize cropping patterns
            """)
            
            st.markdown("### 6.3 When to Update Design Standards")
            
            st.markdown("""
            **Decision Framework:**
            
            | Scenario | P-value | Sen's Slope | Action |
            |----------|---------|-------------|--------|
            | Strong increasing trend | < 0.01 | > 2% of mean per decade | Update immediately, use recent data |
            | Moderate increasing trend | 0.01-0.05 | 1-2% of mean per decade | Update for new critical infrastructure |
            | Weak/No trend | > 0.05 | < 1% of mean per decade | Continue using historical data |
            | Decreasing trend | < 0.05 | Negative | Consider economic optimization |
            
            **Considerations:**
            
            - **Safety-critical structures:** More conservative approach (update with weaker trends)
            - **Economic structures:** Balance cost vs. risk
            - **Time horizon:** Longer design life → more important to account for trends
            - **Uncertainty:** Use confidence intervals for design values
            """)
            
            st.markdown("### 📝 Practice Questions")
            
            result_p5 = QuizEngine.create_multiple_choice(
                "You're designing a new bridge over a river. Historical flow data (40 years) shows: "
                "Mann-Kendall p = 0.02, Sen's slope = +2.5 m³/s/year. Current mean flow = 150 m³/s. "
                "Bridge design life = 75 years. What design flow should you consider?",
                [
                    "Use historical mean of 150 m³/s (most data, most reliable)",
                    "Project 75 years forward: 150 + (2.5 × 75) = 337.5 m³/s",
                    "Use recent 20-year data and project forward 50-75 years",
                    "Add 10% safety factor to 150 m³/s regardless of trend"
                ],
                2,
                {
                    "correct": "✅ Excellent engineering judgment! With a significant trend (p=0.02), the historical "
                              "mean no longer represents future conditions. However, projecting 75 years assumes the "
                              "trend continues unchanged (risky assumption). Best practice: use recent data that reflects "
                              "current regime, project forward conservatively (50-75 years), and include safety factors. "
                              "This balances acknowledgment of the trend with uncertainty about its persistence.",
                    "incorrect": "Think about trend persistence and uncertainty. The historical mean ignores the significant "
                                "trend (p=0.02). Projecting 75 years linearly assumes perfect trend continuation (unrealistic). "
                                "The best approach uses recent data representing current conditions and projects forward "
                                "conservatively with appropriate safety margins."
                },
                f"{self.info.id}_practice_5"
            )
            
            if result_p5:
                st.info("""
                **Engineering Insight:** For long-lived infrastructure with detected trends:
                1. Use recent data (represents current regime)
                2. Project conservatively (don't assume infinite trend continuation)
                3. Add safety factors (account for uncertainty)
                4. Plan for future monitoring and potential upgrades
                """)
                st.markdown("---")
            
            # Additional numerical problem
            st.markdown("**Application Problem:**")
            st.markdown("""
            A city's storm drainage system designed in 1985 based on 1960-1985 rainfall data (mean = 900 mm/year). 
            New analysis (1960-2024, n=65 years) shows:
            - Mann-Kendall: S = +642, p = 0.003
            - Sen's slope: +1.8 mm/year
            - Current mean (2010-2024): 975 mm/year
            
            **Questions:**
            1. Is the trend statistically significant?
            2. How much has rainfall increased since 1985?
            3. Should the city update drainage design standards?
            """)
            
            show_solution_3 = st.checkbox("💡 Show Solution", key="solution_application_problem")
            
            if show_solution_3:
                st.markdown("""
                **Solution:**
                
                **1. Statistical Significance:**
                - P-value = 0.003 < 0.05 → **YES, highly significant trend**
                - S is large and positive → Increasing trend
                
                **2. Rainfall Increase Since 1985:**
                - Years elapsed: 2024 - 1985 = 39 years
                - Increase = Sen's slope × years = 1.8 mm/year × 39 years = **70.2 mm**
                - Original mean: 900 mm
                - Expected current: 900 + 70 = **970 mm** (close to observed 975 mm ✓)
                - **Percentage increase: (70/900) × 100 = 7.8%**
                
                **3. Should Update Standards?**
                **YES** - Strong evidence for update:
                - Trend is highly significant (p = 0.003)
                - Substantial increase (~8% over design period)
                - Current rainfall 975 mm vs design basis 900 mm
                - Existing infrastructure may be undersized by ~8%
                
                **Recommendation:**
                - Update IDF curves using post-1990 or post-2000 data
                - Increase design rainfall by 10-15% for new projects
                - Assess existing critical infrastructure for upgrades
                - Implement monitoring program
                """)

        # Section 7: Python Implementation
        with st.expander("## 7. PYTHON IMPLEMENTATION", expanded=False):
            st.markdown("### 7.1 Mann-Kendall Test Implementation")
            
            st.markdown("""
            **Complete Python Implementation:**
            
            This implementation calculates the Mann-Kendall test statistic and p-value from first principles.
            """)
            
            st.code("""
import numpy as np
from scipy import stats

def mann_kendall_test(data):
    \"\"\"
    Performs Mann-Kendall trend test.
    
    Parameters:
    -----------
    data : array-like
        Time series data (list or numpy array)
    
    Returns:
    --------
    dict : Dictionary containing:
        - S: Mann-Kendall statistic
        - tau: Kendall's tau (normalized S)
        - p_value: Two-tailed p-value
        - trend: 'increasing', 'decreasing', or 'no trend'
        - z_score: Standardized test statistic
    \"\"\"
    n = len(data)
    
    # Step 1: Calculate S statistic
    S = 0
    for i in range(n-1):
        for j in range(i+1, n):
            S += np.sign(data[j] - data[i])
    
    # Step 2: Calculate variance of S
    # Assuming no ties for simplicity
    var_S = n * (n - 1) * (2*n + 5) / 18
    
    # Step 3: Calculate standardized test statistic (Z)
    if S > 0:
        z_score = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        z_score = (S + 1) / np.sqrt(var_S)
    else:
        z_score = 0
    
    # Step 4: Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Step 5: Calculate Kendall's tau
    tau = S / (n * (n - 1) / 2)
    
    # Step 6: Determine trend direction
    if p_value < 0.05 and S > 0:
        trend = 'increasing'
    elif p_value < 0.05 and S < 0:
        trend = 'decreasing'
    else:
        trend = 'no trend'
    
    return {
        'S': S,
        'tau': tau,
        'z_score': z_score,
        'p_value': p_value,
        'trend': trend,
        'significant': p_value < 0.05
    }

# Example Usage
# -------------
# Annual peak discharge data (m³/s)
discharge_data = [120, 125, 118, 135, 142, 138, 155, 160, 
                  165, 172, 168, 185, 190, 195, 205, 198]

# Perform Mann-Kendall test
result = mann_kendall_test(discharge_data)

# Display results
print(f"Mann-Kendall Statistic (S): {result['S']}")
print(f"Kendall's tau: {result['tau']:.4f}")
print(f"Z-score: {result['z_score']:.4f}")
print(f"P-value: {result['p_value']:.6f}")
print(f"Trend: {result['trend']}")
print(f"Significant at α=0.05? {result['significant']}")
            """, language='python')
            
            st.markdown("### 7.2 Sen's Slope Estimator Implementation")
            
            st.code("""
def sens_slope(data):
    \"\"\"
    Calculates Sen's slope estimator for trend magnitude.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    
    Returns:
    --------
    dict : Dictionary containing:
        - slope: Sen's slope (median of all pairwise slopes)
        - intercept: Intercept of trend line
    \"\"\"
    n = len(data)
    slopes = []
    
    # Calculate all pairwise slopes
    for i in range(n-1):
        for j in range(i+1, n):
            slope = (data[j] - data[i]) / (j - i)
            slopes.append(slope)
    
    # Sen's slope is the median
    sens_slope_value = np.median(slopes)
    
    # Calculate intercept (median of: data[i] - slope*i)
    intercepts = [data[i] - sens_slope_value * i for i in range(n)]
    intercept = np.median(intercepts)
    
    return {
        'slope': sens_slope_value,
        'intercept': intercept
    }

# Example Usage
result_sens = sens_slope(discharge_data)

print(f"Sen's Slope: {result_sens['slope']:.4f} m³/s per year")
print(f"Intercept: {result_sens['intercept']:.2f} m³/s")

# Interpret
change_over_10_years = result_sens['slope'] * 10
print(f"\\nPredicted change over 10 years: {change_over_10_years:.2f} m³/s")
            """, language='python')
            
            st.markdown("### 7.3 Using Existing Python Packages")
            
            st.markdown("""
            **Option 1: pymannkendall Package**
            
            Comprehensive package with multiple Mann-Kendall variants.
            """)
            
            st.code("""
# Install: pip install pymannkendall
import pymannkendall as mk

# Perform Mann-Kendall test
result = mk.original_test(discharge_data)

# Result contains:
# result.trend : 'increasing', 'decreasing', 'no trend'
# result.h : True if significant at α=0.05
# result.p : P-value
# result.z : Z-score
# result.Tau : Kendall's tau
# result.s : Mann-Kendall statistic (S)
# result.var_s : Variance of S
# result.slope : Sen's slope
# result.intercept : Intercept

print(f"Trend: {result.trend}")
print(f"Significant: {result.h}")
print(f"P-value: {result.p:.6f}")
print(f"Sen's Slope: {result.slope:.4f}")
            """, language='python')
            
            st.markdown("""
            **Option 2: scipy.stats for basic calculations**
            """)
            
            st.code("""
from scipy.stats import kendalltau

# Kendall's tau correlation
time = list(range(len(discharge_data)))
tau, p_value = kendalltau(time, discharge_data)

print(f"Kendall's tau: {tau:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    if tau > 0:
        print("Significant increasing trend")
    else:
        print("Significant decreasing trend")
else:
    print("No significant trend")
            """, language='python')
            
            st.markdown("### 7.4 Visualization Template")
            
            st.code("""
import matplotlib.pyplot as plt
import numpy as np

def plot_trend_analysis(data, result_mk, result_sens):
    \"\"\"
    Visualizes time series with trend line and statistics.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    result_mk : dict
        Results from Mann-Kendall test
    result_sens : dict
        Results from Sen's slope estimator
    \"\"\"
    n = len(data)
    years = np.arange(1, n + 1)
    
    # Calculate trend line
    trend_line = result_sens['intercept'] + result_sens['slope'] * years
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    ax.plot(years, data, 'o-', color='steelblue', 
            label='Observed Data', linewidth=2, markersize=8)
    
    # Plot trend line
    ax.plot(years, trend_line, '--', color='red', 
            linewidth=2, label=f"Sen's Slope = {result_sens['slope']:.3f}/year")
    
    # Add statistics box
    stats_text = f\"\"\"
Mann-Kendall Test:
  S = {result_mk['S']}
  τ = {result_mk['tau']:.3f}
  p = {result_mk['p_value']:.4f}
  Trend: {result_mk['trend']}
\"\"\"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels
    ax.set_xlabel('Time Index (Years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Discharge (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title('Trend Analysis with Mann-Kendall and Sen\\'s Slope', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example
mk_result = mann_kendall_test(discharge_data)
sens_result = sens_slope(discharge_data)
plot_trend_analysis(discharge_data, mk_result, sens_result)
            """, language='python')

        # Section 8: Course Summary
        with st.expander("## 8. MODULE SUMMARY", expanded=False):
            st.markdown("### 8.1 Key Takeaways")
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); 
                        padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            
            **Essential Concepts Mastered:**
            
            1. **Trends in hydrology** represent systematic long-term changes that affect infrastructure design
            
            2. **Mann-Kendall test** is the standard non-parametric method for trend detection
               - Robust to outliers and non-normality
               - Based on pairwise comparisons
               - Returns p-value for significance testing
            
            3. **Sen's slope estimator** quantifies trend magnitude
               - Median of all pairwise slopes
               - Units: change per unit time
               - Essential for engineering projections
            
            4. **Engineering applications** include:
               - Updating design standards
               - Assessing climate change impacts
               - Water supply planning
               - Risk analysis updates
            
            5. **Critical considerations**:
               - Check for autocorrelation
               - Interpret both significance AND magnitude
               - Validate results with physical reasoning
               - Document methodology thoroughly
            
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 7.2 What You've Learned")
            
            skills = [
                "✅ Distinguish between parametric and non-parametric methods",
                "✅ Calculate Mann-Kendall S statistic by hand",
                "✅ Interpret p-values for trend significance",
                "✅ Calculate Sen's slope and confidence intervals",
                "✅ Make engineering decisions based on trend analysis",
                "✅ Apply trend detection to real hydrologic datasets",
                "✅ Understand when and how to update design standards"
            ]
            
            for skill in skills:
                st.markdown(skill)
            
            st.markdown("### 7.3 Practice Problems Completed")
            
            st.info("""
            Throughout this module, you've worked through:
            - **5 Multiple Choice Questions** testing conceptual understanding
            - **3 Numerical Problems** with step-by-step solutions
            - **1 Interactive Demonstration** with adjustable parameters
            - **Multiple Engineering Applications** with real-world scenarios
            
            These practice questions reinforce your understanding and prepare you for applying
            trend analysis to actual water resources engineering problems.
            """)
            
            st.markdown("### 7.4 Next Steps")
            
            st.markdown("""
            **Continue Your Learning:**
            
            → **Module 8:** Break Point Detection (Pettitt Test)
            - Learn to detect abrupt changes in hydrologic data
            - Distinguish between trends and change points
            - Apply to dam construction and watershed modification scenarios
            
            → **Module 9:** Spatiotemporal Representation
            - Create regional maps of trends
            - Analyze spatial patterns
            - Prioritize infrastructure investments
            
            **For Practice:**
            - Download hydrologic data from USGS or local agencies
            - Apply Mann-Kendall test to your local watershed
            - Calculate Sen's slope for different variables
            - Compare results with nearby stations
            """)
            
            st.success("🎉 **Module 7 Complete!** You've mastered trend detection methods in hydrologic time series.")
            
            return True
        
        # References
        with st.expander("## 📚 REFERENCES AND FURTHER READING", expanded=False):
            st.markdown("""
            **Foundational Papers:**
            
            1. Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*, 13, 245-259.
            
            2. Kendall, M. G. (1975). *Rank Correlation Methods* (4th ed.). Charles Griffin, London.
            
            3. Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. 
               *Journal of the American Statistical Association*, 63(324), 1379-1389.
            
            4. Gilbert, R. O. (1987). *Statistical Methods for Environmental Pollution Monitoring*. 
               Van Nostrand Reinhold Co., New York.
            
            **Hydrologic Applications:**
            
            5. Yue, S., Pilon, P., & Cavadias, G. (2002). Power of the Mann-Kendall and Spearman's 
               rho tests for detecting monotonic trends in hydrological series. 
               *Journal of Hydrology*, 259(1-4), 254-271.
            
            6. Burn, D. H., & Hag Elnur, M. A. (2002). Detection of hydrologic trends and variability. 
               *Journal of Hydrology*, 255(1-4), 107-122.
            
            7. Hirsch, R. M., & Slack, J. R. (1984). A nonparametric trend test for seasonal data 
               with serial dependence. *Water Resources Research*, 20(6), 727-732.
            
            **Climate Change Studies:**
            
            8. IPCC (2021). Climate Change 2021: The Physical Science Basis. 
               Contribution of Working Group I to the Sixth Assessment Report.
            
            9. Milly, P. C. D., et al. (2008). Stationarity is dead: Whither water management? 
               *Science*, 319(5863), 573-574.
            
            **Software and Tools:**
            
            10. Python `pymannkendall` package: https://github.com/mmhs013/pymannkendall
            
            11. R `trend` package: https://cran.r-project.org/package=trend
            
            12. MAKESENS software: Finnish Meteorological Institute
            """)
        
        return None


def main():
    """Standalone module test"""
    st.set_page_config(page_title="Module 9: Trend Detection", layout="wide")
    module = Module09_TrendDetection()
    module.render()


if __name__ == "__main__":
    main()

