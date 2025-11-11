"""
Module 5: GEV Analysis - Simplified for Undergraduates
Show GEV as flexible Gumbel for better extreme value analysis

Author: TA Saurav Bhattarai
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents

class Module05_GEVAnalysis(LearningModule):
    """Module 5: GEV Analysis - Simplified Version"""
    
    def __init__(self):
        objectives = [
            LearningObjective("Understand GEV as flexible Gumbel distribution", "understand"),
            LearningObjective("Recognize three GEV types by shape parameter", "understand"), 
            LearningObjective("Fit GEV to real flood data", "apply"),
            LearningObjective("Compare GEV vs Gumbel results", "analyze"),
            LearningObjective("Calculate design events using GEV", "apply")
        ]
        
        info = ModuleInfo(
            id="module_05",
            title="GEV Analysis: Better than Gumbel",
            description="Learn GEV as an improved Gumbel distribution that captures extreme events better",
            duration_minutes=45,
            prerequisites=["module_01", "module_02", "module_03", "module_04"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=10
        )
        
        super().__init__(info)
        self.flood_data = DataManager.get_flood_data()
    
    def get_slide_titles(self) -> List[str]:
        return [
            "GEV = Flexible Gumbel",
            "The Shape Parameter",
            "Three Types of GEV",
            "Interactive Shape Explorer",
            "Real Data: Gumbel vs GEV",
            "Fitting GEV Step-by-Step",
            "Design Events Comparison",
            "When to Use GEV",
            "Python Quick Demo",
            "Practice Quiz"
        ]
    
    def render_slide(self, slide_num: int) -> Optional[bool]:
        slides = [
            self._slide_gev_concept,
            self._slide_shape_parameter,
            self._slide_three_types,
            self._slide_interactive_explorer,
            self._slide_real_data_comparison,
            self._slide_fitting_steps,
            self._slide_design_comparison,
            self._slide_when_to_use,
            self._slide_python_demo,
            self._slide_quiz
        ]
        
        if slide_num < len(slides):
            return slides[slide_num]()
        return False
    
    def _slide_gev_concept(self) -> Optional[bool]:
        """Slide 1: GEV = Flexible Gumbel"""
        with UIComponents.slide_container("theory"):
            st.markdown("## GEV = Gumbel + Flexibility")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### What You Already Know:")
                UIComponents.highlight_box("""
                **Gumbel Distribution:**
                - Good for most flood analysis
                - Fixed shape (always same curve type)
                - Sometimes misses extreme patterns
                """)
                
                UIComponents.big_number_display("Fixed Shape", "Gumbel Limitation")
                
            with col2:
                st.markdown("### The GEV Improvement:")
                UIComponents.highlight_box("""
                **GEV Distribution:**
                - Gumbel + shape adjustment
                - Can bend curve to fit data better
                - Captures different extreme patterns
                """)
                
                UIComponents.big_number_display("Flexible Shape", "GEV Advantage")
            
            # Simple visual comparison
            x = np.linspace(0, 10, 100)
            
            fig = go.Figure()
            
            # Gumbel (fixed)
            gumbel_pdf = stats.gumbel_r.pdf(x, loc=4, scale=1)
            fig.add_trace(go.Scatter(x=x, y=gumbel_pdf, mode='lines', 
                                   name='Gumbel (Fixed)', line=dict(color='blue', width=3)))
            
            # GEV with different shapes
            for xi, name, color in [(0.2, 'GEV Heavy Tail', 'red'), (-0.2, 'GEV Light Tail', 'green')]:
                gev_pdf = stats.genextreme.pdf(x, -xi, loc=4, scale=1)
                fig.add_trace(go.Scatter(x=x, y=gev_pdf, mode='lines',
                                       name=name, line=dict(color=color, width=3, dash='dash')))
            
            fig.update_layout(title="Gumbel vs GEV Flexibility", height=400)
            fig = PlotTools.apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            UIComponents.highlight_box("**Key Idea**: GEV can adjust its shape to match your specific data!")
        
        return None
    
    def _slide_shape_parameter(self) -> Optional[bool]:
        """Slide 2: The Shape Parameter"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## The Magic Shape Parameter (ξ)")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### One Parameter Controls Everything:")
                
                shape_examples = [
                    ("ξ > 0", "🔴 Heavy Tail", "More extreme events possible"),
                    ("ξ = 0", "🔵 Medium Tail", "Standard Gumbel behavior"),
                    ("ξ < 0", "🟢 Light Tail", "Extreme events are limited")
                ]
                
                for shape_range, description, meaning in shape_examples:
                    UIComponents.highlight_box(f"""
                    **{shape_range}**: {description}
                    {meaning}
                    """)
                
            with col2:
                # Interactive shape parameter demo
                shape_demo = st.slider("Shape Parameter (ξ):", -0.4, 0.4, 0.0, 0.1, key="shape_demo")
                
                x = np.linspace(0, 15, 200)
                try:
                    pdf_vals = stats.genextreme.pdf(x, -shape_demo, loc=5, scale=1.5)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=pdf_vals, mode='lines', fill='tonexty',
                                           name=f'ξ = {shape_demo}'))
                    
                    # Add reference Gumbel
                    gumbel_pdf = stats.gumbel_r.pdf(x, loc=5, scale=1.5)
                    fig.add_trace(go.Scatter(x=x, y=gumbel_pdf, mode='lines',
                                           name='Gumbel (ξ=0)', line=dict(dash='dash', color='gray')))
                    
                    fig.update_layout(title=f"Shape = {shape_demo:.1f}", height=350)
                    fig = PlotTools.apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Live interpretation
                    if shape_demo > 0.1:
                        st.success("🔴 **Heavy Tail**: More extreme floods possible!")
                    elif shape_demo < -0.1:
                        st.info("🟢 **Light Tail**: Extreme floods are limited")
                    else:
                        st.warning("🔵 **Gumbel**: Standard flood behavior")
                        
                except:
                    st.error("Shape parameter out of valid range")
        
        return None
    
    def _slide_three_types(self) -> Optional[bool]:
        """Slide 3: Three Types of GEV"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Three Flavors of GEV")
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            x = np.linspace(0, 12, 200)
            
            with col1:
                st.markdown("### 🟢 Type III: Weibull")
                st.markdown("**ξ < 0** (Shape negative)")
                
                weibull_pdf = stats.genextreme.pdf(x, 0.3, loc=4, scale=1.5)  # xi = -0.3
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=x, y=weibull_pdf, mode='lines', fill='tonexty',
                                        line=dict(color='green')))
                fig1.update_layout(title="Light Tail", height=250, showlegend=False)
                fig1 = PlotTools.apply_theme(fig1)
                st.plotly_chart(fig1, use_container_width=True)
                
                UIComponents.highlight_box("""
                **Characteristics:**
                - Has upper limit
                - Extremes can't exceed certain value
                - Good for bounded phenomena
                """)
                
            with col2:
                st.markdown("### 🔵 Type I: Gumbel")
                st.markdown("**ξ = 0** (No shape parameter)")
                
                gumbel_pdf = stats.gumbel_r.pdf(x, loc=4, scale=1.5)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x, y=gumbel_pdf, mode='lines', fill='tonexty',
                                        line=dict(color='blue')))
                fig2.update_layout(title="Medium Tail", height=250, showlegend=False)
                fig2 = PlotTools.apply_theme(fig2)
                st.plotly_chart(fig2, use_container_width=True)
                
                UIComponents.highlight_box("""
                **Characteristics:**
                - Standard extreme behavior
                - Most common in hydrology
                - Your familiar Gumbel!
                """)
                
            with col3:
                st.markdown("### 🔴 Type II: Fréchet")
                st.markdown("**ξ > 0** (Shape positive)")
                
                frechet_pdf = stats.genextreme.pdf(x, -0.3, loc=4, scale=1.5)  # xi = +0.3
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=x, y=frechet_pdf, mode='lines', fill='tonexty',
                                        line=dict(color='red')))
                fig3.update_layout(title="Heavy Tail", height=250, showlegend=False)
                fig3 = PlotTools.apply_theme(fig3)
                st.plotly_chart(fig3, use_container_width=True)
                
                UIComponents.highlight_box("""
                **Characteristics:**
                - Long heavy tail
                - Very extreme events possible
                - Be careful with extrapolation!
                """)
            
            UIComponents.highlight_box("**The shape parameter (ξ) determines which type you get!**")
        
        return None
    
    def _slide_interactive_explorer(self) -> Optional[bool]:
        """Slide 4: Interactive Shape Explorer"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Explore How Shape Changes Everything")
            
            # Interactive controls
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                shape_value = st.slider("Adjust Shape Parameter (ξ):", -0.5, 0.5, 0.0, 0.05, key="gev_explorer")
                
                # Show current type
                if shape_value > 0.1:
                    current_type = "🔴 Fréchet (Heavy Tail)"
                    behavior = "Extreme events more likely"
                    design_advice = "Use conservative design!"
                elif shape_value < -0.1:
                    current_type = "🟢 Weibull (Light Tail)"
                    behavior = "Extreme events limited"
                    design_advice = "Standard design OK"
                else:
                    current_type = "🔵 Gumbel (Medium Tail)"
                    behavior = "Standard extreme behavior"
                    design_advice = "Normal design procedures"
                
                UIComponents.big_number_display(f"{shape_value:.2f}", "Current ξ")
                st.markdown(f"**Type**: {current_type}")
                st.markdown(f"**Behavior**: {behavior}")
                st.markdown(f"**Design**: {design_advice}")
                
                # Show parameters
                st.markdown("### Fixed Parameters:")
                st.markdown("- Location (μ) = 300 m³/s")
                st.markdown("- Scale (σ) = 100 m³/s")
                st.markdown(f"- **Shape (ξ) = {shape_value:.2f}** ← You control this!")
                
            with col2:
                # Live visualization
                x_range = np.linspace(50, 800, 300)
                
                try:
                    pdf_current = stats.genextreme.pdf(x_range, -shape_value, loc=300, scale=100)
                    
                    fig = go.Figure()
                    
                    # Current GEV
                    fig.add_trace(go.Scatter(x=x_range, y=pdf_current, mode='lines', fill='tonexty',
                                           name=f'GEV (ξ={shape_value:.2f})', 
                                           line=dict(width=4)))
                    
                    # Reference Gumbel
                    gumbel_pdf = stats.gumbel_r.pdf(x_range, loc=300, scale=100)
                    fig.add_trace(go.Scatter(x=x_range, y=gumbel_pdf, mode='lines',
                                           name='Gumbel (ξ=0)', 
                                           line=dict(color='gray', dash='dash', width=3)))
                    
                    fig.update_layout(
                        title=f"GEV Distribution (ξ = {shape_value:.2f})",
                        xaxis_title="Flow (m³/s)",
                        yaxis_title="Probability Density",
                        height=400
                    )
                    fig = PlotTools.apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except:
                    st.error("Invalid parameter range")
            
            # Quick comparison table
            st.markdown("### Quick Reference:")
            
            comparison_data = {
                'Shape (ξ)': ['< -0.1', '≈ 0', '> +0.1'],
                'Type': ['Weibull', 'Gumbel', 'Fréchet'],
                'Tail': ['Light', 'Medium', 'Heavy'],
                'Extremes': ['Limited', 'Standard', 'Very Large'],
                'Design Approach': ['Standard', 'Standard', 'Conservative']
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        
        return None
    
    def _slide_real_data_comparison(self) -> Optional[bool]:
        """Slide 5: Real Data - Gumbel vs GEV"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Real Flood Data: Gumbel vs GEV")
            
            # Use real flood data
            flood_values = self.flood_data['Peak_Flow'].values
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Fit Both Distributions")
                
                # Fit Gumbel
                gumbel_params = stats.gumbel_r.fit(flood_values)
                gumbel_loc, gumbel_scale = gumbel_params
                
                # Fit GEV
                gev_params = stats.genextreme.fit(flood_values)
                gev_shape, gev_loc, gev_scale = gev_params
                
                UIComponents.highlight_box(f"""
                **Gumbel Results:**
                - Location: {gumbel_loc:.1f} m³/s
                - Scale: {gumbel_scale:.1f} m³/s
                - Shape: 0 (fixed)
                """)
                
                UIComponents.highlight_box(f"""
                **GEV Results:**
                - Location: {gev_loc:.1f} m³/s
                - Scale: {gev_scale:.1f} m³/s
                - Shape: {-gev_shape:.3f}
                """)
                
                # Determine GEV type
                if -gev_shape > 0.05:
                    gev_type = "🔴 Fréchet (Heavy tail)"
                elif -gev_shape < -0.05:
                    gev_type = "🟢 Weibull (Light tail)"
                else:
                    gev_type = "🔵 Gumbel-like (Medium tail)"
                
                st.markdown(f"**GEV Type**: {gev_type}")
                
            with col2:
                st.markdown("### Visual Comparison")
                
                # Plot both distributions with data
                fig = go.Figure()
                
                # Histogram of data
                fig.add_trace(go.Histogram(x=flood_values, nbinsx=12, histnorm='probability density',
                                         name='Observed Data', opacity=0.6, marker_color='lightblue'))
                
                # Fitted distributions
                x_range = np.linspace(flood_values.min()*0.8, flood_values.max()*1.2, 200)
                
                gumbel_pdf = stats.gumbel_r.pdf(x_range, *gumbel_params)
                gev_pdf = stats.genextreme.pdf(x_range, *gev_params)
                
                fig.add_trace(go.Scatter(x=x_range, y=gumbel_pdf, mode='lines',
                                       name='Gumbel Fit', line=dict(color='blue', width=3)))
                
                fig.add_trace(go.Scatter(x=x_range, y=gev_pdf, mode='lines',
                                       name='GEV Fit', line=dict(color='red', width=3)))
                
                fig.update_layout(title="Which Fits Better?", height=400)
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            # Simple comparison metrics
            st.markdown("### Which Model is Better?")
            
            # Calculate simple fit measures
            gumbel_ll = np.sum(stats.gumbel_r.logpdf(flood_values, *gumbel_params))
            gev_ll = np.sum(stats.genextreme.logpdf(flood_values, *gev_params))
            
            gumbel_aic = 2*2 - 2*gumbel_ll  # 2 parameters
            gev_aic = 2*3 - 2*gev_ll        # 3 parameters
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                UIComponents.big_number_display(f"{gumbel_aic:.1f}", "Gumbel AIC")
                st.markdown("(Lower is better)")
                
            with col2:
                UIComponents.big_number_display(f"{gev_aic:.1f}", "GEV AIC")
                st.markdown("(Lower is better)")
            
            winner = "GEV" if gev_aic < gumbel_aic else "Gumbel"
            improvement = abs(gev_aic - gumbel_aic)
            
            if winner == "GEV" and improvement > 2:
                st.success(f"🏆 **GEV wins!** Much better fit than Gumbel")
            elif winner == "GEV":
                st.success(f"🏆 **GEV wins!** Slightly better than Gumbel")
            else:
                st.info("📊 **Gumbel adequate** - GEV doesn't add much value")
        
        return None
    
    def _slide_fitting_steps(self) -> Optional[bool]:
        """Slide 6: Fitting GEV Step-by-Step"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Fitting GEV: Simple Steps")
            
            # Show the process step by step
            steps = [
                "1. Load Your Data",
                "2. Fit GEV Distribution", 
                "3. Check the Fit",
                "4. Use for Design"
            ]
            
            selected_step = st.selectbox("Select Step:", steps, key="gev_fitting_steps")
            
            flood_values = self.flood_data['Peak_Flow'].values
            
            if "1. Load" in selected_step:
                col1, col2 = UIComponents.two_column_layout()
                
                with col1:
                    st.markdown("### Your Data:")
                    display_data = self.flood_data[['Year', 'Peak_Flow']].head(8)
                    st.dataframe(display_data, use_container_width=True)
                    
                    UIComponents.big_number_display(f"{len(flood_values)}", "Years of Data")
                    UIComponents.big_number_display(f"{flood_values.max():.0f}", "Largest Flood")
                    
                with col2:
                    st.markdown("### Requirements Check:")
                    requirements = [
                        "✅ Annual maximum values",
                        "✅ At least 20 years of data",
                        "✅ No missing values",
                        "✅ Independent observations"
                    ]
                    
                    for req in requirements:
                        st.markdown(req)
                    
                    UIComponents.highlight_box("**Ready for GEV analysis!**")
            
            elif "2. Fit GEV" in selected_step:
                st.markdown("### One Line of Code!")
                
                st.code("""
# In Python, fitting GEV is super easy:
import scipy.stats as stats

gev_params = stats.genextreme.fit(flood_data)
shape, location, scale = gev_params

print(f"Shape: {-shape:.3f}")      # Convert to hydrology convention
print(f"Location: {location:.1f}")
print(f"Scale: {scale:.1f}")
                """, language='python')
                
                # Show actual results
                gev_params = stats.genextreme.fit(flood_values)
                shape_fit, loc_fit, scale_fit = gev_params
                
                col1, col2 = UIComponents.two_column_layout()
                
                with col1:
                    UIComponents.big_number_display(f"{-shape_fit:.4f}", "Shape (ξ)")
                    UIComponents.big_number_display(f"{loc_fit:.1f}", "Location (μ)")
                    UIComponents.big_number_display(f"{scale_fit:.1f}", "Scale (σ)")
                    
                with col2:
                    # Interpretation
                    if -shape_fit > 0.05:
                        interpretation = "🔴 Fréchet: Heavy tail detected!"
                    elif -shape_fit < -0.05:
                        interpretation = "🟢 Weibull: Light tail detected!"
                    else:
                        interpretation = "🔵 Gumbel-like: Standard behavior"
                    
                    UIComponents.highlight_box(f"**Result**: {interpretation}")
            
            elif "3. Check" in selected_step:
                # Quick visual check
                gev_params = stats.genextreme.fit(flood_values)
                
                fig = go.Figure()
                
                # Data histogram
                fig.add_trace(go.Histogram(x=flood_values, nbinsx=12, histnorm='probability density',
                                         name='Your Data', opacity=0.7))
                
                # GEV fit
                x_range = np.linspace(flood_values.min()*0.8, flood_values.max()*1.2, 200)
                gev_pdf = stats.genextreme.pdf(x_range, *gev_params)
                fig.add_trace(go.Scatter(x=x_range, y=gev_pdf, mode='lines',
                                       name='GEV Fit', line=dict(color='red', width=4)))
                
                fig.update_layout(title="Does GEV Fit Your Data?", height=400)
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                # Simple check
                ks_stat, ks_p = stats.kstest(flood_values, lambda x: stats.genextreme.cdf(x, *gev_params))
                
                if ks_p > 0.05:
                    st.success("✅ **Good fit!** GEV captures your data well")
                else:
                    st.warning("⚠️ **Questionable fit** - consider other options")
                
                st.markdown(f"**Technical note**: p-value = {ks_p:.4f}")
            
            elif "4. Use for Design" in selected_step:
                gev_params = stats.genextreme.fit(flood_values)
                
                st.markdown("### Calculate Design Events")
                
                # Key design events
                design_events = [10, 25, 50, 100]
                
                col1, col2 = UIComponents.two_column_layout()
                
                with col1:
                    for T in design_events:
                        design_value = stats.genextreme.ppf(1-1/T, *gev_params)
                        st.markdown(f"**{T}-year event**: {design_value:.0f} m³/s")
                        
                with col2:
                    # Visual design events
                    fig = go.Figure()
                    
                    # GEV curve
                    T_range = np.logspace(0.3, 2.5, 100)
                    prob_range = 1 - 1/T_range
                    gev_values = stats.genextreme.ppf(prob_range, *gev_params)
                    
                    fig.add_trace(go.Scatter(x=T_range, y=gev_values, mode='lines',
                                           name='GEV', line=dict(color='red', width=3)))
                    
                    # Mark design events
                    for T in design_events:
                        Q = stats.genextreme.ppf(1-1/T, *gev_params)
                        fig.add_trace(go.Scatter(x=[T], y=[Q], mode='markers+text',
                                               text=f'{T}yr', textposition='top center',
                                               marker=dict(size=12, color='blue'),
                                               showlegend=False))
                    
                    fig.update_layout(title="Design Events", height=300, xaxis_type='log')
                    fig = PlotTools.apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_design_comparison(self) -> Optional[bool]:
        """Slide 7: Design Events Comparison"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Gumbel vs GEV: Design Event Comparison")
            
            flood_values = self.flood_data['Peak_Flow'].values
            
            # Fit both distributions
            gumbel_params = stats.gumbel_r.fit(flood_values)
            gev_params = stats.genextreme.fit(flood_values)
            
            # Calculate design events for both
            return_periods = [5, 10, 25, 50, 100, 200]
            
            comparison_data = []
            for T in return_periods:
                prob = 1 - 1/T
                gumbel_value = stats.gumbel_r.ppf(prob, *gumbel_params)
                gev_value = stats.genextreme.ppf(prob, *gev_params)
                difference = gev_value - gumbel_value
                percent_diff = (difference / gumbel_value) * 100
                
                comparison_data.append({
                    'Return Period': f'{T} years',
                    'Gumbel': f'{gumbel_value:.0f}',
                    'GEV': f'{gev_value:.0f}',
                    'Difference': f'{difference:+.0f}',
                    '% Diff': f'{percent_diff:+.1f}%'
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Design Event Comparison")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Highlight differences
                shape_param = -gev_params[0]
                if abs(shape_param) > 0.1:
                    if shape_param > 0:
                        st.warning("🔴 **GEV gives LARGER values** - more conservative design needed!")
                    else:
                        st.info("🟢 **GEV gives SMALLER values** - Gumbel was too conservative")
                else:
                    st.success("🔵 **GEV ≈ Gumbel** - Both give similar results")
                
            with col2:
                st.markdown("### Visual Comparison")
                
                # Plot comparison
                fig = go.Figure()
                
                T_range = np.logspace(0.5, 2.5, 50)
                prob_range = 1 - 1/T_range
                
                gumbel_values = stats.gumbel_r.ppf(prob_range, *gumbel_params)
                gev_values = stats.genextreme.ppf(prob_range, *gev_params)
                
                fig.add_trace(go.Scatter(x=T_range, y=gumbel_values, mode='lines',
                                       name='Gumbel', line=dict(color='blue', width=3)))
                
                fig.add_trace(go.Scatter(x=T_range, y=gev_values, mode='lines',
                                       name='GEV', line=dict(color='red', width=3)))
                
                # Mark common design events
                for T in [10, 50, 100]:
                    gumbel_val = stats.gumbel_r.ppf(1-1/T, *gumbel_params)
                    gev_val = stats.genextreme.ppf(1-1/T, *gev_params)
                    
                    fig.add_trace(go.Scatter(x=[T, T], y=[gumbel_val, gev_val], 
                                           mode='markers+lines',
                                           line=dict(color='gray', dash='dot'),
                                           marker=dict(size=8),
                                           showlegend=False))
                
                fig.update_layout(title="Design Event Comparison", height=400, xaxis_type='log')
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            # Engineering impact
            max_diff = max([abs(float(row['% Diff'][:-1])) for row in comparison_data])
            
            if max_diff > 10:
                UIComponents.highlight_box(f"**Engineering Impact**: Differences up to {max_diff:.0f}% - GEV analysis worthwhile!")
            else:
                UIComponents.highlight_box(f"**Engineering Impact**: Small differences (<{max_diff:.0f}%) - Either method OK")
        
        return None
    
    def _slide_when_to_use(self) -> Optional[bool]:
        """Slide 8: When to Use GEV"""
        with UIComponents.slide_container("theory"):
            st.markdown("## When Should You Use GEV?")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### ✅ Use GEV When:")
                use_gev = [
                    "🎯 **Critical infrastructure** (dams, hospitals)",
                    "📊 **Data looks non-Gumbel** (very skewed)",
                    "⏰ **Long return periods** (100+ years)",
                    "🔍 **Detailed analysis** required",
                    "📋 **Professional standards** require it",
                    "💰 **High economic stakes**"
                ]
                
                for item in use_gev:
                    st.markdown(item)
                
                UIComponents.highlight_box("""
                **Rule of thumb**: If consequences of underestimation are severe, use GEV!
                """)
                
            with col2:
                st.markdown("### 📊 Gumbel Still OK When:")
                use_gumbel = [
                    "🏠 **Standard infrastructure** (small culverts)",
                    "📈 **Data fits Gumbel well**",
                    "⚡ **Quick preliminary** analysis",
                    "📚 **Educational purposes**",
                    "🔄 **Historical consistency** needed",
                    "⏰ **Time constraints**"
                ]
                
                for item in use_gumbel:
                    st.markdown(item)
                
                UIComponents.highlight_box("""
                **Rule of thumb**: For routine design, Gumbel is often sufficient
                """)
            
            # Decision flowchart
            st.markdown("### 🤔 Decision Process:")
            
            decision_flow = [
                "1. **What are you designing?** (Critical vs standard)",
                "2. **What return period?** (>100 years suggests GEV)",
                "3. **How does data look?** (Very skewed suggests GEV)",
                "4. **What do regulations say?** (Some require specific methods)",
                "5. **How much time do you have?** (GEV takes more effort)"
            ]
            
            for step in decision_flow:
                st.markdown(step)
            
            UIComponents.highlight_box("""
            **Bottom Line**: GEV is more accurate, Gumbel is simpler. 
            Choose based on your project needs!
            """)
        
        return None
    
    def _slide_python_demo(self) -> Optional[bool]:
        """Slide 9: Python Quick Demo"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Quick Python Demo")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Complete Analysis in 10 Lines!")
                
                demo_code = '''
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load data (your flood data here)
flows = [334, 203, 567, 285, 349, ...]  

# 2. Fit GEV distribution
gev_params = stats.genextreme.fit(flows)
shape, location, scale = gev_params

# 3. Calculate 100-year design event
design_100 = stats.genextreme.ppf(0.99, *gev_params)
print(f"100-year flood: {design_100:.0f} m³/s")

# 4. That's it! ✅
'''
                
                st.code(demo_code, language='python')
                
                st.markdown("### Try It Yourself:")
                
                st.link_button(
                    "🚀 Open Google Colab Tutorial",
                    "https://colab.research.google.com/drive/1GEVAnalysisComplete",
                    use_container_width=True
                )
                
            with col2:
                st.markdown("### Live Demo with Real Data")
                
                # Actual calculation with our data
                flood_values = self.flood_data['Peak_Flow'].values
                gev_params = stats.genextreme.fit(flood_values)
                
                st.markdown("**Our Flood Data Results:**")
                
                key_events = [10, 50, 100]
                for T in key_events:
                    design_val = stats.genextreme.ppf(1-1/T, *gev_params)
                    st.markdown(f"• **{T}-year event**: {design_val:.0f} m³/s")
                
                # Quick visualization
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(x=flood_values, nbinsx=12, histnorm='probability density',
                                         name='Data', opacity=0.7))
                
                x_range = np.linspace(flood_values.min()*0.8, flood_values.max()*1.2, 100)
                gev_pdf = stats.genextreme.pdf(x_range, *gev_params)
                
                fig.add_trace(go.Scatter(x=x_range, y=gev_pdf, mode='lines',
                                       name='GEV Fit', line=dict(color='red', width=3)))
                
                fig.update_layout(title="Our GEV Analysis", height=300)
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                shape_result = -gev_params[0]
                if shape_result > 0.05:
                    st.error("🔴 Heavy tail - be conservative!")
                elif shape_result < -0.05:
                    st.success("🟢 Light tail - bounded extremes")
                else:
                    st.info("🔵 Standard tail - like Gumbel")
        
        return None
    
    def _slide_quiz(self) -> Optional[bool]:
        """Slide 10: Practice Quiz"""
        with UIComponents.slide_container():
            st.markdown("## Quick Understanding Check")
            
            result = QuizEngine.create_multiple_choice(
                "You fit GEV to flood data and get shape parameter ξ = +0.15. What does this mean for engineering design?",
                [
                    "It's the same as Gumbel, so use standard design procedures",
                    "Heavy tail detected - extreme floods more likely, use conservative design",
                    "Light tail detected - extreme floods are limited, can use smaller design values", 
                    "The parameter is not important, just use the location and scale"
                ],
                1,
                {
                    "correct": "Perfect! ξ = +0.15 > 0 means Fréchet type with heavy tail. This suggests higher probability of very extreme events, so you should be more conservative in your design and consider larger safety factors.",
                    "incorrect": "Remember: ξ > 0 = Heavy tail (Fréchet) = More extreme events possible = Be more conservative in design. The shape parameter is the key difference between GEV and Gumbel!"
                },
                f"{self.info.id}_final_quiz"
            )
            
            if result is True:
                st.success("🎉 Module 5 Complete! You understand GEV as flexible Gumbel!")
                
                # Show what they learned
                UIComponents.highlight_box("""
                **🎯 What You Now Know:**
                - GEV is Gumbel with adjustable shape
                - Shape parameter ξ controls tail behavior
                - ξ > 0: Heavy tail (more extremes)
                - ξ = 0: Gumbel (standard)  
                - ξ < 0: Light tail (limited extremes)
                - Use GEV when you need more accuracy for extreme events
                """)
                
                return True
            
            return None