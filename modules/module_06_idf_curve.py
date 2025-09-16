"""
Module 6: IDF Curves - Intensity Duration Frequency Analysis
Practical approach with calculations and Excel integration

Author: TA Saurav Bhattarai
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from typing import List, Optional, Dict, Tuple

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents, ExcelExporter

class Module06_IDFCurve(LearningModule):
    """Module 6: IDF Curve Analysis"""
    
    def __init__(self):
        objectives = [
            LearningObjective("Understand what Intensity, Duration, and Frequency mean", "understand"),
            LearningObjective("Calculate rainfall intensities from depth data", "apply"),
            LearningObjective("Apply disaggregation ratios to derive short-duration data", "apply"),
            LearningObjective("Create IDF curves using frequency analysis", "apply"),
            LearningObjective("Use IDF curves for engineering design", "apply")
        ]
        
        info = ModuleInfo(
            id="module_06",
            title="IDF Curves: Intensity-Duration-Frequency Analysis",
            description="Learn to create and use IDF curves for rainfall analysis and design",
            duration_minutes=45,
            prerequisites=["module_01", "module_02", "module_03"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=9
        )
        
        super().__init__(info)
        self.base_data = self._generate_base_rainfall_data()
        
    def _generate_base_rainfall_data(self) -> pd.DataFrame:
        """Generate realistic 30-minute annual maximum rainfall data"""
        np.random.seed(2024)
        years = list(range(1990, 2024))  # 34 years of data
        
        # Generate 30-minute rainfall depths (mm) - realistic for urban area
        base_mean = 25  # mm in 30 minutes
        rainfall_30min = []
        
        for year in years:
            # Add some variability and occasional extreme events
            if year in [1998, 2005, 2012, 2019]:  # extreme years
                value = np.random.gamma(shape=4, scale=base_mean/2) * 1.8
            else:
                value = np.random.gamma(shape=3, scale=base_mean/3)
            rainfall_30min.append(max(8, min(80, value)))  # Keep realistic bounds
        
        return pd.DataFrame({
            'Year': years,
            'Rainfall_30min': np.round(rainfall_30min, 1)
        })
    
    def get_slide_titles(self) -> List[str]:
        return [
            "What are IDF Curves?",
            "I-D-F: Breaking it Down",
            "Intensity vs Depth",
            "NOAA Temporal Scaling Method",
            "NOAA Process: Step-by-Step",
            "Frequency Analysis",
            "Creating IDF Curves",
            "Excel Workshop Prep",
            "Practice Problem"
        ]
    
    def render_slide(self, slide_num: int) -> Optional[bool]:
        slides = [
            self._slide_what_are_idf,
            self._slide_idf_breakdown,
            self._slide_intensity_vs_depth,
            self._slide_magic_ratios,
            self._slide_step_calculations,
            self._slide_frequency_analysis,
            self._slide_creating_curves,
            self._slide_excel_prep,
            self._slide_practice_problem
        ]
        
        if slide_num < len(slides):
            return slides[slide_num]()
        return False
    
    def _slide_what_are_idf(self) -> Optional[bool]:
        """Slide 1: What are IDF Curves?"""
        with UIComponents.slide_container("theory"):
            st.markdown("## IDF Curves: The Engineer's Weather Forecast")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### 🌧️ Simple Analogy")
                UIComponents.highlight_box("""
                **Think of IDF curves like a weather menu:**
                
                🕐 **Duration** = How long does it rain?  
                ⛈️ **Intensity** = How hard does it rain?  
                📅 **Frequency** = How often does this happen?
                
                **The Question:** For a 10-year storm lasting 15 minutes, 
                how intense will the rainfall be?
                """)
                
            with col2:
                # Simple conceptual diagram
                st.markdown("### 📊 IDF Curve Preview")
                
                # Mock IDF curves for demonstration
                durations = np.array([5, 10, 15, 30, 60, 120])  # minutes
                intensities_2yr = np.array([120, 100, 85, 60, 40, 25])  # mm/hr
                intensities_10yr = intensities_2yr * 1.5
                intensities_50yr = intensities_2yr * 2.0
                
                fig = go.Figure()
                
                for T, intensities, color in [(2, intensities_2yr, 'blue'), 
                                            (10, intensities_10yr, 'orange'),
                                            (50, intensities_50yr, 'red')]:
                    fig.add_trace(go.Scatter(
                        x=durations, y=intensities,
                        mode='lines+markers',
                        name=f'{T}-year',
                        line=dict(width=3, color=color)
                    ))
                
                fig.update_layout(
                    title="Sample IDF Curves",
                    xaxis_title="Duration (minutes)",
                    yaxis_title="Intensity (mm/hr)",
                    height=350,
                    xaxis_type='log'
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🎯 Why Engineers Need IDF Curves")
            
            applications = [
                "🏗️ **Storm drain design** - Size pipes for 10-year storms",
                "🌊 **Flood prediction** - How much runoff to expect", 
                "🏘️ **Urban planning** - Development impact assessment",
                "⚡ **Infrastructure protection** - Critical facility design"
            ]
            
            for app in applications:
                st.markdown(app)
        
        return None
    
    def _slide_idf_breakdown(self) -> Optional[bool]:
        """Slide 2: I-D-F Breaking it Down"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Understanding I-D-F Components")
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            with col1:
                st.markdown("### ⛈️ INTENSITY (I)")
                UIComponents.highlight_box("""
                **Rate of rainfall**  
                Units: mm/hr or in/hr
                
                **Examples:**
                - Light rain: 2-5 mm/hr
                - Heavy rain: 20-50 mm/hr  
                - Extreme: 100+ mm/hr
                """)
                
                # Interactive intensity calculator
                depth = st.slider("Rainfall Depth (mm):", 1, 50, 15, key="intensity_calc_depth")
                duration_min = st.slider("Duration (minutes):", 5, 120, 30, key="intensity_calc_duration")
                
                intensity = (depth / duration_min) * 60  # Convert to mm/hr
                UIComponents.big_number_display(f"{intensity:.1f}", "mm/hr")
                
            with col2:
                st.markdown("### 🕐 DURATION (D)")
                UIComponents.highlight_box("""
                **How long it rains**  
                Units: minutes or hours
                
                **Typical Values:**
                - Short: 5-15 minutes
                - Medium: 30-60 minutes
                - Long: 2-6 hours
                """)
                
                # Duration impact demonstration
                fixed_depth = 20  # mm
                durations = [5, 15, 30, 60, 120]
                intensities = [(fixed_depth/d)*60 for d in durations]
                
                duration_data = pd.DataFrame({
                    'Duration (min)': durations,
                    'Intensity (mm/hr)': [f"{i:.1f}" for i in intensities]
                })
                
                st.markdown("**Same 20mm depth:**")
                st.dataframe(duration_data, use_container_width=True)
                
            with col3:
                st.markdown("### 📅 FREQUENCY (F)")
                UIComponents.highlight_box("""
                **How often it occurs**  
                Units: Return period (years)
                
                **Design Standards:**
                - Residential: 2-10 years
                - Commercial: 10-25 years
                - Critical: 50-100 years
                """)
                
                # Frequency relationship
                frequencies = [2, 5, 10, 25, 50, 100]
                base_intensity = 40  # mm/hr for 2-year
                freq_intensities = [base_intensity * (1 + 0.3*np.log(f/2)) for f in frequencies]
                
                freq_data = pd.DataFrame({
                    'Return Period': [f"{f}-year" for f in frequencies],
                    'Intensity (mm/hr)': [f"{i:.0f}" for i in freq_intensities]
                })
                
                st.markdown("**30-min duration:**")
                st.dataframe(freq_data, use_container_width=True)
            
            st.markdown("---")
            UIComponents.formula_display("Intensity = Rainfall Depth / Duration", "Basic IDF Relationship")
        
        return None
    
    def _slide_intensity_vs_depth(self) -> Optional[bool]:
        """Slide 3: Intensity vs Depth"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Intensity vs Depth: The Key Conversion")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### 🧮 The Mathematics")
                
                UIComponents.formula_display("I = P / t", "Where:")
                st.markdown("""
                - **I** = Intensity (mm/hr)
                - **P** = Rainfall depth (mm)  
                - **t** = Duration (hours)
                """)
                
                st.markdown("### 🔄 Unit Conversions")
                UIComponents.highlight_box("""
                **Duration in minutes?**
                I = (P / t_min) × 60
                
                **Example:** 
                - P = 25 mm in 30 minutes
                - I = (25/30) × 60 = 50 mm/hr
                """)
                
                # Interactive converter
                st.markdown("### 🧪 Try It Yourself")
                user_depth = st.number_input("Rainfall depth (mm):", 1.0, 100.0, 20.0, 0.5)
                user_duration = st.number_input("Duration (minutes):", 1, 300, 45, 5)
                
                calculated_intensity = (user_depth / user_duration) * 60
                
                st.markdown(f"**Result:** {calculated_intensity:.2f} mm/hr")
                
                # Classification
                if calculated_intensity < 2.5:
                    rain_type = "🌦️ Light Rain"
                elif calculated_intensity < 10:
                    rain_type = "🌧️ Moderate Rain"
                elif calculated_intensity < 50:
                    rain_type = "⛈️ Heavy Rain"
                else:
                    rain_type = "🌪️ Extreme Rain"
                    
                st.markdown(f"**Classification:** {rain_type}")
                
            with col2:
                st.markdown("### 📈 Visualization")
                
                # Create intensity vs duration plot for fixed depth
                fixed_depth = user_depth
                duration_range = np.linspace(5, 120, 50)
                intensity_curve = (fixed_depth / duration_range) * 60
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=duration_range,
                    y=intensity_curve,
                    mode='lines',
                    name=f'{fixed_depth}mm total',
                    line=dict(width=3, color='blue')
                ))
                
                # Mark user's point
                fig.add_trace(go.Scatter(
                    x=[user_duration],
                    y=[calculated_intensity],
                    mode='markers',
                    name='Your calculation',
                    marker=dict(size=12, color='red', symbol='star')
                ))
                
                fig.update_layout(
                    title=f"Intensity vs Duration for {fixed_depth}mm total rainfall",
                    xaxis_title="Duration (minutes)",
                    yaxis_title="Intensity (mm/hr)",
                    height=350
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 💡 Key Insight")
                UIComponents.highlight_box("""
                **Shorter durations = Higher intensities**
                
                Same amount of rain in less time means more intense rainfall!
                """)
        
        return None
    
    def _slide_magic_ratios(self) -> Optional[bool]:
        """Slide 4: NOAA Temporal Scaling Method"""
        with UIComponents.slide_container("theory"):
            st.markdown("## NOAA's Solution: Temporal Scaling Method")

            st.markdown("### 🎯 NOAA's Problem: Limited 5-Minute Data")

            UIComponents.highlight_box("""
            **NOAA Atlas 14 Volume 2 Challenge:**

            📊 **Only 96 stations** had 5-minute precipitation data
            📊 **994 stations** had hourly precipitation data
            🎯 **Goal:** Provide 5-minute frequency estimates everywhere

            **Solution:** Develop temporal scaling ratios to convert 1-hour data to sub-hourly estimates
            """)

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 📋 NOAA Atlas 14 Temporal Ratios")

                st.markdown("**🔗 Reference:** [NOAA Atlas 14 Volume 2](https://www.weather.gov/media/owp/hdsc_documents/Atlas14_Volume2.pdf)")

                # Display the actual NOAA ratios (for sub-hourly durations only)
                ratios_data = {
                    'Duration': ['5-minute', '10-minute', '15-minute', '30-minute', '60-minute'],
                    'Technical Paper 40': [0.29, 0.45, 0.57, 0.79, 1.00],
                    'NOAA Northern': ['0.261-0.325', '0.386-0.481', '0.475-0.619', '0.712-0.819', '1.000'],
                    'NOAA Southern': ['0.214-0.293', '0.340-0.441', '0.423-0.585', '0.685-0.802', '1.000']
                }

                ratios_df = pd.DataFrame(ratios_data)
                st.dataframe(ratios_df, use_container_width=True)

                UIComponents.formula_display("P_t = P_60min × Ratio_t", "NOAA Temporal Scaling Formula")

                st.markdown("### 🧮 Example Calculation")
                st.markdown("**For 1990 data (60-min maximum = 64mm):**")
                st.markdown("• 5-minute: 64 × 0.29 = **18.6 mm**")
                st.markdown("• 15-minute: 64 × 0.57 = **36.5 mm**")
                st.markdown("• 30-minute: 64 × 0.79 = **50.6 mm**")
                st.markdown("• 60-minute: 64 × 1.00 = **64.0 mm** (original)")

            with col2:
                st.markdown("### 🔬 The Scientific Method")

                st.markdown("**Step 1: Regional Analysis**")
                st.markdown("• Analyzed 96 stations with 5-minute data")
                st.markdown("• Calculated ratios between sub-hourly and hourly maxima")
                st.markdown("• Developed regional scaling factors")

                st.markdown("**Step 2: Validation**")
                st.markdown("• Tested ratios against independent data")
                st.markdown("• Verified consistency across climate regions")
                st.markdown("• Published in NOAA Atlas 14 Volume 2")

                st.markdown("**Step 3: Implementation**")
                st.markdown("• Apply ratios to ALL annual maximum 60-min data")
                st.markdown("• Create annual maximum series for each duration")
                st.markdown("• Use same statistical distribution for all durations")

                # Interactive ratio calculator
                st.markdown("### 🧪 Try the Ratios")
                base_60min = st.slider("60-min annual maximum (mm):", 20, 100, 64, 1, key="noaa_calc")

                st.markdown("**Using Technical Paper 40 ratios:**")
                ratios = [0.29, 0.45, 0.57, 0.79, 1.00]
                durations = ['5-min', '10-min', '15-min', '30-min', '60-min']

                for dur, ratio in zip(durations, ratios):
                    scaled_value = base_60min * ratio
                    st.markdown(f"• **{dur}:** {scaled_value:.1f} mm")

            st.markdown("### 🔑 Key Advantages of NOAA Method")

            advantages = [
                "**Scientifically based** - Derived from actual meteorological data analysis",
                "**Regionally validated** - Different ratios for different climate regions",
                "**Widely accepted** - Standard practice in US engineering",
                "**Quality controlled** - Extensive peer review and validation",
                "**Consistent approach** - Same methodology across entire United States"
            ]

            for advantage in advantages:
                st.markdown(f"✅ {advantage}")

        return None
    
    def _slide_step_calculations(self) -> Optional[bool]:
        """Slide 5: NOAA Temporal Scaling Process - Step by Step"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## NOAA Temporal Scaling Process: Complete Workflow")

            st.markdown("### 📋 Phase 1: Data Preparation")

            UIComponents.highlight_box("""
            **Input Required:**
            • 60-minute annual maximum precipitation data (multiple years)
            • Minimum 20-30 years recommended for robust statistics
            • Quality-controlled, homogeneous dataset
            """)

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 📊 Our 60-minute Annual Maxima")

                # Show sample of base data (converted to 60-min equivalent)
                display_data = self.base_data.copy()
                display_data['Rainfall_60min'] = display_data['Rainfall_30min'] * 1.3  # Convert 30min to 60min equivalent
                display_data = display_data[['Year', 'Rainfall_60min']].head(10)

                st.dataframe(display_data.round(1), use_container_width=True)
                UIComponents.big_number_display(f"{len(self.base_data)}", "Years of Data")
                UIComponents.big_number_display(f"{display_data['Rainfall_60min'].max():.1f}", "Max 60-min (mm)")

            with col2:
                # Histogram of 60-min data
                rainfall_60min = self.base_data['Rainfall_30min'] * 1.3
                fig = px.histogram(x=rainfall_60min, nbins=12,
                                 title="60-min Annual Maximum Distribution")
                fig.update_layout(height=250, xaxis_title="60-min Rainfall (mm)", yaxis_title="Frequency")
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🔢 Phase 2: Temporal Scaling")

            st.markdown("**Apply NOAA temporal ratios to each year's 60-minute maximum:**")

            # NOAA temporal scaling ratios (sub-hourly only)
            durations = [5, 10, 15, 30, 60]
            noaa_ratios = [0.29, 0.45, 0.57, 0.79, 1.00]

            # Create example calculation table
            sample_year = 1990
            sample_60min = 64.0  # mm

            scaling_example = []
            for dur, ratio in zip(durations, noaa_ratios):
                scaled_value = sample_60min * ratio
                intensity = (scaled_value / dur) * 60
                scaling_example.append({
                    'Duration': f"{dur}-min",
                    'NOAA Ratio': ratio,
                    'Rainfall (mm)': f"{scaled_value:.1f}",
                    'Intensity (mm/hr)': f"{intensity:.1f}"
                })

            scaling_df = pd.DataFrame(scaling_example)
            st.dataframe(scaling_df, use_container_width=True)

            st.markdown(f"**Example:** For {sample_year} (60-min maximum = {sample_60min} mm)")

            st.markdown("### 📈 Phase 3: Statistical Analysis")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("**🎯 Key Steps:**")
                analysis_steps = [
                    "**Fit Distribution to 60-min data** - Find best probability distribution (GEV, Log-Normal, etc.)",
                    "**Parameter Estimation** - Calculate location, scale, shape parameters",
                    "**Goodness-of-Fit Testing** - Validate distribution choice statistically",
                    "**Apply Same Distribution** - Use identical distribution for all durations"
                ]

                for i, step in enumerate(analysis_steps, 1):
                    st.markdown(f"{i}. {step}")

                st.markdown("**🔑 NOAA Assumption:**")
                UIComponents.highlight_box("""
                **Same statistical distribution applies to all durations**

                This allows scaling of entire annual maximum series rather than individual quantiles
                """)

            with col2:
                st.markdown("**📊 Distribution Example:**")

                # Generate sample data for visualization
                np.random.seed(42)
                sample_data = np.random.gamma(2.5, 15, 50)  # Sample 60-min data
                sorted_sample = np.sort(sample_data)[::-1]

                # Simple return period estimation
                ranks = np.arange(1, len(sorted_sample) + 1)
                return_periods = (len(sorted_sample) + 1) / ranks

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=return_periods[:15],  # Show top 15 events
                    y=sorted_sample[:15],
                    mode='markers',
                    name='60-min Annual Maxima',
                    marker=dict(size=8, color='blue')
                ))

                fig.update_layout(
                    title="Frequency Analysis Example",
                    xaxis_title="Return Period (years)",
                    yaxis_title="60-min Rainfall (mm)",
                    height=300,
                    xaxis_type='log'
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🎯 Phase 4: Frequency Analysis & IDF Creation")

            workflow_steps = [
                "**Extract Quantiles:** Calculate precipitation depths for target return periods (2, 5, 10, 25, 50, 100 years)",
                "**Build IDF Table:** Organize results in duration vs return period matrix",
                "**Apply Scaling:** Use temporal ratios to convert 60-min quantiles to other durations",
                "**Generate Curves:** Plot final intensity-duration-frequency relationships"
            ]

            for i, step in enumerate(workflow_steps, 1):
                st.markdown(f"{i}. {step}")

            UIComponents.highlight_box("""
            **🎓 Manual Implementation Strategy:**

            1. Use our continuous hourly dataset to extract 60-min annual maxima
            2. Apply NOAA temporal scaling ratios in Excel
            3. Perform distribution fitting in Python/Colab
            4. Extract design quantiles and build final IDF curves
            """)

        return None
    
    def _slide_frequency_analysis(self) -> Optional[bool]:
        """Slide 6: Frequency Analysis for All Durations"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Frequency Analysis: All Durations at Once")
            
            if 'idf_complete_data' not in st.session_state:
                st.error("Please complete the previous slide first!")
                return None
                
            complete_data = st.session_state['idf_complete_data']
            durations = [5, 10, 15, 30, 60, 120]
            design_periods = [2, 5, 10, 25, 50, 100]
            
            st.markdown("### 📊 Design Intensity Table")
            
            # Calculate design intensities for all combinations
            design_intensities = {}
            
            for dur in durations:
                intensity_column = f'Intensity_{dur}min'
                intensity_data = complete_data[intensity_column].values
                sorted_data, return_periods, _ = AnalysisTools.weibull_positions(intensity_data)
                
                design_intensities[dur] = {}
                for T in design_periods:
                    design_val = np.interp(T, return_periods[::-1], sorted_data[::-1])
                    design_intensities[dur][T] = design_val
            
            # Create the IDF table
            idf_table = pd.DataFrame(index=design_periods)
            idf_table.index.name = 'Return Period (years)'
            
            for dur in durations:
                column_name = f'{dur} min'
                idf_table[column_name] = [design_intensities[dur][T] for T in design_periods]
            
            # Display table with formatting
            st.dataframe(idf_table.round(1), use_container_width=True)
            
            # Store for Excel export
            st.session_state['idf_table'] = idf_table
            
            st.markdown("### 🎯 Key Observations")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("**Patterns in the table:**")
                observations = [
                    "📉 **Intensity decreases** with longer duration",
                    "📈 **Intensity increases** with longer return period", 
                    "⚡ **5-minute intensities** are highest",
                    "💧 **2-hour intensities** are lowest"
                ]
                for obs in observations:
                    st.markdown(obs)
                    
                # Quick comparison
                intensity_5min_2yr = design_intensities[5][2]
                intensity_120min_2yr = design_intensities[120][2]
                ratio = intensity_5min_2yr / intensity_120min_2yr
                
                UIComponents.highlight_box(f"""
                **Example:** 2-year event  
                5-min: {intensity_5min_2yr:.1f} mm/hr  
                120-min: {intensity_120min_2yr:.1f} mm/hr  
                Ratio: {ratio:.1f}:1
                """)
                
            with col2:
                # Matrix heatmap
                import plotly.express as px
                
                # Prepare data for heatmap
                heatmap_data = []
                for T in design_periods:
                    for dur in durations:
                        heatmap_data.append({
                            'Return Period': f'{T}-year',
                            'Duration': f'{dur} min',
                            'Intensity': design_intensities[dur][T]
                        })
                
                heatmap_df = pd.DataFrame(heatmap_data)
                
                # Create pivot for heatmap
                pivot_data = heatmap_df.pivot(index='Return Period', columns='Duration', values='Intensity')
                
                fig = px.imshow(
                    pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    color_continuous_scale='Blues',
                    title="IDF Design Intensities (mm/hr)",
                    text_auto='.1f'
                )
                
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_creating_curves(self) -> Optional[bool]:
        """Slide 7: Creating the IDF Curves"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Creating the Famous IDF Curves!")
            
            if 'idf_table' not in st.session_state:
                st.error("Please complete the frequency analysis first!")
                return None
                
            idf_table = st.session_state['idf_table']
            durations = [5, 10, 15, 30, 60, 120]
            
            st.markdown("### 📈 The Complete IDF Curve Family")
            
            # Create the IDF curves plot
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, (T, row) in enumerate(idf_table.iterrows()):
                intensities = [row[f'{dur} min'] for dur in durations]
                
                fig.add_trace(go.Scatter(
                    x=durations,
                    y=intensities,
                    mode='lines+markers',
                    name=f'{T}-year',
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Complete IDF Curves",
                xaxis_title="Duration (minutes)",
                yaxis_title="Intensity (mm/hr)",
                height=500,
                xaxis_type='log',
                yaxis_type='log',
                legend=dict(x=0.7, y=0.95)
            )
            
            fig = PlotTools.apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🎯 How to Read IDF Curves")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("**Interactive Example:**")
                
                selected_return = st.selectbox(
                    "Select return period:",
                    [2, 5, 10, 25, 50, 100],
                    index=2
                )
                
                selected_duration = st.selectbox(
                    "Select duration:",
                    durations,
                    index=2
                )
                
                # Get the intensity
                intensity_value = idf_table.loc[selected_return, f'{selected_duration} min']
                
                UIComponents.big_number_display(f"{intensity_value:.1f}", "mm/hr")
                
                st.markdown(f"""
                **Interpretation:**  
                A **{selected_return}-year** storm lasting **{selected_duration} minutes** 
                will have an intensity of **{intensity_value:.1f} mm/hr**
                """)
                
                # Practical calculation
                total_rainfall = (intensity_value * selected_duration) / 60
                st.markdown(f"**Total rainfall:** {total_rainfall:.1f} mm in {selected_duration} minutes")
                
            with col2:
                st.markdown("**Reading Steps:**")
                
                steps = [
                    "1️⃣ **Choose your design return period** (how rare?)",
                    "2️⃣ **Select your time of concentration** (how long?)",
                    "3️⃣ **Read the intensity** from the curve",
                    "4️⃣ **Calculate total rainfall** if needed",
                    "5️⃣ **Use for runoff calculations**"
                ]
                
                for step in steps:
                    st.markdown(step)
                
                UIComponents.highlight_box("""
                **Pro Tip:** Most urban drainage systems are designed for 
                10-year return periods with durations equal to time of concentration.
                """)
            
            st.markdown("### 💾 Data Export")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                if st.button("📊 Download IDF Table (Excel)", use_container_width=True):
                    excel_data = ExcelExporter.create_idf_template(idf_table, durations)
                    st.download_button(
                        label="Click to Download Excel File",
                        data=excel_data,
                        file_name="idf_curves_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
            with col2:
                st.markdown("**Excel file contains:**")
                excel_contents = [
                    "• Complete IDF table",
                    "• Chart template", 
                    "• Calculation formulas",
                    "• Design examples"
                ]
                for content in excel_contents:
                    st.markdown(content)
        
        return None
    
    def _slide_excel_prep(self) -> Optional[bool]:
        """Slide 8: Excel Workshop Preparation"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Manual Implementation: NOAA Temporal Scaling Method")

            # Add prominent notice about manual implementation
            UIComponents.highlight_box("""
            **🎯 Manual Implementation Approach:**

            📊 **Use our 75-year hourly dataset** to extract 60-minute annual maxima
            🔧 **Apply NOAA temporal scaling ratios** in Excel for all durations
            🔗 **Perform distribution fitting** in Python/Google Colab
            📈 **Build professional IDF curves** following NOAA methodology

            **Learn the complete process step-by-step!**
            """)

            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### 📋 NOAA Implementation Steps")

                workshop_steps = [
                    "**Step 1:** Extract 60-min annual maxima from hourly data",
                    "**Step 2:** Apply NOAA temporal scaling ratios",
                    "**Step 3:** Create annual maximum series for each duration",
                    "**Step 4:** Fit probability distribution to 60-min data",
                    "**Step 5:** Extract quantiles for design return periods",
                    "**Step 6:** Apply same distribution to all durations",
                    "**Step 7:** Build final IDF curves and tables"
                ]

                for i, step in enumerate(workshop_steps, 1):
                    st.markdown(f"{step}")

                st.markdown("### 🧮 NOAA Temporal Scaling Formulas")

                UIComponents.formula_display("=B2*0.29", "5-min scaling (Technical Paper 40)")
                UIComponents.formula_display("=B2*0.45", "10-min scaling")
                UIComponents.formula_display("=B2*0.57", "15-min scaling")
                UIComponents.formula_display("=B2*0.79", "30-min scaling")

                st.markdown("**📖 Reference:** [NOAA Atlas 14 Volume 2](https://www.weather.gov/media/owp/hdsc_documents/Atlas14_Volume2.pdf)")
                
            with col2:
                st.markdown("### 📊 Excel Template")
                
                # Create sample template
                template_data = {
                    'Year': [1990, 1991, 1992, '...', 2023],
                    '30min_Rain': [22.5, 18.3, 35.7, '...', 28.1],
                    '5min_Rain': ['=B2*0.29', '=B3*0.29', '=B4*0.29', '...', '=B35*0.29'],
                    '5min_Intensity': ['=(C2/5)*60', '=(C3/5)*60', '=(C4/5)*60', '...', '=(C35/5)*60']
                }
                
                template_df = pd.DataFrame(template_data)
                st.dataframe(template_df, use_container_width=True)
                
                st.markdown("### 📈 Expected Chart")
                
                # Show what the final chart should look like
                durations = np.array([5, 10, 15, 30, 60, 120])
                
                # Sample curves for demonstration
                intensities_2yr = np.array([75, 58, 48, 35, 23, 15])
                intensities_10yr = intensities_2yr * 1.4
                intensities_50yr = intensities_2yr * 1.8
                
                fig = go.Figure()
                
                for T, intensities, color in [(2, intensities_2yr, 'blue'), 
                                            (10, intensities_10yr, 'orange'),
                                            (50, intensities_50yr, 'red')]:
                    fig.add_trace(go.Scatter(
                        x=durations, y=intensities,
                        mode='lines+markers',
                        name=f'{T}-year',
                        line=dict(width=3)
                    ))
                
                fig.update_layout(
                    title="Target IDF Curves",
                    xaxis_title="Duration (minutes)",
                    yaxis_title="Intensity (mm/hr)",
                    height=300,
                    xaxis_type='log'
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📥 Workshop Materials")
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            with col1:
                st.markdown("**60-min Rainfall Data**")

                # Check if idf.xlsx exists in notebooks folder
                import os
                idf_file_path = "notebooks/idf.xlsx"

                if os.path.exists(idf_file_path):
                    with open(idf_file_path, "rb") as file:
                        st.download_button(
                            "📊 Download IDF Data (Excel)",
                            file.read(),
                            "idf.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    st.markdown("**📄 Contains:**")
                    st.markdown("• 34 years of 60-min data")
                    st.markdown("• Statistics summary")
                    st.markdown("• Excel template")
                    st.markdown("• Complete instructions")
                else:
                    st.error("IDF data file not found. Please check notebooks/idf.xlsx")
                    
            with col2:
                st.markdown("**Distribution Analysis**")
                UIComponents.highlight_box("""
                **🔗 Google Colab Notebook**

                Use this notebook to find the best-fit probability distribution for your rainfall data:
                """)

                st.markdown("**[📊 Open Distribution Analysis Colab](https://colab.research.google.com/drive/1t-Sz6p3xeyxV74efFzu6_gFsigLAkbHz?usp=sharing)**")

                st.markdown("**📋 Steps:**")
                st.markdown("1. Click the Colab link above")
                st.markdown("2. Upload your downloaded data")
                st.markdown("3. Run the analysis")
                st.markdown("4. Get best distribution for IDF curves")

                st.markdown("**🎯 Benefits:**")
                st.markdown("• Compare multiple distributions")
                st.markdown("• Statistical goodness-of-fit tests")
                st.markdown("• Professional IDF parameters")
                    
            with col3:
                st.markdown("**Excel Template**")

                UIComponents.highlight_box("""
                **📋 Excel Formulas Guide**

                **Disaggregation ratios:**
                • 5-min: =B2*0.25
                • 10-min: =B2*0.35
                • 15-min: =B2*0.45
                • 30-min: =B2*0.65
                • 120-min: =B2*1.30

                **Intensity calculations:**
                • I = (P/Duration)*60
                """)

                st.markdown("**📈 Workflow:**")
                st.markdown("1. Download 60-min data (left)")
                st.markdown("2. Apply disaggregation formulas")
                st.markdown("3. Calculate intensities")
                st.markdown("4. Use Colab for distribution analysis")
                st.markdown("5. Create final IDF curves")
            
            UIComponents.highlight_box("""
            **🎯 Workshop Goals:**
            - Understand every step of IDF curve creation
            - Practice Excel formulas for hydrologic analysis
            - Create professional IDF charts
            - Apply to real engineering problems
            """)
        
        return None
    
    
    def _slide_practice_problem(self) -> Optional[bool]:
        """Slide 9: Practice Problem - Conceptual Understanding"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Practice Problem: Understanding IDF Applications")

            st.markdown("### 🎯 Engineering Scenario")

            UIComponents.highlight_box("""
            **You are consulting on storm drainage design for a new shopping center.**

            **Project details:**
            - Shopping center (commercial development)
            - Catchment area: 2.5 hectares
            - Time of concentration: 20 minutes
            - Runoff coefficient: 0.85
            - Local standards require 25-year design event
            """)

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 🧠 Conceptual Analysis")

                st.markdown("**Step 1: Understanding the Parameters**")
                st.markdown("• **Return period (25 years)**: How rare should our design storm be?")
                st.markdown("• **Duration (20 minutes)**: How long does the critical storm last?")
                st.markdown("• **Area (2.5 hectares)**: Size of the contributing watershed")
                st.markdown("• **Runoff coefficient (0.85)**: High for commercial area with lots of pavement")

                st.markdown("**Step 2: IDF Curve Application**")
                st.markdown("• Find the intersection of 25-year return period and 20-minute duration")
                st.markdown("• Read the corresponding rainfall intensity")
                st.markdown("• Use this intensity for drainage system design")

                st.markdown("**Step 3: Design Considerations**")
                st.markdown("• Apply rational method: Q = CIA")
                st.markdown("• Size pipes to handle calculated flow")
                st.markdown("• Consider safety factors and future conditions")

            with col2:
                st.markdown("### 🤔 Discussion Questions")

                st.markdown("**Think about these aspects:**")

                discussion_points = [
                    "**Climate considerations**: How might future climate affect our 25-year design?",
                    "**Duration selection**: Why is time of concentration important for duration choice?",
                    "**Return period**: Should critical infrastructure use longer return periods?",
                    "**Data quality**: How does 75 years of rainfall data improve our confidence?",
                    "**Regional differences**: Would IDF curves be different in other climates?"
                ]

                for point in discussion_points:
                    st.markdown(f"• {point}")

                st.markdown("### 📊 Data Analysis Exercise")

                UIComponents.highlight_box("""
                **Using the provided rainfall dataset:**

                1. **Explore patterns**: Use the 75 years of hourly data to find seasonal trends
                2. **Create pivot tables**: Analyze rainfall by decade, season, and hour
                3. **Compare methods**: Use both Excel and Google Colab for analysis
                4. **Validate results**: Check if your IDF curves make physical sense
                """)

            st.markdown("### 🏆 Learning Objectives Review")

            learning_checks = [
                "✅ **Understand IDF components**: Can you explain Intensity, Duration, and Frequency?",
                "✅ **Data application**: Can you use 75 years of rainfall data for analysis?",
                "✅ **Tool proficiency**: Are you comfortable with both Excel and statistical software?",
                "✅ **Engineering judgment**: Can you critically evaluate IDF curve results?"
            ]

            for check in learning_checks:
                st.markdown(check)

            st.markdown("### 🚀 Next Steps")

            UIComponents.highlight_box("""
            **Continue your IDF learning journey:**

            🔍 **Advanced Analysis**: Use the Google Colab notebook for distribution fitting
            📊 **Comparative Studies**: Analyze how IDF curves vary by region
            🌡️ **Climate Studies**: Investigate long-term trends in rainfall intensity
            🏗️ **Design Applications**: Apply IDF curves to real engineering projects

            **Remember**: IDF curves are fundamental tools that connect meteorology with engineering design!
            """)
        
        return None

    def _slide_quiz(self) -> Optional[bool]:
        """Final quiz - placeholder for now since not in slide list"""
        with UIComponents.slide_container():
            st.markdown("## Knowledge Check")
            
            result = QuizEngine.create_multiple_choice(
                "A 25-year storm with 15-minute duration has an intensity of 80 mm/hr. For a 2-hectare area with runoff coefficient 0.6, what is the peak flow using the rational method?",
                [
                    "267 L/s",
                    "400 L/s", 
                    "133 L/s",
                    "800 L/s"
                ],
                0,
                {
                    "correct": "Perfect! Q = C×I×A = 0.6 × 80 × 20,000 / 3600 = 267 L/s. Remember to convert hectares to m² and mm/hr to proper units.",
                    "incorrect": "Use Q = C×I×A where C=0.6, I=80 mm/hr, A=2 hectares=20,000 m². Result: Q = 0.6×80×20,000/3600 = 267 L/s."
                },
                f"{self.info.id}_final_quiz"
            )
            
            if result is True:
                st.success("🎉 Module 6 Complete! You've mastered IDF curve analysis!")
                return True
            
            return None