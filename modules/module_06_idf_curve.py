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
            "The Magic Ratios",
            "Step-by-Step Calculations",
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
        """Slide 4: The Magic Ratios - Disaggregation"""
        with UIComponents.slide_container("theory"):
            st.markdown("## The Magic Ratios: Creating Short Duration Data")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### 🎯 The Problem")
                UIComponents.highlight_box("""
                **We have:** 30-minute rainfall data  
                **We need:** 5, 10, 15-minute data for complete IDF curves
                
                **Solution:** Use empirical ratios from research studies
                """)
                
                st.markdown("### 🔢 Standard Ratios (US Practice)")
                
                # Display the standard ratios
                ratios_data = {
                    'Duration': ['5 min', '10 min', '15 min', '30 min', '60 min', '120 min'],
                    'Ratio': [0.29, 0.54, 0.70, 1.00, 0.81, 0.66],
                    'Source': ['Research', 'Research', 'Research', 'Base Data', 'Calculated', 'Calculated']
                }
                
                ratios_df = pd.DataFrame(ratios_data)
                st.dataframe(ratios_df, use_container_width=True)
                
                UIComponents.formula_display("P_t = P_30min × Ratio_t", "Disaggregation Formula")
                
            with col2:
                st.markdown("### 🧮 Live Calculator")
                
                # Interactive disaggregation calculator
                base_30min = st.slider("30-min rainfall (mm):", 5, 60, 25, 1, key="disagg_base")
                
                st.markdown("**Calculated values:**")
                
                durations = [5, 10, 15, 30, 60, 120]
                ratios = [0.29, 0.54, 0.70, 1.00, 0.81, 0.66]
                
                calc_data = []
                for dur, ratio in zip(durations, ratios):
                    if dur <= 30:
                        rainfall = base_30min * ratio
                    else:
                        rainfall = base_30min * ratio  # These are extension ratios
                    
                    intensity = (rainfall / dur) * 60
                    calc_data.append({
                        'Duration': f"{dur} min",
                        'Rainfall (mm)': f"{rainfall:.1f}",
                        'Intensity (mm/hr)': f"{intensity:.1f}"
                    })
                
                calc_df = pd.DataFrame(calc_data)
                st.dataframe(calc_df, use_container_width=True)
                
                # Visualization
                fig = go.Figure()
                
                plot_durations = [d for d in durations]
                plot_intensities = [float(row['Intensity (mm/hr)']) for row in calc_data]
                
                fig.add_trace(go.Scatter(
                    x=plot_durations,
                    y=plot_intensities,
                    mode='lines+markers',
                    name='Single Event',
                    line=dict(width=3)
                ))
                
                fig.update_layout(
                    title="Intensity-Duration for Single Event",
                    xaxis_title="Duration (minutes)",
                    yaxis_title="Intensity (mm/hr)",
                    height=300,
                    xaxis_type='log'
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### ⚠️ Important Notes")
            UIComponents.highlight_box("""
            **These ratios are:**
            - Based on meteorological research
            - Vary slightly by geographic region
            - Standard practice in engineering
            - Used when detailed data isn't available
            """)
        
        return None
    
    def _slide_step_calculations(self) -> Optional[bool]:
        """Slide 5: Step-by-Step Calculations"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Complete IDF Calculation: Step by Step")
            
            st.markdown("### 📊 Our Base Data (30-minute annual maximums)")
            
            # Show sample of base data
            display_data = self.base_data.head(10)
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.dataframe(display_data, use_container_width=True)
                UIComponents.big_number_display(f"{len(self.base_data)}", "Years of Data")
                UIComponents.big_number_display(f"{self.base_data['Rainfall_30min'].max():.1f}", "Max (mm)")
                
            with col2:
                # Histogram of base data
                fig = px.histogram(self.base_data, x='Rainfall_30min', nbins=12,
                                 title="30-min Annual Maximum Distribution")
                fig.update_layout(height=250)
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔢 Step 1: Apply Disaggregation Ratios")
            
            # Calculate all durations for all years
            durations = [5, 10, 15, 30, 60, 120]
            ratios = [0.29, 0.54, 0.70, 1.00, 0.81, 0.66]
            
            # Create complete dataset
            complete_data = pd.DataFrame({'Year': self.base_data['Year']})
            
            for dur, ratio in zip(durations, ratios):
                complete_data[f'Rain_{dur}min'] = self.base_data['Rainfall_30min'] * ratio
                complete_data[f'Intensity_{dur}min'] = (complete_data[f'Rain_{dur}min'] / dur) * 60
            
            st.markdown("**Sample of calculated data:**")
            sample_display = complete_data[['Year', 'Rain_5min', 'Rain_15min', 'Rain_30min', 'Rain_60min']].head(8)
            st.dataframe(sample_display.round(1), use_container_width=True)
            
            st.markdown("### 📈 Step 2: Frequency Analysis for Each Duration")
            
            selected_duration = st.selectbox(
                "Choose duration to analyze:",
                [f"{d} minutes" for d in durations],
                index=2
            )
            
            dur_value = int(selected_duration.split()[0])
            intensity_column = f'Intensity_{dur_value}min'
            intensity_data = complete_data[intensity_column].values
            
            # Perform Weibull analysis
            sorted_data, return_periods, plotting_positions = AnalysisTools.weibull_positions(intensity_data)
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown(f"**Frequency Analysis for {selected_duration}:**")
                
                # Show top events
                freq_results = pd.DataFrame({
                    'Rank': range(1, min(8, len(sorted_data) + 1)),
                    'Intensity (mm/hr)': sorted_data[:7].round(1),
                    'Return Period (years)': return_periods[:7].round(1)
                })
                
                st.dataframe(freq_results, use_container_width=True)
                
                # Design values
                design_periods = [2, 5, 10, 25, 50]
                st.markdown("**Design Intensities:**")
                for T in design_periods:
                    design_intensity = np.interp(T, return_periods[::-1], sorted_data[::-1])
                    st.markdown(f"• **{T}-year:** {design_intensity:.1f} mm/hr")
                    
            with col2:
                # Frequency plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=return_periods,
                    y=sorted_data,
                    mode='markers',
                    name='Data',
                    marker=dict(size=8, color='blue')
                ))
                
                # Add design event markers
                for T in design_periods:
                    design_val = np.interp(T, return_periods[::-1], sorted_data[::-1])
                    fig.add_trace(go.Scatter(
                        x=[T], y=[design_val],
                        mode='markers+text',
                        text=f'{T}yr',
                        textposition='top center',
                        marker=dict(size=10, color='red', symbol='star'),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title=f"Frequency Analysis - {selected_duration}",
                    xaxis_title="Return Period (years)",
                    yaxis_title="Intensity (mm/hr)",
                    height=350,
                    xaxis_type='log'
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
            # Store complete dataset for next slides
            st.session_state['idf_complete_data'] = complete_data
        
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
            st.markdown("## Excel Workshop: Let's Build IDF Curves Together!")

            # Add prominent notice about data and Colab
            UIComponents.highlight_box("""
            **🎯 Workshop Resources Available:**

            📊 **Real 60-minute rainfall data** for download in notebooks folder
            🔗 **Google Colab notebook** for advanced distribution analysis
            📋 **Excel templates** with formulas ready to use

            **Everything you need to create professional IDF curves!**
            """)

            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### 📋 Workshop Steps")
                
                workshop_steps = [
                    "**Step 1:** Load 30-minute base data",
                    "**Step 2:** Apply disaggregation ratios",
                    "**Step 3:** Calculate intensities", 
                    "**Step 4:** Perform frequency analysis",
                    "**Step 5:** Create design intensity table",
                    "**Step 6:** Plot IDF curves",
                    "**Step 7:** Verify results"
                ]
                
                for i, step in enumerate(workshop_steps, 1):
                    st.markdown(f"{step}")
                
                st.markdown("### 🧮 Key Formulas for Excel")
                
                UIComponents.formula_display("=B2*0.29", "5-min disaggregation")
                UIComponents.formula_display("=(C2/5)*60", "5-min intensity")
                UIComponents.formula_display("=RANK(B2,$B$2:$B$35,0)", "Weibull rank")
                UIComponents.formula_display("=A2/(35+1)", "Plotting position")
                UIComponents.formula_display("=1/D2", "Return period")
                
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