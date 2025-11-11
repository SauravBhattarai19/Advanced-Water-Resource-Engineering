"""
Module 1: Data Exploration & Weibull Analysis
Slide-based learning module for classroom presentation

Author: TA Saurav Bhattarai
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents, ExcelExporter

class Module01_DataExploration(LearningModule):
    """Module 1: Data Exploration & Weibull Analysis"""
    
    def __init__(self):
        objectives = [
            LearningObjective("Visualize hydrologic time series data", "understand"),
            LearningObjective("Apply Weibull plotting position method", "apply"),
            LearningObjective("Calculate return periods from data", "apply"),
            LearningObjective("Interpret frequency analysis results", "analyze")
        ]
        
        info = ModuleInfo(
            id="module_01",
            title="Data Exploration & Weibull Analysis",
            description="Learn to analyze hydrologic data and estimate frequencies using the Weibull method",
            duration_minutes=30,
            prerequisites=[],
            learning_objectives=objectives,
            difficulty="beginner",
            total_slides=8
        )
        
        super().__init__(info)
        self.data = DataManager.get_precipitation_data()
    
    def get_slide_titles(self) -> List[str]:
        return [
            "Introduction",
            "The Dataset", 
            "Data Visualization",
            "Weibull Method Theory",
            "Step-by-Step Analysis",
            "Frequency Plot",
            "Engineering Applications",
            "Knowledge Check"
        ]
    
    def render_slide(self, slide_num: int) -> Optional[bool]:
        """Render specific slide content"""
        
        if slide_num == 0:
            return self._slide_introduction()
        elif slide_num == 1:
            return self._slide_dataset()
        elif slide_num == 2:
            return self._slide_visualization()
        elif slide_num == 3:
            return self._slide_theory()
        elif slide_num == 4:
            return self._slide_analysis()
        elif slide_num == 5:
            return self._slide_frequency_plot()
        elif slide_num == 6:
            return self._slide_applications()
        elif slide_num == 7:
            return self._slide_quiz()
        
        return False
    
    def _slide_introduction(self) -> Optional[bool]:
        """Slide 1: Introduction"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Why Study Hydrologic Frequency Analysis?")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                UIComponents.highlight_box("""
                **Engineering Applications:**
                - Storm drain design
                - Flood risk assessment  
                - Infrastructure planning
                - Climate analysis
                """)
                
            with col2:
                UIComponents.big_number_display("44", "Years of Data")
                UIComponents.big_number_display("1980-2023", "Study Period")
            
            st.markdown("---")
            st.markdown("**Today's Goal:** Learn to estimate how often extreme events occur using historical data")
        
        return None
    
    def _slide_dataset(self) -> Optional[bool]:
        """Slide 2: The Dataset"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Annual Maximum Precipitation Data")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                # Show data table
                st.markdown("**Sample Data (Recent Years):**")
                display_data = self.data.tail(8)[['Year', 'Annual_Max_Precip']]
                display_data.columns = ['Year', 'Precipitation (mm)']
                st.dataframe(display_data, use_container_width=True)
                
            with col2:
                # Basic statistics
                precip_values = self.data['Annual_Max_Precip'].values
                
                UIComponents.big_number_display(f"{np.mean(precip_values):.1f} mm", "Average")
                UIComponents.big_number_display(f"{np.max(precip_values):.0f} mm", "Maximum")
                UIComponents.big_number_display(f"{np.min(precip_values):.0f} mm", "Minimum")
        
        return None
    
    def _slide_visualization(self) -> Optional[bool]:
        """Slide 3: Data Visualization"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Visualizing the Data")
            
            precip_values = self.data['Annual_Max_Precip'].values
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                # Time series
                fig_ts = px.line(self.data, x='Year', y='Annual_Max_Precip',
                               title='Time Series (1980-2023)', markers=True)
                fig_ts.update_layout(height=300, showlegend=False)
                fig_ts = PlotTools.apply_theme(fig_ts)
                st.plotly_chart(fig_ts, use_container_width=True)
                
            with col2:
                # Histogram
                fig_hist = px.histogram(x=precip_values, nbins=10, 
                                      title='Distribution')
                fig_hist.update_layout(height=300, showlegend=False)
                fig_hist.update_xaxes(title='Precipitation (mm)')
                fig_hist.update_yaxes(title='Frequency')
                fig_hist = PlotTools.apply_theme(fig_hist)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            UIComponents.highlight_box("**Key Insight:** Data shows natural variability - some years much wetter than others")
        
        return None
    
    def _slide_theory(self) -> Optional[bool]:
        """Slide 4: Weibull Method Theory"""
        with UIComponents.slide_container("theory"):
            st.markdown("## The Weibull Plotting Position Method")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("**Step-by-Step Process:**")
                st.markdown("""
                1. **Sort** data (largest to smallest)
                2. **Assign ranks** (1, 2, 3, ...)  
                3. **Calculate probability:** P = m/(n+1)
                4. **Calculate return period:** T = 1/P
                """)
                
                UIComponents.formula_display("P = m/(n+1)", "Plotting Position Formula")
                UIComponents.formula_display("T = 1/P", "Return Period Formula")
                
            with col2:
                st.markdown("**Key Variables:**")
                st.markdown("""
                - **m** = rank of observation
                - **n** = total number of observations  
                - **P** = exceedance probability
                - **T** = return period (years)
                """)
                
                UIComponents.highlight_box("""
                **Example:** If P = 0.04, then T = 25 years  
                *This event has a 4% chance each year*
                """)
        
        return None
    
    def _slide_analysis(self) -> Optional[bool]:
        """Slide 5: Step-by-Step Analysis"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Weibull Analysis Results")
            
            # Perform analysis
            precip_values = self.data['Annual_Max_Precip'].values
            sorted_data, return_periods, plotting_positions = AnalysisTools.weibull_positions(precip_values)
            
            # Create results table
            results_df = pd.DataFrame({
                'Rank': range(1, min(11, len(sorted_data) + 1)),
                'Precipitation (mm)': np.round(sorted_data[:10], 1),
                'Probability P': np.round(plotting_positions[:10], 4),
                'Return Period T': np.round(return_periods[:10], 1)
            })
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("**Top 10 Events:**")
                st.dataframe(results_df, use_container_width=True)
                
            with col2:
                st.markdown("**Key Results:**")
                UIComponents.big_number_display(f"{sorted_data[0]:.1f} mm", "Largest Event")
                UIComponents.big_number_display(f"{return_periods[0]:.1f} years", "Return Period")
                
                # Design event estimates
                design_events = [2, 5, 10, 25, 50]
                st.markdown("**Design Event Estimates:**")
                for T in design_events:
                    value = np.interp(T, return_periods[::-1], sorted_data[::-1])
                    st.markdown(f"• **{T}-year:** {value:.1f} mm")
        
        return None
    
    def _slide_frequency_plot(self) -> Optional[bool]:
        """Slide 6: Frequency Plot"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Frequency Analysis Plot")
            
            precip_values = self.data['Annual_Max_Precip'].values
            fig = PlotTools.frequency_plot(precip_values, "Precipitation Frequency Curve")
            
            # Add design event markers
            design_events = [2, 5, 10, 25, 50, 100]
            sorted_data, return_periods, _ = AnalysisTools.weibull_positions(precip_values)
            
            for T in design_events:
                value = np.interp(T, return_periods[::-1], sorted_data[::-1])
                fig.add_trace(go.Scatter(
                    x=[T], y=[value],
                    mode='markers+text',
                    text=f'{T}yr',
                    textposition='top center',
                    marker=dict(size=12, color='red', symbol='star'),
                    name=f'{T}-year event',
                    showlegend=False
                ))
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            UIComponents.highlight_box("""
            **Reading the Plot:** Higher return periods = rarer events with larger precipitation amounts
            """)
        
        return None
    
    def _slide_applications(self) -> Optional[bool]:
        """Slide 7: Engineering Applications"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Engineering Design Applications")
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            # Calculate design values
            precip_values = self.data['Annual_Max_Precip'].values
            sorted_data, return_periods, _ = AnalysisTools.weibull_positions(precip_values)
            
            val_10yr = np.interp(10, return_periods[::-1], sorted_data[::-1])
            val_25yr = np.interp(25, return_periods[::-1], sorted_data[::-1])
            val_100yr = np.interp(100, return_periods[::-1], sorted_data[::-1])
            
            with col1:
                UIComponents.big_number_display(f"{val_10yr:.1f} mm", "10-Year Design")
                st.markdown("**Residential Areas**")
                st.markdown("• Storm drains")  
                st.markdown("• Parking lots")
                st.markdown("• Small culverts")
                
            with col2:
                UIComponents.big_number_display(f"{val_25yr:.1f} mm", "25-Year Design")
                st.markdown("**Commercial Areas**")
                st.markdown("• Shopping centers")
                st.markdown("• Office buildings") 
                st.markdown("• Major roadways")
                
            with col3:
                UIComponents.big_number_display(f"{val_100yr:.1f} mm", "100-Year Design")
                st.markdown("**Critical Infrastructure**")
                st.markdown("• Hospitals")
                st.markdown("• Emergency services")
                st.markdown("• Major bridges")
            
            st.markdown("---")
            
            # Excel download
            if st.button("📥 Download Excel Template for Practice"):
                excel_data = ExcelExporter.create_weibull_template(self.data)
                st.download_button(
                    label="Click to Download",
                    data=excel_data,
                    file_name="weibull_analysis_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        return None
    
    def _slide_quiz(self) -> Optional[bool]:
        """Slide 8: Knowledge Check"""
        with UIComponents.slide_container():
            st.markdown("## Knowledge Check")
            
            result = QuizEngine.create_multiple_choice(
                "In the Weibull method, if an event has plotting position P = 0.02, what is its return period and annual exceedance probability?",
                [
                    "T = 50 years, Annual probability = 2%",
                    "T = 2 years, Annual probability = 50%", 
                    "T = 0.02 years, Annual probability = 98%",
                    "T = 25 years, Annual probability = 4%"
                ],
                0,
                {
                    "correct": "Perfect! T = 1/P = 1/0.02 = 50 years. This means 2% annual chance of exceedance.",
                    "incorrect": "Remember: Return period T = 1/P. If P = 0.02, then T = 50 years and annual probability = 2%."
                },
                f"{self.info.id}_final_quiz"
            )
            
            if result is True:
                st.success("🎉 Module 1 Complete! You've mastered data exploration and Weibull analysis.")
                return True
            
            return None