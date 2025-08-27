"""
Module 3: Risk, Reliability & Return Periods
Based on original comprehensive learning path

Author: TA Saurav Bhattarai
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents

class Module03_RiskAnalysis(LearningModule):
    """Module 3: Risk, Reliability & Return Periods"""
    
    def __init__(self):
        objectives = [
            LearningObjective("Calculate lifetime risk for infrastructure", "apply"),
            LearningObjective("Understand reliability vs risk relationships", "understand"),
            LearningObjective("Apply risk concepts to design standards", "apply"),
            LearningObjective("Evaluate design adequacy using risk analysis", "analyze")
        ]
        
        info = ModuleInfo(
            id="module_03",
            title="Risk, Reliability & Return Periods",
            description="Master risk analysis for engineering design and infrastructure planning",
            duration_minutes=40,
            prerequisites=["module_01", "module_02"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=8
        )
        
        super().__init__(info)
        self.precip_data = DataManager.get_precipitation_data()
        self.flood_data = DataManager.get_flood_data()
    
    def get_slide_titles(self) -> List[str]:
        return [
            "Risk vs Reliability",
            "Mathematical Relationships",
            "Interactive Risk Calculator",
            "Design Standards", 
            "Frequency Analysis Comparison",
            "Risk Assessment Table",
            "Real-World Applications",
            "Knowledge Check"
        ]
    
    def render_slide(self, slide_num: int) -> Optional[bool]:
        if slide_num == 0:
            return self._slide_concepts()
        elif slide_num == 1:
            return self._slide_math()
        elif slide_num == 2:
            return self._slide_calculator()
        elif slide_num == 3:
            return self._slide_standards()
        elif slide_num == 4:
            return self._slide_frequency_comparison()
        elif slide_num == 5:
            return self._slide_risk_table()
        elif slide_num == 6:
            return self._slide_applications()
        elif slide_num == 7:
            return self._slide_quiz()
        
        return False
    
    def _slide_concepts(self) -> Optional[bool]:
        """Slide 1: Risk vs Reliability"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Risk vs Reliability")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Risk (R)")
                UIComponents.highlight_box("""
                **Probability that design event is EXCEEDED during lifetime**
                
                - What we want to minimize
                - Represents potential failure
                - Used for safety assessment
                """)
                
                UIComponents.big_number_display("HIGH RISK", "BAD")
                
            with col2:
                st.markdown("### Reliability (Rel)")
                UIComponents.highlight_box("""
                **Probability that design event is NOT EXCEEDED during lifetime**
                
                - What we want to maximize
                - Represents successful operation
                - Used for performance assessment
                """)
                
                UIComponents.big_number_display("HIGH RELIABILITY", "GOOD")
            
            UIComponents.formula_display("Risk + Reliability = 1", "Complementary Relationship")
        
        return None
    
    def _slide_math(self) -> Optional[bool]:
        """Slide 2: Mathematical Relationships"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Key Formulas")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Basic Relationships")
                UIComponents.formula_display("P = 1/T", "Annual Probability")
                UIComponents.formula_display("R = 1 - (1-P)ⁿ", "Lifetime Risk")
                UIComponents.formula_display("Rel = (1-P)ⁿ", "Reliability")
                
            with col2:
                st.markdown("### Where:")
                st.markdown("""
                - **P** = Annual exceedance probability
                - **T** = Return period (years)
                - **n** = Design life (years)
                - **R** = Lifetime risk
                - **Rel** = Reliability
                """)
                
                UIComponents.highlight_box("""
                **Key Insight:** Risk increases with longer design life!
                """)
        
        return None
    
    def _slide_calculator(self) -> Optional[bool]:
        """Slide 3: Interactive Risk Calculator"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Risk Calculator")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                design_return_period = st.selectbox(
                    "Design return period:", 
                    [2, 5, 10, 25, 50, 100, 200, 500, 1000],
                    index=4,
                    key="risk_calc_T"
                )
                design_life = st.slider("Design life (years):", 1, 100, 50, key="risk_calc_n")
                
                # Calculations
                annual_prob = 1/design_return_period
                lifetime_risk = 1-(1-annual_prob)**design_life
                reliability = (1-annual_prob)**design_life
                
                UIComponents.big_number_display(f"{annual_prob:.4f}", "Annual Probability")
                UIComponents.big_number_display(f"{lifetime_risk:.3f}", f"{design_life}-Year Risk")
                UIComponents.big_number_display(f"{reliability:.3f}", f"{design_life}-Year Reliability")
                
                # Risk level indicator
                if lifetime_risk < 0.1:
                    risk_level = "🟢 Low Risk"
                elif lifetime_risk < 0.3:
                    risk_level = "🟡 Moderate Risk" 
                else:
                    risk_level = "🔴 High Risk"
                
                st.markdown(f"**Risk Level:** {risk_level}")
                
            with col2:
                # Risk evolution plot
                life_range = np.arange(1, 101)
                risks = 1-(1-annual_prob)**life_range
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=life_range, 
                    y=risks, 
                    mode='lines', 
                    name=f'{design_return_period}-Year Design',
                    line=dict(color='#e74c3c', width=4)
                ))
                
                # Current design point
                fig.add_trace(go.Scatter(
                    x=[design_life], 
                    y=[lifetime_risk], 
                    mode='markers', 
                    name='Current Design',
                    marker=dict(size=15, color='#3498db', symbol='star')
                ))
                
                # Risk guidelines
                for level, color, label in [(0.1, '#27ae60', '10%'), (0.2, '#f39c12', '20%'), (0.5, '#e74c3c', '50%')]:
                    fig.add_hline(y=level, line_dash="dash", line_color=color, 
                                 annotation_text=f"{label} Risk")
                
                fig.update_layout(
                    title='Risk vs Design Life',
                    xaxis_title='Design Life (years)',
                    yaxis_title='Lifetime Risk',
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_standards(self) -> Optional[bool]:
        """Slide 4: Design Standards"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Engineering Design Standards")
            
            standards_data = {
                'Facility Type': ['Residential', 'Commercial', 'Critical Infrastructure', 'High-Risk Facilities'],
                'Typical Design T': ['10-25 years', '25-50 years', '100-500 years', '500+ years'], 
                'Acceptable 50-Year Risk': ['≤ 40%', '≤ 20%', '≤ 10%', '≤ 5%'],
                'Examples': [
                    'Houses, small roads',
                    'Shopping centers, offices', 
                    'Hospitals, schools',
                    'Nuclear plants, dams'
                ]
            }
            
            standards_df = pd.DataFrame(standards_data)
            st.dataframe(standards_df, use_container_width=True)
            
            UIComponents.highlight_box("""
            **Key Principle:** Higher consequences require lower acceptable risk levels (longer return periods)
            """)
            
            # Quick risk check for common designs
            st.markdown("### Quick Risk Assessment:")
            col1, col2, col3 = UIComponents.three_column_layout()
            
            designs = [(10, 'Residential'), (25, 'Commercial'), (100, 'Critical')]
            
            for i, (T, category) in enumerate(designs):
                risk_50 = 1-(1-1/T)**50
                status = "✅ Acceptable" if risk_50 <= [0.4, 0.2, 0.1][i] else "⚠️ Too High"
                
                with [col1, col2, col3][i]:
                    UIComponents.big_number_display(f"{T}", f"{category}")
                    st.markdown(f"50-year risk: {risk_50:.2f}")
                    st.markdown(f"Status: {status}")
        
        return None
    
    def _slide_frequency_comparison(self) -> Optional[bool]:
        """Slide 5: Frequency Analysis Comparison"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Comparing Precipitation & Flood Risk")
            
            col1, col2 = UIComponents.two_column_layout()
            
            # Analyze precipitation data
            precip_values = self.precip_data['Annual_Max_Precip'].values
            sorted_precip, T_precip, _ = AnalysisTools.weibull_positions(precip_values)
            
            # Analyze flood data
            flood_values = self.flood_data['Peak_Flow'].values
            sorted_flood, T_flood, _ = AnalysisTools.weibull_positions(flood_values)
            
            with col1:
                fig_precip = go.Figure()
                fig_precip.add_trace(go.Scatter(
                    x=T_precip, 
                    y=sorted_precip, 
                    mode='markers+lines',
                    name='Precipitation',
                    marker=dict(size=8, color='#3498db')
                ))
                
                fig_precip.update_layout(
                    title='Precipitation Frequency',
                    xaxis_title='Return Period (years)',
                    yaxis_title='Precipitation (mm)',
                    xaxis_type='log',
                    height=350
                )
                fig_precip = PlotTools.apply_theme(fig_precip)
                st.plotly_chart(fig_precip, use_container_width=True)
                
            with col2:
                fig_flood = go.Figure()
                fig_flood.add_trace(go.Scatter(
                    x=T_flood, 
                    y=sorted_flood, 
                    mode='markers+lines',
                    name='Flood',
                    marker=dict(size=8, color='#e74c3c')
                ))
                
                fig_flood.update_layout(
                    title='Flood Frequency',
                    xaxis_title='Return Period (years)',
                    yaxis_title='Peak Flow (m³/s)',
                    xaxis_type='log',
                    height=350
                )
                fig_flood = PlotTools.apply_theme(fig_flood)
                st.plotly_chart(fig_flood, use_container_width=True)
        
        return None
    
    def _slide_risk_table(self) -> Optional[bool]:
        """Slide 6: Risk Assessment Table"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Design Event Risk Comparison")
            
            return_periods = [2, 5, 10, 25, 50, 100]
            risk_data = []
            
            # Get data for interpolation
            precip_values = self.precip_data['Annual_Max_Precip'].values
            flood_values = self.flood_data['Peak_Flow'].values
            sorted_precip, T_precip, _ = AnalysisTools.weibull_positions(precip_values)
            sorted_flood, T_flood, _ = AnalysisTools.weibull_positions(flood_values)
            
            for T in return_periods:
                annual_risk = 1/T
                risk_25 = 1-(1-annual_risk)**25
                risk_50 = 1-(1-annual_risk)**50
                
                # Interpolate design values
                precip_val = np.interp(T, T_precip[::-1], sorted_precip[::-1])
                flood_val = np.interp(T, T_flood[::-1], sorted_flood[::-1])
                
                risk_data.append({
                    'Return Period': f'{T} years',
                    'Annual Risk': f'{annual_risk:.4f}',
                    '25-Year Risk': f'{risk_25:.3f}',
                    '50-Year Risk': f'{risk_50:.3f}',
                    'Design Precip (mm)': f'{precip_val:.1f}',
                    'Design Flow (m³/s)': f'{flood_val:.0f}'
                })
            
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True)
            
            UIComponents.highlight_box("""
            **Key Insight:** Notice how lifetime risk increases dramatically with design life, 
            even for the same annual probability.
            """)
        
        return None
    
    def _slide_applications(self) -> Optional[bool]:
        """Slide 7: Real-World Applications"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Risk-Based Design Applications")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Case Study 1: Storm Drain")
                UIComponents.highlight_box("""
                **Scenario:** Residential storm drain
                - Design: 10-year storm
                - Life: 25 years
                - Risk = 1-(1-0.1)^25 = 0.36 (36%)
                - **Decision:** Acceptable for residential
                """)
                
                st.markdown("### Case Study 2: Hospital")
                UIComponents.highlight_box("""
                **Scenario:** Critical facility drainage
                - Design: 100-year storm
                - Life: 50 years  
                - Risk = 1-(1-0.01)^50 = 0.39 (39%)
                - **Decision:** Too high! Need 500-year design
                """)
                
            with col2:
                st.markdown("### Decision Framework")
                decision_steps = [
                    "1. Identify facility importance",
                    "2. Determine acceptable risk level",
                    "3. Calculate lifetime risk for design",
                    "4. Compare with standard",
                    "5. Adjust design if needed"
                ]
                
                for step in decision_steps:
                    st.markdown(f"- {step}")
                
                UIComponents.formula_display("Economics ↔ Safety", "Balance Design Costs vs Risk")
        
        return None
    
    def _slide_quiz(self) -> Optional[bool]:
        """Slide 8: Knowledge Check"""
        with UIComponents.slide_container():
            st.markdown("## Knowledge Check")
            
            result = QuizEngine.create_multiple_choice(
                "A storm drain designed for a 10-year storm with a 25-year design life has what lifetime risk, and is this acceptable for residential infrastructure?",
                [
                    "10% risk, acceptable",
                    "36% risk, acceptable", 
                    "25% risk, not acceptable",
                    "64% risk, not acceptable"
                ],
                1,
                {
                    "correct": "Correct! Risk = 1-(1-0.1)^25 = 0.36 (36%). This is acceptable for residential infrastructure (≤40% threshold).",
                    "incorrect": "Use R = 1-(1-P)^n where P=1/10=0.1, n=25. Risk = 1-(0.9)^25 = 0.36 or 36%. For residential areas, up to 40% is typically acceptable."
                },
                f"{self.info.id}_final_quiz"
            )
            
            if result is True:
                st.success("Module 3 Complete! You've mastered risk and reliability analysis.")
                return True
            
            return None