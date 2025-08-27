"""
Module 2: Understanding Probability Concepts
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

class Module02_Probability(LearningModule):
    """Module 2: Understanding Probability Concepts"""
    
    def __init__(self):
        objectives = [
            LearningObjective("Define probability in engineering context", "understand"),
            LearningObjective("Distinguish frequency from probability", "analyze"),
            LearningObjective("Calculate exceedance vs non-exceedance probability", "apply"),
            LearningObjective("Apply probability to risk assessment", "apply")
        ]
        
        info = ModuleInfo(
            id="module_02",
            title="Understanding Probability Concepts",
            description="Master fundamental probability concepts for hydrologic engineering",
            duration_minutes=35,
            prerequisites=["module_01"],
            learning_objectives=objectives,
            difficulty="beginner",
            total_slides=7
        )
        
        super().__init__(info)
        self.data = DataManager.get_precipitation_data()
    
    def get_slide_titles(self) -> List[str]:
        return [
            "What is Probability?",
            "Probability vs Frequency",
            "Types of Probability", 
            "Interactive Calculator",
            "Engineering Context",
            "Common Misconceptions",
            "Knowledge Check"
        ]
    
    def render_slide(self, slide_num: int) -> Optional[bool]:
        if slide_num == 0:
            return self._slide_definition()
        elif slide_num == 1:
            return self._slide_vs_frequency()
        elif slide_num == 2:
            return self._slide_types()
        elif slide_num == 3:
            return self._slide_calculator()
        elif slide_num == 4:
            return self._slide_engineering()
        elif slide_num == 5:
            return self._slide_misconceptions()
        elif slide_num == 6:
            return self._slide_quiz()
        
        return False
    
    def _slide_definition(self) -> Optional[bool]:
        """Slide 1: What is Probability?"""
        with UIComponents.slide_container("theory"):
            st.markdown("## What is Probability?")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("**Probability** quantifies uncertainty:")
                
                UIComponents.big_number_display("0", "Impossible")
                UIComponents.big_number_display("0.5", "Equal Chance")
                UIComponents.big_number_display("1", "Certain")
                
            with col2:
                UIComponents.highlight_box("""
                **In Water Resources Engineering:**
                
                - Design flood probabilities for dam safety
                - Storm frequency for drainage design  
                - Drought probabilities for water supply
                - Equipment failure rates for reliability
                """)
            
            UIComponents.formula_display("0 ≤ P ≤ 1", "Probability Range")
        
        return None
    
    def _slide_vs_frequency(self) -> Optional[bool]:
        """Slide 2: Probability vs Frequency"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Probability vs Frequency: Key Distinction")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### Frequency (Observed)")
                UIComponents.highlight_box("""
                - **Historical count** of occurrences
                - **Sample-dependent** (varies with data period)
                - **Changes** as you collect more data
                - **Example:** "5 floods in 50 years"
                - **Purpose:** Estimate probability from data
                """)
                
            with col2:
                st.markdown("### Probability (Theoretical)")
                UIComponents.highlight_box("""
                - **Long-term expectation** of occurrence  
                - **Population parameter** (true underlying value)
                - **Estimated** from frequency data
                - **Example:** "P(flood) = 0.1 per year"
                - **Purpose:** Make predictions and design decisions
                """)
        
        return None
    
    def _slide_types(self) -> Optional[bool]:
        """Slide 3: Types of Probability"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Exceedance vs Non-Exceedance")
            
            # Interactive threshold demo
            precip_data = self.data['Annual_Max_Precip'].values
            threshold = st.slider(
                "Select threshold (mm):", 
                float(min(precip_data)), 
                float(max(precip_data)), 
                60.0,
                key="prob_threshold_slide3"
            )
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                exceedance_count = sum(1 for p in precip_data if p >= threshold)
                exceedance_prob = exceedance_count / len(precip_data)
                
                UIComponents.big_number_display(f"{exceedance_prob:.3f}", f"P(X ≥ {threshold:.0f})")
                st.markdown("**Exceedance Probability**")
                st.markdown("Used for flood/storm design")
                
            with col2:
                non_exceedance_prob = 1 - exceedance_prob
                
                UIComponents.big_number_display(f"{non_exceedance_prob:.3f}", f"P(X < {threshold:.0f})")
                st.markdown("**Non-Exceedance Probability**")
                st.markdown("Used for drought/low-flow analysis")
            
            # Visualization
            fig = px.histogram(x=precip_data, nbins=15, title="Probability Visualization")
            fig.add_vline(x=threshold, line_dash="dash", line_color="red")
            fig.update_layout(height=300)
            fig = PlotTools.apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        return None
    
    def _slide_calculator(self) -> Optional[bool]:
        """Slide 4: Interactive Calculator"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Interactive Probability Calculator")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("**Scenario: Flood Events**")
                event_count = st.slider("Floods above threshold:", 0, 20, 4, key="flood_count")
                total_years = st.slider("Years of record:", 20, 100, 50, key="total_years")
                
                if total_years > 0 and event_count <= total_years:
                    frequency = event_count / total_years
                    annual_chance = frequency * 100
                    return_period = 1/frequency if frequency > 0 else float('inf')
                    
                    UIComponents.big_number_display(f"{frequency:.3f}", "Probability")
                    UIComponents.big_number_display(f"{annual_chance:.1f}%", "Annual Chance")
                    UIComponents.big_number_display(f"{return_period:.1f} yrs", "Return Period")
                    
            with col2:
                st.markdown("**Key Relationships:**")
                UIComponents.formula_display("P = Events / Total Years", "Probability from Frequency")
                UIComponents.formula_display("T = 1 / P", "Return Period")
                UIComponents.formula_display("Annual Chance = P × 100%", "Percentage")
                
                UIComponents.highlight_box("""
                **Engineering Interpretation:**
                This event has a probability P of occurring in any given year.
                """)
        
        return None
    
    def _slide_engineering(self) -> Optional[bool]:
        """Slide 5: Engineering Context"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Probability in Engineering Design")
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            with col1:
                UIComponents.big_number_display("2%", "100-Year Event")
                st.markdown("**Critical Infrastructure**")
                st.markdown("- Dams")
                st.markdown("- Nuclear facilities") 
                st.markdown("- Major hospitals")
                
            with col2:
                UIComponents.big_number_display("10%", "10-Year Event")
                st.markdown("**Standard Infrastructure**")
                st.markdown("- Residential areas")
                st.markdown("- Storm drains")
                st.markdown("- Small culverts")
                
            with col3:
                UIComponents.big_number_display("50%", "2-Year Event")
                st.markdown("**Minor Infrastructure**")
                st.markdown("- Parking lots")
                st.markdown("- Rural roads")
                st.markdown("- Agricultural drainage")
            
            UIComponents.highlight_box("""
            **Design Philosophy:** Higher consequences require lower acceptable probabilities (rarer design events)
            """)
        
        return None
    
    def _slide_misconceptions(self) -> Optional[bool]:
        """Slide 6: Common Misconceptions"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Common Misconceptions")
            
            misconceptions = [
                {
                    "myth": "A 100-year flood occurs exactly every 100 years",
                    "reality": "It has a 1% chance each year - could happen multiple times in a decade or not at all for 200 years"
                },
                {
                    "myth": "After a 100-year flood, we're safe for another 99 years",
                    "reality": "Each year has the same 1% probability, independent of previous events"
                },
                {
                    "myth": "Short-term frequency equals long-term probability",
                    "reality": "Probability is estimated from frequency, but represents long-term expectation"
                }
            ]
            
            for i, item in enumerate(misconceptions):
                with st.expander(f"Misconception #{i+1}"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"**Myth:** {item['myth']}")
                    with col2:
                        st.markdown(f"**Reality:** {item['reality']}")
        
        return None
    
    def _slide_quiz(self) -> Optional[bool]:
        """Slide 7: Knowledge Check"""
        with UIComponents.slide_container():
            st.markdown("## Knowledge Check")
            
            result = QuizEngine.create_multiple_choice(
                "A 100-year flood occurs twice in a 30-year record. An engineer says 'It's no longer a 100-year flood, now it's a 15-year flood.' What's wrong with this reasoning?",
                [
                    "The calculation is wrong - it should be 30/2 = 15 years",
                    "Nothing is wrong - the return period has indeed changed", 
                    "The 100-year designation refers to long-term probability, not short-term frequency",
                    "The data must be incorrect if this happened"
                ],
                2,
                {
                    "correct": "Correct! The '100-year flood' designation refers to long-term probability (P=0.01). Short-term variations are normal due to natural randomness. You need much longer records to update probability estimates.",
                    "incorrect": "Remember: Return period refers to long-term average recurrence interval based on probability, not short-term observed frequency. Natural variability causes deviations from expected patterns."
                },
                f"{self.info.id}_final_quiz"
            )
            
            if result is True:
                st.success("🎉 Module 2 Complete! You understand probability concepts.")
                return True
            
            return None