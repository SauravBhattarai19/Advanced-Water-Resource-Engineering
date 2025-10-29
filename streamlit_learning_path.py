"""
Water Resources Engineering: Frequency Analysis Learning Path
Main Application Entry Point - Updated with IDF Curve Analysis Module

Author: TA Saurav Bhattarai
Course: Advanced Water Resources Engineering  
Instructor: Dr. Rocky Talchabhadel
Institution: Jackson State University
Date: Fall 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional
import importlib
import sys
from pathlib import Path

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

# Import base classes
from base_classes import LearningModule, ModuleInfo, SlideNavigator
from utilities import DataManager, AnalysisTools, UIComponents

# Import individual modules
try:
    from module_01_data_exploration import Module01_DataExploration
    from module_02_probability import Module02_Probability
    from module_03_risk_analysis import Module03_RiskAnalysis
    from module_04_distributions import Module04_Distributions
    from module_05_gev_analysis import Module05_GEVAnalysis
    from module_06_idf_curve import Module06_IDFCurve
    from module_exam_answers import ModuleExamAnswers
    from module_midterm_answers import ModuleMidtermAnswers
    from module_09_trend_detection import Module09_TrendDetection
    from module_10_breakpoint_detection import Module10_BreakpointDetection
    from module_11_spatiotemporal import Module11_Spatiotemporal
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.info("Please ensure all module files are in the modules/ directory")

# Configure Streamlit
st.set_page_config(
    page_title="Water Resources Frequency Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
UIComponents.load_presentation_css()

class LearningPathApp:
    """Main application class for the learning path"""
    
    def __init__(self):
        self.modules = self._initialize_modules()
        self._initialize_session_state()
    
    def _initialize_modules(self) -> List[LearningModule]:
        """Initialize all learning modules"""
        return [
            Module01_DataExploration(),
            Module02_Probability(),
            Module03_RiskAnalysis(),
            Module04_Distributions(),
            Module05_GEVAnalysis(),
            Module06_IDFCurve(),
            ModuleExamAnswers(),  # Module 7: Quiz 1 answer key and review
            ModuleMidtermAnswers(),  # Module 8: Midterm exam solutions
            Module09_TrendDetection(),  # Module 9: Trend analysis with Mann-Kendall test
            Module10_BreakpointDetection(),  # Module 10: Change point detection with Pettitt test
            Module11_Spatiotemporal()  # Module 11: Spatiotemporal mapping and regional analysis
        ]
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'completed_modules' not in st.session_state:
            st.session_state.completed_modules = set()
        if 'current_slide' not in st.session_state:
            st.session_state.current_slide = {}
        if 'learning_path_started' not in st.session_state:
            st.session_state.learning_path_started = False
    
    def render_header(self, show_full_header=False):
        """Render application header - full version only on main page"""
        if show_full_header:
            st.markdown("""
            <div class="main-header">
                <h1>Advanced Water Resources Engineering</h1>
            </div>
            """, unsafe_allow_html=True)

            # Course info with better styling
            st.markdown("""
            <div class="course-info">
                <strong>📚 Advanced Water Resources Engineering</strong> |
                <strong>👨‍🏫 Dr. Rocky Talchabhadel</strong> |
                <strong>🎓 Jackson State University</strong> |
                <strong>📅 Fall 2025</strong>
            </div>
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 8px 15px;
                border-radius: 8px;
                text-align: center;
                margin: 10px 0;
                font-size: 14px;
            ">
                <strong>Developed by TA Saurav Bhattarai</strong> • <em>Assisted by Claude</em> • 🔓 All modules accessible anytime
            </div>
            """, unsafe_allow_html=True)
        else:
            # No header for module pages - keep it completely clean
            pass
    
    def render_course_overview(self):
        """Render course overview and learning path"""
        if not st.session_state.learning_path_started:
            st.markdown("## 🎯 Course Learning Path")

            # Prominent go to modules button
            if st.button("🚀 Go to Modules", use_container_width=True, type="primary"):
                st.session_state.learning_path_started = True
                st.rerun()

            st.markdown("---")

            # Clean learning path visualization
            path_data = []
            for i, module in enumerate(self.modules):
                path_data.append({
                    'Module': f"Module {i+1}",
                    'Title': module.info.title
                })

            path_df = pd.DataFrame(path_data)
            st.dataframe(path_df, use_container_width=True)
            
            st.markdown("### 🎓 Overall Course Objectives")
            
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("**Foundation Concepts:**")
                foundation_objectives = [
                    "Master data visualization and exploratory analysis",
                    "Understand probability theory in engineering context", 
                    "Apply risk analysis to infrastructure design",
                    "Select appropriate probability distributions"
                ]
                for obj in foundation_objectives:
                    st.markdown(f"• {obj}")
                
            with col2:
                st.markdown("**Advanced Applications:**")
                advanced_objectives = [
                    "Understand GEV as flexible Gumbel distribution",
                    "Apply shape parameter to control tail behavior",
                    "Create IDF curves from rainfall data",
                    "Apply disaggregation ratios for short-duration analysis",
                    "Use IDF curves for engineering design decisions",
                    "Detect monotonic trends using Mann-Kendall test",
                    "Identify abrupt change points with Pettitt test",
                    "Create spatiotemporal maps for regional assessment",
                    "Review and master all concepts through interactive exam solutions"
                ]
                for obj in advanced_objectives:
                    st.markdown(f"• {obj}")
            
            UIComponents.highlight_box("""
            **📚 Learning Path Flexibility:**
            All modules are accessible anytime! While there's a recommended sequence,
            you can explore topics in any order based on your interests and needs.
            """)
            
            # Simple course statistics
            total_slides = sum(module.info.total_slides for module in self.modules)

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                UIComponents.big_number_display(str(len(self.modules)), "Total Modules")

            with col2:
                UIComponents.big_number_display(str(total_slides), "Total Slides")
    
    def render_module_selector(self) -> Optional[LearningModule]:
        """Render module selection at top of sidebar"""
        # Home button first
        if st.sidebar.button("🏠 Back to Home", use_container_width=True):
            st.session_state.learning_path_started = False
            st.rerun()
            return None

        st.sidebar.markdown("## 🎯 Select Module")

        # Module options only (no Home in dropdown)
        module_options = []
        for i, module in enumerate(self.modules):
            module_options.append(f"Module {i+1}: {module.info.title}")

        # Simple selectbox with modules only - defaults to Module 1 (index 0)
        selected_idx = st.sidebar.selectbox(
            "Choose a module:",
            range(len(module_options)),
            format_func=lambda x: module_options[x],
            key="module_selector"
        )

        # Handle module selection directly
        selected_module = self.modules[selected_idx] if selected_idx < len(self.modules) else None

        # Show minimal module information
        if selected_module:
            st.sidebar.markdown("### 📋 Module Info")
            st.sidebar.markdown(f"**Slides:** {selected_module.info.total_slides}")

        return selected_module

    def render_simple_sidebar(self):
        """Render simplified sidebar without progress tracking"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📚 All Available Modules")

        # Simple module list - completely clean
        for i, module in enumerate(self.modules):
            st.sidebar.markdown(f"**Module {i+1}:** {module.info.title}")
    
    def render_course_completion(self):
        """Render course completion celebration"""
        if len(st.session_state.completed_modules) == len(self.modules):
            st.balloons()
            st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                <h1>🎉 Congratulations! 🎉</h1>
                <h2>Course Completed Successfully!</h2>
                <p>You have mastered frequency analysis in water resources engineering!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Course completion summary
            st.markdown("## 📜 Course Completion Summary")
            
            completion_data = []
            for i, module in enumerate(self.modules):
                completion_data.append({
                    'Module': f"Module {i+1}",
                    'Title': module.info.title,
                    'Difficulty': module.info.difficulty.title(),
                    'Duration': f"{module.info.duration_minutes} min",
                    'Status': "✅ Completed"
                })
            
            completion_df = pd.DataFrame(completion_data)
            st.dataframe(completion_df, use_container_width=True)
            
            # Skills mastered
            col1, col2 = UIComponents.two_column_layout()
            
            with col1:
                st.markdown("### 🛠️ Technical Skills Mastered")
                skills = [
                    "Data exploration and visualization",
                    "Weibull plotting position analysis",
                    "Probability theory applications",
                    "Risk and reliability assessment",
                    "Distribution fitting and selection",
                    "GEV analysis and parameter estimation",
                    "IDF curve development and application",
                    "Excel proficiency for hydrologic analysis",
                    "Python programming for hydrology",
                    "Statistical model assessment"
                ]
                for skill in skills:
                    st.markdown(f"✅ {skill}")
                    
            with col2:
                st.markdown("### 🏗️ Engineering Applications")
                applications = [
                    "Infrastructure design standards",
                    "Flood frequency analysis",
                    "Dam safety evaluation",
                    "Bridge and culvert design",
                    "Storm drainage system sizing",
                    "Urban runoff analysis",
                    "Risk-based decision making",
                    "Climate change adaptation",
                    "Insurance and catastrophe modeling",
                    "Regulatory compliance assessment"
                ]
                for app in applications:
                    st.markdown(f"✅ {app}")
            
            # Next steps
            UIComponents.highlight_box("""
            **🚀 Next Steps:**
            - Apply these techniques to your capstone project
            - Create IDF curves for your local area
            - Explore advanced topics: non-stationarity, climate change impacts
            - Consider graduate studies in water resources engineering
            - Join professional organizations (ASCE, AWRA)
            - Continue learning through conferences and workshops
            """)
    
    def run(self):
        """Main application entry point"""
        # Show course overview if not started
        if not st.session_state.learning_path_started:
            # Render full header only on main page
            self.render_header(show_full_header=True)
            self.render_course_overview()
            return

        # No header for module pages - completely clean interface

        # Render module selector first, then sidebar info
        selected_module = self.render_module_selector()
        self.render_simple_sidebar()

        if selected_module:
            # Show simple slide progress
            current_slide_key = f"slide_{selected_module.info.id}"
            current_slide = st.session_state.get(current_slide_key, 0)

            # Simple slide indicator
            st.markdown(f"**Slide {current_slide + 1} of {selected_module.info.total_slides}**")

            # Render module without header (since title is already in sidebar)
            module_completed = selected_module.render(show_header=False)

            # Simple completion message
            if module_completed:
                st.success("🎉 Module completed! Great work!")
                st.info("Select another module from the sidebar to continue learning.")
        else:
            # Welcome message when no module selected
            st.markdown("## 🎯 Welcome to Frequency Analysis!")
            st.markdown("**Select any module from the sidebar to begin learning.**")

            # Show simple module overview
            st.markdown("### Available Modules:")
            for i, module in enumerate(self.modules):
                st.markdown(f"**Module {i+1}**: {module.info.title}")

def main():
    """Application entry point"""
    try:
        app = LearningPathApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check that all module files are properly installed and accessible.")

if __name__ == "__main__":
    main()