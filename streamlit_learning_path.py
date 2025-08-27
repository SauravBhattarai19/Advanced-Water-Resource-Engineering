"""
Water Resources Engineering: Frequency Analysis Learning Path
Main Application Entry Point

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
            Module04_Distributions()
        ]
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'completed_modules' not in st.session_state:
            st.session_state.completed_modules = set()
        if 'current_slide' not in st.session_state:
            st.session_state.current_slide = {}
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div class="main-header">
            <h1>Water Resources Frequency Analysis</h1>
            <p>Interactive Learning Path for Engineering Students</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Course info
        st.markdown("""
        <div class="course-info">
            <strong>📚 Advanced Water Resources Engineering</strong> | 
            <strong>👨‍🏫 Dr. Rocky Talchabhadel</strong> | 
            <strong>🎓 Jackson State University</strong> | 
            <strong>📅 Fall 2025</strong><br>
            <em>Developed by TA Saurav Bhattarai</em>
        </div>
        """, unsafe_allow_html=True)
    
    def render_progress_sidebar(self):
        """Render progress tracking in sidebar"""
        st.sidebar.markdown("## 📊 Course Progress")
        
        completed_count = len(st.session_state.completed_modules)
        total_count = len(self.modules)
        progress = completed_count / total_count if total_count > 0 else 0
        
        st.sidebar.progress(progress)
        st.sidebar.markdown(f"**{completed_count}/{total_count} modules completed ({progress*100:.0f}%)**")
        
        # Module status
        st.sidebar.markdown("### Module Status")
        for i, module in enumerate(self.modules):
            status_icon = "✅" if module.info.id in st.session_state.completed_modules else "📖"
            difficulty_badge = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}[module.info.difficulty]
            
            st.sidebar.markdown(f"""
            {status_icon} **Module {i+1}:** {module.info.title}  
            {difficulty_badge} {module.info.difficulty.title()} | ⏱️ {module.info.duration_minutes}min
            """)
    
    def render_module_selector(self) -> Optional[LearningModule]:
        """Render module selection interface"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🎯 Select Module")
        
        module_options = [f"Module {i+1}: {mod.info.title}" for i, mod in enumerate(self.modules)]
        
        selected_idx = st.sidebar.selectbox(
            "Choose a module:",
            range(len(module_options)),
            format_func=lambda x: module_options[x]
        )
        
        return self.modules[selected_idx] if selected_idx is not None else None
    
    def run(self):
        """Main application entry point"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_progress_sidebar()
        selected_module = self.render_module_selector()
        
        if selected_module:
            # Render module
            module_completed = selected_module.render()
            
            # Update progress
            if module_completed:
                st.session_state.completed_modules.add(selected_module.info.id)
                st.success("Module completed! Great work!")
                st.balloons()

def main():
    """Application entry point"""
    app = LearningPathApp()
    app.run()

if __name__ == "__main__":
    main()