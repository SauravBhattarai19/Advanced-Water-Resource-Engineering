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
            Module06_IDFCurve()  # New IDF Curve module
        ]
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'completed_modules' not in st.session_state:
            st.session_state.completed_modules = set()
        if 'current_slide' not in st.session_state:
            st.session_state.current_slide = {}
        if 'learning_path_started' not in st.session_state:
            st.session_state.learning_path_started = False
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div class="main-header">
            <h1>Water Resources Frequency Analysis</h1>
            <p>Complete Interactive Learning Path - All 6 Modules Accessible</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Course info
        st.markdown("""
        <div class="course-info">
            <strong>📚 Advanced Water Resources Engineering</strong> | 
            <strong>👨‍🏫 Dr. Rocky Talchabhadel</strong> | 
            <strong>🎓 Jackson State University</strong> | 
            <strong>📅 Fall 2025</strong><br>
            <em>Developed by TA Saurav Bhattarai | 🔓 All modules accessible anytime</em>
        </div>
        """, unsafe_allow_html=True)
    
    def render_course_overview(self):
        """Render course overview and learning path"""
        if not st.session_state.learning_path_started:
            st.markdown("## 🎯 Course Learning Path")
            
            # Learning path visualization
            path_data = []
            for i, module in enumerate(self.modules):
                status = "✅ Completed" if module.info.id in st.session_state.completed_modules else "📖 Available"
                difficulty_icon = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}[module.info.difficulty]
                
                path_data.append({
                    'Module': f"Module {i+1}",
                    'Title': module.info.title,
                    'Difficulty': f"{difficulty_icon} {module.info.difficulty.title()}",
                    'Duration': f"⏱️ {module.info.duration_minutes} min",
                    'Status': status,
                    'Builds On': ", ".join(module.info.prerequisites) if module.info.prerequisites else "None"
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
                    "Use IDF curves for engineering design decisions"
                ]
                for obj in advanced_objectives:
                    st.markdown(f"• {obj}")
            
            UIComponents.highlight_box("""
            **📚 Learning Path Flexibility:**  
            All modules are accessible anytime! While there's a recommended sequence, 
            you can explore topics in any order based on your interests and needs.
            """)
            
            # Total course statistics
            total_slides = sum(module.info.total_slides for module in self.modules)
            total_duration = sum(module.info.duration_minutes for module in self.modules)
            
            col1, col2, col3 = UIComponents.three_column_layout()
            
            with col1:
                UIComponents.big_number_display(str(len(self.modules)), "Total Modules")
                
            with col2:
                UIComponents.big_number_display(str(total_slides), "Total Slides")
                
            with col3:
                UIComponents.big_number_display(f"{total_duration} min", "Total Duration")
            
            if st.button("🚀 Start Learning Path", use_container_width=True):
                st.session_state.learning_path_started = True
                st.rerun()
    
    def render_progress_sidebar(self):
        """Render progress tracking in sidebar"""
        st.sidebar.markdown("## 📊 Course Progress")
        
        completed_count = len(st.session_state.completed_modules)
        total_count = len(self.modules)
        progress = completed_count / total_count if total_count > 0 else 0
        
        st.sidebar.progress(progress)
        st.sidebar.markdown(f"**{completed_count}/{total_count} modules completed ({progress*100:.0f}%)**")
        
        # Module status with enhanced information
        st.sidebar.markdown("### Module Status")
        for i, module in enumerate(self.modules):
            status_icon = "✅" if module.info.id in st.session_state.completed_modules else "📖"
            difficulty_badge = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}[module.info.difficulty]
            
            st.sidebar.markdown(f"""
            {status_icon} **Module {i+1}:** {module.info.title}  
            {difficulty_badge} {module.info.difficulty.title()} | ⏱️ {module.info.duration_minutes}min
            """)
            
            # Show prerequisites as info (not requirement)
            if module.info.prerequisites:
                st.sidebar.markdown(f"*Builds on: {', '.join(module.info.prerequisites)}*", 
                                  help="Recommended background - not required")
    
    def render_module_selector(self) -> Optional[LearningModule]:
        """Render module selection interface"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🎯 Select Module")
        
        # All modules are available - no locking
        module_options = []
        
        for i, module in enumerate(self.modules):
            status = "✅" if module.info.id in st.session_state.completed_modules else "📖"
            module_options.append(f"{status} Module {i+1}: {module.info.title}")
        
        selected_idx = st.sidebar.selectbox(
            "Choose a module:",
            range(len(self.modules)),
            format_func=lambda x: module_options[x]
        )
        
        selected_module = self.modules[selected_idx] if selected_idx is not None else None
        
        # Show module information
        if selected_module:
            st.sidebar.markdown("### 📋 Module Information")
            st.sidebar.markdown(f"**Duration:** {selected_module.info.duration_minutes} minutes")
            st.sidebar.markdown(f"**Slides:** {selected_module.info.total_slides}")
            st.sidebar.markdown(f"**Difficulty:** {selected_module.info.difficulty.title()}")
            
            if selected_module.info.prerequisites:
                st.sidebar.markdown(f"**Builds on:** {', '.join(selected_module.info.prerequisites)}")
                st.sidebar.info("💡 Prerequisites are recommended but not required")
            
            # Learning objectives
            st.sidebar.markdown("**Learning Objectives:**")
            for obj in selected_module.info.learning_objectives:
                icon = "✅" if obj.completed else "🎯"
                st.sidebar.markdown(f"{icon} {obj.description}")
        
        return selected_module
    
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
        # Render header
        self.render_header()
        
        # Show course overview if not started
        if not st.session_state.learning_path_started:
            self.render_course_overview()
            return
        
        # Render sidebar
        self.render_progress_sidebar()
        selected_module = self.render_module_selector()
        
        # Check for course completion
        if len(st.session_state.completed_modules) == len(self.modules):
            self.render_course_completion()
            return
        
        if selected_module:
            # Show module progress in main area
            current_slide_key = f"slide_{selected_module.info.id}"
            current_slide = st.session_state.get(current_slide_key, 0)
            
            # Module progress indicator
            progress_pct = (current_slide + 1) / selected_module.info.total_slides
            st.markdown(f"**Module Progress:** {progress_pct:.0%} ({current_slide + 1}/{selected_module.info.total_slides} slides)")
            st.progress(progress_pct)
            
            # Render module
            module_completed = selected_module.render()
            
            # Update progress
            if module_completed:
                st.session_state.completed_modules.add(selected_module.info.id)
                
                # Mark learning objectives as completed
                for obj in selected_module.info.learning_objectives:
                    obj.completed = True
                
                st.success("🎉 Module completed! Great work!")
                
                # Check if this completes the entire course
                if len(st.session_state.completed_modules) == len(self.modules):
                    st.success("🏆 CONGRATULATIONS! You have completed the entire course!")
                    st.balloons()
                else:
                    # Suggest next module in sequence
                    current_module_idx = None
                    for i, m in enumerate(self.modules):
                        if m.info.id == selected_module.info.id:
                            current_module_idx = i
                            break
                    
                    if current_module_idx is not None and current_module_idx < len(self.modules) - 1:
                        next_module = self.modules[current_module_idx + 1]
                        st.info(f"🎯 Continue your learning journey with **{next_module.info.title}**")
                    else:
                        st.info("🎉 You've completed the entire learning path! Great work!")
        else:
            # No module selected or available
            st.markdown("## 🎯 Welcome to Frequency Analysis!")
            st.markdown("**All modules are accessible!** Select any module from the sidebar to begin learning.")
            
            # Show overall progress
            completed_count = len(st.session_state.completed_modules)
            total_count = len(self.modules)
            
            if completed_count > 0:
                st.markdown(f"### Your Progress: {completed_count}/{total_count} modules completed")
                
                # Show completed modules
                completed_modules = [m for m in self.modules if m.info.id in st.session_state.completed_modules]
                if completed_modules:
                    st.markdown("**Completed Modules:**")
                    for i, module in enumerate(self.modules):
                        if module.info.id in st.session_state.completed_modules:
                            st.markdown(f"✅ Module {i+1}: {module.info.title}")
            else:
                st.markdown("### 🚀 Ready to Start!")
                st.markdown("Choose any module from the sidebar. While there's a recommended learning sequence, you're free to explore topics in any order!")
                
                # Show recommended sequence
                st.markdown("**Recommended Learning Sequence:**")
                for i, module in enumerate(self.modules):
                    difficulty_icon = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}[module.info.difficulty]
                    st.markdown(f"{difficulty_icon} **Module {i+1}**: {module.info.title} ({module.info.duration_minutes} min)")
                    if module.info.prerequisites:
                        st.markdown(f"   *Builds on: {', '.join(module.info.prerequisites)}*")

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