"""
Base Classes for Water Resources Learning Modules
Abstract classes and core functionality

Author: TA Saurav Bhattarai
"""

import streamlit as st
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd

@dataclass
class LearningObjective:
    """Represents a single learning objective"""
    description: str
    level: str  # 'remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'
    completed: bool = False

@dataclass
class ModuleInfo:
    """Contains metadata about a learning module"""
    id: str
    title: str
    description: str
    duration_minutes: int
    prerequisites: List[str]
    learning_objectives: List[LearningObjective]
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    total_slides: int = 0

@dataclass
class Slide:
    """Represents a single slide in a module"""
    id: str
    title: str
    content_type: str  # 'theory', 'interactive', 'quiz', 'demo'
    
class SlideNavigator:
    """Handles slide navigation within modules"""
    
    def __init__(self, module_id: str, total_slides: int):
        self.module_id = module_id
        self.total_slides = total_slides
        self.slide_key = f"slide_{module_id}"
        
        # Initialize current slide
        if self.slide_key not in st.session_state:
            st.session_state[self.slide_key] = 0
    
    @property
    def current_slide(self) -> int:
        return st.session_state.get(self.slide_key, 0)
    
    def next_slide(self):
        if self.current_slide < self.total_slides - 1:
            st.session_state[self.slide_key] += 1
            st.rerun()
    
    def prev_slide(self):
        if self.current_slide > 0:
            st.session_state[self.slide_key] -= 1
            st.rerun()
    
    def go_to_slide(self, slide_num: int):
        if 0 <= slide_num < self.total_slides:
            st.session_state[self.slide_key] = slide_num
            st.rerun()
    
    def render_navigation(self, slide_titles: List[str]):
        """Render slide navigation controls"""
        current = self.current_slide
        
        # Top navigation bar
        col1, col2, col3, col4, col5 = st.columns([1, 2, 6, 2, 1])
        
        with col1:
            if st.button("⬅", disabled=current==0, key=f"prev_{self.module_id}"):
                self.prev_slide()
        
        with col2:
            st.markdown(f"**{current + 1}/{self.total_slides}**")
        
        with col3:
            st.markdown(f"<h2 style='text-align: center; margin: 0;'>{slide_titles[current]}</h2>", 
                       unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"**{((current + 1)/self.total_slides*100):.0f}%**")
        
        with col5:
            if st.button("➡", disabled=current==self.total_slides-1, key=f"next_{self.module_id}"):
                self.next_slide()
        
        # Progress bar
        progress = (current + 1) / self.total_slides
        st.progress(progress)
        
        return current

class QuizEngine:
    """Handles quiz creation and assessment"""
    
    @staticmethod
    def create_multiple_choice(question: str, options: List[str], correct_idx: int, 
                             explanation: Dict[str, str], quiz_id: str) -> Optional[bool]:
        """Create multiple choice quiz"""
        
        st.markdown('<div class="quiz-slide">', unsafe_allow_html=True)
        st.markdown(f"### {question}")
        
        # Store answer in session state
        answer_key = f"quiz_answer_{quiz_id}"
        if answer_key not in st.session_state:
            st.session_state[answer_key] = None
        
        # Radio buttons for options
        selected = st.radio(
            "Select your answer:",
            options,
            key=f"quiz_radio_{quiz_id}",
            index=None
        )
        
        # Submit button
        if st.button("Submit Answer", key=f"submit_{quiz_id}"):
            if selected is None:
                st.warning("Please select an answer first!")
                return None
            
            is_correct = selected == options[correct_idx]
            st.session_state[answer_key] = is_correct
            
            if is_correct:
                st.success(f"✅ Correct! {explanation.get('correct', '')}")
            else:
                st.error(f"❌ Incorrect. {explanation.get('incorrect', '')}")
                st.info(f"The correct answer is: **{options[correct_idx]}**")
            
            st.markdown('</div>', unsafe_allow_html=True)
            return is_correct
        
        st.markdown('</div>', unsafe_allow_html=True)
        return None

class LearningModule(ABC):
    """Abstract base class for all learning modules"""
    
    def __init__(self, module_info: ModuleInfo):
        self.info = module_info
        self.navigator = SlideNavigator(module_info.id, module_info.total_slides)
        self.completed = False
    
    def check_prerequisites(self, completed_modules: set) -> bool:
        """Check if all prerequisites are met"""
        return all(prereq in completed_modules for prereq in self.info.prerequisites)
    
    @abstractmethod
    def get_slide_titles(self) -> List[str]:
        """Return list of slide titles"""
        pass
    
    @abstractmethod
    def render_slide(self, slide_num: int) -> Optional[bool]:
        """Render specific slide, return True if module completed"""
        pass
    
    def render(self, show_header=False) -> bool:
        """Main render method for the module"""
        # Optional module header
        if show_header:
            st.markdown(f"""
            <div class="module-header">
                <h1>{self.info.title}</h1>
                <div class="module-meta">
                    <span class="difficulty difficulty-{self.info.difficulty}">{self.info.difficulty.upper()}</span>
                    <span class="duration">{self.info.duration_minutes} minutes</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Navigation
        slide_titles = self.get_slide_titles()
        current_slide = self.navigator.render_navigation(slide_titles)

        # Render current slide
        module_completed = self.render_slide(current_slide)

        return module_completed or False