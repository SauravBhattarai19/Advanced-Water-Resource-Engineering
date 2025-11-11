"""
Module: Exam Answer Key & Learning Review
Interactive answer key with detailed explanations and learning reinforcement

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

class ModuleExamAnswers(LearningModule):
    """Interactive Exam Answer Key & Learning Review"""

    def __init__(self):
        objectives = [
            LearningObjective("Review exam solutions and understand key concepts", "understand"),
            LearningObjective("Identify common mistakes and how to avoid them", "analyze"),
            LearningObjective("Reinforce learning through detailed explanations", "apply"),
            LearningObjective("Practice additional problems for mastery", "apply")
        ]

        info = ModuleInfo(
            id="module_exam_answers",
            title="📝 Exam Answer Key & Review",
            description="Interactive answer key with detailed explanations and learning reinforcement",
            duration_minutes=25,
            prerequisites=["module_01", "module_02", "module_03", "module_04", "module_05", "module_06"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=6
        )

        super().__init__(info)

    def get_slide_titles(self) -> List[str]:
        return [
            "Exam Overview & Instructions",
            "Section A: MCQ Answers (1-5)",
            "Section A: MCQ Answers (6-10)",
            "Section B: Numerical Solutions",
            "Common Mistakes & Learning Tips",
            "Practice Problems & Next Steps"
        ]

    def render_slide(self, slide_num: int) -> Optional[bool]:
        slides = [
            self._slide_overview,
            self._slide_mcq_1_5,
            self._slide_mcq_6_10,
            self._slide_numerical_solutions,
            self._slide_learning_tips,
            self._slide_practice_problems
        ]

        if slide_num < len(slides):
            return slides[slide_num]()
        return False

    def _slide_overview(self) -> Optional[bool]:
        """Slide 1: Exam Overview & Instructions"""
        with UIComponents.slide_container("theory"):
            st.markdown("## 📝 Exam Answer Key & Learning Review")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 🎯 Exam Structure")
                UIComponents.highlight_box("""
                **Section A: Multiple Choice (30 marks)**
                - 10 questions × 3 marks each
                - Topics: Weibull, probability, risk, distributions, GEV, IDF

                **Section B: Numerical Problems (20 marks)**
                - 3 problems: IDF application (8 marks), Risk analysis (6 marks), Return period (6 marks)
                - Focus on practical calculations and engineering applications
                """)

                st.markdown("### 📊 Score Breakdown")
                score_data = {
                    'Section': ['Section A (MCQ)', 'Section B (Numerical)', 'Total'],
                    'Questions': ['10 questions', '3 problems', '13 total'],
                    'Marks': ['30 marks', '20 marks', '50 marks'],
                    'Time': ['~10 minutes', '~10 minutes', '20 minutes']
                }
                score_df = pd.DataFrame(score_data)
                st.dataframe(score_df, use_container_width=True)

            with col2:
                st.markdown("### 🔍 How to Use This Answer Key")

                usage_steps = [
                    "**🤔 Try First**: Attempt questions before looking at answers",
                    "**✅ Check Answers**: Compare your responses with correct answers",
                    "**📖 Read Explanations**: Understand the reasoning behind each answer",
                    "**🔄 Review Mistakes**: Focus on questions you got wrong",
                    "**💡 Learn Tips**: Apply the learning strategies provided",
                    "**🎯 Practice More**: Use additional problems for reinforcement"
                ]

                for step in usage_steps:
                    st.markdown(step)

                UIComponents.highlight_box("""
                **💡 Learning Strategy:**
                Don't just memorize answers - understand the WHY behind each solution.
                This will help you tackle similar problems in future exams and real engineering work.
                """)

            st.markdown("### 🎓 Learning Objectives for This Review")

            col1, col2, col3 = UIComponents.three_column_layout()

            with col1:
                st.markdown("**🧠 Knowledge Reinforcement**")
                st.markdown("• Solidify understanding of key concepts")
                st.markdown("• Connect theory to practical applications")
                st.markdown("• Identify knowledge gaps")

            with col2:
                st.markdown("**🔧 Problem-Solving Skills**")
                st.markdown("• Master step-by-step calculation methods")
                st.markdown("• Learn to avoid common mistakes")
                st.markdown("• Develop engineering intuition")

            with col3:
                st.markdown("**📈 Exam Performance**")
                st.markdown("• Improve time management")
                st.markdown("• Build confidence for future assessments")
                st.markdown("• Prepare for advanced coursework")

        return None

    def _slide_mcq_1_5(self) -> Optional[bool]:
        """Slide 2: MCQ Answers 1-5"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Section A: Multiple Choice Questions (1-5)")

            # Question 1
            with st.expander("**Question 1:** Weibull Plotting Position", expanded=True):
                st.markdown("**Question:** If you have 25 years of data and an event is ranked 5th largest, what is its plotting position P?")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) P = 5/25 = 0.20")
                    st.markdown("B) P = 5/26 = 0.19 ✅")
                    st.markdown("C) P = 5/24 = 0.21")
                    st.markdown("D) P = 4/25 = 0.16")

                with col2:
                    st.success("**Correct Answer: B) 0.19**")

                    UIComponents.highlight_box("""
                    **Solution:**
                    Use Weibull formula: P = m/(n+1)
                    • m = rank = 5
                    • n = total years = 25
                    • P = 5/(25+1) = 5/26 = 0.192 ≈ 0.19

                    **Key Point:** Always use (n+1) in denominator!
                    """)

            # Question 2
            with st.expander("**Question 2:** Probability vs Frequency"):
                st.markdown("**Question:** Which statement best distinguishes probability from frequency?")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) Probability and frequency are the same thing")
                    st.markdown("B) Frequency is observed count; probability is long-term expectation ✅")
                    st.markdown("C) Probability changes with more data; frequency stays constant")
                    st.markdown("D) Frequency is theoretical; probability is observed")

                with col2:
                    st.success("**Correct Answer: B**")

                    UIComponents.highlight_box("""
                    **Key Distinction:**
                    • **Frequency**: Historical count (e.g., "5 floods in 50 years")
                    • **Probability**: Long-term expectation (e.g., "P(flood) = 0.1")

                    Frequency is used to *estimate* probability from data.
                    """)

            # Question 3
            with st.expander("**Question 3:** Risk and Reliability"):
                st.markdown("**Question:** If the risk of failure is 0.35, what is the reliability?")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) 0.35")
                    st.markdown("B) 0.65 ✅")
                    st.markdown("C) 1.35")
                    st.markdown("D) Cannot be determined")

                with col2:
                    st.success("**Correct Answer: B) 0.65**")

                    UIComponents.highlight_box("""
                    **Solution:**
                    Risk + Reliability = 1
                    Reliability = 1 - Risk = 1 - 0.35 = 0.65

                    **Meaning:** 65% chance system will NOT fail during design life.
                    """)

            # Question 4
            with st.expander("**Question 4:** PDF vs CDF"):
                st.markdown("**Question:** What is the main difference between PDF and CDF?")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) PDF shows probability density; CDF shows cumulative probability")
                    st.markdown("B) PDF is for continuous data; CDF is for discrete data")
                    st.markdown("C) PDF values can exceed 1; CDF values cannot exceed 1")
                    st.markdown("D) Both A and C are correct ✅")

                with col2:
                    st.success("**Correct Answer: D) Both A and C**")

                    UIComponents.highlight_box("""
                    **Both statements are true:**
                    • PDF: f(x) = probability density (can be > 1)
                    • CDF: F(x) = cumulative probability (always ≤ 1)

                    **Remember:** PDF height ≠ probability. Area under PDF = probability.
                    """)

            # Question 5
            with st.expander("**Question 5:** Goodness-of-Fit Test"):
                st.markdown("**Question:** Given KS test results, which distribution provides the best fit (α = 0.05)?")

                # Show the table
                test_data = {
                    'Distribution': ['Normal', 'Log-Normal', 'Exponential', 'Gumbel'],
                    'KS p-value': [0.12, 0.45, 0.02, 0.38],
                    'Accept?': ['✅ Yes', '✅ Yes (Best)', '❌ No', '✅ Yes']
                }
                test_df = pd.DataFrame(test_data)
                st.dataframe(test_df, use_container_width=True)

                st.success("**Correct Answer: B) Log-Normal**")

                UIComponents.highlight_box("""
                **KS Test Rule:**
                • Accept if p-value > α = 0.05
                • Choose highest p-value among acceptable distributions
                • Log-Normal has highest p-value (0.45) = best fit
                • Exponential rejected (0.02 < 0.05)
                """)

        return None

    def _slide_mcq_6_10(self) -> Optional[bool]:
        """Slide 3: MCQ Answers 6-10"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Section A: Multiple Choice Questions (6-10)")

            # Question 6
            with st.expander("**Question 6:** GEV Shape Parameter", expanded=True):
                st.markdown("**Question:** GEV distribution with shape parameter ξ = +0.25 indicates:")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) Light tail (Weibull type) - extremes are bounded")
                    st.markdown("B) Standard tail (Gumbel type) - normal extreme behavior")
                    st.markdown("C) Heavy tail (Fréchet type) - extreme events more likely ✅")
                    st.markdown("D) The distribution fitting failed")

                with col2:
                    st.success("**Correct Answer: C) Heavy tail (Fréchet)**")

                    UIComponents.highlight_box("""
                    **GEV Shape Parameter (ξ):**
                    • ξ > 0: Fréchet (Heavy tail) - More extreme events
                    • ξ = 0: Gumbel (Standard) - Normal behavior
                    • ξ < 0: Weibull (Light tail) - Bounded extremes

                    **Engineering Impact:** ξ = +0.25 means be more conservative in design!
                    """)

            # Question 7
            with st.expander("**Question 7:** IDF Curves"):
                st.markdown("**Question:** In IDF curves, what does the 'I' represent?")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) Total rainfall depth (mm)")
                    st.markdown("B) Rainfall intensity (mm/hr) ✅")
                    st.markdown("C) Infrastructure design standard")
                    st.markdown("D) Infiltration rate (mm/hr)")

                with col2:
                    st.success("**Correct Answer: B) Rainfall intensity**")

                    UIComponents.highlight_box("""
                    **IDF Components:**
                    • **I** = Intensity (mm/hr) - Rate of rainfall
                    • **D** = Duration (min/hr) - How long it rains
                    • **F** = Frequency (years) - How often it occurs

                    **Key:** Intensity ≠ total depth. It's the *rate* of rainfall.
                    """)

            # Question 8
            with st.expander("**Question 8:** 100-Year Flood Misconception"):
                st.markdown("**Question:** A '100-year flood' occurred twice in a 10-year period. Which statement is correct?")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) It's no longer a 100-year flood; it's now a 5-year flood")
                    st.markdown("B) The 100-year designation is wrong and should be recalculated")
                    st.markdown("C) This is natural variability; the long-term probability remains 1% ✅")
                    st.markdown("D) This indicates climate change has altered the flood regime")

                with col2:
                    st.success("**Correct Answer: C) Natural variability**")

                    UIComponents.highlight_box("""
                    **Common Misconception Clarified:**
                    • "100-year" means 1% annual probability, not exact timing
                    • Each year is independent - past events don't affect future
                    • Short-term clustering is normal randomness
                    • Need long-term data to update probability estimates

                    **Analogy:** Getting two heads in a row doesn't make a coin biased!
                    """)

            # Question 9
            with st.expander("**Question 9:** Critical Infrastructure Design"):
                st.markdown("**Question:** Why are longer return periods generally selected for critical infrastructure?")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) To minimize the cost of construction")
                    st.markdown("B) To reduce the probability of failure and increase safety ✅")
                    st.markdown("C) Because rainfall data is always available for long durations")
                    st.markdown("D) To simplify hydrologic analysis")

                with col2:
                    st.success("**Correct Answer: B) Increase safety**")

                    UIComponents.highlight_box("""
                    **Design Philosophy:**
                    Longer return period = Lower probability = Higher safety

                    **Typical Standards:**
                    • Residential: 10-25 years
                    • Commercial: 25-50 years
                    • Critical: 100-500+ years

                    **Rule:** Higher consequences require lower acceptable risk.
                    """)

            # Question 10
            with st.expander("**Question 10:** Rainfall Intensity Calculation"):
                st.markdown("**Question:** If 30 mm of rain falls in 45 minutes, what is the rainfall intensity?")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Options:**")
                    st.markdown("A) 30 mm/hr")
                    st.markdown("B) 40 mm/hr ✅")
                    st.markdown("C) 45 mm/hr")
                    st.markdown("D) 67 mm/hr")

                with col2:
                    st.success("**Correct Answer: B) 40 mm/hr**")

                    UIComponents.highlight_box("""
                    **Step-by-step calculation:**

                    I = Depth / Time
                    • Depth = 30 mm
                    • Time = 45 min = 0.75 hr
                    • I = 30/0.75 = 40 mm/hr

                    **Quick method:** (30 mm / 45 min) × 60 = 40 mm/hr
                    """)

            # Section A Summary
            st.markdown("---")
            st.markdown("### 📊 Section A Summary")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("**Correct Answers:**")
                answers = [
                    "1. B) 0.19", "2. B) Frequency vs probability", "3. B) 0.65",
                    "4. D) Both A and C", "5. B) Log-Normal", "6. C) Heavy tail",
                    "7. B) Rainfall intensity", "8. C) Natural variability",
                    "9. B) Increase safety", "10. B) 40 mm/hr"
                ]
                for ans in answers:
                    st.markdown(f"✅ {ans}")

            with col2:
                st.markdown("**Key Topics Covered:**")
                topics = [
                    "Weibull plotting positions", "Probability concepts", "Risk analysis",
                    "Distribution theory", "Goodness-of-fit testing", "GEV analysis",
                    "IDF curves", "Return period interpretation", "Design standards",
                    "Intensity calculations"
                ]
                for topic in topics:
                    st.markdown(f"📖 {topic}")

        return None

    def _slide_numerical_solutions(self) -> Optional[bool]:
        """Slide 4: Numerical Solutions"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Section B: Numerical Problem Solutions")

            # Problem 1: IDF Application
            with st.expander("**Problem 1:** IDF Curve Application (8 marks)", expanded=True):
                st.markdown("**Given:** I = 85 mm/hr, A = 3.5 hectares, C = 0.75, Q = (1/3600) × C × I × A")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Step-by-step solution:**")

                    st.markdown("**Step 1: Identify given values**")
                    st.code("""
I = 85 mm/hr (rainfall intensity)
A = 3.5 hectares (drainage area)
C = 0.75 (runoff coefficient)
                    """)

                    st.markdown("**Step 2: Apply rational method with proper units**")
                    st.code("""
Q = (1/3600) × C × I × A
Where:
- I is in mm/hr
- A is in hectares
- Q will be in m³/s
                    """)

                    st.markdown("**Step 3: Calculate discharge**")
                    st.code("""
Q = (1/3600) × 0.75 × 85 × 3.5
Q = (1/3600) × 223.125
Q = 0.062 m³/s
                    """)

                    st.markdown("**Step 4: Convert to L/s**")
                    st.code("""
Q = 0.062 m³/s × 1000 L/m³ = 62 L/s
                    """)

                with col2:
                    st.success("**Final Answer: Q = 62 L/s**")

                    st.markdown("**Marking Scheme (8 marks):**")
                    marking = [
                        "Identifying given values (1 mark)",
                        "Formula setup with correct units (2 marks)",
                        "Calculation steps (3 marks)",
                        "Unit conversion to L/s (1 mark)",
                        "Final answer (1 mark)"
                    ]
                    for mark in marking:
                        st.markdown(f"• {mark}")

                    UIComponents.highlight_box("""
                    **Key Points:**
                    • Use Q = (1/3600) × C × I × A when I is in mm/hr and A is in hectares
                    • Rational method assumes steady-state flow
                    • This discharge size would need ~250mm diameter pipe
                    """)

            # Problem 2: Risk Analysis
            with st.expander("**Problem 2:** Risk and Reliability Analysis (6 marks)"):
                st.markdown("**Given:** T = 25 years, n = 40 years, criterion ≤ 30%")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Solutions:**")

                    st.markdown("**(a) Annual probability:**")
                    st.code("P = 1/T = 1/25 = 0.04 or 4%")

                    st.markdown("**(b) Lifetime risk:**")
                    st.code("""
R = 1-(1-P)^n = 1-(1-0.04)^40
R = 1-(0.96)^40 = 1-0.195 = 0.805 or 80.5%
                    """)

                    st.markdown("**(c) Reliability:**")
                    st.code("""
Rel = (1-P)^n = (0.96)^40 = 0.195 or 19.5%
Check: R + Rel = 0.805 + 0.195 = 1.000 ✓
                    """)

                with col2:
                    st.error("**Design Assessment: FAILS criterion**")

                    UIComponents.highlight_box("""
                    **Design Adequacy:**
                    • Calculated risk: 80.5%
                    • Acceptable criterion: ≤ 30%
                    • **Conclusion:** Design does NOT meet criteria
                    • **Recommendation:** Use 100-year design instead

                    **Why it fails:** 25-year design over 40-year life = very high risk!
                    """)

                    st.markdown("**Marking Scheme (6 marks):**")
                    marking = [
                        "Annual probability (1 mark)",
                        "Risk calculation (2 marks)",
                        "Reliability calculation (1 mark)",
                        "Adequacy assessment (2 marks)"
                    ]
                    for mark in marking:
                        st.markdown(f"• {mark}")

            # Problem 3: Return Period
            with st.expander("**Problem 3:** Return Period Calculation (6 marks)"):
                st.markdown("**Given:** n = 35 years, m = 3 (3rd largest), magnitude = 450 m³/s")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Solutions:**")

                    st.markdown("**(a) Plotting position:**")
                    st.code("""
P = m/(n+1) = 3/(35+1) = 3/36 = 0.0833
                    """)

                    st.markdown("**(b) Return period:**")
                    st.code("""
T = 1/P = 1/0.0833 = 12.0 years
                    """)

                    st.markdown("**(c) Practical meaning:**")
                    st.markdown("""
                    **Statistical interpretation:** A flood of 450 m³/s or larger has an 8.33% chance
                    of occurring in any given year.

                    **Engineering interpretation:** This is approximately a 12-year design event,
                    suitable for standard infrastructure requiring ~10-15 year protection.

                    **Important note:** This doesn't mean it occurs exactly every 12 years -
                    it's an average recurrence interval based on probability.
                    """)

                with col2:
                    st.success("**Answers:**")
                    st.markdown("• **(a)** P = 0.0833")
                    st.markdown("• **(b)** T = 12.0 years")
                    st.markdown("• **(c)** See detailed explanation")

                    st.markdown("**Marking Scheme (6 marks):**")
                    marking = [
                        "Plotting position (2 marks)",
                        "Return period calculation (2 marks)",
                        "Practical interpretation (2 marks)"
                    ]
                    for mark in marking:
                        st.markdown(f"• {mark}")

                    UIComponents.highlight_box("""
                    **Engineering Context:**
                    A 12-year event is suitable for:
                    • Residential drainage
                    • Small culverts
                    • Local road crossings
                    • Standard infrastructure
                    """)

        return None

    def _slide_learning_tips(self) -> Optional[bool]:
        """Slide 5: Common Mistakes & Learning Tips"""
        with UIComponents.slide_container("theory"):
            st.markdown("## 🎯 Common Mistakes & Learning Tips")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### ❌ Common Student Errors")

                errors = [
                    {
                        "mistake": "Using m/n instead of m/(n+1) for Weibull",
                        "fix": "Always remember the +1 in denominator for Weibull plotting position"
                    },
                    {
                        "mistake": "Forgetting unit conversions in rational method",
                        "fix": "Check units carefully: mm/hr with m² needs conversion factor"
                    },
                    {
                        "mistake": "Confusing risk and reliability definitions",
                        "fix": "Risk = probability of failure, Reliability = probability of success"
                    },
                    {
                        "mistake": "Thinking PDF values are probabilities",
                        "fix": "PDF = density (can be >1), CDF = actual probability (≤1)"
                    },
                    {
                        "mistake": "Misinterpreting GEV shape parameter sign",
                        "fix": "ξ > 0 = heavy tail (more extremes), ξ < 0 = light tail (bounded)"
                    }
                ]

                for i, error in enumerate(errors, 1):
                    with st.expander(f"**Error #{i}:** {error['mistake']}"):
                        st.error(f"**Mistake:** {error['mistake']}")
                        st.success(f"**Fix:** {error['fix']}")

            with col2:
                st.markdown("### ✅ Study Strategies")

                strategies = [
                    "**Practice Unit Conversions**: They appear frequently in water resources",
                    "**Understand Physical Meaning**: Don't just memorize formulas",
                    "**Check Answer Reasonableness**: Does your result make engineering sense?",
                    "**Master Key Relationships**: Risk + Reliability = 1, T = 1/P, etc.",
                    "**Practice Mental Math**: Basic calculations without calculator",
                    "**Read Questions Carefully**: Identify what's given vs. what's asked"
                ]

                for strategy in strategies:
                    st.markdown(f"💡 {strategy}")

                st.markdown("### 📚 Key Formulas to Memorize")

                formulas = {
                    "Weibull plotting position": "P = m/(n+1)",
                    "Return period": "T = 1/P",
                    "Lifetime risk": "R = 1-(1-P)ⁿ",
                    "Reliability": "Rel = (1-P)ⁿ",
                    "Intensity": "I = Depth/Time",
                    "Rational method": "Q = (1/3600) × C × I × A"
                }

                for concept, formula in formulas.items():
                    st.markdown(f"**{concept}:** `{formula}`")

            st.markdown("---")
            st.markdown("### 🧠 Exam Success Tips")

            col1, col2, col3 = UIComponents.three_column_layout()

            with col1:
                st.markdown("**⏰ Time Management**")
                tips_time = [
                    "Spend ~1 min per MCQ",
                    "Start with questions you know",
                    "Don't get stuck on one problem",
                    "Save 2-3 min to review"
                ]
                for tip in tips_time:
                    st.markdown(f"• {tip}")

            with col2:
                st.markdown("**🧮 Calculation Strategy**")
                tips_calc = [
                    "Show all work clearly",
                    "Check units at each step",
                    "Round sensibly (2-3 digits)",
                    "Verify final answers"
                ]
                for tip in tips_calc:
                    st.markdown(f"• {tip}")

            with col3:
                st.markdown("**🎯 General Strategy**")
                tips_general = [
                    "Read entire question first",
                    "Identify given vs. unknown",
                    "Choose appropriate formula",
                    "Double-check calculations"
                ]
                for tip in tips_general:
                    st.markdown(f"• {tip}")

            UIComponents.highlight_box("""
            **🏆 Final Advice:** The best way to improve is practice! Work through similar problems
            until the methods become automatic. Focus on understanding concepts, not just memorizing answers.
            """)

        return None

    def _slide_practice_problems(self) -> Optional[bool]:
        """Slide 6: Practice Problems & Next Steps"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## 🎯 Additional Practice & Next Steps")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 📝 Quick Practice Problems")

                with st.expander("**Practice Problem 1:** Weibull Analysis"):
                    st.markdown("""
                    **Problem:** From 30 years of annual maximum precipitation data,
                    the 7th largest event is 65 mm. Calculate the plotting position and return period.
                    """)

                    if st.button("Show Solution", key="practice1"):
                        st.success("""
                        **Solution:**
                        • P = m/(n+1) = 7/(30+1) = 7/31 = 0.226
                        • T = 1/P = 1/0.226 = 4.4 years

                        **Interpretation:** 65 mm event has ~4.4 year return period
                        """)

                with st.expander("**Practice Problem 2:** Risk Calculation"):
                    st.markdown("""
                    **Problem:** A bridge is designed for a 50-year flood with a 75-year design life.
                    What is the lifetime risk?
                    """)

                    if st.button("Show Solution", key="practice2"):
                        st.success("""
                        **Solution:**
                        • P = 1/50 = 0.02
                        • R = 1-(1-0.02)^75 = 1-(0.98)^75 = 1-0.223 = 0.777 or 77.7%

                        **Interpretation:** 77.7% chance of experiencing design flood during bridge life
                        """)

                with st.expander("**Practice Problem 3:** IDF Application"):
                    st.markdown("""
                    **Problem:** From IDF curves, a 25-year, 15-minute storm has intensity 120 mm/hr.
                    Calculate peak flow for 2.8 hectares with C = 0.80.
                    """)

                    if st.button("Show Solution", key="practice3"):
                        st.success("""
                        **Solution:**
                        • Given: I = 120 mm/hr, A = 2.8 hectares, C = 0.80
                        • Q = (1/3600) × C × I × A
                        • Q = (1/3600) × 0.80 × 120 × 2.8
                        • Q = (1/3600) × 268.8 = 0.075 m³/s = 75 L/s
                        """)

            with col2:
                st.markdown("### 🚀 Next Steps in Your Learning")

                st.markdown("**📚 Continue Learning:**")
                next_topics = [
                    "**Non-stationary analysis** - Accounting for climate change",
                    "**Multivariate analysis** - Joint probability of events",
                    "**Uncertainty quantification** - Confidence intervals",
                    "**Regional frequency analysis** - Using data from multiple sites",
                    "**Design storm modeling** - Creating synthetic rainfall events"
                ]

                for topic in next_topics:
                    st.markdown(f"• {topic}")

                st.markdown("**🔧 Practical Applications:**")
                applications = [
                    "Apply to your capstone project",
                    "Analyze local rainfall/flood data",
                    "Design drainage for real site",
                    "Compare different design standards",
                    "Study climate change impacts"
                ]

                for app in applications:
                    st.markdown(f"• {app}")

                st.markdown("**📖 Additional Resources:**")

                st.link_button(
                    "📊 NOAA Atlas 14 Precipitation Data",
                    "https://hdsc.nws.noaa.gov/hdsc/pfds/",
                    use_container_width=True
                )

                st.link_button(
                    "🌊 USGS WaterWatch - Real-time Data",
                    "https://waterwatch.usgs.gov/",
                    use_container_width=True
                )

                st.link_button(
                    "📚 HEC-HMS Modeling Software",
                    "https://www.hec.usace.army.mil/software/hec-hms/",
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown("### 🎓 Course Completion Reflection")

            UIComponents.highlight_box("""
            **🎉 Congratulations on completing the exam review!**

            You've now mastered:
            • Data exploration and frequency analysis
            • Probability theory for engineering applications
            • Risk and reliability assessment
            • Distribution fitting and selection
            • GEV analysis for extreme events
            • IDF curve development and application

            **These skills will serve you well in:**
            • Capstone projects and future coursework
            • Professional engineering practice
            • Graduate studies in water resources
            • Consulting and design work
            • Research and development

            **Keep practicing and stay curious about water resources engineering!**
            """)

            # Final quiz
            st.markdown("### 🔄 Quick Confidence Check")

            confidence_check = st.radio(
                "How confident do you feel about the exam topics after this review?",
                [
                    "😟 Still need more practice",
                    "😐 Somewhat confident",
                    "😊 Confident and ready",
                    "🤓 Very confident, could teach others"
                ]
            )

            if confidence_check:
                if "Still need" in confidence_check:
                    st.info("💡 Consider reviewing specific modules where you had difficulties. Practice more problems!")
                elif "Somewhat confident" in confidence_check:
                    st.success("👍 Good progress! A bit more practice will boost your confidence.")
                elif "Confident and ready" in confidence_check:
                    st.success("🎉 Excellent! You're well-prepared. Review the formulas one more time.")
                else:
                    st.success("🌟 Outstanding! Consider helping classmates who might need assistance.")

        return True  # Mark module as complete

    def render(self, show_header=False) -> bool:
        """Override render to add custom styling"""
        # Add custom CSS for answer key
        st.markdown("""
        <style>
        .answer-correct {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 10px;
            margin: 10px 0;
        }
        .answer-incorrect {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 10px;
            margin: 10px 0;
        }
        .solution-box {
            background-color: #e2f3ff;
            border: 1px solid #0066cc;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        return super().render(show_header)