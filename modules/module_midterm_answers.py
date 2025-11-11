"""
Module: Midterm Exam Answer Key & Solutions
Interactive answer key with detailed explanations for all sections

Author: TA Saurav Bhattarai
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

from base_classes import LearningModule, ModuleInfo, LearningObjective
from utilities import UIComponents

class ModuleMidtermAnswers(LearningModule):
    """Interactive Midterm Exam Answer Key"""

    def __init__(self):
        objectives = [
            LearningObjective("Review midterm exam solutions", "understand"),
            LearningObjective("Master key concepts through detailed explanations", "analyze"),
            LearningObjective("Learn from common mistakes", "apply")
        ]

        info = ModuleInfo(
            id="module_midterm_answers",
            title="📝 Midterm Exam Solutions",
            description="Complete answer key with detailed explanations for all sections",
            duration_minutes=40,
            prerequisites=["module_01", "module_02", "module_03", "module_04", "module_05"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=8
        )

        super().__init__(info)

    def get_slide_titles(self) -> List[str]:
        return [
            "Exam Overview",
            "Section A: MCQ (1-8)",
            "Section A: MCQ (9-15)",
            "Section B.1: IDF Curves",
            "Section B.2: Frequency Plot",
            "Section C: Numerical Problems",
            "Section D: Theoretical Questions",
            "Summary & Practice"
        ]

    def render_slide(self, slide_num: int) -> Optional[bool]:
        slides = [
            self._slide_overview,
            self._slide_mcq_1_8,
            self._slide_mcq_9_15,
            self._slide_idf_curves,
            self._slide_frequency_plot,
            self._slide_numerical,
            self._slide_theoretical,
            self._slide_summary
        ]

        if slide_num < len(slides):
            return slides[slide_num]()
        return False

    def _slide_overview(self) -> Optional[bool]:
        """Slide 1: Overview"""
        with UIComponents.slide_container("theory"):
            st.markdown("## 📝 Midterm Exam Answer Key")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 🎯 Exam Structure (100 marks)")
                UIComponents.highlight_box("""
                **Section A: Multiple Choice (30 marks)**
                - 15 questions × 2 marks each

                **Section B: Figure-Based (25 marks)**
                - B.1: IDF Curves (15 marks)
                - B.2: Frequency Plot (10 marks)

                **Section C: Numerical Problems (30 marks)**
                - Problem 1: Weibull Analysis (8 marks)
                - Problem 2: Risk Analysis (8 marks)
                - Problem 3: GEV Distribution (7 marks)
                - Problem 4: IDF Application (7 marks)

                **Section D: Short Answer (15 marks)**
                - 3 theoretical questions × 5 marks each
                """)

            with col2:
                st.markdown("### 📊 Time Management (60 minutes)")
                time_data = {
                    'Section': ['A: MCQ', 'B: Figures', 'C: Numerical', 'D: Theory', 'Review'],
                    'Time (min)': [15, 10, 20, 12, 3],
                    'Marks': [30, 25, 30, 15, '-']
                }
                st.dataframe(pd.DataFrame(time_data), use_container_width=True)

                UIComponents.highlight_box("""
                **💡 Usage Tips:**
                - Try questions first before viewing answers
                - Focus on understanding the reasoning
                - Review detailed explanations for mistakes
                - Practice similar problems for mastery
                """)

        return None

    def _slide_mcq_1_8(self) -> Optional[bool]:
        """Slide 2: MCQ 1-8"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Section A: Multiple Choice (1-8)")

            # Q1
            with st.expander("**Q1:** Weibull Plotting Position", expanded=True):
                st.markdown("**Question:** What does P = m/(n+1) estimate?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) Non-exceedance probability")
                    st.markdown("B) Exceedance probability ✅")
                    st.markdown("C) Mean return period")
                    st.markdown("D) CDF value")
                with col2:
                    st.success("**Answer: B**")
                    UIComponents.highlight_box("""
                    Weibull formula estimates **exceedance probability**:
                    - P = probability event is exceeded
                    - Used for ranking extreme events
                    - Return period T = 1/P
                    """)

            # Q2
            with st.expander("**Q2:** Probability vs Frequency"):
                st.markdown("**Question:** Which correctly distinguishes probability from frequency?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) Probability is observed; frequency is theoretical")
                    st.markdown("B) Frequency is count; probability is long-term expectation ✅")
                    st.markdown("C) They are identical")
                    st.markdown("D) Probability for continuous; frequency for discrete")
                with col2:
                    st.success("**Answer: B**")
                    UIComponents.highlight_box("""
                    **Key distinction:**
                    - **Frequency**: Observed count (e.g., 5 floods in 50 years)
                    - **Probability**: Long-term expectation (e.g., P = 0.1)
                    """)

            # Q3
            with st.expander("**Q3:** Lifetime Risk Calculation"):
                st.markdown("**Question:** Bridge designed for 100-year flood, 50-year life. Lifetime risk?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) 0.01 or 1%")
                    st.markdown("B) 0.39 or 39% ✅")
                    st.markdown("C) 0.50 or 50%")
                    st.markdown("D) 0.99 or 99%")
                with col2:
                    st.success("**Answer: B) 39%**")
                    UIComponents.highlight_box("""
                    **Solution:**
                    - P = 1/100 = 0.01
                    - R = 1-(1-P)^n = 1-(0.99)^50
                    - R = 1-0.605 = 0.395 ≈ 39%
                    """)

            # Q4
            with st.expander("**Q4:** PDF vs CDF"):
                st.markdown("**Question:** Key difference between PDF and CDF?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) PDF = density; CDF = cumulative probability ✅")
                    st.markdown("B) PDF for discrete; CDF for continuous")
                    st.markdown("C) PDF ≤ 1; CDF can exceed 1")
                    st.markdown("D) PDF integrates to ∞; CDF to 1")
                with col2:
                    st.success("**Answer: A**")
                    UIComponents.highlight_box("""
                    **Remember:**
                    - PDF: f(x) = probability density (can be > 1)
                    - CDF: F(x) = P(X ≤ x) (always ≤ 1)
                    - Area under PDF = probability
                    """)

            # Q5
            with st.expander("**Q5:** Exceedance Probability"):
                st.markdown("**Question:** To calculate P(X ≥ 75) for continuous variable, use:")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) PDF: f(75)")
                    st.markdown("B) CDF: F(75)")
                    st.markdown("C) Complement: 1 - F(75) ✅")
                    st.markdown("D) Integral to 75")
                with col2:
                    st.success("**Answer: C**")
                    UIComponents.highlight_box("""
                    **Exceedance probability:**
                    - P(X ≥ a) = 1 - F(a)
                    - F(a) = P(X ≤ a) = non-exceedance
                    - 1 - F(a) = exceedance probability
                    """)

            # Q6
            with st.expander("**Q6:** GEV Shape Parameter (+0.30)"):
                st.markdown("**Question:** GEV with ξ = +0.30 indicates:")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) Weibull - bounded extremes")
                    st.markdown("B) Gumbel - standard behavior")
                    st.markdown("C) Fréchet - heavy tail ✅")
                    st.markdown("D) Not suitable")
                with col2:
                    st.success("**Answer: C) Fréchet**")
                    UIComponents.highlight_box("""
                    **GEV shape parameter:**
                    - ξ > 0: Fréchet (heavy tail, more extremes)
                    - ξ = 0: Gumbel (standard)
                    - ξ < 0: Weibull (bounded)
                    **Design impact:** Be more conservative!
                    """)

            # Q7
            with st.expander("**Q7:** Gumbel Shape Parameter"):
                st.markdown("**Question:** Which ξ makes GEV = Gumbel?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) ξ < 0")
                    st.markdown("B) ξ = 0 ✅")
                    st.markdown("C) ξ > 0")
                    st.markdown("D) ξ = 1")
                with col2:
                    st.success("**Answer: B) ξ = 0**")
                    UIComponents.highlight_box("""
                    When shape parameter = 0:
                    - GEV reduces to Gumbel distribution
                    - Standard extreme value behavior
                    - Most common for flood/rainfall analysis
                    """)

            # Q8
            with st.expander("**Q8:** IDF Intensity vs Duration"):
                st.markdown("**Question:** Why does intensity decrease as duration increases?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) Longer storms produce less rainfall")
                    st.markdown("B) Same depth over longer time = lower rate ✅")
                    st.markdown("C) Measurement errors increase")
                    st.markdown("D) Shorter durations less accurate")
                with col2:
                    st.success("**Answer: B**")
                    UIComponents.highlight_box("""
                    **Physical explanation:**
                    - Intensity = Depth / Duration
                    - Maximum rainfall depths don't scale linearly
                    - Same total depth spread over more time
                    - Results in lower average rate (intensity)
                    """)

        return None

    def _slide_mcq_9_15(self) -> Optional[bool]:
        """Slide 3: MCQ 9-15"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Section A: Multiple Choice (9-15)")

            # Q9
            with st.expander("**Q9:** NOAA Temporal Scaling", expanded=True):
                st.markdown("**Question:** NOAA ratio 0.79 for 30-min. If 60-min = 50mm, what is 30-min?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) 39.5 mm ✅")
                    st.markdown("B) 50.0 mm")
                    st.markdown("C) 63.3 mm")
                    st.markdown("D) 29.0 mm")
                with col2:
                    st.success("**Answer: A) 39.5 mm**")
                    UIComponents.highlight_box("""
                    **Solution:**
                    P₃₀ = P₆₀ × Ratio
                    P₃₀ = 50 × 0.79 = 39.5 mm
                    """)

            # Q10
            with st.expander("**Q10:** Annual Exceedance Probability"):
                st.markdown("**Question:** For 10-year return period, annual exceedance probability?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) 0.01")
                    st.markdown("B) 0.10 ✅")
                    st.markdown("C) 0.90")
                    st.markdown("D) 1.00")
                with col2:
                    st.success("**Answer: B) 0.10**")
                    UIComponents.highlight_box("""
                    P = 1/T = 1/10 = 0.10 or 10%
                    """)

            # Q11
            with st.expander("**Q11:** Risk and Reliability"):
                st.markdown("**Question:** If risk over 25 years = 0.40, what is reliability?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) 0.40")
                    st.markdown("B) 0.60 ✅")
                    st.markdown("C) 0.25")
                    st.markdown("D) Cannot determine")
                with col2:
                    st.success("**Answer: B) 0.60**")
                    UIComponents.highlight_box("""
                    Reliability = 1 - Risk
                    Rel = 1 - 0.40 = 0.60 or 60%
                    """)

            # Q12
            with st.expander("**Q12:** US Flood Frequency Standard"):
                st.markdown("**Question:** Which distribution per Bulletin 17C?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) Normal")
                    st.markdown("B) Gumbel")
                    st.markdown("C) Log-Pearson Type III ✅")
                    st.markdown("D) Exponential")
                with col2:
                    st.success("**Answer: C**")
                    UIComponents.highlight_box("""
                    **Bulletin 17C (USGS):**
                    - Standard: Log-Pearson Type III
                    - Most used in US for flood frequency
                    - Handles skewed data well
                    """)

            # Q13
            with st.expander("**Q13:** 100-Year Flood Misconception"):
                st.markdown("**Question:** 100-year flood occurs twice in 5 years. Which is correct?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) No longer 100-year flood")
                    st.markdown("B) Violates probability theory")
                    st.markdown("C) Natural variability; each year 1% independent ✅")
                    st.markdown("D) Cannot occur for 95 years")
                with col2:
                    st.success("**Answer: C**")
                    UIComponents.highlight_box("""
                    **Key concept:**
                    - 100-year = 1% annual probability
                    - Each year is independent
                    - Short-term clustering is normal randomness
                    - Past events don't affect future probability
                    """)

            # Q14
            with st.expander("**Q14:** Critical Infrastructure Design"):
                st.markdown("**Question:** Why longer return periods for critical infrastructure?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) Reduce construction costs")
                    st.markdown("B) Minimize failure probability, increase safety ✅")
                    st.markdown("C) Unlimited budgets")
                    st.markdown("D) Insurance requirements only")
                with col2:
                    st.success("**Answer: B**")
                    UIComponents.highlight_box("""
                    **Design philosophy:**
                    - Longer T → Lower P → Higher safety
                    - Hospitals: 500+ year events
                    - Residential: 10-25 years
                    - Higher consequences = lower acceptable risk
                    """)

            # Q15
            with st.expander("**Q15:** IDF 'I' Meaning"):
                st.markdown("**Question:** In IDF curves, what does 'I' represent?")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("A) Total rainfall depth (mm)")
                    st.markdown("B) Rainfall intensity (mm/hr) ✅")
                    st.markdown("C) Infrastructure design standard")
                    st.markdown("D) Infiltration rate")
                with col2:
                    st.success("**Answer: B**")
                    UIComponents.highlight_box("""
                    **IDF components:**
                    - **I**: Intensity (mm/hr) - rainfall rate
                    - **D**: Duration (min/hr)
                    - **F**: Frequency (return period)
                    """)

            st.markdown("---")
            st.markdown("### ✅ Section A Answers: B B B A C C B B A B B C C B B")

        return None

    def _slide_idf_curves(self) -> Optional[bool]:
        """Slide 4: IDF Curve Problems"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Section B.1: IDF Curve Analysis (15 marks)")

            # Q1
            with st.expander("**Question 1 (3 marks):** Reading IDF Curve", expanded=True):
                st.markdown("**Find intensity for 25-year, 30-minute storm**")
                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("**From graph:**")
                    st.markdown("- Return period: 25-year (purple line)")
                    st.markdown("- Duration: 30 minutes")
                    st.markdown("- Read intersection point")
                with col2:
                    st.success("**Answer: I = 64 mm/hr**")
                    UIComponents.highlight_box("""
                    **How to read:**
                    1. Find 25-year curve (purple)
                    2. Locate 30-min on x-axis
                    3. Read y-value at intersection
                    **Result: 64 mm/hr**
                    """)

            # Q2
            with st.expander("**Question 2 (5 marks):** Rational Method Calculation"):
                st.markdown("**Given:** A = 1.8 ha, C = 0.80, Tc = 15 min, T = 10 year")
                st.markdown("**Find:** Peak discharge Q (L/s)")

                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("**Step 1:** Read intensity from IDF")
                    st.code("I = 73 mm/hr (10-year, 15-min)")

                    st.markdown("**Step 2:** Convert area")
                    st.code("A = 1.8 ha = 18,000 m²")

                    st.markdown("**Step 3:** Apply rational method")
                    st.code("""Q = (1/3600) × C × I × A
Q = (1/3600) × 0.80 × 73 × 18000
Q = (1/3600) × 1,051,200
Q = 292 L/s""")

                with col2:
                    st.success("**Answer: Q = 292 L/s**")
                    UIComponents.highlight_box("""
                    **Marking:**
                    - Read I correctly (1 mark)
                    - Convert area (1 mark)
                    - Formula setup (1 mark)
                    - Calculation (1 mark)
                    - Final answer (1 mark)
                    """)

            # Q3
            with st.expander("**Question 3 (3 marks):** Percentage Increase"):
                st.markdown("**Compare 15-min storm: 10-year vs 100-year**")

                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("**From IDF curves:**")
                    st.code("""10-year, 15-min: I = 73 mm/hr
100-year, 15-min: I = 103 mm/hr

Increase = (103-73)/73 × 100%
         = 30/73 × 100%
         = 41.1%""")

                with col2:
                    st.success("**Answer: 41.1% increase**")
                    UIComponents.highlight_box("""
                    **Interpretation:**
                    - 100-year event is 41% more intense
                    - Significant increase for safety margin
                    - Important for infrastructure design
                    """)

            # Q4
            with st.expander("**Question 4 (4 marks):** Physical Explanation"):
                st.markdown("**Why does intensity decrease with duration?**")

                st.success("**Answer:**")
                UIComponents.highlight_box("""
                **Physical explanation:**

                As storm duration increases, rainfall intensity decreases because:

                1. **Storm mechanics**: Short-duration storms are typically from intense
                   convective cells that cannot sustain high rates for extended periods

                2. **Water supply limitation**: Atmosphere can deliver maximum moisture
                   at high rates for brief periods, but sustained delivery requires
                   lower average rates

                3. **Mathematical relationship**: I = Depth/Duration. While total depth
                   increases with duration, it doesn't increase proportionally, resulting
                   in lower average intensity

                4. **Practical observation**: A 5-minute cloudburst can be extremely
                   intense, but such intensity cannot be maintained for hours

                **Marking:**
                - Physical explanation (2 marks)
                - Storm characteristics (1 mark)
                - Mathematical reasoning (1 mark)
                """)

        return None

    def _slide_frequency_plot(self) -> Optional[bool]:
        """Slide 5: Frequency Plot Analysis"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Section B.2: Flood Frequency Plot (10 marks)")

            with st.expander("**Complete Question (10 marks)**", expanded=True):
                st.markdown("**Given:** Gumbel fit Q = 300 + 85×ln(T)")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**(a) 25-year design flood (2 marks)**")
                    st.code("""Q₂₅ = 573.6 m³/s""")

                    st.markdown("**(b) 100-year design flood (2 marks)**")
                    st.code("""Q₁₀₀ =  691 m³/s""")

                    st.markdown("**(c) Factor of safety (2 marks)**")
                    st.code("""FOS = Q₁₀₀/Q₂₅
    = 691/574
    = 1.20""")

                with col2:
                    st.success("**Answers:**")
                    st.markdown("- **(a)** Q₂₅ = 574 m³/s")
                    st.markdown("- **(b)** Q₁₀₀ = 691 m³/s")
                    st.markdown("- **(c)** FOS = 1.20")
                    st.markdown("- **(d)** See below")

                    UIComponents.highlight_box("""
                    **Interpretation:**
                    100-year flood is 20% larger than 25-year.
                    This provides safety margin for critical structures.
                    """)

                st.markdown("**(d) Goodness of fit assessment (4 marks)**")
                UIComponents.highlight_box("""
                **Assessment of Gumbel fit quality:**

                The Gumbel distribution provides a **bad fit** to the observed data:

                The observed data points are not aligned with the fitted line.

                The slope of the fitted line is not the same as the slope of the observed data points.
                """)

        return None

    def _slide_numerical(self) -> Optional[bool]:
        """Slide 6: Numerical Problems"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Section C: Numerical Problems (30 marks)")

            # Problem 1
            with st.expander("**Problem 1:** Weibull Analysis (8 marks)", expanded=True):
                st.markdown("**Given:** n=30 years, m=4 (4th largest), rainfall=85mm")

                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("**(a) Plotting position (3 marks)**")
                    st.code("""P = m/(n+1)
  = 4/(30+1)
  = 4/31
  = 0.129 or 12.9%""")

                    st.markdown("**(b) Return period (2 marks)**")
                    st.code("""T = 1/P
  = 1/0.129
  = 7.75 years""")

                    st.markdown("**(c) Annual exceedance (2 marks)**")
                    st.code("P = 0.129 or 12.9% (same as plotting position)")

                with col2:
                    st.success("**Answers:**")
                    st.markdown("- **(a)** P = 0.129")
                    st.markdown("- **(b)** T = 7.75 years")
                    st.markdown("- **(c)** P = 12.9%")
                    st.markdown("- **(d)** See explanation")

                st.markdown("**(d) Practical meaning (1 mark)**")
                UIComponents.highlight_box("""
                An 85mm rainfall event has approximately 13% chance of occurring
                in any given year. On average, such events occur about once every
                7-8 years. Suitable for moderate infrastructure design (residential
                drainage, small culverts).
                """)

            # Problem 2
            with st.expander("**Problem 2:** Risk & Reliability (8 marks)"):
                st.markdown("**Given:** T=50 years, n=30 years, acceptable risk ≤ 30%")

                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("**(a) Annual probability (2 marks)**")
                    st.code("P = 1/T = 1/50 = 0.02 or 2%")

                    st.markdown("**(b) Lifetime risk (3 marks)**")
                    st.code("""R = 1-(1-P)^n
  = 1-(1-0.02)^30
  = 1-(0.98)^30
  = 1-0.545
  = 0.455 or 45.5%""")

                    st.markdown("**(c) Reliability (2 marks)**")
                    st.code("""Rel = (1-P)^n = 0.545 or 54.5%
or: Rel = 1-R = 1-0.455 = 0.545""")

                with col2:
                    st.success("**Answers:**")
                    st.markdown("- **(a)** P = 0.02 or 2%")
                    st.markdown("- **(b)** R = 0.455 or 45.5%")
                    st.markdown("- **(c)** Rel = 0.545 or 54.5%")
                    st.markdown("- **(d)** FAILS criterion")

                    st.markdown("**(d) Design adequacy (1 mark)**")
                    st.error("""**DOES NOT meet criterion**

Calculated: 45.5% > Acceptable: 30%

Design is inadequate. Recommend
100-year design standard instead.""")

            # Problem 3
            with st.expander("**Problem 3:** GEV Distribution (7 marks)"):
                st.markdown("**Given:** ξ=+0.20, μ=450 m³/s, σ=120 m³/s")

                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("**(a) Distribution type (2 marks)**")
                    st.success("**Fréchet (Type II)**")
                    st.markdown("Since ξ = +0.20 > 0")

                    st.markdown("**(b) Tail behavior (2 marks)**")
                    UIComponents.highlight_box("""
                    Positive shape parameter indicates **heavy-tailed** distribution:
                    - More extreme events are possible
                    - Probability of very large floods is higher
                    - Upper tail decays more slowly than Gumbel
                    """)

                with col2:
                    st.markdown("**(c) Design implications (3 marks)**")
                    UIComponents.highlight_box("""
                    **Engineering implications:**

                    Should be **MORE conservative** in design:

                    1. **Higher design values**: Extreme events
                       larger than Gumbel would predict

                    2. **Larger safety factors**: Account for
                       heavy tail behavior

                    3. **Conservative extrapolation**: Predictions
                       beyond data range are less certain

                    4. **Risk assessment**: Consider consequences
                       of underestimating extreme events

                    Fréchet suggests potential for rare but
                    extremely large floods.
                    """)

            # Problem 4
            with st.expander("**Problem 4:** IDF Disaggregation (7 marks)"):
                st.markdown("**Given:** P₆₀ = 85mm (50-year), NOAA ratios provided")

                col1, col2 = UIComponents.two_column_layout()
                with col1:
                    st.markdown("**(a) 15-min depth (2 marks)**")
                    st.code("""P₁₅ = P₆₀ × Ratio₁₅
    = 85 × 0.57
    = 48.45 mm""")

                    st.markdown("**(b) 15-min intensity (2 marks)**")
                    st.code("""I₁₅ = P₁₅/(15/60)
    = 48.45/0.25
    = 193.8 mm/hr""")

                    st.markdown("**(c) 30-min depth (2 marks)**")
                    st.code("""P₃₀ = P₆₀ × Ratio₃₀
    = 85 × 0.79
    = 67.15 mm""")

                with col2:
                    st.success("**Answers:**")
                    st.markdown("- **(a)** P₁₅ = 48.45 mm")
                    st.markdown("- **(b)** I₁₅ = 193.8 mm/hr")
                    st.markdown("- **(c)** P₃₀ = 67.15 mm")
                    st.markdown("- **(d)** See explanation")

                    st.markdown("**30-min intensity:**")
                    st.code("""I₃₀ = 67.15/(30/60)
    = 67.15/0.5
    = 134.3 mm/hr""")

                    st.markdown("**(d) Which has higher intensity? (1 mark)**")
                    UIComponents.highlight_box("""
                    **15-minute has higher intensity (193.8 > 134.3)**

                    Shorter durations always have higher intensities
                    because the same (or similar) rainfall depth is
                    concentrated over a shorter time period.
                    """)

        return None

    def _slide_theoretical(self) -> Optional[bool]:
        """Slide 7: Theoretical Questions"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Section D: Short Answer Questions (15 marks)")

            st.markdown("### 📝 For detailed answers to all theoretical questions, please refer to:")
            
            st.markdown("#### 🔗 [**Midterm Exam Solutions Document**](https://docs.google.com/document/d/1lPA3819mLV1jhK6rz-jeXszTl7uPMe0l/edit?usp=sharing&ouid=102407295970298031075&rtpof=true&sd=true)")
            
            UIComponents.highlight_box("""
            This document contains complete solutions for:
            - **Question 1:** Hydrological & Hydrodynamic Modeling (5 marks)
            - **Question 2:** Dam and Reservoir Design (5 marks) 
            - **Question 3:** Open Channel Flow Equations (5 marks)
            """)

        return None

    def _slide_summary(self) -> Optional[bool]:
        """Slide 8: Summary & Practice"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## 📊 Exam Summary & Additional Practice")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### ✅ Complete Answer Key")

                st.markdown("**Section A (MCQ):**")
                st.code("""1.B  2.B  3.B  4.A  5.C
6.C  7.B  8.B  9.A  10.B
11.B 12.C 13.C 14.B 15.B""")

                st.markdown("**Section B.1 (IDF):**")
                st.markdown("- Q1: 64 mm/hr")
                st.markdown("- Q2: 292 L/s")
                st.markdown("- Q3: 41.1% increase")
                st.markdown("- Q4: Physical explanation provided")

                st.markdown("**Section B.2 (Frequency):**")
                st.markdown("- (a) 574 m³/s")
                st.markdown("- (b) 691 m³/s")
                st.markdown("- (c) FOS = 1.20")
                st.markdown("- (d) Good fit assessment")

                st.markdown("**Section C (Numerical):**")
                st.markdown("- P1: P=0.129, T=7.75 yr")
                st.markdown("- P2: R=45.5%, Fails")
                st.markdown("- P3: Fréchet, be conservative")
                st.markdown("- P4: 48.45mm, 193.8 mm/hr")

            with col2:
                st.markdown("### 🎯 Key Formulas")
                formulas = {
                    "Weibull": "P = m/(n+1)",
                    "Return period": "T = 1/P",
                    "Lifetime risk": "R = 1-(1-P)ⁿ",
                    "Reliability": "Rel = (1-P)ⁿ = 1-R",
                    "Intensity": "I = Depth/Duration × 60",
                    "Rational method": "Q = (1/3600)×C×I×A",
                    "NOAA scaling": "Pₜ = P₆₀ × Ratio",
                    "Gumbel": "Q = μ + σ×ln(T)"
                }
                for name, formula in formulas.items():
                    st.markdown(f"**{name}:** `{formula}`")

            st.markdown("---")
            st.markdown("### 🚀 Study Tips")

            col1, col2, col3 = UIComponents.three_column_layout()

            with col1:
                st.markdown("**Common Mistakes:**")
                st.markdown("- Forgetting (n+1) in Weibull")
                st.markdown("- Wrong units in rational method")
                st.markdown("- Confusing risk vs reliability")
                st.markdown("- Misreading IDF graphs")

            with col2:
                st.markdown("**Time Management:**")
                st.markdown("- 1 min per MCQ")
                st.markdown("- 5-7 min per numerical")
                st.markdown("- Don't get stuck")
                st.markdown("- Review at end")

            with col3:
                st.markdown("**Success Strategy:**")
                st.markdown("- Show all work")
                st.markdown("- Check units")
                st.markdown("- Verify reasonableness")
                st.markdown("- Practice similar problems")

            st.markdown("---")

            UIComponents.highlight_box("""
            **🎓 Final Advice:**

            - Understand concepts, don't just memorize
            - Practice is key to mastery
            - Review formulas regularly
            - Focus on areas where you struggled
            - Ask questions if anything is unclear

            **Good luck with your studies! 🌟**
            """)

        return True

    def render(self, show_header=False) -> bool:
        """Override render to add custom styling"""
        st.markdown("""
        <style>
        .solution-box {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

        return super().render(show_header)
