"""
Module 8: Spatio-Temporal Analysis in Water Resources
Trend Analysis, Break Point Tests, and Mapping for Engineering Applications

Author: TA Saurav Bhattarai
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
import datetime

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents

class Module08_SpatioTemporal(LearningModule):
    """Module 8: Spatio-Temporal Analysis in Water Resources"""

    def __init__(self):
        objectives = [
            LearningObjective("Understand spatio-temporal analysis in water resources", "understand"),
            LearningObjective("Apply trend analysis to hydrologic data", "apply"),
            LearningObjective("Detect change points in water resource systems", "analyze"),
            LearningObjective("Create spatio-temporal maps for engineering decisions", "create")
        ]

        info = ModuleInfo(
            id="module_08",
            title="Spatio-Temporal Analysis in Water Resources",
            description="Trend analysis, change point detection, and mapping for water resources engineering",
            duration_minutes=35,
            prerequisites=["module_01"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=9
        )

        super().__init__(info)

    def get_slide_titles(self) -> List[str]:
        return [
            "Why Do We Need Spatio-Temporal Analysis?",
            "How Do We Find Trends in Water Data?",
            "When Should We Use Trend Analysis?",
            "How Do We Detect Sudden Changes?",
            "Which Test Should We Use?",
            "How Do We Use Change Points in Design?",
            "How Do We Create Useful Maps?",
            "What Are Real Engineering Examples?",
            "How Do We Apply This in Practice?"
        ]

    def render_slide(self, slide_num: int) -> Optional[bool]:
        slides = [
            self._slide_intro,
            self._slide_trend_analysis,
            self._slide_discussion_trends,
            self._slide_breakpoint_tests,
            self._slide_change_detection,
            self._slide_discussion_change,
            self._slide_spatiotemporal_mapping,
            self._slide_engineering_applications,
            self._slide_discussion_applications
        ]

        if slide_num < len(slides):
            return slides[slide_num]()
        return False

    def _slide_intro(self) -> Optional[bool]:
        """Slide 1: Why Do We Need Spatio-Temporal Analysis?"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Why Do We Need Spatio-Temporal Analysis?")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### ❓ Engineering Question:")
                st.markdown("**Your city wants to build a new bridge. Should you use 50 years of flood data or just the recent 20 years?**")

                st.markdown("### 💡 The Answer:")
                st.markdown("**It depends!** If flood patterns changed 25 years ago, using all 50 years gives wrong answers.")

                st.markdown("### 🎯 What We Actually Do:")
                practices = [
                    "**Check if patterns changed over time** (trend analysis)",
                    "**Find when changes happened** (change point tests)",
                    "**See where changes occurred** (spatial mapping)",
                    "**Design using correct data** (engineering decisions)"
                ]

                for practice in practices:
                    st.markdown(f"• {practice}")

            with col2:
                st.markdown("### 📊 Real Example: Why This Matters")

                # Create sample spatio-temporal data
                years = list(range(1990, 2025))
                locations = ['Upstream', 'Midstream', 'Downstream']

                # Simulate rainfall trends
                np.random.seed(42)
                rainfall_data = []

                for i, year in enumerate(years):
                    # Upstream: increasing trend
                    upstream = 800 + 2*i + np.random.normal(0, 50)
                    # Midstream: stable
                    midstream = 900 + np.random.normal(0, 40)
                    # Downstream: decreasing trend
                    downstream = 1000 - 1.5*i + np.random.normal(0, 45)

                    rainfall_data.extend([
                        {'Year': year, 'Location': 'Upstream', 'Rainfall': max(600, upstream)},
                        {'Year': year, 'Location': 'Midstream', 'Rainfall': max(700, midstream)},
                        {'Year': year, 'Location': 'Downstream', 'Rainfall': max(750, downstream)}
                    ])

                df = pd.DataFrame(rainfall_data)

                fig = px.line(df, x='Year', y='Rainfall', color='Location',
                             title='Watershed Rainfall: Different Locations, Different Stories')
                fig.update_layout(height=350)
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                st.success("**Engineering Impact:** Upstream needs bigger drains, downstream needs less!")

            st.markdown("### ✅ Three Questions We Answer:")

            col1, col2, col3 = UIComponents.three_column_layout()

            with col1:
                st.markdown("#### ❓ Is it changing?")
                st.markdown("• Are floods getting bigger?")
                st.markdown("• Is rainfall increasing?")
                st.markdown("• **Tool: Trend Analysis**")

            with col2:
                st.markdown("#### ❓ When did it change?")
                st.markdown("• Dam built in 1995?")
                st.markdown("• Climate shift in 2000?")
                st.markdown("• **Tool: Change Point Tests**")

            with col3:
                st.markdown("#### ❓ Where is it happening?")
                st.markdown("• Which neighborhoods?")
                st.markdown("• Upstream or downstream?")
                st.markdown("• **Tool: Spatial Maps**")

        return None

    def _slide_trend_analysis(self) -> Optional[bool]:
        """Slide 2: Trend Analysis in Hydrology"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Trend Analysis in Hydrology")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 📊 What is Trend Analysis?")

                UIComponents.highlight_box("""
                **Purpose:** Detect if hydrologic variables are systematically increasing or decreasing over time.

                **Common Variables:**
                • Annual rainfall
                • Peak flood flows
                • Drought indices
                • Temperature patterns
                """)

                st.markdown("### 🧮 Mann-Kendall Test")
                st.markdown("**Most common method in hydrology:**")

                st.markdown("**Step 1:** Calculate trend statistic (S)")
                st.markdown("**Step 2:** Compute p-value")
                st.markdown("**Step 3:** Interpret results")

                UIComponents.formula_display("H₀: No trend vs H₁: Trend exists", "Hypothesis Testing")

                st.markdown("**Interpretation:**")
                st.markdown("• p < 0.05: **Significant trend**")
                st.markdown("• p ≥ 0.05: **No significant trend**")

            with col2:
                st.markdown("### 🔧 Interactive Trend Example")

                # Interactive controls
                trend_strength = st.slider("Trend Strength:", -3, 3, 1, 1)
                noise_level = st.slider("Data Noise:", 0.1, 2.0, 0.5, 0.1)

                # Generate synthetic data
                np.random.seed(42)
                years = np.arange(1990, 2025)
                n_years = len(years)

                # Create trend with noise
                base_flow = 100
                trend_component = trend_strength * np.arange(n_years)
                noise = np.random.normal(0, noise_level * 20, n_years)
                annual_flow = base_flow + trend_component + noise

                # Create dataframe
                flow_df = pd.DataFrame({
                    'Year': years,
                    'Annual_Flow': annual_flow
                })

                # Plot
                fig = go.Figure()

                # Data points
                fig.add_trace(go.Scatter(
                    x=flow_df['Year'],
                    y=flow_df['Annual_Flow'],
                    mode='markers+lines',
                    name='Annual Flow',
                    line=dict(color='blue', width=2)
                ))

                # Trend line
                z = np.polyfit(years, annual_flow, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=years,
                    y=p(years),
                    mode='lines',
                    name=f'Trend Line (slope={z[0]:.2f})',
                    line=dict(color='red', width=3, dash='dash')
                ))

                fig.update_layout(
                    title="Stream Flow Trend Analysis",
                    xaxis_title="Year",
                    yaxis_title="Annual Flow (m³/s)",
                    height=350
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Simple trend interpretation
                if abs(trend_strength) >= 2:
                    if trend_strength > 0:
                        st.success("🔺 **Strong Increasing Trend** - Flow is significantly rising")
                    else:
                        st.error("🔻 **Strong Decreasing Trend** - Flow is significantly declining")
                elif abs(trend_strength) >= 1:
                    if trend_strength > 0:
                        st.info("📈 **Moderate Increasing Trend** - Flow is rising")
                    else:
                        st.warning("📉 **Moderate Decreasing Trend** - Flow is declining")
                else:
                    st.info("➡️ **No Clear Trend** - Flow appears stable")

            st.markdown("### 🎯 Engineering Implications")

            implications = [
                "**Infrastructure Sizing** - Increasing flows may require larger pipes/channels",
                "**Flood Risk** - Rising trends indicate higher future flood risk",
                "**Water Supply** - Decreasing trends may affect water availability",
                "**Design Standards** - May need to update design criteria"
            ]

            for imp in implications:
                st.markdown(f"• {imp}")

        return None

    def _slide_discussion_trends(self) -> Optional[bool]:
        """Slide 3: Discussion - Trend Applications"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Discussion: Trend Analysis Applications")

            result = QuizEngine.create_multiple_choice(
                "You're designing a new bridge over a river. Historical flow data shows a significant increasing trend (p = 0.02) in annual peak flows over the past 30 years. What should you do?",
                [
                    "Use historical average flows for design - trends don't matter for infrastructure",
                    "Design for higher flows than historical average to account for the trend",
                    "Ignore the trend since 30 years isn't enough data",
                    "Use only the most recent 5 years of data"
                ],
                1,
                {
                    "correct": "Excellent! An increasing trend means future flows will likely be higher than historical averages. Conservative engineering practice requires designing for these higher expected flows to ensure safety.",
                    "incorrect": "Think about engineering safety. If flows are increasing over time, using only historical averages or ignoring the trend could lead to undersized infrastructure. We need to account for future conditions in our design."
                },
                f"{self.info.id}_trends_quiz"
            )

            if result is True:
                st.markdown("---")
                st.markdown("### 💡 Key Engineering Insights")

                insights = [
                    "**Conservative Design:** Always design for future conditions, not just past",
                    "**Climate Change:** Many trends are related to changing climate patterns",
                    "**Safety Factor:** Increasing trends require additional safety margins",
                    "**Economic Impact:** Under-designed infrastructure is more expensive to replace"
                ]

                for insight in insights:
                    st.markdown(f"• {insight}")

            return None

    def _slide_breakpoint_tests(self) -> Optional[bool]:
        """Slide 4: Break Point Tests - Detecting Changes"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Break Point Tests - Detecting Changes")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 🔍 What are Break Points?")

                st.markdown("**Most common: Pettitt Test** - finds the year when patterns suddenly shifted.")

                st.markdown("### 🎯 Real Examples:")
                examples = [
                    "**Dam built in 1995** → Lower peak flows after 1995",
                    "**Shopping mall built** → Higher runoff after construction",
                    "**Climate shift in 2000** → More intense storms after 2000",
                    "**Forest fire** → Changed watershed response"
                ]

                for example in examples:
                    st.markdown(f"• {example}")

                st.markdown("### 📊 How to Read Results:")
                st.markdown("• **p < 0.05:** Change is real")
                st.markdown("• **p ≥ 0.05:** No clear change point")
                st.markdown("• **Year:** When change happened")

            with col2:
                st.markdown("### 📊 Break Point Example")

                # Create example with clear break point
                years = np.arange(1970, 2020)
                np.random.seed(123)

                # Before break point (1970-1995): lower mean
                period1 = np.random.normal(50, 8, 26)  # 1970-1995
                # After break point (1996-2019): higher mean
                period2 = np.random.normal(75, 8, 24)  # 1996-2019

                annual_rainfall = np.concatenate([period1, period2])

                # Create dataframe
                rainfall_df = pd.DataFrame({
                    'Year': years,
                    'Rainfall': annual_rainfall,
                    'Period': ['Before 1996' if y < 1996 else 'After 1996' for y in years]
                })

                # Plot with break point highlighted
                fig = go.Figure()

                # Before break point
                before_data = rainfall_df[rainfall_df['Period'] == 'Before 1996']
                fig.add_trace(go.Scatter(
                    x=before_data['Year'],
                    y=before_data['Rainfall'],
                    mode='markers+lines',
                    name='Before 1996',
                    line=dict(color='blue', width=2)
                ))

                # After break point
                after_data = rainfall_df[rainfall_df['Period'] == 'After 1996']
                fig.add_trace(go.Scatter(
                    x=after_data['Year'],
                    y=after_data['Rainfall'],
                    mode='markers+lines',
                    name='After 1996',
                    line=dict(color='red', width=2)
                ))

                # Add vertical line at break point
                fig.add_vline(x=1996, line_dash="dash", line_color="orange", line_width=3)
                fig.add_annotation(x=1996, y=85, text="Break Point<br>1996", showarrow=True)

                # Add means
                fig.add_hline(y=period1.mean(), line_dash="dot", line_color="blue",
                             annotation_text=f"Mean Before: {period1.mean():.1f} mm")
                fig.add_hline(y=period2.mean(), line_dash="dot", line_color="red",
                             annotation_text=f"Mean After: {period2.mean():.1f} mm")

                fig.update_layout(
                    title="Break Point Detection Example",
                    xaxis_title="Year",
                    yaxis_title="Annual Rainfall (mm)",
                    height=400
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                st.success(f"**Change Detected:** Mean increased by {period2.mean() - period1.mean():.1f} mm in 1996")

            st.markdown("### ✅ What Do We Do with Change Points?")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("**Before Change Point:**")
                st.markdown("• Shows old conditions")
                st.markdown("• Good for historical analysis")
                st.markdown("• Don't use for new designs")

            with col2:
                st.markdown("**After Change Point:**")
                st.markdown("• Shows current conditions")
                st.markdown("• Use for new infrastructure")
                st.markdown("• Update risk assessments")

            st.warning("⚠️ **Never mix data from before and after a change point - you'll get wrong answers!**")

        return None

    def _slide_change_detection(self) -> Optional[bool]:
        """Slide 5: Change Point Detection Methods"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Change Point Detection Methods")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 🧪 Pettitt Test")

                st.markdown("**Why it's popular:**")
                st.markdown("• Works with any type of water data")
                st.markdown("• Finds the most likely year of change")
                st.markdown("• Tells you if change is real (p-value)")
                st.markdown("• No complicated assumptions")

                st.markdown("**Interpretation:**")
                st.markdown("• p < 0.05: **Significant change point**")
                st.markdown("• p ≥ 0.05: **No significant change**")

                st.markdown("### 🎯 When to Use")
                use_cases = [
                    "**Dam construction** - Before/after flow analysis",
                    "**Urbanization** - Runoff pattern changes",
                    "**Climate events** - El Niño/La Niña effects",
                    "**Land use change** - Agricultural to urban"
                ]

                for case in use_cases:
                    st.markdown(f"• {case}")

            with col2:
                st.markdown("### 🔧 Interactive Change Detection")

                # Controls for demonstration
                change_year = st.slider("Change Year:", 1985, 2005, 1995)
                change_magnitude = st.slider("Change Magnitude:", 0, 30, 15)

                # Generate data with user-defined change point
                years = np.arange(1975, 2015)
                np.random.seed(456)

                rainfall = []
                for year in years:
                    if year < change_year:
                        rainfall.append(100 + np.random.normal(0, 10))
                    else:
                        rainfall.append(100 + change_magnitude + np.random.normal(0, 10))

                # Create simple visualization
                fig = go.Figure()

                # Plot data
                colors = ['blue' if y < change_year else 'red' for y in years]
                fig.add_trace(go.Scatter(
                    x=years,
                    y=rainfall,
                    mode='markers+lines',
                    name='Annual Data',
                    marker=dict(color=colors),
                    line=dict(color='gray', width=1)
                ))

                # Add change point line
                fig.add_vline(x=change_year, line_dash="dash", line_color="orange", line_width=3)

                fig.update_layout(
                    title=f"Change Point at {change_year}",
                    xaxis_title="Year",
                    yaxis_title="Rainfall (mm)",
                    height=350
                )
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Calculate basic statistics
                before_data = [r for i, r in enumerate(rainfall) if years[i] < change_year]
                after_data = [r for i, r in enumerate(rainfall) if years[i] >= change_year]

                st.markdown("**Results:**")
                st.markdown(f"• Before {change_year}: {np.mean(before_data):.1f} ± {np.std(before_data):.1f} mm")
                st.markdown(f"• After {change_year}: {np.mean(after_data):.1f} ± {np.std(after_data):.1f} mm")
                st.markdown(f"• **Change:** {np.mean(after_data) - np.mean(before_data):.1f} mm")

            st.markdown("### ⚠️ Engineering Cautions")

            cautions = [
                "**False Positives** - Random variations can look like change points",
                "**Multiple Changes** - Some methods only detect one change",
                "**Gradual Changes** - These tests work best for sudden changes",
                "**Data Quality** - Bad data can create false change points"
            ]

            for caution in cautions:
                st.markdown(f"• {caution}")

        return None

    def _slide_discussion_change(self) -> Optional[bool]:
        """Slide 6: Discussion - Change Detection"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Discussion: Change Point Detection")

            result = QuizEngine.create_multiple_choice(
                "A Pettitt test on 50 years of flood data shows a significant change point (p = 0.01) in 1998. You're designing flood protection for a new development. Which data should you primarily use?",
                [
                    "All 50 years of data to get the most robust statistics",
                    "Only data after 1998 since the flood regime has changed",
                    "Only data before 1998 to be conservative",
                    "Just the most recent 10 years regardless of the change point"
                ],
                1,
                {
                    "correct": "Correct! A significant change point means the flood regime fundamentally changed in 1998. Using only post-1998 data ensures your design reflects current conditions rather than outdated historical patterns.",
                    "incorrect": "Think about what a change point means. If flood patterns changed significantly in 1998, mixing old and new data gives an average that doesn't represent either period well. For current design, use data that represents current conditions."
                },
                f"{self.info.id}_change_quiz"
            )

            if result is True:
                st.markdown("---")
                st.markdown("### 🏗️ Engineering Applications")

                col1, col2 = UIComponents.two_column_layout()

                with col1:
                    st.markdown("**Flood Management:**")
                    st.markdown("• Update flood maps after change points")
                    st.markdown("• Revise dam operating rules")
                    st.markdown("• Adjust insurance rates")

                with col2:
                    st.markdown("**Water Supply:**")
                    st.markdown("• Reassess reservoir capacity")
                    st.markdown("• Update drought contingency plans")
                    st.markdown("• Modify water allocation policies")

            return None

    def _slide_spatiotemporal_mapping(self) -> Optional[bool]:
        """Slide 7: Spatio-Temporal Mapping"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Spatio-Temporal Mapping")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 🗺️ What is Spatio-Temporal Mapping?")

                st.markdown("**Step 1:** Analyze trends at each weather station")
                st.markdown("**Step 2:** Color-code the results on a map")
                st.markdown("**Step 3:** Show decision-makers where problems are")

                st.markdown("### 📊 Types of Maps")

                map_types = [
                    "**Trend Maps** - Show where trends are increasing/decreasing",
                    "**Change Point Maps** - Display when changes occurred",
                    "**Anomaly Maps** - Highlight unusual patterns",
                    "**Risk Maps** - Show areas of concern"
                ]

                for map_type in map_types:
                    st.markdown(f"• {map_type}")

                st.markdown("### 🎯 Engineering Uses")
                st.markdown("• **Infrastructure Planning** - Where to build new facilities")
                st.markdown("• **Risk Assessment** - Identify vulnerable areas")
                st.markdown("• **Resource Allocation** - Prioritize investments")
                st.markdown("• **Policy Development** - Support regulations")

            with col2:
                st.markdown("### 📈 Example: Regional Rainfall Trends")

                # Create sample spatial data
                np.random.seed(789)

                # Define grid of locations
                lats = np.linspace(30, 35, 6)
                lons = np.linspace(-90, -85, 6)

                # Create trend data
                trend_data = []
                for i, lat in enumerate(lats):
                    for j, lon in enumerate(lons):
                        # Create spatial pattern - more positive trends in north
                        base_trend = (lat - 30) * 0.8 - 2
                        # Add some randomness
                        trend = base_trend + np.random.normal(0, 0.5)

                        # Determine significance (simplified)
                        p_value = 0.5 * (1 - abs(trend)/3)
                        significant = p_value < 0.05

                        trend_data.append({
                            'Latitude': lat,
                            'Longitude': lon,
                            'Trend': trend,
                            'Significant': significant,
                            'P_Value': max(0.01, p_value)
                        })

                trend_df = pd.DataFrame(trend_data)

                # Create trend map
                fig = px.scatter(trend_df,
                               x='Longitude', y='Latitude',
                               color='Trend',
                               size=[abs(t)*5+10 for t in trend_df['Trend']],
                               color_continuous_scale='RdBu_r',
                               color_continuous_midpoint=0,
                               title='Rainfall Trend Map (mm/year)')

                # Add significance markers
                significant_data = trend_df[trend_df['Significant']]
                fig.add_trace(go.Scatter(
                    x=significant_data['Longitude'],
                    y=significant_data['Latitude'],
                    mode='markers',
                    marker=dict(symbol='circle-open', size=15, color='black', line=dict(width=2)),
                    name='Significant (p<0.05)',
                    showlegend=True
                ))

                fig.update_layout(height=400)
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Map Interpretation:**")
                st.markdown("🔴 **Red:** Decreasing rainfall trends")
                st.markdown("🔵 **Blue:** Increasing rainfall trends")
                st.markdown("⚫ **Black circles:** Statistically significant")

            st.markdown("### 🔧 Map Creation Process")

            process_steps = [
                "**1. Data Collection** - Gather spatial time series data",
                "**2. Trend Analysis** - Calculate trends for each location",
                "**3. Significance Testing** - Determine statistical confidence",
                "**4. Visualization** - Create clear, interpretable maps",
                "**5. Validation** - Check results make physical sense"
            ]

            for step in process_steps:
                st.markdown(step)

        return None

    def _slide_engineering_applications(self) -> Optional[bool]:
        """Slide 8: Engineering Applications"""
        with UIComponents.slide_container("theory"):
            st.markdown("## Real-World Engineering Applications")

            col1, col2 = UIComponents.two_column_layout()

            with col1:
                st.markdown("### 🏗️ Infrastructure Planning")

                st.markdown("**The Problem:**")
                st.markdown("Houston needed new storm drains after repeated flooding.")

                st.markdown("**The Analysis:**")
                st.markdown("• Trend test: Rainfall up 15% since 2000")
                st.markdown("• Change point: 2005 (more urbanization)")
                st.markdown("• Maps: East side hit hardest")

                st.markdown("**The Solution:**")
                st.markdown("• Design 20% bigger pipes city-wide")
                st.markdown("• Start with east side neighborhoods")
                st.markdown("• Use post-2005 rainfall data only")

                st.markdown("### 🌊 Flood Risk Assessment")

                flood_applications = [
                    "**Flood Map Updates** - Incorporate trend information",
                    "**Early Warning Systems** - Use spatial patterns",
                    "**Insurance Pricing** - Risk-based premiums",
                    "**Land Use Planning** - Restrict development in high-risk areas"
                ]

                for app in flood_applications:
                    st.markdown(f"• {app}")

            with col2:
                st.markdown("### 💧 Water Resource Management")

                st.markdown("**The Problem:**")
                st.markdown("California reservoir running low more often.")

                st.markdown("**The Analysis:**")
                st.markdown("• Trend: Inflows down 2% per year")
                st.markdown("• Change point: 2010 (climate pattern shift)")
                st.markdown("• Maps: Northern areas most affected")

                st.markdown("**The Solution:**")
                st.markdown("• Cut water allocations 10%")
                st.markdown("• Focus conservation efforts up north")
                st.markdown("• Plan for even drier conditions")

                st.markdown("### 🎯 Decision Support")

                decision_applications = [
                    "**Budget Allocation** - Prioritize based on risk maps",
                    "**Design Standards** - Update codes using trend data",
                    "**Emergency Planning** - Pre-position resources",
                    "**Climate Adaptation** - Long-term infrastructure planning"
                ]

                for app in decision_applications:
                    st.markdown(f"• {app}")

            st.markdown("### 📋 Implementation Framework")

            col1, col2, col3 = UIComponents.three_column_layout()

            with col1:
                st.markdown("#### 🔍 Assessment Phase")
                st.markdown("• Collect quality data")
                st.markdown("• Perform trend analysis")
                st.markdown("• Detect change points")
                st.markdown("• Create spatial maps")

            with col2:
                st.markdown("#### 📊 Analysis Phase")
                st.markdown("• Validate results")
                st.markdown("• Assess significance")
                st.markdown("• Identify patterns")
                st.markdown("• Quantify changes")

            with col3:
                st.markdown("#### 🎯 Action Phase")
                st.markdown("• Update design standards")
                st.markdown("• Modify operations")
                st.markdown("• Implement monitoring")
                st.markdown("• Communicate findings")

        return None

    def _slide_discussion_applications(self) -> Optional[bool]:
        """Slide 9: Discussion - Real-World Applications"""
        with UIComponents.slide_container("interactive"):
            st.markdown("## Discussion: Real-World Applications")

            result = QuizEngine.create_multiple_choice(
                "Your city's water treatment plant was designed in 1985 based on population projections. Recent spatio-temporal analysis shows: 1) Significant increasing trend in water demand since 2010, 2) Change point in 2010, 3) Spatial analysis shows demand growth concentrated in eastern suburbs. What's the most appropriate engineering response?",
                [
                    "Expand the entire treatment plant capacity uniformly",
                    "Build a new smaller plant in the eastern suburbs to handle growth",
                    "Implement water restrictions city-wide to reduce demand",
                    "Wait for more data since trends might reverse"
                ],
                1,
                {
                    "correct": "Excellent engineering thinking! Since growth is spatially concentrated in eastern suburbs and started in 2010, a targeted solution (new plant in that area) is more efficient than expanding the entire system. This addresses the specific spatial pattern identified in the analysis.",
                    "incorrect": "Consider the spatial component. The analysis shows growth is concentrated in eastern suburbs, not city-wide. A targeted engineering solution addressing the specific location of increased demand is more efficient and cost-effective than system-wide changes."
                },
                f"{self.info.id}_applications_quiz"
            )

            if result is True:
                st.markdown("---")
                st.markdown("### 🎓 Course Summary: Key Takeaways")

                takeaways = [
                    "**Trend Analysis** reveals long-term changes requiring design updates",
                    "**Change Point Detection** identifies when to split data for analysis",
                    "**Spatial Mapping** shows where problems are concentrated",
                    "**Engineering Decisions** should be based on statistical evidence",
                    "**Targeted Solutions** are more efficient than uniform approaches"
                ]

                for takeaway in takeaways:
                    st.markdown(f"✅ {takeaway}")

                st.markdown("### 🚀 Next Steps")

                checklist = [
                    "**Before any design:** Check for trends and change points",
                    "**If you find trends:** Design for future conditions, not past",
                    "**If you find change points:** Use only recent data",
                    "**If patterns vary by location:** Target solutions where needed most",
                    "**Always:** Make sure your statistics support your engineering decisions"
                ]

                for item in checklist:
                    st.markdown(f"✅ {item}")

                return True  # Mark module complete

            return None