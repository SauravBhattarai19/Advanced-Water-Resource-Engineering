"""
Module 11: Spatiotemporal Representation in Water Resources
Mapping Trends and Change Points for Engineering Decision-Making

Author: TA Saurav Bhattarai
Course: Advanced Water Resources Engineering
Institution: Jackson State University
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
from scipy import stats

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents


class Module11_Spatiotemporal(LearningModule):
    """Module 11: Spatiotemporal Representation in Water Resources"""

    def __init__(self):
        objectives = [
            LearningObjective("Understand spatiotemporal analysis in water resources", "understand"),
            LearningObjective("Create spatial maps of trends and change points", "create"),
            LearningObjective("Interpret regional patterns for engineering decisions", "evaluate"),
            LearningObjective("Apply spatiotemporal analysis to infrastructure planning", "apply")
        ]

        info = ModuleInfo(
            id="module_11",
            title="Spatiotemporal Representation in Water Resources",
            description="Spatial visualization and analysis of hydrologic trends and change points for regional assessment",
            duration_minutes=35,
            prerequisites=["module_09", "module_10"],
            learning_objectives=objectives,
            difficulty="intermediate",
            total_slides=1  # Paper-like format
        )

        super().__init__(info)

    def get_slide_titles(self) -> List[str]:
        return ["Spatiotemporal Analysis: A Comprehensive Guide"]

    def render_slide(self, slide_num: int) -> Optional[bool]:
        """Render the complete paper-like module"""
        return self._render_complete_module()

    def _render_complete_module(self) -> Optional[bool]:
        """Render complete module in paper-like academic format"""
        
        # Module title and metadata
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;'>
            <h1 style='margin: 0; color: white;'>Module 9: Spatiotemporal Representation in Water Resources</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
                Mapping and Visualizing Hydrologic Patterns for Regional Engineering Decisions
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Abstract
        with st.expander("📄 **ABSTRACT**", expanded=True):
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; 
                        border-left: 4px solid #43e97b;'>
            
            **Objective:** This module synthesizes trend detection (Module 7) and change point analysis (Module 8)
            into a spatiotemporal framework for regional water resources assessment. We demonstrate how to create,
            interpret, and apply spatial visualizations of temporal patterns to support engineering decision-making.
            
            **Significance:** While point-based analyses reveal local patterns, spatiotemporal representation enables
            identification of regional trends, watershed-scale impacts, and prioritization of infrastructure investments
            across multiple locations.
            
            **Methods:** Integration of Mann-Kendall trend tests, Pettitt change point detection, and spatial
            interpolation techniques to create comprehensive regional assessments. Emphasis on practical
            visualization methods for communicating results to stakeholders and decision-makers.
            
            **Applications:** Infrastructure planning, climate change impact assessment, water allocation policies,
            flood risk mapping, and regional water resources management.
            
            **Keywords:** Spatiotemporal analysis, hydrologic mapping, regional assessment, trend visualization,
            change point mapping, GIS integration, water resources planning
            
            </div>
            """, unsafe_allow_html=True)

        # Section 1: Introduction
        with st.expander("## 1. INTRODUCTION TO SPATIOTEMPORAL ANALYSIS", expanded=False):
            st.markdown("### 1.1 From Point Analysis to Regional Understanding")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Limitations of Single-Station Analysis:**
                
                Modules 7 and 8 equipped you with powerful tools for analyzing hydrologic time series at individual
                locations. However, water resources challenges rarely respect single-point boundaries:
                
                • **Floods affect entire watersheds**, not single gages
                • **Droughts span regional scales**, impacting multiple water systems
                • **Climate change impacts vary spatially**, requiring area-wide assessment
                • **Infrastructure planning requires regional perspective**, not isolated decisions
                • **Investment priorities need spatial context**, allocating resources where most needed
                
                **The Spatiotemporal Paradigm:**
                
                Spatiotemporal analysis combines:
                - **Spatial dimension:** Where patterns occur (geographic location)
                - **Temporal dimension:** How patterns change over time (trends, change points)
                - **Integration:** Understanding coupled space-time dynamics
                
                **Engineering Value:**
                
                1. **Identify Regional Patterns:**
                   - Which watersheds showing similar trends?
                   - Are change points clustered geographically?
                   - Do patterns follow elevation gradients?
                
                2. **Prioritize Interventions:**
                   - Where are changes most severe?
                   - Which areas need immediate attention?
                   - How to allocate limited budgets?
                
                3. **Understand Physical Processes:**
                   - What's driving observed patterns?
                   - Are changes due to local or regional factors?
                   - Can we predict future changes?
                
                4. **Communicate Effectively:**
                   - Maps speak to decision-makers
                   - Visual patterns easier to grasp
                   - Support policy development
                """)
            
            with col2:
                st.markdown("**Conceptual Framework:**")
                
                # Create conceptual diagram
                np.random.seed(42)
                
                # Generate synthetic spatial pattern
                n_stations = 12
                lats = np.array([32.5, 32.5, 32.5, 32.5, 33.5, 33.5, 33.5, 33.5, 34.5, 34.5, 34.5, 34.5])
                lons = np.array([-91.5, -90.5, -89.5, -88.5, -91.5, -90.5, -89.5, -88.5, -91.5, -90.5, -89.5, -88.5])
                
                # Trend values (mm/year) - spatial pattern
                trends = np.array([2.5, 2.8, 1.2, 0.5, 3.1, 3.5, 1.8, 0.8, 2.9, 3.2, 1.5, 0.6])
                
                # Significance
                p_values = np.array([0.001, 0.002, 0.045, 0.15, 0.001, 0.001, 0.03, 0.12, 0.002, 0.001, 0.04, 0.18])
                significant = p_values < 0.05
                
                fig = go.Figure()
                
                # Add all points
                fig.add_trace(go.Scatter(
                    x=lons, y=lats,
                    mode='markers',
                    marker=dict(
                        size=np.abs(trends) * 10,
                        color=trends,
                        colorscale='RdBu_r',
                        showscale=True,
                        colorbar=dict(title="Trend<br>(mm/yr)"),
                        cmin=-3, cmax=3,
                        line=dict(
                            width=2,
                            color=['black' if s else 'gray' for s in significant]
                        )
                    ),
                    text=[f"Station {i+1}<br>Trend: {t:.1f} mm/yr<br>p={p:.3f}" 
                          for i, (t, p) in enumerate(zip(trends, p_values))],
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title="Example: Regional Rainfall Trend Pattern",
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=400,
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                fig = PlotTools.apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Pattern Visible:**
                - Strong increasing trends in western stations (blue, large)
                - Weak/insignificant trends in eastern stations (small)
                - Black outlines indicate statistical significance
                - Clear west-to-east gradient
                
                **Engineering Insight:** Western watersheds need updated drainage design;
                eastern areas can continue with historical statistics.
                """)
            
            st.markdown("### 1.2 Types of Spatiotemporal Maps in Hydrology")
            
            st.markdown("""
            **1. Trend Magnitude Maps:**
            - Display Sen's slope at each location
            - Color intensity shows trend strength
            - Symbol size can represent significance level
            - **Use:** Identify regions with strongest changes
            
            **2. Trend Significance Maps:**
            - Show p-values from Mann-Kendall test
            - Binary (significant/not) or continuous scale
            - Often overlaid with magnitude
            - **Use:** Determine where changes are statistically robust
            
            **3. Change Point Location Maps:**
            - Display year of detected change (τ) at each station
            - Color represents timing of change
            - Size represents magnitude of change
            - **Use:** Identify spatial clusters of synchronous changes
            
            **4. Change Direction Maps:**
            - Show whether changes are increasing or decreasing
            - Often binary color scheme (red/blue)
            - Can include magnitude information
            - **Use:** Identify regions with opposing trends
            
            **5. Composite Risk Maps:**
            - Combine multiple factors (trend, change point, variability)
            - Create risk scores or categories
            - Support decision-making directly
            - **Use:** Prioritize infrastructure investments
            
            **6. Temporal Animation Maps:**
            - Show how spatial patterns evolve over time
            - Dynamic visualization of changes
            - Powerful for presentations
            - **Use:** Communicate complex patterns to stakeholders
            """)

        # Section 2: Creating Trend Maps
        with st.expander("## 2. CREATING TREND MAPS: METHODOLOGY", expanded=False):
            st.markdown("### 2.1 Data Requirements and Quality Control")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Minimum Requirements:**
                
                **1. Spatial Coverage:**
                - Multiple stations (minimum 5-10)
                - Reasonably distributed across area of interest
                - Representative of different watershed conditions
                - Avoid clustering (stations too close)
                
                **2. Temporal Requirements:**
                - Common period of record (ideally)
                - At least 20-30 years per station
                - Acceptable gaps: < 10% missing data
                - Quality-controlled observations
                
                **3. Data Homogeneity:**
                - Consistent measurement methods
                - No station relocations (or documented adjustments)
                - Similar data quality across network
                - Comparable recording intervals
                
                **Quality Control Procedures:**
                
                ```
                Step 1: Visual Inspection
                  • Plot each time series
                  • Identify obvious errors
                  • Check for suspicious jumps
                
                Step 2: Statistical Tests
                  • Apply Pettitt test to each station
                  • Identify data quality issues
                  • Distinguish real changes from artifacts
                
                Step 3: Regional Consistency
                  • Compare neighboring stations
                  • Check for isolated outliers
                  • Validate with meteorological data
                
                Step 4: Documentation
                  • Record all adjustments
                  • Note stations excluded
                  • Document quality flags
                ```
                """)
            
            with col2:
                st.markdown("""
                **Dealing with Common Issues:**
                
                **Missing Data:**
                - **< 5% missing:** Accept as-is for Mann-Kendall
                - **5-10% missing:** Acceptable with caution
                - **> 10% missing:** Consider infilling or exclude
                - **Infilling methods:** Regional regression, spatial interpolation
                
                **Unequal Record Lengths:**
                - **Option 1:** Use common period (loses data)
                - **Option 2:** Use all available (unequal sample sizes)
                - **Recommendation:** Common period for direct comparison
                - **Sensitivity:** Test both approaches
                
                **Change Points Due to Station Issues:**
                - **Problem:** Gage relocation mimics real change
                - **Detection:** Inconsistent with neighbors
                - **Solution:** Apply homogenization methods
                - **Documentation:** Critical to note adjustments
                
                **Spatial Autocorrelation:**
                - **Issue:** Nearby stations not independent
                - **Effect:** Overestimates regional significance
                - **Test:** Calculate Moran's I statistic
                - **Correction:** Account for spatial dependence
                
                **Example QC Flag System:**
                
                | Flag | Meaning | Action |
                |------|---------|--------|
                | A | Excellent quality | Use as-is |
                | B | Good quality, minor gaps | Use with caution |
                | C | Questionable, > 10% missing | Exclude or infill |
                | D | Poor quality, known issues | Exclude |
                | E | Station relocated | Apply homogenization |
                """)
            
            st.markdown("### 2.2 Step-by-Step Workflow")
            
            st.markdown("""
            **Complete Analysis Workflow:**
            
            ---
            
            #### **PHASE 1: DATA PREPARATION**
            
            ```
            Step 1.1: Assemble Dataset
              • Download data from all stations
              • Organize in consistent format
              • Include metadata (lat/lon, elevation, etc.)
            
            Step 1.2: Quality Control
              • Apply QC procedures (Section 2.1)
              • Flag or exclude problematic stations
              • Document all decisions
            
            Step 1.3: Synchronize Records
              • Identify common period
              • Standardize time intervals
              • Handle missing values
            
            Step 1.4: Visual Exploration
              • Plot all time series
              • Check for regional coherence
              • Identify potential outliers
            ```
            
            ---
            
            #### **PHASE 2: STATISTICAL ANALYSIS**
            
            ```
            Step 2.1: Trend Detection at Each Station
              FOR each station DO:
                • Apply Mann-Kendall test
                • Calculate Sen's slope
                • Determine 95% confidence intervals
                • Record: τ, p-value, β, CI_lower, CI_upper
              END FOR
            
            Step 2.2: Change Point Detection at Each Station
              FOR each station DO:
                • Apply Pettitt test
                • Record: K_τ, τ, p-value
                • If significant: calculate pre/post means
              END FOR
            
            Step 2.3: Compile Results
              • Create results table
              • Include all test statistics
              • Add spatial coordinates
              • Compute additional metrics (e.g., trend/mean ratio)
            
            Step 2.4: Regional Summary Statistics
              • Percentage of stations with significant trends
              • Distribution of trend magnitudes
              • Spatial autocorrelation of trends
              • Average change point year
            ```
            
            ---
            
            #### **PHASE 3: SPATIAL VISUALIZATION**
            
            ```
            Step 3.1: Create Base Maps
              • Set up geographic coordinate system
              • Add watershed boundaries
              • Include relevant features (rivers, cities)
            
            Step 3.2: Generate Trend Map
              • Plot points at station locations
              • Color by Sen's slope magnitude
              • Size by significance level
              • Add legend and north arrow
            
            Step 3.3: Generate Change Point Map
              • Color by year of change
              • Size by magnitude of change
              • Include only significant change points
            
            Step 3.4: Create Composite Maps
              • Overlay trends and change points
              • Add isopleth lines (if appropriate)
              • Include confidence information
            ```
            
            ---
            
            #### **PHASE 4: INTERPRETATION**
            
            ```
            Step 4.1: Identify Spatial Patterns
              • Clusters of similar trends
              • Gradients (elevation, distance)
              • Anomalous stations
            
            Step 4.2: Physical Validation
              • Match patterns to known causes
              • Check consistency with climate data
              • Validate with land use changes
            
            Step 4.3: Regional Coherence Assessment
              • Compare neighboring stations
              • Test for spatial autocorrelation
              • Identify regional signals vs. local noise
            
            Step 4.4: Engineering Implications
              • Where are updates needed?
              • What infrastructure at risk?
              • How to prioritize investments?
            ```
            
            ---
            
            #### **PHASE 5: COMMUNICATION**
            
            ```
            Step 5.1: Prepare Presentation Materials
              • High-quality maps for reports
              • Summary statistics tables
              • Interactive visualizations (if applicable)
            
            Step 5.2: Document Methodology
              • Data sources and processing
              • Statistical methods applied
              • Quality control procedures
              • Limitations and assumptions
            
            Step 5.3: Stakeholder Communication
              • Translate statistics to impacts
              • Recommend specific actions
              • Provide uncertainty ranges
              • Develop monitoring plan
            ```
            """)

        # Section 3: Interactive Demonstration
        with st.expander("## 3. INTERACTIVE REGIONAL ANALYSIS", expanded=False):
            st.markdown("### 3.1 Simulated Regional Network")
            
            st.markdown("""
            **Scenario:** A regional water resources agency manages 15 streamflow gaging stations across a 
            10,000 km² watershed. They need to assess whether flood frequencies should be updated due to 
            changes over the past 40 years.
            """)
            
            # Generate synthetic regional network
            np.random.seed(42)
            n_stations = 15
            
            # Station locations (grid with some jitter)
            base_lats = np.repeat([31.0, 31.5, 32.0, 32.5, 33.0], 3)
            base_lons = np.tile([-92.0, -91.0, -90.0], 5)
            lats = base_lats + np.random.normal(0, 0.1, n_stations)
            lons = base_lons + np.random.normal(0, 0.1, n_stations)
            
            # Station IDs
            station_ids = [f"STN-{i+1:02d}" for i in range(n_stations)]
            
            # Generate realistic trends with spatial pattern
            # Northern stations: increasing trends
            # Southern stations: decreasing or no trends
            # Create latitude gradient
            lat_effect = (lats - lats.min()) / (lats.max() - lats.min())  # 0 to 1
            base_trends = 3.0 * lat_effect - 1.0  # -1 to +2 m³/s per year
            trends = base_trends + np.random.normal(0, 0.5, n_stations)
            
            # Calculate p-values (stronger trends more significant)
            # Use inverse relationship: larger |trend| → smaller p-value
            p_values = 0.15 / (1 + np.abs(trends) * 2) + np.random.uniform(0, 0.05, n_stations)
            p_values = np.clip(p_values, 0.001, 0.5)
            significant_trend = p_values < 0.05
            
            # Generate change point data
            # Some stations have change points, others don't
            has_changepoint = np.random.random(n_stations) < 0.6  # 60% have change points
            change_years = np.where(has_changepoint, 
                                   np.random.randint(1995, 2010, n_stations),
                                   0)
            change_p_values = np.where(has_changepoint,
                                       np.random.uniform(0.001, 0.08, n_stations),
                                       0.5)
            significant_change = change_p_values < 0.05
            
            # Create DataFrame
            network_df = pd.DataFrame({
                'Station': station_ids,
                'Latitude': lats,
                'Longitude': lons,
                'Trend': trends,
                'Trend_PValue': p_values,
                'Trend_Significant': significant_trend,
                'ChangePoint_Year': change_years,
                'ChangePoint_PValue': change_p_values,
                'ChangePoint_Significant': significant_change
            })
            
            # Display data table
            st.markdown("### 3.2 Network Characteristics")
            
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                # Show first few stations
                display_df = network_df[['Station', 'Latitude', 'Longitude', 'Trend', 
                                        'Trend_PValue', 'ChangePoint_Year', 'ChangePoint_PValue']].copy()
                display_df['Trend'] = display_df['Trend'].round(2)
                display_df['Trend_PValue'] = display_df['Trend_PValue'].round(3)
                display_df['ChangePoint_PValue'] = display_df['ChangePoint_PValue'].round(3)
                st.dataframe(display_df, use_container_width=True, height=400)
            
            with col2:
                # Summary statistics
                st.markdown("**Network Summary:**")
                st.metric("Total Stations", n_stations)
                st.metric("Significant Trends", f"{significant_trend.sum()} ({100*significant_trend.sum()/n_stations:.0f}%)")
                st.metric("Significant Change Points", f"{significant_change.sum()} ({100*significant_change.sum()/n_stations:.0f}%)")
                
                st.markdown("**Trend Statistics:**")
                st.markdown(f"• Mean trend: {trends.mean():.2f} m³/s/year")
                st.markdown(f"• Range: {trends.min():.2f} to {trends.max():.2f}")
                st.markdown(f"• Increasing: {(trends > 0).sum()} stations")
                st.markdown(f"• Decreasing: {(trends < 0).sum()} stations")
            
            st.markdown("### 3.3 Spatial Visualization: Trend Map")
            
            # Create comprehensive trend map
            fig = go.Figure()
            
            # All stations - color by trend, size by significance
            marker_sizes = np.where(significant_trend, 15, 8)
            marker_symbols = np.where(trends > 0, 'triangle-up', 'triangle-down')
            
            # Separate significant and non-significant for legend
            sig_df = network_df[network_df['Trend_Significant']]
            nonsig_df = network_df[~network_df['Trend_Significant']]
            
            # Non-significant stations
            if len(nonsig_df) > 0:
                fig.add_trace(go.Scatter(
                    x=nonsig_df['Longitude'],
                    y=nonsig_df['Latitude'],
                    mode='markers',
                    name='Not Significant',
                    marker=dict(
                        size=8,
                        color=nonsig_df['Trend'],
                        colorscale='RdBu_r',
                        cmin=-2, cmax=2,
                        showscale=False,
                        opacity=0.5,
                        line=dict(width=1, color='gray')
                    ),
                    text=[f"{row['Station']}<br>Trend: {row['Trend']:.2f} m³/s/yr<br>p={row['Trend_PValue']:.3f} (NS)" 
                          for _, row in nonsig_df.iterrows()],
                    hovertemplate='%{text}<extra></extra>'
                ))
            
            # Significant stations
            if len(sig_df) > 0:
                fig.add_trace(go.Scatter(
                    x=sig_df['Longitude'],
                    y=sig_df['Latitude'],
                    mode='markers',
                    name='Significant (p<0.05)',
                    marker=dict(
                        size=15,
                        color=sig_df['Trend'],
                        colorscale='RdBu_r',
                        cmin=-2, cmax=2,
                        showscale=True,
                        colorbar=dict(
                            title="Trend<br>(m³/s/yr)",
                            x=1.02
                        ),
                        line=dict(width=2, color='black')
                    ),
                    text=[f"{row['Station']}<br>Trend: {row['Trend']:.2f} m³/s/yr<br>p={row['Trend_PValue']:.3f} ***" 
                          for _, row in sig_df.iterrows()],
                    hovertemplate='%{text}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Regional Trend Map: Annual Peak Flow (1985-2024)",
                xaxis_title="Longitude (°W)",
                yaxis_title="Latitude (°N)",
                height=500,
                hovermode='closest',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray', scaleanchor="x", scaleratio=1)
            )
            
            fig = PlotTools.apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 3.4 Spatial Visualization: Change Point Map")
            
            # Create change point map
            fig2 = go.Figure()
            
            # Stations with significant change points
            cp_sig_df = network_df[network_df['ChangePoint_Significant']]
            cp_nonsig_df = network_df[~network_df['ChangePoint_Significant'] & (network_df['ChangePoint_Year'] > 0)]
            
            # Non-significant change points
            if len(cp_nonsig_df) > 0:
                fig2.add_trace(go.Scatter(
                    x=cp_nonsig_df['Longitude'],
                    y=cp_nonsig_df['Latitude'],
                    mode='markers',
                    name='Detected (Not Significant)',
                    marker=dict(
                        size=8,
                        color=cp_nonsig_df['ChangePoint_Year'],
                        colorscale='Viridis',
                        showscale=False,
                        opacity=0.4,
                        line=dict(width=1, color='gray')
                    ),
                    text=[f"{row['Station']}<br>τ={row['ChangePoint_Year']:.0f}<br>p={row['ChangePoint_PValue']:.3f} (NS)" 
                          for _, row in cp_nonsig_df.iterrows()],
                    hovertemplate='%{text}<extra></extra>'
                ))
            
            # Significant change points
            if len(cp_sig_df) > 0:
                fig2.add_trace(go.Scatter(
                    x=cp_sig_df['Longitude'],
                    y=cp_sig_df['Latitude'],
                    mode='markers',
                    name='Significant (p<0.05)',
                    marker=dict(
                        size=15,
                        color=cp_sig_df['ChangePoint_Year'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title="Change<br>Point<br>Year",
                            x=1.02
                        ),
                        line=dict(width=2, color='black')
                    ),
                    text=[f"{row['Station']}<br>τ={row['ChangePoint_Year']:.0f}<br>p={row['ChangePoint_PValue']:.3f} ***" 
                          for _, row in cp_sig_df.iterrows()],
                    hovertemplate='%{text}<extra></extra>'
                ))
            
            # Stations with no detected change point
            no_cp_df = network_df[network_df['ChangePoint_Year'] == 0]
            if len(no_cp_df) > 0:
                fig2.add_trace(go.Scatter(
                    x=no_cp_df['Longitude'],
                    y=no_cp_df['Latitude'],
                    mode='markers',
                    name='No Change Point',
                    marker=dict(
                        size=8,
                        color='lightgray',
                        symbol='x',
                        line=dict(width=1, color='gray')
                    ),
                    text=[f"{row['Station']}<br>No change point detected" 
                          for _, row in no_cp_df.iterrows()],
                    hovertemplate='%{text}<extra></extra>'
                ))
            
            fig2.update_layout(
                title="Regional Change Point Map: Timing of Abrupt Changes",
                xaxis_title="Longitude (°W)",
                yaxis_title="Latitude (°N)",
                height=500,
                hovermode='closest',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray', scaleanchor="x", scaleratio=1)
            )
            
            fig2 = PlotTools.apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("### 3.5 Pattern Interpretation")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Observed Spatial Patterns:**")
                
                st.markdown(f"""
                **Trend Analysis:**
                - Clear **north-south gradient** in trends
                - Northern stations: predominantly increasing flows
                - Southern stations: stable or decreasing flows
                - {significant_trend.sum()}/{n_stations} stations with significant trends
                
                **Physical Interpretation:**
                - Likely related to **latitudinal climate patterns**
                - Northern watersheds: increasing precipitation
                - Southern watersheds: potential drought intensification
                - Consistent with regional climate studies
                
                **Change Point Analysis:**
                - {significant_change.sum()}/{n_stations} stations with significant change points
                - Most changes clustered in **late 1990s to early 2000s**
                - Suggests **regional climate shift** rather than local causes
                - Timing consistent with documented climate regime change
                """)
            
            with col2:
                st.markdown("**Engineering Implications:**")
                
                st.markdown(f"""
                **Design Standard Updates:**
                
                **Northern Region (Increasing Trends):**
                - ✅ Update flood frequency curves upward
                - ✅ Use post-change point data for new designs
                - ✅ Increase culvert/bridge capacities by 10-20%
                - ✅ Prioritize upgrades in areas with strongest trends
                
                **Southern Region (Stable/Decreasing):**
                - ✅ May continue using historical statistics
                - ✅ Monitor for emergence of trends
                - ⚠️ Consider water supply implications of decreases
                - ⚠️ Plan for potential drought impacts
                
                **Regional Coordination:**
                - Develop **watershed-scale strategies**
                - Share data and resources across jurisdictions
                - Implement **consistent design standards** within regions
                - Establish **monitoring network** for ongoing assessment
                
                **Investment Prioritization:**
                1. Northern stations with strong increasing trends
                2. Stations near critical infrastructure
                3. Areas with significant change points
                4. High-consequence failure locations
                """)
            
            st.markdown("### 3.6 Statistical Significance of Regional Pattern")
            
            # Test for spatial trend in trends (meta-analysis!)
            from scipy import stats
            
            # Correlation between latitude and trend
            lat_trend_corr = stats.pearsonr(lats, trends)
            
            st.markdown(f"""
            **Question:** Is the north-south gradient statistically significant?
            
            **Test:** Pearson correlation between latitude and trend magnitude
            
            **Result:**
            - Correlation coefficient: r = {lat_trend_corr[0]:.3f}
            - P-value: p = {lat_trend_corr[1]:.4f}
            - **Conclusion:** {' Highly significant north-south gradient' if lat_trend_corr[1] < 0.01 else 'Significant north-south gradient' if lat_trend_corr[1] < 0.05 else 'No significant spatial pattern'}
            
            This confirms that the observed spatial pattern is **not due to chance** but represents a 
            real regional phenomenon requiring coordinated engineering response.
            """)

        # Section 4: Engineering Applications
        with st.expander("## 4. ENGINEERING APPLICATIONS AND CASE STUDIES", expanded=False):
            st.markdown("### 4.1 Infrastructure Planning and Prioritization")
            
            st.markdown("""
            **Case Study: Regional Drainage System Master Plan**
            
            **Situation:**
            A metropolitan water district covering 500 km² with 25 major drainage basins needs to update
            its master plan. Budget allows upgrading 5 basins in next 5 years. How to prioritize?
            
            **Analysis Approach:**
            
            1. **Conduct Spatiotemporal Analysis:**
               - Install/utilize 25 rain gages (one per basin)
               - Collect 30+ years of rainfall data
               - Perform Mann-Kendall tests on each gage
               - Apply Pettitt tests to detect change points
               - Create spatial maps of results
            
            2. **Develop Risk Score:**
               ```
               Risk Score = w₁×(Trend Magnitude) + w₂×(Significance Level) + 
                           w₃×(Change Point Indicator) + w₄×(Population Density) +
                           w₅×(Critical Infrastructure)
               
               Where weights (w) determined by stakeholder priorities
               ```
            
            3. **Create Priority Map:**
               - Overlay risk scores on basin map
               - Identify top 5 basins
               - Validate with flooding history
               - Adjust for political/social factors
            
            4. **Results:**
               - **High Priority (Years 1-2):**
                 - Basin A: Strongest increasing trend (4.2 mm/yr, p<0.001)
                 - Basin B: Recent change point (2010, p=0.003) + hospital
               - **Medium Priority (Years 3-4):**
                 - Basin C: Moderate trend + downtown area
                 - Basin D: Significant change point + school district
               - **Next Phase (Years 5+):**
                 - Basin E: Emerging trend, monitor closely
                 - Basins F-Z: Continue with maintenance
            
            5. **Economic Impact:**
               - Targeted approach saves $15M vs. uniform upgrades
               - Reduces flood damages by $50M over 20 years
               - Optimizes limited capital budget
               - Politically defensible (data-driven)
            """)
            
            st.markdown("### 4.2 Climate Change Adaptation Planning")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Application: State-Wide Vulnerability Assessment**
                
                **Objective:**
                Identify regions most vulnerable to changing flood patterns
                for adaptation planning and resource allocation.
                
                **Methodology:**
                
                **Step 1: Data Assembly**
                - Compile data from 150+ USGS gages statewide
                - Minimum 40 years of record
                - Quality-control all stations
                
                **Step 2: Temporal Analysis**
                - Mann-Kendall trend tests (all stations)
                - Pettitt change point detection (all stations)
                - Calculate magnitude of changes
                
                **Step 3: Spatial Mapping**
                - Create statewide trend map
                - Overlay with watershed boundaries
                - Identify regional clusters
                
                **Step 4: Vulnerability Classification**
                
                | Category | Criteria | Response |
                |----------|----------|----------|
                | High Vulnerability | p<0.01, trend>20% | Immediate action |
                | Moderate | 0.01<p<0.05, trend 10-20% | Plan upgrades |
                | Low | p>0.05, trend<10% | Continue monitoring |
                | Decreasing | Negative trends | Water supply focus |
                """)
            
            with col2:
                st.markdown("""
                **Step 5: Regional Strategies**
                
                **High Vulnerability Regions:**
                - Update IDF curves immediately
                - Revise floodplain maps
                - Implement green infrastructure
                - Enhance early warning systems
                - Restrict development in high-risk areas
                
                **Moderate Vulnerability Regions:**
                - Schedule infrastructure assessments
                - Plan capital improvements
                - Update emergency response plans
                - Educate public about risks
                
                **Low Vulnerability Regions:**
                - Continue existing programs
                - Maintain monitoring network
                - Re-assess every 5 years
                
                **Benefits of Spatiotemporal Approach:**
                
                ✅ **Efficient Resource Allocation:**
                - Focus limited funds where most needed
                - Avoid unnecessary upgrades in stable regions
                - Maximize return on investment
                
                ✅ **Political Support:**
                - Data-driven decisions defensible
                - Clear visual communication
                - Transparent prioritization
                
                ✅ **Adaptive Management:**
                - Framework for ongoing updates
                - Responsive to new data
                - Flexible to changing conditions
                """)
            
            st.markdown("### 4.3 Water Allocation Policy Development")
            
            st.markdown("""
            **Case Study: Interstate Water Compact Revision**
            
            **Background:**
            A water compact between three states allocates river flows based on historical (1950-1980) data.
            Downstream state claims upstream withdrawals are increasing, causing shortages.
            
            **Spatiotemporal Analysis:**
            
            **1. Data Collection:**
            - Upstream gages (n=12): Tributaries in State A
            - Mainstem gages (n=8): State B
            - Downstream gages (n=6): State C
            - Period: 1950-2023 (73 years)
            
            **2. Trend Analysis by State:**
            
            | State | Mean Trend | Significant | Interpretation |
            |-------|------------|-------------|----------------|
            | A (Upstream) | -0.8 m³/s/yr | 7/12 stations | Decreasing natural flows |
            | B (Midstream) | -1.2 m³/s/yr | 6/8 stations | Decreasing + withdrawals |
            | C (Downstream) | -1.8 m³/s/yr | 5/6 stations | Compounded decreases |
            
            **3. Change Point Analysis:**
            - State A: Changes in late 1990s (drought onset)
            - State B: Changes in early 2000s (increased irrigation)
            - State C: Changes progressive through period
            
            **4. Spatial Pattern:**
            - Clear downstream amplification of decreases
            - Both climate (natural decrease) AND human (withdrawals) factors
            - Disproportionate impact on downstream state
            
            **5. Resolution:**
            - Compact revised to use post-2000 hydrology
            - Proportional reductions in all states
            - Adaptive management framework established
            - Annual re-assessment of conditions
            - Drought response plan implemented
            
            **Outcome:**
            - More equitable allocation
            - Reduced interstate conflict
            - Sustainable water use
            - Regular monitoring ensures ongoing fairness
            
            **Key Lesson:** Spatiotemporal analysis revealed that **both** upstream impacts AND 
            natural climate variability were responsible, preventing unfair blame and enabling 
            collaborative solution.
            """)

        # Section 5: Best Practices and Software
        with st.expander("## 5. BEST PRACTICES AND IMPLEMENTATION", expanded=False):
            st.markdown("### 5.1 Data Management and Documentation")
            
            st.markdown("""
            **Comprehensive Data Management System:**
            
            **1. Database Structure:**
            
            ```sql
            -- Stations table
            CREATE TABLE stations (
                station_id VARCHAR PRIMARY KEY,
                latitude FLOAT,
                longitude FLOAT,
                elevation FLOAT,
                watershed_area FLOAT,
                start_date DATE,
                end_date DATE,
                data_quality_flag VARCHAR
            );
            
            -- Time series data table
            CREATE TABLE timeseries (
                station_id VARCHAR,
                date DATE,
                value FLOAT,
                flag VARCHAR,
                PRIMARY KEY (station_id, date),
                FOREIGN KEY (station_id) REFERENCES stations(station_id)
            );
            
            -- Analysis results table
            CREATE TABLE analysis_results (
                station_id VARCHAR,
                analysis_type VARCHAR,
                parameter VARCHAR,
                value FLOAT,
                p_value FLOAT,
                confidence_lower FLOAT,
                confidence_upper FLOAT,
                analysis_date DATE,
                PRIMARY KEY (station_id, analysis_type, parameter)
            );
            ```
            
            **2. Metadata Requirements:**
            
            Essential information to maintain:
            - Station location (lat/lon with datum)
            - Instrumentation type and history
            - Calibration records
            - Known issues or data quality concerns
            - Land use in watershed
            - Any watershed modifications
            - Data processing methods
            - Contact information for data provider
            
            **3. Quality Flags:**
            
            ```
            A = Approved (passed all QC)
            E = Estimated (infilled missing data)
            P = Provisional (not yet fully reviewed)
            Q = Questionable (potential issues noted)
            R = Rejected (known errors, exclude from analysis)
            ```
            
            **4. Version Control:**
            
            - Track all dataset versions
            - Document changes between versions
            - Maintain reproducibility
            - Archive historical analyses
            
            **5. Documentation Standards:**
            
            Every analysis should include:
            - **README file:** Overview, data sources, methods
            - **CHANGELOG:** Updates and modifications
            - **METADATA:** Detailed station information
            - **METHODS:** Statistical procedures applied
            - **CODE:** Scripts used for analysis
            - **RESULTS:** Summary tables and figures
            - **REPORT:** Interpretation and conclusions
            """)
            
            st.markdown("### 5.2 Software and Tools")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Python Workflow:**
                
                ```python
                import pandas as pd
                import geopandas as gpd
                import numpy as np
                from scipy import stats
                import matplotlib.pyplot as plt
                import seaborn as sns
                import plotly.express as px
                
                # 1. Load station data
                stations = gpd.read_file('stations.shp')
                
                # 2. Perform trend analysis
                results = []
                for idx, row in stations.iterrows():
                    station_id = row['station_id']
                    data = pd.read_csv(f'data/{station_id}.csv')
                    
                    # Mann-Kendall test
                    mk_result = mann_kendall_test(data['value'])
                    
                    # Sen's slope
                    slope = sens_slope(data['value'])
                    
                    results.append({
                        'station_id': station_id,
                        'trend': slope,
                        'p_value': mk_result.p,
                        'significant': mk_result.p < 0.05
                    })
                
                # 3. Merge with spatial data
                results_df = pd.DataFrame(results)
                stations = stations.merge(results_df, on='station_id')
                
                # 4. Create map
                fig = px.scatter_map(
                    stations,
                    lat='latitude',
                    lon='longitude',
                    color='trend',
                    size='significant',
                    hover_data=['station_id', 'trend', 'p_value'],
                    color_continuous_scale='RdBu_r',
                    title='Regional Trend Map'
                )
                fig.show()
                
                # 5. Export results
                stations.to_file('results/trend_map.shp')
                results_df.to_csv('results/trend_results.csv')
                ```
                
                **Key Python Packages:**
                - `geopandas`: Spatial data handling
                - `pymannkendall`: Trend detection
                - `scipy.stats`: Statistical tests
                - `plotly`/`folium`: Interactive maps
                - `statsmodels`: Advanced statistics
                """)
            
            with col2:
                st.markdown("""
                **R Workflow:**
                
                ```r
                library(sf)
                library(trend)
                library(ggplot2)
                library(dplyr)
                library(leaflet)
                
                # 1. Load station data
                stations <- st_read("stations.shp")
                
                # 2. Perform trend analysis
                results <- stations %>%
                  rowwise() %>%
                  mutate(
                    data = list(read.csv(paste0("data/", station_id, ".csv"))),
                    mk_test = list(mk.test(data$value)),
                    trend = sens.slope(data$value)$estimates,
                    p_value = mk_test$p.value,
                    significant = p_value < 0.05
                  )
                
                # 3. Create static map
                ggplot(results) +
                  geom_sf(aes(color = trend, size = significant)) +
                  scale_color_gradient2(low = "red", mid = "white", high = "blue") +
                  labs(title = "Regional Trend Map") +
                  theme_minimal()
                
                # 4. Create interactive map
                leaflet(results) %>%
                  addTiles() %>%
                  addCircleMarkers(
                    radius = ~ifelse(significant, 10, 5),
                    color = ~colorNumeric("RdBu", trend)(trend),
                    popup = ~paste0("Station: ", station_id, 
                                   "<br>Trend: ", round(trend, 2),
                                   "<br>P-value: ", round(p_value, 3))
                  )
                
                # 5. Export
                st_write(results, "results/trend_map.shp")
                write.csv(results, "results/trend_results.csv")
                ```
                
                **Key R Packages:**
                - `sf`: Simple features for spatial data
                - `trend`: Trend and homogeneity tests
                - `sp`/`raster`: Spatial analysis
                - `leaflet`: Interactive maps
                - `tmap`: Thematic maps
                """)
            
            st.markdown("### 5.3 Quality Assurance Checklist")
            
            st.markdown("""
            **Before Publishing Results:**
            
            ☐ **Data Quality**
              - [ ] All stations passed QC procedures
              - [ ] Missing data documented and handled appropriately
              - [ ] Outliers investigated and validated
              - [ ] Homogeneity tests performed
            
            ☐ **Statistical Validity**
              - [ ] Appropriate tests selected for data characteristics
              - [ ] Assumptions checked (independence, etc.)
              - [ ] Multiple testing corrections applied if needed
              - [ ] Confidence intervals calculated
            
            ☐ **Spatial Considerations**
              - [ ] Adequate spatial coverage
              - [ ] Spatial autocorrelation assessed
              - [ ] Regional coherence verified
              - [ ] Coordinate systems consistent
            
            ☐ **Physical Plausibility**
              - [ ] Results consistent with known processes
              - [ ] Patterns make physical sense
              - [ ] Validated against independent data
              - [ ] Anomalies investigated
            
            ☐ **Documentation**
              - [ ] Methods fully described
              - [ ] Code archived and commented
              - [ ] Data sources cited
              - [ ] Assumptions stated explicitly
              - [ ] Limitations acknowledged
            
            ☐ **Visualization**
              - [ ] Maps clearly labeled
              - [ ] Legends comprehensive
              - [ ] Color schemes appropriate
              - [ ] Scale bars and north arrows included
              - [ ] High resolution for publication
            
            ☐ **Communication**
              - [ ] Executive summary prepared
              - [ ] Technical report complete
              - [ ] Stakeholder presentation ready
              - [ ] Recommendations actionable
            """)

        # Section 6: Knowledge Check
        with st.expander("## 6. SYNTHESIS AND ASSESSMENT", expanded=False):
            st.markdown("### 6.1 Integration of Modules 7, 8, and 9")
            
            st.markdown("""
            **Comprehensive Framework for Non-Stationarity Assessment:**
            
            This module synthesizes the analytical tools from Modules 7 and 8 into a complete
            workflow for regional water resources assessment:
            
            ---
            
            **MODULE 7: Trend Detection**
            - **What it does:** Detects gradual, monotonic changes over time
            - **Key method:** Mann-Kendall test + Sen's slope
            - **Output:** Trend magnitude, direction, significance
            - **When to use:** Suspected gradual climate change, urbanization effects
            
            ↓ *Apply at each station in network*
            
            **MODULE 9: Spatiotemporal Representation (Current)**
            - **What it does:** Maps spatial patterns of trends
            - **Key method:** Spatial visualization, interpolation, regional statistics
            - **Output:** Trend maps, regional patterns, priority areas
            - **Value added:** Regional perspective, investment prioritization
            
            ---
            
            **MODULE 8: Change Point Detection**
            - **What it does:** Identifies abrupt shifts in statistical properties
            - **Key method:** Pettitt test
            - **Output:** Change point location (τ), significance, magnitude
            - **When to use:** Known watershed modifications, suspected regime shifts
            
            ↓ *Apply at each station in network*
            
            **MODULE 9: Spatiotemporal Representation (Current)**
            - **What it does:** Maps change point locations and timing
            - **Key method:** Spatial visualization of change years
            - **Output:** Change point maps, temporal clusters
            - **Value added:** Identify regional vs. local changes
            
            ---
            
            **INTEGRATED WORKFLOW:**
            
            ```
            1. DATA PREPARATION (All Modules)
               └─ Quality control, synchronization, documentation
            
            2. POINT ANALYSIS (Modules 7 & 8)
               ├─ Trend detection at each station
               └─ Change point detection at each station
            
            3. SPATIAL ANALYSIS (Module 9)
               ├─ Create trend maps
               ├─ Create change point maps
               ├─ Identify regional patterns
               └─ Calculate spatial statistics
            
            4. INTERPRETATION (Module 9)
               ├─ Physical validation
               ├─ Regional coherence assessment
               └─ Engineering implications
            
            5. APPLICATION (Module 9)
               ├─ Prioritize infrastructure investments
               ├─ Update design standards regionally
               ├─ Develop monitoring strategies
               └─ Support policy development
            ```
            """)
            
            st.markdown("### 6.2 Comprehensive Case Study Assessment")
            
            result1 = QuizEngine.create_multiple_choice(
                "A regional analysis of 20 stations shows: (1) 60% have significant increasing trends in "
                "northern half, (2) 40% have significant change points clustered around year 2005 in central "
                "region, (3) Southern stations show no significant patterns. What is the most appropriate "
                "engineering response?",
                [
                    "Update design standards uniformly across entire region using post-2005 data",
                    "Maintain existing standards; patterns may reverse in future",
                    "Implement differentiated approach: North (trend-based updates), Center (post-2005 data), South (historical data)",
                    "Focus only on stations with both trends AND change points"
                ],
                2,
                {
                    "correct": "✅ Excellent regional thinking! The spatiotemporal analysis reveals three distinct "
                              "subregions requiring different approaches. Northern stations need trend-based projections, "
                              "central stations should split at the 2005 change point, and southern stations can "
                              "continue with historical statistics. A uniform approach would either over-design in the "
                              "south (wasting money) or under-design in the north (safety risk). This demonstrates why "
                              "spatiotemporal analysis is critical for efficient, safe regional water resources management.",
                    "incorrect": "Consider that different regions show different patterns, requiring tailored responses. "
                                "Think about the three distinct patterns revealed: increasing trends (north), change points "
                                "(center), and stability (south). Each region needs an approach specific to its observed "
                                "pattern. A uniform approach ignores valuable spatial information."
                },
                f"{self.info.id}_quiz1"
            )
            
            if result1:
                st.markdown("---")
                st.markdown("### 6.3 Final Application Challenge")
                
                result2 = QuizEngine.create_multiple_choice(
                    "You created a spatiotemporal trend map showing significant increasing trends in 12 of 18 "
                    "urban watersheds, but no trends in 15 of 15 rural watersheds in the same region. What does "
                    "this pattern most likely indicate, and what should you recommend?",
                    [
                        "Urban data quality is poor; re-analyze with better data",
                        "Urbanization is driving the trends; recommend green infrastructure in urban areas and updated urban drainage design",
                        "It's a statistical artifact from multiple testing; apply Bonferroni correction",
                        "Rural areas need more monitoring stations to detect their trends"
                    ],
                    1,
                    {
                        "correct": "✅ Outstanding inference! The stark urban/rural contrast strongly suggests urbanization "
                                  "as the physical cause - increased impervious surfaces, altered drainage patterns, and "
                                  "reduced infiltration in cities lead to increased runoff. This isn't a statistical issue "
                                  "(clear spatial pattern) or data quality problem (affects both urban and rural equally). "
                                  "Recommendations should address the cause: green infrastructure to restore natural "
                                  "infiltration and updated design standards for urban areas. This exemplifies how "
                                  "spatiotemporal analysis reveals causal mechanisms, not just patterns.",
                        "incorrect": "Think about what would cause such a clear spatial pattern aligned with land use. "
                                    "When patterns correlate with a known physical factor (urbanization), it's rarely "
                                    "a data quality or statistical artifact issue. Consider the physical processes in "
                                    "urban vs. rural watersheds and how they'd affect hydrology differently."
                    },
                    f"{self.info.id}_quiz2"
                )
                
                if result2:
                    st.success("🎉 Module 9 Complete! You've mastered spatiotemporal analysis for water resources engineering.")
                    st.success("🎓 **Modules 7, 8, and 9 Complete!** You now have a comprehensive toolkit for analyzing non-stationarity in hydrologic systems.")
                    return True
        
        # References
        with st.expander("## 📚 REFERENCES AND RESOURCES", expanded=False):
            st.markdown("""
            **Spatiotemporal Analysis Methods:**
            
            1. Cressie, N., & Wikle, C. K. (2011). *Statistics for Spatio-Temporal Data*. 
               John Wiley & Sons.
               [Comprehensive theoretical foundation]
            
            2. Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982). Techniques of trend analysis for 
               monthly water quality data. *Water Resources Research*, 18(1), 107-121.
               [Classic paper on regional trend assessment]
            
            **Hydrologic Applications:**
            
            3. Douglas, E. M., Vogel, R. M., & Kroll, C. N. (2000). Trends in floods and low flows 
               in the United States: impact of spatial correlation. *Journal of Hydrology*, 240(1-2), 90-105.
               [Addressing spatial correlation in regional analysis]
            
            4. Burn, D. H., & Elnur, M. A. H. (2002). Detection of hydrologic trends and variability. 
               *Journal of Hydrology*, 255(1-4), 107-122.
               [Methods for regional homogeneity]
            
            5. McCabe, G. J., & Wolock, D. M. (2002). A step increase in streamflow in the conterminous 
               United States. *Geophysical Research Letters*, 29(24), 2185.
               [National-scale change point analysis]
            
            **Climate Change Context:**
            
            6. Kunkel, K. E., et al. (2013). Monitoring and understanding trends in extreme storms: 
               State of knowledge. *Bulletin of the American Meteorological Society*, 94(4), 499-514.
               [Review of precipitation trend detection]
            
            7. Peterson, T. C., et al. (2008). Why weather and climate extremes matter. *Weather and 
               Climate Extremes*, 1, 1-2.
               [Motivation for extremes analysis]
            
            **Mapping and Visualization:**
            
            8. Tennekes, M. (2018). tmap: Thematic maps in R. *Journal of Statistical Software*, 84(6), 1-39.
               [R package for spatial visualization]
            
            9. Kahle, D., & Wickham, H. (2013). ggmap: Spatial visualization with ggplot2. 
               *The R Journal*, 5(1), 144-161.
               [Advanced mapping in R]
            
            **Engineering Guidelines:**
            
            10. ASCE (2017). *Manual of Practice No. 28: Hydrology Handbook* (3rd ed.). 
                American Society of Civil Engineers.
                [Standard reference for engineering hydrology]
            
            11. U.S. Army Corps of Engineers (2019). *Engineering and Design: Hydrologic Frequency Analysis*.
                EM 1110-2-1415.
                [Federal guidelines including non-stationarity]
            
            **Software and Tools:**
            
            12. QGIS Development Team. QGIS Geographic Information System. https://qgis.org
                [Free, open-source GIS software]
            
            13. ArcGIS by Esri. https://www.esri.com/en-us/arcgis/about-arcgis/overview
                [Commercial GIS platform]
            
            14. Python libraries: `geopandas`, `folium`, `plotly`, `cartopy`
            
            15. R packages: `sf`, `tmap`, `leaflet`, `mapview`, `ggplot2`
            """)
        
        return None


def main():
    """Standalone module test"""
    st.set_page_config(page_title="Module 11: Spatiotemporal Analysis", layout="wide")
    module = Module11_Spatiotemporal()
    module.render()


if __name__ == "__main__":
    main()

