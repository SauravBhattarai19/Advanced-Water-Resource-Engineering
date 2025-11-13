"""
Module 11: Spatiotemporal Representation in Water Resources
Understanding How Water Data Varies in Time and Space

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
from datetime import datetime, timedelta

from base_classes import LearningModule, ModuleInfo, LearningObjective, QuizEngine
from utilities import DataManager, AnalysisTools, PlotTools, UIComponents


class Module11_Spatiotemporal(LearningModule):
    """Module 11: Spatiotemporal Representation in Water Resources"""

    def __init__(self):
        objectives = [
            LearningObjective("Understand temporal data representation", "understand"),
            LearningObjective("Understand spatial data representation", "understand"),
            LearningObjective("Understand spatiotemporal (NetCDF) data structure", "understand"),
            LearningObjective("Choose appropriate data representation methods", "apply")
        ]

        info = ModuleInfo(
            id="module_11",
            title="Spatiotemporal Representation in Water Resources",
            description="Understanding how water data varies in time, space, and both",
            duration_minutes=20,
            prerequisites=["module_01"],
            learning_objectives=objectives,
            difficulty="beginner",
            total_slides=1
        )

        super().__init__(info)

    def get_slide_titles(self) -> List[str]:
        return ["Data Representation: Temporal → Spatial → Spatiotemporal"]

    def render_slide(self, slide_num: int) -> Optional[bool]:
        """Render the complete module"""
        return self._render_complete_module()

    def _render_complete_module(self) -> Optional[bool]:
        """Render complete module with simple, clear examples"""

        # Module title
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;'>
            <h1 style='margin: 0; color: white;'>📊 Module 11: Data Representation</h1>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1em;'>
                Temporal, Spatial, and Spatiotemporal
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Introduction
        with st.expander("🎯 **WHAT YOU'LL LEARN**", expanded=True):
            st.markdown("""
            Water resources data can be represented in three ways:

            ### 1. ⏰ TEMPORAL (Time Only)
            - Data at **ONE location** over **MANY times**
            - Example: Daily rainfall at Station A for one year
            - Visualize with: Line plots, bar charts

            ### 2. 🗺️ SPATIAL (Space Only)
            - Data at **MANY locations** at **ONE time**
            - Example: Rainfall at 10 stations on January 15, 2024
            - Visualize with: Maps (points or grids)

            ### 3. 🎲 SPATIOTEMPORAL (Time + Space)
            - Data at **MANY locations** over **MANY times**
            - Example: Daily rainfall at 10 stations for one year
            - Format: NetCDF files (3D data cubes)

            ---

            **Why This Matters:**
            Different types of data require different visualization and analysis approaches.
            """)

        # ==========================================
        # PART 1: TEMPORAL DATA
        # ==========================================
        st.markdown("## ⏰ Part 1: TEMPORAL DATA")
        st.markdown("**One location, many time points**")

        # Generate sample data
        np.random.seed(42)
        days = 365
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

        # Daily rainfall with seasonality
        seasonal = 30 + 20 * np.sin(2 * np.pi * np.arange(days) / 365)
        daily_rainfall = np.maximum(0, np.random.gamma(2, seasonal/10) - 2)

        daily_df = pd.DataFrame({
            'Date': dates,
            'Rainfall': daily_rainfall
        })

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Daily Data - Line Plot")

            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=daily_df['Date'],
                y=daily_df['Rainfall'],
                mode='lines',
                fill='tozeroy',
                line=dict(color='steelblue', width=1.5),
                fillcolor='rgba(70, 130, 180, 0.3)'
            ))

            fig_daily.update_layout(
                title="Daily Rainfall at Station A (2024)",
                xaxis_title="Date",
                yaxis_title="Rainfall (mm/day)",
                height=350,
                showlegend=False
            )

            fig_daily = PlotTools.apply_theme(fig_daily)
            st.plotly_chart(fig_daily, use_container_width=True)

        with col2:
            st.markdown("### Data Structure")
            st.code("""
Date       | Rainfall
-----------|---------
2024-01-01 | 12.5 mm
2024-01-02 | 8.3 mm
2024-01-03 | 0.0 mm
...        | ...
2024-12-31 | 15.2 mm

Total: 365 rows
            """)

            st.info("""
            **Key Point:**
            - ONE location (Station A)
            - MANY times (365 days)
            - Shows how rainfall **changes over time**
            """)

        # Monthly aggregation
        daily_df['Month'] = daily_df['Date'].dt.to_period('M')
        monthly_df = daily_df.groupby('Month').agg({'Rainfall': 'sum'}).reset_index()
        monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
        monthly_df['Month_Name'] = monthly_df['Month'].dt.strftime('%b')

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Monthly Data - Bar Chart")

            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_df['Month_Name'],
                y=monthly_df['Rainfall'],
                marker_color='teal',
                text=monthly_df['Rainfall'].round(0),
                textposition='outside'
            ))

            fig_monthly.update_layout(
                title="Monthly Total Rainfall (2024)",
                xaxis_title="Month",
                yaxis_title="Total Rainfall (mm/month)",
                height=350,
                showlegend=False
            )

            fig_monthly = PlotTools.apply_theme(fig_monthly)
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col2:
            st.markdown("### Data Structure")
            st.code("""
Month | Rainfall
------|----------
Jan   | 245 mm
Feb   | 198 mm
Mar   | 312 mm
...   | ...
Dec   | 267 mm

Total: 12 rows
            """)

            st.info("""
            **Key Point:**
            - Same location
            - Aggregated to monthly
            - Easier to see **seasonal pattern**
            """)

        st.markdown("---")

        # ==========================================
        # PART 2: SPATIAL DATA
        # ==========================================
        st.markdown("## 🗺️ Part 2: SPATIAL DATA")
        st.markdown("**Many locations, one time point**")

        # Generate station data
        np.random.seed(123)
        n_stations = 12

        # Station locations
        lats = np.array([32.0, 32.0, 32.0, 32.0, 32.5, 32.5, 32.5, 32.5, 33.0, 33.0, 33.0, 33.0])
        lons = np.array([-90.5, -90.0, -89.5, -89.0, -90.5, -90.0, -89.5, -89.0, -90.5, -90.0, -89.5, -89.0])

        # Add jitter
        lats = lats + np.random.normal(0, 0.05, n_stations)
        lons = lons + np.random.normal(0, 0.05, n_stations)

        # Rainfall on ONE day - spatial pattern (higher in west)
        base_rainfall = 50 - 15 * (lons + 90)
        rainfall_spatial = base_rainfall + np.random.normal(0, 5, n_stations)
        rainfall_spatial = np.maximum(0, rainfall_spatial)

        station_df = pd.DataFrame({
            'Station': [f'STN-{i+1:02d}' for i in range(n_stations)],
            'Lat': lats,
            'Lon': lons,
            'Rainfall': rainfall_spatial
        })

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Station/Point Data - Map")

            fig_spatial = go.Figure()

            fig_spatial.add_trace(go.Scatter(
                x=station_df['Lon'],
                y=station_df['Lat'],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=station_df['Rainfall'],
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Rainfall<br>(mm)"),
                    line=dict(width=1, color='black')
                ),
                text=station_df['Station'],
                textposition='top center',
                textfont=dict(size=8),
                hovertemplate='<b>%{text}</b><br>Rainfall: %{marker.color:.1f} mm<extra></extra>'
            ))

            fig_spatial.update_layout(
                title="Rainfall on January 15, 2024 (One Day Snapshot)",
                xaxis_title="Longitude (°)",
                yaxis_title="Latitude (°)",
                height=400,
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )

            fig_spatial = PlotTools.apply_theme(fig_spatial)
            st.plotly_chart(fig_spatial, use_container_width=True)

        with col2:
            st.markdown("### Data Structure")
            st.code("""
Station | Lat   | Lon    | Rain
--------|-------|--------|-----
STN-01  | 32.05 | -90.48 | 42.3
STN-02  | 31.98 | -90.02 | 35.8
STN-03  | 32.03 | -89.51 | 28.5
...     | ...   | ...    | ...

Total: 12 rows
            """)

            st.info("""
            **Key Point:**
            - MANY locations (12 stations)
            - ONE time (Jan 15, 2024)
            - Shows **spatial variation**
            - Darker blue = more rain
            """)

            st.dataframe(
                station_df[['Station', 'Rainfall']].round(1),
                use_container_width=True,
                height=200
            )

        # Gridded/Raster data
        st.markdown("### Gridded Data (TIFF/Raster) - Continuous Surface")

        # Create synthetic grid
        x = np.linspace(-90.5, -89.0, 100)
        y = np.linspace(32.0, 33.0, 100)
        X, Y = np.meshgrid(x, y)

        # Elevation pattern
        elevation = 100 + 80 * (X + 90) + 120 * (Y - 32) + 20 * np.sin(10 * X) * np.cos(10 * Y)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_grid = go.Figure(data=go.Contour(
                x=x,
                y=y,
                z=elevation,
                colorscale='Viridis',
                colorbar=dict(title="Elevation<br>(m)"),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=8, color='white')
                )
            ))

            fig_grid.update_layout(
                title="Elevation Map (DEM) - One Time Point",
                xaxis_title="Longitude (°)",
                yaxis_title="Latitude (°)",
                height=400,
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )

            fig_grid = PlotTools.apply_theme(fig_grid)
            st.plotly_chart(fig_grid, use_container_width=True)

        with col2:
            st.markdown("### Grid Structure")
            st.code("""
Grid: 100 × 100 cells
= 10,000 pixels

Each cell:
- Lat position
- Lon position
- Elevation value

File format:
- GeoTIFF (.tif)
- Each pixel = value
            """)

            st.info("""
            **Key Point:**
            - Continuous surface
            - Every pixel has a value
            - ONE time snapshot
            - Shows spatial variation
            """)

        st.markdown("---")

        # ==========================================
        # PART 3: SPATIOTEMPORAL DATA (NetCDF)
        # ==========================================
        st.markdown("## 🎲 Part 3: SPATIOTEMPORAL DATA (NetCDF)")
        st.markdown("**Many locations, many time points**")

        st.markdown("""
        ### What is NetCDF?

        **NetCDF** = Network Common Data Form

        **Think of it as a 3D DATA CUBE:**
        - **X-axis:** Longitude (space)
        - **Y-axis:** Latitude (space)
        - **Z-axis:** Time
        """)

        # Create visual representation
        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.markdown("### 3D Data Cube Visualization")

            # Create a simple 3D representation
            st.code("""
    TIME DIMENSION (365 days)
    ↓ ↓ ↓

    Day 1       Day 2       Day 3       ...    Day 365
    ┌─────┐    ┌─────┐    ┌─────┐            ┌─────┐
    │GRID │    │GRID │    │GRID │            │GRID │
    │100x │    │100x │    │100x │            │100x │
    │120  │    │120  │    │120  │            │120  │
    └─────┘    └─────┘    └─────┘            └─────┘

    Each GRID = Spatial map (Lat × Lon)

    DIMENSIONS:
    - latitude:  100 points (32°N to 34°N)
    - longitude: 120 points (91°W to 88°W)
    - time:      365 days (Jan 1 to Dec 31)

    Total data points: 100 × 120 × 365 = 4,380,000
            """, language="text")

            # Generate sample spatiotemporal data for visualization
            n_times = 12  # 12 months
            n_lats = 10
            n_lons = 10

            times = pd.date_range('2024-01-01', periods=n_times, freq='M')
            lats_grid = np.linspace(32, 33, n_lats)
            lons_grid = np.linspace(-90.5, -89.5, n_lons)

            # Create sample data for one time slice
            X_grid, Y_grid = np.meshgrid(lons_grid, lats_grid)
            sample_data = 30 + 20 * np.sin(2 * np.pi * 6 / 12) - 10 * (X_grid + 90)

            st.markdown("### Example: One Time Slice (June 2024)")

            fig_slice = go.Figure(data=go.Heatmap(
                x=lons_grid,
                y=lats_grid,
                z=sample_data,
                colorscale='Blues',
                colorbar=dict(title="Rainfall<br>(mm)")
            ))

            fig_slice.update_layout(
                title="Rainfall Grid - June 2024",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=300
            )

            fig_slice = PlotTools.apply_theme(fig_slice)
            st.plotly_chart(fig_slice, use_container_width=True)

        with col2:
            st.markdown("### NetCDF Structure")

            st.code("""
netcdf rainfall_2024 {
dimensions:
    time = 365 ;
    lat = 100 ;
    lon = 120 ;

variables:
    float precip(time, lat, lon);
        units = "mm/day" ;

    float lat(lat);
        units = "degrees_north" ;

    float lon(lon);
        units = "degrees_east" ;

    int time(time);
        units = "days since
                 2024-01-01" ;
}
            """, language="text")

            st.info("""
            **File Extension:** `.nc` or `.nc4`

            **Contains:**
            - Coordinates (lat, lon, time)
            - Data values (precipitation)
            - Metadata (units, descriptions)
            """)

        st.markdown("### How to Extract Data from NetCDF")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Extract ONE time slice (spatial map)")
            st.code("""
import xarray as xr

# Open file
ds = xr.open_dataset('rainfall.nc')

# Get data for Jan 15, 2024
day_map = ds['precip'].sel(
    time='2024-01-15'
)
# Returns: 100 × 120 grid

day_map.plot()
            """, language="python")

        with col2:
            st.markdown("#### Extract ONE location (time series)")
            st.code("""
import xarray as xr

# Open file
ds = xr.open_dataset('rainfall.nc')

# Get time series at one point
point_ts = ds['precip'].sel(
    lat=32.5,
    lon=-90.0,
    method='nearest'
)
# Returns: 365 values

point_ts.plot()
            """, language="python")

        st.markdown("### Common NetCDF Datasets")

        datasets = pd.DataFrame({
            'Dataset': ['ERA5', 'CHIRPS', 'PRISM', 'NLDAS-2'],
            'Type': ['Reanalysis', 'Satellite', 'Observations', 'Model'],
            'Resolution': ['~31 km', '5 km', '4 km', '12 km'],
            'Coverage': ['Global', 'Global', 'USA only', 'USA only'],
            'Period': ['1950-now', '1981-now', '1895-now', '1979-now']
        })

        st.dataframe(datasets, use_container_width=True, hide_index=True)

        st.success("""
        **Summary:**
        - **Temporal:** Line plots and bar charts for ONE location over time
        - **Spatial:** Maps (points or grids) for MANY locations at ONE time
        - **Spatiotemporal:** NetCDF files = 3D cubes with space AND time
        """)

        st.markdown("---")

        # ==========================================
        # KNOWLEDGE CHECK
        # ==========================================
        st.markdown("## 🎓 Knowledge Check")

        result1 = QuizEngine.create_multiple_choice(
            "You have daily rainfall measurements at Station A for the year 2024. What type of data is this?",
            [
                "Spatial - it's a map",
                "Temporal - one location, many times",
                "Spatiotemporal - it has coordinates",
                "None of the above"
            ],
            1,
            {
                "correct": "✅ Correct! This is TEMPORAL data because you have ONE location (Station A) "
                          "measured over MANY times (365 days in 2024).",
                "incorrect": "Think about the dimensions: You have data from just ONE station (no spatial variation) "
                            "but MANY time points (daily for one year). This is temporal data."
            },
            f"{self.info.id}_quiz1"
        )

        if result1:
            st.markdown("---")

            result2 = QuizEngine.create_multiple_choice(
                "You have a GeoTIFF file showing elevation across a watershed. What type of data is this?",
                [
                    "Temporal - elevation changes over time",
                    "Spatial - many locations at one time",
                    "Spatiotemporal - it's a grid",
                    "It depends on the file size"
                ],
                1,
                {
                    "correct": "✅ Exactly! Elevation is SPATIAL data - it varies across MANY locations "
                              "but is measured at ONE point in time (elevation doesn't change rapidly).",
                    "incorrect": "Elevation is a property that varies across space (different elevations at different "
                                "locations) but is relatively constant in time. This is spatial data."
                },
                f"{self.info.id}_quiz2"
            )

            if result2:
                st.markdown("---")

                result3 = QuizEngine.create_multiple_choice(
                    "A NetCDF file has dimensions [time: 365, lat: 50, lon: 60]. How many total data values does it contain?",
                    [
                        "365 values",
                        "3,000 values (50 × 60)",
                        "1,095,000 values (365 × 50 × 60)",
                        "It depends on the variable"
                    ],
                    2,
                    {
                        "correct": "✅ Perfect! Total values = time × lat × lon = 365 × 50 × 60 = 1,095,000. "
                                  "This is a 3D cube: 365 time steps, each containing a 50×60 spatial grid.",
                        "incorrect": "Remember: NetCDF is a 3D cube. Multiply ALL three dimensions together: "
                                    "time × latitude × longitude = total number of values."
                    },
                    f"{self.info.id}_quiz3"
                )

                if result3:
                    st.success("🎉 Congratulations! You've completed Module 11!")
                    st.balloons()

                    st.info("""
                    **What You've Learned:**

                    ✅ **Temporal data** = One location, many times → Use line/bar charts

                    ✅ **Spatial data** = Many locations, one time → Use maps (points or grids)

                    ✅ **Spatiotemporal data** = Many locations, many times → Use NetCDF (3D cubes)

                    ✅ How to visualize each type appropriately

                    ✅ Understanding NetCDF structure and how to extract data

                    **Next Steps:** Apply these concepts to real water resources datasets!
                    """)
                    return True

        return None


def main():
    """Standalone module test"""
    st.set_page_config(page_title="Module 11: Data Representation", layout="wide")
    module = Module11_Spatiotemporal()
    module.render()


if __name__ == "__main__":
    main()
