# Module 11: Quick Reference Guide

## For Students: What You'll Learn

### 🎯 Main Goal
Understand how water data is organized: in **time**, in **space**, or in **both**!

---

## Three Types of Data

### 1. ⏰ TEMPORAL DATA (Time Only)
**One location → Many time points**

**What you'll see:**
- Line plot: Daily rainfall for 365 days
- Bar chart: Monthly totals
- Histogram: Distribution of annual values

**Real example:** 
"I measure rainfall every day at my house for a year"

**File format:** CSV, Excel spreadsheet

---

### 2. 🗺️ SPATIAL DATA (Space Only)
**Many locations → One time point**

**What you'll see:**
- **Point data**: 12 weather stations on a map (colored by rainfall)
- **Gridded data**: Elevation map (like a digital photograph)

**Real example:**
"Today's rainfall amount at 100 different weather stations"

**File formats:**
- Points: CSV with lat/lon
- Grids: GeoTIFF (.tif file)

---

### 3. 🎲 SPATIOTEMPORAL DATA (Space + Time)
**Many locations → Many time points**

**What you'll see:**
- The "DATA CUBE" concept (3D!)
- Interactive slicing:
  - Pick a day → see map
  - Pick a location → see time series

**Real example:**
"Satellite measures rainfall every day over entire USA for 10 years"

**File format:** NetCDF (.nc file)

**Structure:**
```
Dimensions: 100 lon × 100 lat × 3650 days
Total values: 36,500,000 data points!
```

---

## Interactive Demonstrations

### Demo 1: Temporal
- See 365 days of rainfall
- Watch it aggregate to months
- View distribution histogram

### Demo 2: Spatial
- 12 stations on map
- Color shows rainfall amount
- See elevation contours

### Demo 3: Spatiotemporal 
**YOU CONTROL:**
- Slider: Pick a day (1-30)
- Slider: Pick a location
- See BOTH views update!

---

## Quiz Questions

**Q1:** River flow at one gage for 2 years?
→ TEMPORAL ✓

**Q2:** Elevation map file (.tif)?
→ SPATIAL ✓

**Q3:** How many values in 100×100 grid for 10 years?
→ 100 × 100 × 3650 = 36,500,000 ✓

---

## Quick Reference Table

| Type | When to Use | How Big | Common Files |
|------|-------------|---------|--------------|
| **Temporal** | Tracking one location over time | Small | .csv, .xlsx |
| **Spatial** | Snapshot across area | Medium | .tif, .shp |
| **Spatiotemporal** | Tracking area over time | HUGE | .nc, .grib |

---

## Real-World Datasets You'll Encounter

### Temporal
- Stream gage data (USGS)
- Weather station records
- Your own field measurements

### Spatial
- Elevation maps (DEM)
- Land cover maps
- Soil property maps

### Spatiotemporal
- **TRMM**: Satellite rainfall (daily, global)
- **ERA5**: Climate reanalysis (hourly, everything)
- **CMIP6**: Future climate projections

---

## Tools You'll Use

### Python
```python
import xarray as xr

# Open NetCDF
data = xr.open_dataset('rainfall.nc')

# Get time series at point
ts = data.sel(lat=32.5, lon=-90.0)

# Get map for one day
map = data.sel(time='2024-01-15')
```

### Online Sources
- NASA Earthdata: earthdata.nasa.gov
- NOAA Climate: ncdc.noaa.gov
- Copernicus: climate.copernicus.eu

---

## Key Takeaways

1. **1D (Temporal)** = Array of values over time
2. **2D (Spatial)** = Grid of values across space
3. **3D (Spatiotemporal)** = Cube with time as 3rd dimension

**Remember:** You can "slice" the cube any way you want!
- Horizontal slice = map for one day
- Vertical slice = time series at one point
- Average = reduce dimensions

---

## Common Student Questions

**Q: Why is NetCDF better than many CSV files?**
A: One .nc file = organized 3D cube. Otherwise you'd need 365 separate CSV files for one year!

**Q: How do I visualize 3D data?**
A: You don't! You slice it into 2D (maps) or 1D (time series) views.

**Q: What's the difference between .tif and .nc?**
A: 
- .tif = 2D image (spatial only)
- .nc = 3D+ cube (spatial + temporal + more)

**Q: Do I need GIS software?**
A: Not always! Python (xarray) can handle NetCDF directly.

---

## Module Duration
⏱️ **30 minutes** (much shorter than old version!)

## Difficulty
📚 **Beginner** (no prerequisites beyond Module 1)

## What's Next?
After this module, you'll be ready to:
- Download satellite data
- Extract time series
- Create maps
- Understand climate datasets

**You're building the foundation for modern water resources analysis!** 🎓
