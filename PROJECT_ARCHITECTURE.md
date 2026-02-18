# Extinction Horizon: Project Architecture

## Objective
Build a **mismatch analysis + predictive modeling** system to identify:
1. **Conservation Resilience Gap**: Species with high biological risk but low protection
2. **Extinction Horizon Forecast**: Which species will move to higher threat categories by 2035

---

## Data Architecture (5 Layers)

### Layer 1: Species Baseline (IUCN)
- **Source**: IUCN Red List API + Spatial Downloads
- **Key Fields**: 
  - `scientificName`, `redlistCategory`, `populationTrend`
  - `generationLength`, `habitatTypes`, `threats_encoded`
  - `range_area_km2`, `countries`

### Layer 2: Protection Supply (WDPA)
- **Source**: Protected Planet WDPA
- **Derived Metric**: `protected_area_coverage_pct`
  - Calculated via spatial intersection: `(range ∩ protected_areas) / range_area`

### Layer 3: Threat Intensity (HFI)
- **Source**: NASA SEDAC Human Footprint
- **Derived Metrics**:
  - `mean_hfi_2020`: Average HFI across species range
  - `hfi_change_2009_2020`: Rate of human encroachment
  - `hfi_velocity`: % change per year

### Layer 4: Climate Pressure (WorldClim)
- **Source**: WorldClim BioClim + CMIP6 Projections
- **Derived Metrics**:
  - `temp_change_2020_2050`: ΔT in habitat
  - `precip_seasonality_change`
  - `climate_velocity_kmyr`: Speed of isotherm shift

### Layer 5: Habitat Loss (Hansen)
- **Source**: Global Forest Watch / Hansen
- **Derived Metrics**:
  - `forest_loss_2015_2023_pct`: % of range deforested
  - `forest_loss_rate_annual`: Yearly % loss

---

## Feature Engineering (The "DataNation" Move)

### 1. Biological Vulnerability Index (0-1)
```python
vulnerability_index = (
    0.30 * normalize(hfi_velocity) +
    0.25 * normalize(forest_loss_rate) +
    0.20 * normalize(climate_velocity) +
    0.15 * (1 - normalize(generation_length)) +  # Slow reproduction = higher risk
    0.10 * threat_count_normalized
)
```

### 2. Conservation Capacity Score (0-1)
```python
capacity_score = (
    0.50 * protected_area_coverage_pct +
    0.30 * country_conservation_funding_pct +
    0.20 * (1 - mean_hfi_2020_normalized)  # Lower human pressure = better capacity
)
```

### 3. Resilience Gap (Mismatch Score)
```python
resilience_gap = vulnerability_index - capacity_score
# High vulnerability + Low capacity = Large positive gap = PRIORITY
```

---

## Model Architecture

### Model 1: Red List Forecaster (Random Forest / Gradient Boosting)
**Task**: Predict 2035 IUCN Category

**Target Variable**: 
- Current: `redlistCategory_2024` (LC=0, NT=1, VU=2, EN=3, CR=4)
- Future: `predicted_redlistCategory_2035`

**Features (X)**:
- `hfi_velocity`, `forest_loss_rate`, `climate_velocity`
- `protected_area_coverage_pct`
- `generation_length`, `range_area_km2`
- `populationTrend_encoded` (Decreasing=1, Stable=0)
- `threat_count`, `habitat_specificity`

**Training Strategy**:
- Train on species with **historical status changes** (2004→2014→2024)
- Validate on held-out species
- Test R² target: ~0.40-0.60 (explainable variance)

**Output**: 
- "142 species currently listed as 'Least Concern' will likely move to 'Vulnerable' by 2035"

---

### Model 2: Forgotten Species Identifier (Residual Analysis)
**Task**: Find species with highest `resilience_gap`

**Method**:
1. Scatter plot: `vulnerability_index` (X) vs `capacity_score` (Y)
2. Fit linear regression: `capacity_score ~ vulnerability_index`
3. Calculate residuals: `residual = actual_capacity - predicted_capacity`
4. Flag species with **large negative residuals** (high risk, low protection)

**Key Insight** (like DataNation):
> "The Jaguar has vulnerability=0.72 but capacity=0.68 (residual=-0.04). The Glass Frog has vulnerability=0.81 but capacity=0.22 (residual=-0.59). The Glass Frog is systematically neglected."

---

## Deliverables (Competition-Ready)

### 1. Extinction Horizon Index 2035
A ranked table of top 50 species by `resilience_gap`:

| Rank | Species | Vulnerability | Capacity | Gap | Current Status | Predicted 2035 |
|------|---------|--------------|----------|-----|----------------|----------------|
| 1 | *Atelopus zeteki* | 0.89 | 0.12 | 0.77 | CR | EX |
| 2 | *Nannophryne cophotis* | 0.84 | 0.19 | 0.65 | EN | CR |

### 2. Feature Importance Analysis
- Which threats drive vulnerability most? (Deforestation vs Climate vs HFI)
- Which interventions close the gap fastest? (Protected areas vs Corridor creation)

### 3. Geographic Hotspots
- Heatmap: "The Amazon Basin has 89 'forgotten species' vs 12 in Africa"
- Country-level mismatch: "Colombia has high vulnerability but low capacity"

### 4. Sector-Level Analysis (Like DataNation's Cluster Framework)
- Taxonomic groups: Amphibians vs Mammals vs Birds
- Threat types: Deforestation-driven vs Climate-driven vs Overexploitation
- Conservation efficiency: $ per species saved

### 5. Temporal Trends
- "Species in the top decile of `hfi_velocity` declined 3.2x faster than those in the bottom decile"

---

## Technical Stack

- **Geospatial**: `geopandas`, `rasterio`, `rasterstats`
- **ML**: `scikit-learn`, `xgboost`, `lightgbm`
- **Tracking**: `mlflow` (log all experiments)
- **Viz**: `matplotlib`, `seaborn`, `plotly`

---

## Next Steps (Week 1)

1. **Download Data** (Days 1-2)
   - IUCN species traits CSV
   - WDPA global shapefile
   - HFI 2009 + 2020 rasters
   - WorldClim current + 2050 rasters
   - Hansen forest loss tiles (Amazon, SE Asia, Congo)

2. **Build Feature Pipeline** (Days 3-4)
   - Script: `01_extract_zonal_stats.py`
   - For each species range polygon:
     - Calculate mean HFI, forest loss %, protected coverage
     - Extract BioClim variables
   - Output: `species_features.csv`

3. **Engineer Indices** (Day 5)
   - Script: `02_calculate_indices.py`
   - Compute vulnerability, capacity, resilience gap
   - Output: `species_with_gap_scores.csv`

4. **Train Models** (Days 6-7)
   - Script: `03_train_forecaster.py`
   - Train Red List Forecaster with cross-validation
   - Log experiments with MLflow
   - Script: `04_residual_analysis.py`
   - Fit regression, calculate residuals, rank "forgotten species"

---

## Validation Strategy (Critical for Credibility)

1. **Temporal Validation**: 
   - Train on 2004-2014 data, predict 2024 status
   - Compare predictions to actual 2024 Red List

2. **Cross-Validation**:
   - K-Fold by taxonomic family (prevent data leakage)

3. **Sensitivity Analysis**:
   - Re-run with different index weights
   - Report: "Top 20 species stable across 4+ weighting schemes"

4. **Expert Validation**:
   - Compare "forgotten species" list to EDGE of Existence priorities
   - Cite alignment: "78% of our top 30 match EDGE priorities"

---

## Why This Wins

✅ Uses their **exact winning architecture** (mismatch + residuals)  
✅ **Feature engineering** at their level (vulnerability indices, gap scores)  
✅ **Two complementary models** (forecaster + anomaly detection)  
✅ **Transparent methodology** (every metric is explainable)  
✅ **Actionable outputs** (ranked priority list for conservationists)  
✅ **Rigorous validation** (temporal split, sensitivity analysis)  

You're not just "doing ML on wildlife data."  
You're building **The INFORM Severity Index for biodiversity.**
