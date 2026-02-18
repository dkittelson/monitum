#!/bin/bash
# Run complete Extinction Horizon pipeline

echo "========================================================================"
echo "EXTINCTION HORIZON - FULL PIPELINE"
echo "========================================================================"
echo ""

# Check Python environment
echo "📋 Checking Python environment..."
python --version
echo ""

# Install required packages
echo "📦 Installing required packages..."
pip install -q geopandas pandas numpy scikit-learn matplotlib seaborn rasterio tqdm mlflow joblib
echo "✅ Packages installed"
echo ""

# Step 1: Extract zonal statistics
echo "========================================================================"
echo "STEP 1: Extract Zonal Statistics"
echo "========================================================================"
python scripts/02_extract_zonal_stats.py
if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed!"
    exit 1
fi
echo ""

# Step 2: Calculate indices
echo "========================================================================"
echo "STEP 2: Calculate Vulnerability & Capacity Indices"
echo "========================================================================"
python scripts/03_calculate_indices.py
if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed!"
    exit 1
fi
echo ""

# Step 3: Train models
echo "========================================================================"
echo "STEP 3: Train Machine Learning Models"
echo "========================================================================"
python scripts/04_train_models.py
if [ $? -ne 0 ]; then
    echo "❌ Step 3 failed!"
    exit 1
fi
echo ""

# Summary
echo "========================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 Outputs:"
echo "   Data:    data/processed/"
echo "   Models:  models/"
echo "   Results: results/"
echo ""
echo "🔍 View results:"
echo "   - Feature importance: results/feature_importance_redlist.png"
echo "   - Resilience gap analysis: results/resilience_gap_analysis.png"
echo "   - Forgotten species: results/forgotten_species_list.csv"
echo ""
echo "📈 View MLflow experiments:"
echo "   mlflow ui"
echo "   Then open: http://localhost:5000"
echo ""
