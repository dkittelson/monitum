#!/bin/bash
# Setup script for Monitum environment

echo "🌿 Monitum Environment Setup"
echo "=============================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^monitum "; then
    echo "✅ Environment 'monitum' already exists"
    echo ""
    echo "To activate it, run:"
    echo "    conda activate monitum"
    echo ""
    echo "To remove and reinstall, run:"
    echo "    conda env remove -n monitum"
    echo "    bash setup_environment.sh"
else
    echo "📦 Creating conda environment 'monitum'..."
    conda env create -f environment.yml
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Environment created successfully!"
        echo ""
        echo "To activate it, run:"
        echo "    conda activate monitum"
    else
        echo ""
        echo "❌ Environment creation failed"
        exit 1
    fi
fi

echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate monitum"
echo "2. (Optional) Authenticate with Google Earth Engine: python scripts/00_setup_gee.py"
echo "3. Run the pipeline: bash scripts/run_pipeline.sh"
