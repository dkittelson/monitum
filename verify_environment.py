#!/usr/bin/env python3
"""
Verify that all required packages for Monitum are installed correctly.
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:25s} {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:25s} NOT FOUND")
        return False

def main():
    print("=" * 60)
    print("Monitum Environment Verification")
    print("=" * 60)
    print()
    
    packages = [
        # Core data science
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        
        # Geospatial
        ('geopandas', 'geopandas'),
        ('rasterio', 'rasterio'),
        ('rasterstats', 'rasterstats'),
        ('shapely', 'shapely'),
        ('fiona', 'fiona'),
        ('pyproj', 'pyproj'),
        
        # Google Earth Engine
        ('earthengine-api', 'ee'),
        ('geemap', 'geemap'),
        
        # ML & Tracking
        ('mlflow', 'mlflow'),
        ('joblib', 'joblib'),
        
        # Visualization
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        
        # Utilities
        ('tqdm', 'tqdm'),
    ]
    
    print("Checking packages:")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("=" * 60)
    print(f"Results: {success_count} installed, {fail_count} missing")
    print("=" * 60)
    
    if fail_count > 0:
        print()
        print("⚠️  Some packages are missing. Try:")
        print("    conda activate monitum")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    else:
        print()
        print("🎉 All packages installed successfully!")
        print()
        print("You're ready to run the pipeline:")
        print("    bash scripts/run_pipeline.sh")
        sys.exit(0)

if __name__ == "__main__":
    main()
