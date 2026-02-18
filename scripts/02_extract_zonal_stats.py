"""
Extract Zonal Statistics from Rasters
Calculates mean values of climate, human footprint, and protection for each species range

This script processes:
1. Climate variables (WorldClim baseline + future) → climate velocity
2. Human Footprint (2010 + 2020) → human pressure velocity  
3. Protected area coverage (WDPA) → protection score
4. Forest loss (optional, if Hansen data available)

Output: species_features_raw.csv with all extracted values
"""

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data/raw"
DATA_PROCESSED = BASE_DIR / "data/processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Climate variables to extract (key BioClim variables)
BIOCLIM_VARS = {
    'bio_1': 'annual_mean_temp',
    'bio_5': 'max_temp_warmest_month',
    'bio_6': 'min_temp_coldest_month',
    'bio_12': 'annual_precipitation',
    'bio_15': 'precipitation_seasonality'
}

def extract_raster_stats(geometry, raster_path, stat='mean'):
    """
    Extract statistics from raster for a given geometry
    
    Args:
        geometry: Shapely geometry object
        raster_path: Path to raster file
        stat: Statistic to calculate ('mean', 'sum', 'median', 'std')
    
    Returns:
        float: Calculated statistic or None if error
    """
    try:
        with rasterio.open(raster_path) as src:
            # Mask raster with geometry
            out_image, out_transform = mask(src, [geometry], crop=True, nodata=src.nodata)
            
            # Remove nodata values
            data = out_image[0]
            if src.nodata is not None:
                data = data[data != src.nodata]
            
            # Calculate statistic
            if len(data) == 0:
                return None
            
            if stat == 'mean':
                return float(np.mean(data))
            elif stat == 'sum':
                return float(np.sum(data))
            elif stat == 'median':
                return float(np.median(data))
            elif stat == 'std':
                return float(np.std(data))
            else:
                return float(np.mean(data))
                
    except Exception as e:
        print(f"  ⚠️  Error extracting from {raster_path.name}: {e}")
        return None


def calculate_protected_coverage(species_geom, wdpa_gdf):
    """
    Calculate percentage of species range within protected areas
    
    Args:
        species_geom: Species range geometry
        wdpa_gdf: GeoDataFrame of protected areas
    
    Returns:
        float: Percentage protected (0-100)
    """
    try:
        # Find intersecting protected areas
        intersecting = wdpa_gdf[wdpa_gdf.intersects(species_geom)]
        
        if len(intersecting) == 0:
            return 0.0
        
        # Calculate intersection area
        species_area = species_geom.area
        protected_area = intersecting.union_all().intersection(species_geom).area
        
        coverage = (protected_area / species_area) * 100
        return min(coverage, 100.0)  # Cap at 100%
        
    except Exception as e:
        print(f"  ⚠️  Error calculating protection: {e}")
        return None


def process_species_batch(species_gdf, wdpa_gdf=None, batch_name="species"):
    """
    Extract all features for a batch of species
    
    Args:
        species_gdf: GeoDataFrame with species ranges
        wdpa_gdf: GeoDataFrame with protected areas (optional)
        batch_name: Name for progress display
    
    Returns:
        DataFrame with extracted features
    """
    
    print(f"\n🔬 Processing {len(species_gdf)} {batch_name}...")
    print("=" * 70)
    
    # Paths to rasters
    worldclim_baseline = DATA_RAW / "worldclim/baseline_1970_2000"
    worldclim_future = DATA_RAW / "worldclim/wc2.1_2.5m_bioc_HadGEM3-GC31-LL_ssp245_2041-2060.tif"
    hfi_2010 = DATA_RAW / "human_footprint/hii_2010-01-01.tif"
    hfi_2020 = DATA_RAW / "human_footprint/hii_2020-01-01.tif"
    
    # Check which datasets are available
    has_worldclim = worldclim_baseline.exists()
    has_worldclim_future = worldclim_future.exists()
    has_hfi = hfi_2010.exists() and hfi_2020.exists()
    has_wdpa = wdpa_gdf is not None
    
    print(f"📊 Available datasets:")
    print(f"   {'✅' if has_worldclim else '❌'} WorldClim Baseline")
    print(f"   {'✅' if has_worldclim_future else '❌'} WorldClim Future")
    print(f"   {'✅' if has_hfi else '❌'} Human Footprint (2010 & 2020)")
    print(f"   {'✅' if has_wdpa else '❌'} WDPA Protected Areas")
    print()
    
    results = []
    
    for idx, row in tqdm(species_gdf.iterrows(), total=len(species_gdf), desc="Extracting features"):
        try:
            # Get species info
            species_name = row.get('binomial', row.get('sci_name', f'species_{idx}'))
            iucn_category = row.get('category', row.get('code', 'Unknown'))
            geometry = row.geometry
            
            # Initialize feature dict
            features = {
                'species': species_name,
                'iucn_category': iucn_category,
                'range_area_km2': geometry.area / 1e6 if hasattr(geometry, 'area') else None
            }
            
            # Extract WorldClim baseline (current climate)
            if has_worldclim:
                for var_code, var_name in BIOCLIM_VARS.items():
                    raster_file = worldclim_baseline / f"wc2.1_2.5m_{var_code}.tif"
                    if raster_file.exists():
                        value = extract_raster_stats(geometry, raster_file)
                        features[f'{var_name}_baseline'] = value
            
            # Extract WorldClim future (to calculate velocity)
            # Note: Future file has all 19 BioClim vars in one multi-band raster
            if has_worldclim_future:
                try:
                    with rasterio.open(worldclim_future) as src:
                        # Extract Bio1 (band 1), Bio5 (band 5), Bio6 (band 6), Bio12 (band 12)
                        band_mapping = {1: 'annual_mean_temp', 5: 'max_temp_warmest_month', 
                                       6: 'min_temp_coldest_month', 12: 'annual_precipitation'}
                        for band_num, var_name in band_mapping.items():
                            out_image, _ = mask(src, [geometry], crop=True, nodata=src.nodata, indexes=band_num)
                            data = out_image[0]
                            if src.nodata is not None:
                                data = data[data != src.nodata]
                            if len(data) > 0:
                                features[f'{var_name}_future'] = float(np.mean(data))
                except Exception as e:
                    print(f"  ⚠️  Error extracting future climate for {species_name}: {e}")
            
            # Extract Human Footprint
            if has_hfi:
                hfi_2010_val = extract_raster_stats(geometry, hfi_2010)
                hfi_2020_val = extract_raster_stats(geometry, hfi_2020)
                features['hfi_2010'] = hfi_2010_val
                features['hfi_2020'] = hfi_2020_val
            
            # Calculate protected area coverage
            if has_wdpa:
                features['protected_area_coverage_pct'] = calculate_protected_coverage(geometry, wdpa_gdf)
            
            results.append(features)
            
        except Exception as e:
            print(f"\n⚠️  Error processing {species_name}: {e}")
            results.append({
                'species': species_name,
                'iucn_category': iucn_category,
                'error': str(e)
            })
    
    df = pd.DataFrame(results)
    print(f"\n✅ Extracted features for {len(df)} species")
    print(f"   Features per species: {len(df.columns)}")
    print(f"   Missing values: {df.isnull().sum().sum()} / {df.size}")
    
    return df


def main():
    """Main execution"""
    
    print("=" * 70)
    print("ZONAL STATISTICS EXTRACTION")
    print("=" * 70)
    
    # Load WDPA (if available)
    print("\n📂 Loading WDPA Protected Areas...")
    wdpa_gdf = None
    wdpa_shp = DATA_RAW / "protected_areas"
    
    # Try to find WDPA shapefile (might be .gdb or .shp)
    wdpa_candidates = list(wdpa_shp.glob("*.shp")) + list(wdpa_shp.glob("*.gdb"))
    if wdpa_candidates:
        try:
            wdpa_path = wdpa_candidates[0]
            print(f"   Loading: {wdpa_path.name}")
            wdpa_gdf = gpd.read_file(wdpa_path)
            print(f"   ✅ Loaded {len(wdpa_gdf):,} protected areas")
        except Exception as e:
            print(f"   ⚠️  Error loading WDPA: {e}")
            print(f"   Continuing without protected area data...")
    else:
        print("   ⚠️  WDPA shapefile not found")
        print("   Continuing without protected area data...")
    
    # Process Amphibians
    print("\n" + "=" * 70)
    print("AMPHIBIANS")
    print("=" * 70)
    
    amphibians_path = DATA_RAW / "iucn_amphibians" / "AMPHIBIANS_PART1.shp"
    if amphibians_path.exists():
        print(f"📂 Loading: {amphibians_path}")
        amphibians = gpd.read_file(amphibians_path)
        print(f"   Loaded {len(amphibians):,} amphibian species")
        
        # Extract features
        amphibians_features = process_species_batch(amphibians, wdpa_gdf, "amphibians")
        
        # Save
        output_path = DATA_PROCESSED / "amphibians_features_raw.csv"
        amphibians_features.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print(f"❌ Not found: {amphibians_path}")
    
    # Process Mammals
    print("\n" + "=" * 70)
    print("MAMMALS")
    print("=" * 70)
    
    mammals_parts = [
        DATA_RAW / "iucn_mammals" / "MAMMALS_PART1.shp",
        DATA_RAW / "iucn_mammals" / "MAMMALS_PART2.shp"
    ]
    
    mammals_list = []
    for part_path in mammals_parts:
        if part_path.exists():
            print(f"📂 Loading: {part_path}")
            mammals_list.append(gpd.read_file(part_path))
    
    if mammals_list:
        mammals = pd.concat(mammals_list, ignore_index=True)
        print(f"   Loaded {len(mammals):,} mammal species (combined)")
        
        # Extract features
        mammals_features = process_species_batch(mammals, wdpa_gdf, "mammals")
        
        # Save
        output_path = DATA_PROCESSED / "mammals_features_raw.csv"
        mammals_features.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print("❌ No mammal shapefiles found")
    
    # Process Reptiles
    print("\n" + "=" * 70)
    print("REPTILES")
    print("=" * 70)
    
    reptiles_parts = [
        DATA_RAW / "iucn_reptiles" / "REPTILES_PART1.shp",
        DATA_RAW / "iucn_reptiles" / "REPTILES_PART2.shp"
    ]
    
    reptiles_list = []
    for part_path in reptiles_parts:
        if part_path.exists():
            print(f"📂 Loading: {part_path}")
            reptiles_list.append(gpd.read_file(part_path))
    
    if reptiles_list:
        reptiles = pd.concat(reptiles_list, ignore_index=True)
        print(f"   Loaded {len(reptiles):,} reptile species (combined)")
        
        # Extract features
        reptiles_features = process_species_batch(reptiles, wdpa_gdf, "reptiles")
        
        # Save
        output_path = DATA_PROCESSED / "reptiles_features_raw.csv"
        reptiles_features.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print("❌ No reptile shapefiles found")
    
    print("\n" + "=" * 70)
    print("✅ ZONAL STATISTICS EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Output files:")
    print(f"   - {DATA_PROCESSED / 'amphibians_features_raw.csv'}")
    print(f"   - {DATA_PROCESSED / 'mammals_features_raw.csv'}")
    print(f"   - {DATA_PROCESSED / 'reptiles_features_raw.csv'}")
    print(f"\n🚀 Next step: Run scripts/03_calculate_indices.py")


if __name__ == "__main__":
    main()
