"""
Extract Forest Loss Statistics from Google Earth Engine
Uses Hansen Global Forest Change dataset to calculate deforestation rates per species
"""

import ee
import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys

# Initialize Earth Engine (you may need to specify project)
try:
    ee.Initialize()
    print("✅ Earth Engine initialized successfully")
except Exception as e:
    print(f"⚠️  Initialize failed: {e}")
    print("\nTrying with cloud project...")
    print("If you get a project error, run:")
    print("  ee.Initialize(project='your-project-id')")
    print("\nOr create a free project at: https://code.earthengine.google.com/")
    sys.exit(1)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data/raw"
DATA_PROCESSED = BASE_DIR / "data/processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Load Hansen Global Forest Change
print("\n📡 Loading Hansen Global Forest Change dataset...")
hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')

# Extract key bands
treecover2000 = hansen.select(['treecover2000'])  # Tree cover in 2000
loss = hansen.select(['loss'])  # Binary: was forest lost?
lossyear = hansen.select(['lossyear'])  # Year of loss (0-23 = 2000-2023)

print("✅ Hansen dataset loaded")
print("   - Tree cover 2000")
print("   - Forest loss 2001-2023")
print("")

def process_species_batch(species_gdf, batch_name="amphibians"):
    """
    Extract forest loss stats for each species range
    
    Args:
        species_gdf: GeoDataFrame with species ranges
        batch_name: Name for output file
    
    Returns:
        DataFrame with forest loss metrics per species
    """
    
    # Store results
    results = []
    total_species = len(species_gdf)
    
    print(f"\n🐸 Processing {total_species} {batch_name}...")
    print("=" * 70)
    
    # Iterate through each species
    for idx, row in species_gdf.iterrows():
        try:
            # Get species info
            species_name = row.get('binomial', row.get('sci_name', f'species_{idx}'))
            
            # Convert ICUN geometry to Earth Engine 
            geom = ee.Geometry(row.geometry.__geo_interface__)
            
            # Calculate forest cover in 2000
            forest_2000 = treecover2000.gte(10)  # >=10% canopy = forest
            forest_area_2000 = forest_2000.multiply(ee.Image.pixelArea()).reduceRegion( # Forest Pixel (0, 1) x 900m^2
                reducer=ee.Reducer.sum(), # sums up all pixels
                geometry=geom, # clips to species boundary
                scale=30,  # Hansen is 30m x 30m per pixel
                maxPixels=1e13
            ).get('treecover2000')
            
            # Calculate forest loss --> 1 if loss, 0 if none
            forest_loss = loss.multiply(ee.Image.pixelArea()).reduceRegion( # Loss x 900m^2
                reducer=ee.Reducer.sum(), # sums up all pixels
                geometry=geom, # clips to species boundary
                scale=30,
                maxPixels=1e13
            ).get('loss')
            
            # Get values in km^2
            forest_2000_km2 = ee.Number(forest_area_2000).divide(1e6).getInfo() if forest_area_2000 else 0 
            loss_km2 = ee.Number(forest_loss).divide(1e6).getInfo() if forest_loss else 0
            
            # Calculate rates
            loss_pct = (loss_km2 / forest_2000_km2 * 100) if forest_2000_km2 > 0 else 0
            loss_rate_annual = loss_pct / 23  # 2001-2023 = 23 years
            
            # Display results
            results.append({
                'species': species_name,
                'forest_2000_km2': round(forest_2000_km2, 2),
                'forest_loss_2001_2023_km2': round(loss_km2, 2),
                'forest_loss_pct': round(loss_pct, 2),
                'forest_loss_rate_annual': round(loss_rate_annual, 4)
            })
            
            # Progress
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{total_species} species...")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {species_name}: {e}")
            results.append({
                'species': species_name,
                'forest_2000_km2': None,
                'forest_loss_2001_2023_km2': None,
                'forest_loss_pct': None,
                'forest_loss_rate_annual': None,
                'error': str(e)
            })
    
    df = pd.DataFrame(results)
    print(f"\n✅ Processed {len(df)} species")
    print(f"   Mean forest loss: {df['forest_loss_pct'].mean():.2f}%")
    print(f"   Max forest loss: {df['forest_loss_pct'].max():.2f}%")
    
    return df


def main():
    """Main execution"""
    
    print("=" * 70)
    print("FOREST LOSS EXTRACTION - GOOGLE EARTH ENGINE")
    print("=" * 70)
    
    # Process Amphibians
    print("\n📂 Loading amphibians shapefile...")
    amphibians_path = DATA_RAW / "iucn_amphibians" / "AMPHIBIANS_PART1.shp"
    
    if amphibians_path.exists():
        amphibians = gpd.read_file(amphibians_path)
        print(f"   Loaded {len(amphibians)} amphibian species")
        
        # Extract forest loss
        amphibians_forest = process_species_batch(amphibians, "amphibians")
        
        # Save
        output_path = DATA_PROCESSED / "amphibians_forest_loss.csv"
        amphibians_forest.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print(f"   ⚠️  File not found: {amphibians_path}")
    
    # Process Mammals
    print("\n📂 Loading mammals shapefiles...")
    mammals_part1 = DATA_RAW / "iucn_mammals" / "MAMMALS_PART1.shp"
    mammals_part2 = DATA_RAW / "iucn_mammals" / "MAMMALS_PART2.shp"
    
    mammals_list = []
    if mammals_part1.exists():
        mammals_list.append(gpd.read_file(mammals_part1))
    if mammals_part2.exists():
        mammals_list.append(gpd.read_file(mammals_part2))
    
    if mammals_list:
        mammals = pd.concat(mammals_list, ignore_index=True)
        print(f"   Loaded {len(mammals)} mammal species")
        
        # Extract forest loss
        mammals_forest = process_species_batch(mammals, "mammals")
        
        # Save
        output_path = DATA_PROCESSED / "mammals_forest_loss.csv"
        mammals_forest.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print(f"   ⚠️  No mammal shapefiles found")
    
    # Process Reptiles
    print("\n📂 Loading reptiles shapefiles...")
    reptiles_part1 = DATA_RAW / "iucn_reptiles" / "REPTILES_PART1.shp"
    reptiles_part2 = DATA_RAW / "iucn_reptiles" / "REPTILES_PART2.shp"
    
    reptiles_list = []
    if reptiles_part1.exists():
        reptiles_list.append(gpd.read_file(reptiles_part1))
    if reptiles_part2.exists():
        reptiles_list.append(gpd.read_file(reptiles_part2))
    
    if reptiles_list:
        reptiles = pd.concat(reptiles_list, ignore_index=True)
        print(f"   Loaded {len(reptiles)} reptile species")
        
        # Extract forest loss
        reptiles_forest = process_species_batch(reptiles, "reptiles")
        
        # Save
        output_path = DATA_PROCESSED / "reptiles_forest_loss.csv"
        reptiles_forest.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print(f"   ⚠️  No reptile shapefiles found")
    
    print("\n" + "=" * 70)
    print("✅ FOREST LOSS EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
