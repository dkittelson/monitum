"""
Calculate Vulnerability, Capacity, and Resilience Gap Indices

This script takes the raw features from zonal statistics and engineers the core indices:
1. Biological Vulnerability Index - combines threat metrics
2. Conservation Capacity Score - combines protection metrics  
3. Resilience Gap - the mismatch between vulnerability and capacity

Output: species_with_indices.csv ready for ML modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data/processed"

def calculate_climate_velocity(df):
    """
    Calculate rate of climate change
    
    climate_velocity = (future_temp - baseline_temp) / years
    """
    print("\n🌡️  Calculating climate velocity...")
    
    # Temperature velocity (°C per year)
    if 'annual_mean_temp_baseline' in df.columns and 'annual_mean_temp_future' in df.columns:
        df['temp_velocity'] = (df['annual_mean_temp_future'] - df['annual_mean_temp_baseline']) / 50
        print(f"   Mean temp velocity: {df['temp_velocity'].mean():.4f} °C/year")
    else:
        df['temp_velocity'] = np.nan
        print("   ⚠️  Temperature data not available")
    
    # Precipitation velocity (mm per year)
    if 'annual_precipitation_baseline' in df.columns and 'annual_precipitation_future' in df.columns:
        df['precip_velocity'] = (df['annual_precipitation_future'] - df['annual_precipitation_baseline']) / 50
        print(f"   Mean precip velocity: {df['precip_velocity'].mean():.2f} mm/year")
    else:
        df['precip_velocity'] = np.nan
        print("   ⚠️  Precipitation data not available")
    
    return df


def calculate_human_footprint_velocity(df):
    """
    Calculate rate of human encroachment
    
    hfi_velocity = (hfi_2020 - hfi_2010) / 10 years
    """
    print("\n🏙️  Calculating human footprint velocity...")
    
    if 'hfi_2010' in df.columns and 'hfi_2020' in df.columns:
        df['hfi_velocity'] = (df['hfi_2020'] - df['hfi_2010']) / 10
        df['hfi_change_pct'] = ((df['hfi_2020'] - df['hfi_2010']) / (df['hfi_2010'] + 1)) * 100
        
        print(f"   Mean HFI velocity: {df['hfi_velocity'].mean():.4f} points/year")
        print(f"   Mean HFI change: {df['hfi_change_pct'].mean():.2f}%")
        print(f"   Species with increasing HFI: {(df['hfi_velocity'] > 0).sum()} ({(df['hfi_velocity'] > 0).sum() / len(df) * 100:.1f}%)")
    else:
        df['hfi_velocity'] = np.nan
        df['hfi_change_pct'] = np.nan
        print("   ⚠️  Human footprint data not available")
    
    return df


def calculate_vulnerability_index(df):
    """
    Composite index of biological vulnerability
    
    Components (normalized 0-1):
    - Climate velocity (30%)
    - Human footprint velocity (30%)
    - Current human pressure (20%)
    - Temperature seasonality (10%)
    - Range size (10% - smaller = more vulnerable)
    """
    print("\n🦎 Calculating Biological Vulnerability Index...")
    
    scaler = MinMaxScaler()
    
    # Initialize component dict
    vulnerability_components = {}
    weights = {}
    
    # Climate velocity component
    if 'temp_velocity' in df.columns and df['temp_velocity'].notna().sum() > 0:
        vulnerability_components['climate'] = scaler.fit_transform(
            df[['temp_velocity']].fillna(df['temp_velocity'].median())
        ).flatten()
        weights['climate'] = 0.30
        print(f"   ✅ Climate velocity (weight: {weights['climate']:.2f})")
    
    # Human footprint velocity
    if 'hfi_velocity' in df.columns and df['hfi_velocity'].notna().sum() > 0:
        vulnerability_components['hfi_velocity'] = scaler.fit_transform(
            df[['hfi_velocity']].fillna(0)  # Fill missing with 0 (no change)
        ).flatten()
        weights['hfi_velocity'] = 0.30
        print(f"   ✅ HFI velocity (weight: {weights['hfi_velocity']:.2f})")
    
    # Current human pressure
    if 'hfi_2020' in df.columns and df['hfi_2020'].notna().sum() > 0:
        vulnerability_components['hfi_current'] = scaler.fit_transform(
            df[['hfi_2020']].fillna(df['hfi_2020'].median())
        ).flatten()
        weights['hfi_current'] = 0.20
        print(f"   ✅ Current HFI (weight: {weights['hfi_current']:.2f})")
    
    # Range size (inversed - smaller range = higher vulnerability)
    if 'range_area_km2' in df.columns and df['range_area_km2'].notna().sum() > 0:
        # Log transform for skewed distribution
        log_range = np.log10(df['range_area_km2'].fillna(df['range_area_km2'].median()) + 1)
        vulnerability_components['range_size'] = 1 - scaler.fit_transform(
            log_range.values.reshape(-1, 1)
        ).flatten()
        weights['range_size'] = 0.20
        print(f"   ✅ Range size (inversed) (weight: {weights['range_size']:.2f})")
    
    # Calculate weighted vulnerability index
    if vulnerability_components:
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        df['vulnerability_index'] = sum(
            vulnerability_components[k] * normalized_weights[k]
            for k in vulnerability_components.keys()
        )
        
        print(f"\n   📊 Vulnerability Index Statistics:")
        print(f"      Mean: {df['vulnerability_index'].mean():.3f}")
        print(f"      Std:  {df['vulnerability_index'].std():.3f}")
        print(f"      Min:  {df['vulnerability_index'].min():.3f}")
        print(f"      Max:  {df['vulnerability_index'].max():.3f}")
    else:
        df['vulnerability_index'] = np.nan
        print("   ⚠️  Insufficient data for vulnerability index")
    
    return df


def calculate_capacity_score(df):
    """
    Composite score of conservation capacity
    
    Components (normalized 0-1):
    - Protected area coverage (70%)
    - Low human footprint (30% - higher score for lower HFI)
    """
    print("\n🛡️  Calculating Conservation Capacity Score...")
    
    scaler = MinMaxScaler()
    
    capacity_components = {}
    weights = {}
    
    # Protected area coverage
    if 'protected_area_coverage_pct' in df.columns and df['protected_area_coverage_pct'].notna().sum() > 0:
        # Already 0-100, convert to 0-1
        capacity_components['protection'] = df['protected_area_coverage_pct'].fillna(0) / 100
        weights['protection'] = 0.70
        print(f"   ✅ Protected area coverage (weight: {weights['protection']:.2f})")
        print(f"      Mean coverage: {df['protected_area_coverage_pct'].mean():.1f}%")
    else:
        print("   ⚠️  Protected area data not available - using placeholder")
        capacity_components['protection'] = np.zeros(len(df))
        weights['protection'] = 0.70
    
    # Low human footprint (inverted - lower HFI = higher capacity)
    if 'hfi_2020' in df.columns and df['hfi_2020'].notna().sum() > 0:
        normalized_hfi = scaler.fit_transform(
            df[['hfi_2020']].fillna(df['hfi_2020'].median())
        ).flatten()
        capacity_components['low_pressure'] = 1 - normalized_hfi
        weights['low_pressure'] = 0.30
        print(f"   ✅ Low human pressure (weight: {weights['low_pressure']:.2f})")
    
    # Calculate weighted capacity score
    if capacity_components:
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        df['capacity_score'] = sum(
            capacity_components[k] * normalized_weights[k]
            for k in capacity_components.keys()
        )
        
        print(f"\n   📊 Capacity Score Statistics:")
        print(f"      Mean: {df['capacity_score'].mean():.3f}")
        print(f"      Std:  {df['capacity_score'].std():.3f}")
        print(f"      Min:  {df['capacity_score'].min():.3f}")
        print(f"      Max:  {df['capacity_score'].max():.3f}")
    else:
        df['capacity_score'] = np.nan
        print("   ⚠️  Insufficient data for capacity score")
    
    return df


def calculate_resilience_gap(df):
    """
    Calculate the mismatch between vulnerability and capacity
    
    resilience_gap = vulnerability_index - capacity_score
    
    Positive gap = High vulnerability, Low capacity = PRIORITY
    Negative gap = Low vulnerability, High capacity = Well protected
    """
    print("\n⚖️  Calculating Resilience Gap...")
    
    if 'vulnerability_index' in df.columns and 'capacity_score' in df.columns:
        df['resilience_gap'] = df['vulnerability_index'] - df['capacity_score']
        
        print(f"   Mean gap: {df['resilience_gap'].mean():.3f}")
        print(f"   Gap range: [{df['resilience_gap'].min():.3f}, {df['resilience_gap'].max():.3f}]")
        
        # Identify forgotten species (top 10% gap)
        threshold_90 = df['resilience_gap'].quantile(0.90)
        forgotten = df['resilience_gap'] >= threshold_90
        print(f"\n   🚨 'Forgotten Species' (top 10% gap, resilience_gap >= {threshold_90:.3f}):")
        print(f"      Count: {forgotten.sum()} species")
        
        if forgotten.sum() > 0:
            top_forgotten = df[forgotten].nlargest(10, 'resilience_gap')[['species', 'iucn_category', 'vulnerability_index', 'capacity_score', 'resilience_gap']]
            print(f"\n      Top 10:")
            for idx, row in top_forgotten.iterrows():
                print(f"         {row['species'][:40]:40} | {row['iucn_category']:5} | Gap: {row['resilience_gap']:.3f}")
    else:
        df['resilience_gap'] = np.nan
        print("   ⚠️  Cannot calculate gap - missing vulnerability or capacity data")
    
    return df


def main():
    """Main execution"""
    
    print("=" * 70)
    print("CALCULATE VULNERABILITY, CAPACITY, AND RESILIENCE GAP INDICES")
    print("=" * 70)
    
    # Process each taxonomic group
    for taxa in ['amphibians', 'mammals', 'reptiles']:
        input_file = DATA_PROCESSED / f"{taxa}_features_raw.csv"
        
        if not input_file.exists():
            print(f"\n⚠️  Skipping {taxa}: {input_file} not found")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"{taxa.upper()}")
        print("=" * 70)
        
        # Load data
        print(f"\n📂 Loading: {input_file}")
        df = pd.read_csv(input_file)
        print(f"   Loaded {len(df):,} species with {len(df.columns)} features")
        
        # Calculate derived metrics
        df = calculate_climate_velocity(df)
        df = calculate_human_footprint_velocity(df)
        
        # Calculate composite indices
        df = calculate_vulnerability_index(df)
        df = calculate_capacity_score(df)
        df = calculate_resilience_gap(df)
        
        # Save enriched dataset
        output_file = DATA_PROCESSED / f"{taxa}_with_indices.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")
        print(f"   Total features: {len(df.columns)}")
    
    print("\n" + "=" * 70)
    print("✅ INDEX CALCULATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Output files:")
    print(f"   - {DATA_PROCESSED / 'amphibians_with_indices.csv'}")
    print(f"   - {DATA_PROCESSED / 'mammals_with_indices.csv'}")
    print(f"\n🚀 Next step: Run scripts/04_train_models.py")


if __name__ == "__main__":
    main()
