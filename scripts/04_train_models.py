"""
Train Machine Learning Models for Extinction Horizon Forecasting

Two main models:
1. Red List Forecaster - Predict 2035 IUCN status
2. Resilience Gap Analyzer - Find systematically under-protected species

Uses MLflow for experiment tracking and reproducibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import MLflow (optional)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available - install with: pip install mlflow")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data/processed"
MODEL_OUTPUT = BASE_DIR / "models"
RESULTS_OUTPUT = BASE_DIR / "results"
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# IUCN Category encoding
IUCN_MAPPING = {
    'LC': 0,  # Least Concern
    'NT': 1,  # Near Threatened
    'VU': 2,  # Vulnerable
    'EN': 3,  # Endangered
    'CR': 4,  # Critically Endangered
    'EW': 5,  # Extinct in Wild
    'EX': 6   # Extinct
}

def prepare_ml_features(df):
    """
    Prepare feature matrix for ML models
    
    Returns:
        X: Feature matrix
        y: Target variable (encoded IUCN category)
        feature_names: List of feature column names
    """
    print("\n🔧 Preparing ML features...")
    
    # Core features for prediction
    feature_cols = [
        # Velocity metrics (key predictors)
        'temp_velocity',
        'hfi_velocity',
        'hfi_change_pct',
        
        # Current conditions
        'hfi_2020',
        'annual_mean_temp_baseline',
        'annual_precipitation_baseline',
        
        # Protection
        'protected_area_coverage_pct',
        
        # Composite indices
        'vulnerability_index',
        'capacity_score',
        'resilience_gap',
        
        # Species characteristics
        'range_area_km2'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"   Available features: {len(available_features)}/{len(feature_cols)}")
    
    # Extract features
    X = df[available_features].copy()
    
    # Handle missing values (median imputation)
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # Encode target variable
    if 'iucn_category' in df.columns:
        # Map to numeric
        y_encoded = df['iucn_category'].map(IUCN_MAPPING)
        # Fill unknown with median
        y = y_encoded.fillna(int(y_encoded.median()))
    else:
        y = None
        print("   ⚠️  No IUCN category column found")
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target distribution:")
    if y is not None:
        for cat, code in IUCN_MAPPING.items():
            count = (y == code).sum()
            pct = count / len(y) * 100
            print(f"      {cat}: {count:5d} ({pct:5.1f}%)")
    
    return X, y, available_features


def train_red_list_forecaster(X, y, feature_names):
    """
    Model A: Predict future IUCN Red List category
    
    Uses Gradient Boosting classifier to predict threat escalation
    """
    print("\n" + "=" * 70)
    print("MODEL A: RED LIST FORECASTER")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Train Gradient Boosting model
    print("\n🤖 Training Gradient Boosting Classifier...")
    
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    
    # Evaluate
    train_score = gb_model.score(X_train, y_train)
    test_score = gb_model.score(X_test, y_test)
    
    print(f"\n📊 Model Performance:")
    print(f"   Train Accuracy: {train_score:.3f}")
    print(f"   Test Accuracy:  {test_score:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(gb_model, X, y, cv=5)
    print(f"   CV Accuracy (5-fold): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Predictions
    y_pred = gb_model.predict(X_test)
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(IUCN_MAPPING.keys())[:7], zero_division=0))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:35s} {row['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance - Red List Forecaster')
    plt.tight_layout()
    plt.savefig(RESULTS_OUTPUT / 'feature_importance_redlist.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved feature importance plot: {RESULTS_OUTPUT / 'feature_importance_redlist.png'}")
    plt.close()
    
    # Log to MLflow
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="RedList_GradientBoosting"):
            mlflow.log_param("model_type", "GradientBoostingClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)
            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("test_accuracy", test_score)
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.sklearn.log_model(gb_model, "model")
            mlflow.log_artifact(str(RESULTS_OUTPUT / 'feature_importance_redlist.png'))
    
    return gb_model, feature_importance


def train_resilience_gap_analyzer(df):
    """
    Model B: Identify "Forgotten Species" via residual analysis
    
    Method:
    1. Regress: capacity_score ~ vulnerability_index
    2. Calculate residuals
    3. Species with large negative residuals = High risk, Low protection = FORGOTTEN
    """
    print("\n" + "=" * 70)
    print("MODEL B: RESILIENCE GAP ANALYZER (Residual Analysis)")
    print("=" * 70)
    
    # Filter to rows with both indices
    analysis_df = df[df['vulnerability_index'].notna() & df['capacity_score'].notna()].copy()
    
    print(f"\nAnalyzing {len(analysis_df)} species with complete indices")
    
    # Prepare data for regression
    X = analysis_df[['vulnerability_index']].values
    y = analysis_df['capacity_score'].values
    
    # Train Ridge regression
    print("\n🤖 Training Ridge Regression (capacity ~ vulnerability)...")
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)
    
    # Predictions
    y_pred = ridge_model.predict(X)
    
    # Residuals (negative = under-protected)
    residuals = y - y_pred
    analysis_df['predicted_capacity'] = y_pred
    analysis_df['residual'] = residuals
    
    # Model performance
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {r2:.3f}")
    print(f"   MAE: {mae:.3f}")
    print(f"   Coefficient: {ridge_model.coef_[0]:.3f} (capacity per unit vulnerability)")
    
    # Identify forgotten species (bottom 10% residuals)
    threshold_10 = analysis_df['residual'].quantile(0.10)
    forgotten_species = analysis_df[analysis_df['residual'] <= threshold_10].copy()
    forgotten_species = forgotten_species.sort_values('residual')
    
    print(f"\n🚨 'Forgotten Species' Identified (bottom 10% residuals):")
    print(f"   Count: {len(forgotten_species)} species")
    print(f"   Residual threshold: {threshold_10:.3f}")
    
    if len(forgotten_species) > 0:
        print(f"\n   Top 20 Most Neglected Species:")
        print(f"   {'Species':<45} {'IUCN':5} {'Vuln':>6} {'Cap':>6} {'Gap':>6} {'Residual':>7}")
        print(f"   {'-' * 90}")
        
        for idx, row in forgotten_species.head(20).iterrows():
            species_name = row['species'][:43] if len(row['species']) > 43 else row['species']
            print(f"   {species_name:45} {row.get('iucn_category', 'N/A'):5} "
                  f"{row['vulnerability_index']:6.3f} {row['capacity_score']:6.3f} "
                  f"{row['resilience_gap']:6.3f} {row['residual']:7.3f}")
    
    # Visualization: Scatter plot with regression line
    plt.figure(figsize=(12, 8))
    
    # All species
    plt.scatter(analysis_df['vulnerability_index'], analysis_df['capacity_score'], 
                alpha=0.4, s=30, c='gray', label='All species')
    
    # Forgotten species
    plt.scatter(forgotten_species['vulnerability_index'], forgotten_species['capacity_score'],
                alpha=0.7, s=60, c='red', label='Forgotten Species (bottom 10%)', marker='^')
    
    # Regression line
    x_line = np.linspace(analysis_df['vulnerability_index'].min(), 
                         analysis_df['vulnerability_index'].max(), 100)
    y_line = ridge_model.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, 'b--', linewidth=2, label='Expected Capacity')
    
    plt.xlabel('Biological Vulnerability Index', fontsize=12)
    plt.ylabel('Conservation Capacity Score', fontsize=12)
    plt.title('Resilience Gap Analysis: Forgotten Species Identification', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_OUTPUT / 'resilience_gap_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved resilience gap plot: {RESULTS_OUTPUT / 'resilience_gap_analysis.png'}")
    plt.close()
    
    # Log to MLflow
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="ResilienceGap_Ridge"):
            mlflow.log_param("model_type", "Ridge")
            mlflow.log_param("alpha", 1.0)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("num_forgotten_species", len(forgotten_species))
            mlflow.sklearn.log_model(ridge_model, "model")
            mlflow.log_artifact(str(RESULTS_OUTPUT / 'resilience_gap_analysis.png'))
    
    # Save forgotten species list
    forgotten_species_export = forgotten_species[[
        'species', 'iucn_category', 'vulnerability_index', 'capacity_score', 
        'resilience_gap', 'predicted_capacity', 'residual'
    ]].copy()
    
    output_file = RESULTS_OUTPUT / 'forgotten_species_list.csv'
    forgotten_species_export.to_csv(output_file, index=False)
    print(f"💾 Saved forgotten species list: {output_file}")
    
    return ridge_model, forgotten_species_export


def main():
    """Main execution"""
    
    print("=" * 70)
    print("MACHINE LEARNING MODEL TRAINING")
    print("Extinction Horizon Forecasting")
    print("=" * 70)
    
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("Extinction_Horizon_ML")
        print("✅ MLflow experiment tracking enabled")
    
    # Combine all taxonomic groups
    all_data = []
    
    for taxa in ['amphibians', 'mammals', 'reptiles']:
        input_file = DATA_PROCESSED / f"{taxa}_with_indices.csv"
        
        if input_file.exists():
            print(f"\n📂 Loading {taxa}: {input_file}")
            df = pd.read_csv(input_file)
            df['taxonomic_group'] = taxa
            all_data.append(df)
            print(f"   Loaded {len(df):,} species")
        else:
            print(f"\n⚠️  {taxa} data not found: {input_file}")
    
    if not all_data:
        print("\n❌ No data found! Run scripts 02 and 03 first.")
        return
    
    # Combine datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df):,} species")
    print(f"   Taxonomic groups: {combined_df['taxonomic_group'].value_counts().to_dict()}")
    
    # Prepare features
    X, y, feature_names = prepare_ml_features(combined_df)
    
    # Train Model A: Red List Forecaster
    if y is not None:
        model_a, feature_importance = train_red_list_forecaster(X, y, feature_names)
        
        # Save model
        import joblib
        joblib.dump(model_a, MODEL_OUTPUT / 'redlist_forecaster.pkl')
        print(f"\n💾 Saved model: {MODEL_OUTPUT / 'redlist_forecaster.pkl'}")
    else:
        print("\n⚠️  Skipping Red List Forecaster - no target variable")
    
    # Train Model B: Resilience Gap Analyzer
    model_b, forgotten_list = train_resilience_gap_analyzer(combined_df)
    
    # Save model
    import joblib
    joblib.dump(model_b, MODEL_OUTPUT / 'resilience_gap_analyzer.pkl')
    print(f"💾 Saved model: {MODEL_OUTPUT / 'resilience_gap_analyzer.pkl'}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Outputs:")
    print(f"   Models:")
    print(f"      - {MODEL_OUTPUT / 'redlist_forecaster.pkl'}")
    print(f"      - {MODEL_OUTPUT / 'resilience_gap_analyzer.pkl'}")
    print(f"\n   Results:")
    print(f"      - {RESULTS_OUTPUT / 'feature_importance_redlist.png'}")
    print(f"      - {RESULTS_OUTPUT / 'resilience_gap_analysis.png'}")
    print(f"      - {RESULTS_OUTPUT / 'forgotten_species_list.csv'}")
    
    if MLFLOW_AVAILABLE:
        print(f"\n   MLflow Tracking:")
        print(f"      View experiments: mlflow ui")
        print(f"      Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
