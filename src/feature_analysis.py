"""
Feature importance and financial factors analysis.
This is the most important module where the majority of analyses are performed.
It concretely analyzes the effect of financial ratios on bankruptcy or non-bankruptcy.
Identifies key financial indicators that predict bankruptcy.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from src.models import train_logistic_regression, train_random_forest, train_knn
from src.data_loader import preprocess_data


def analyze_logistic_regression_features(model, feature_names, top_n=20, save_path=None):
    """
    Analyze feature importance using Logistic Regression model coefficients and determine the role (effect direction)
    played by each factor in bankruptcy or non-bankruptcy, then plot them for the report with colors:
    green for protective effect (decreases risk), red for risk effect (increases risk).
    Excludes horizon indicator features (year_2, year_3, etc.) to focus on financial factors.

    Note: This analysis combines all prediction horizons (see analyze_all_models_by_horizon for horizon-specific analysis)
    """
    if (not hasattr(model, "coef_")):
        print("This model does not have coefficients")
        return None
    
    # Exclude horizon indicator features (year_2, year_3, year_4, year_5) as they are not financial ratios
    financial_features = [f for f in feature_names if not f.startswith('year_')]
    financial_indices = [i for i, f in enumerate(feature_names) if not f.startswith('year_')]
    
    coefficients = np.abs(model.coef_[0])
    
    feature_importance = pd.DataFrame({
        "feature": financial_features,
        "coefficient": model.coef_[0][financial_indices],
        "importance": coefficients[financial_indices]
    }).sort_values("importance", ascending=False)
    
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    colors = ["red" if x < 0 else "green" for x in top_features["coefficient"]]
    plt.barh(range(top_n), top_features["importance"], color=colors)
    plt.yticks(range(top_n), top_features["feature"].values)
    plt.xlabel("Absolute Coefficient Value (Importance)", fontsize=11, fontweight="bold")
    plt.title(f"Top {top_n} Financial Factors - Logistic Regression\n(Green: Decreases risk, Red: Increases risk)", 
              fontsize=12, fontweight="bold", pad=15)
    plt.gca().invert_yaxis()
    
    # Save plot if path is provided
    # os.makedirs ensures the directory exists before saving (creates it if needed)
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()
    
    return feature_importance


def analyze_random_forest_features(model, feature_names, top_n=20, save_path=None):
    """
    Same idea as for LR, but analyzes feature importance from Random Forest and plots it for the report.
    Excludes horizon indicator features (year_2, year_3, etc.) to focus on financial factors.

    Note: This analysis combines all prediction horizons (see analyze_all_models_by_horizon for horizon-specific analysis)
    """
    if (not hasattr(model, "feature_importances_")):
        print("This model does not have feature_importances_")
        return None
    
    # Exclude horizon indicator features (year_2, year_3, year_4, year_5) again
    financial_features = [f for f in feature_names if not f.startswith('year_')]
    financial_indices = [i for i, f in enumerate(feature_names) if not f.startswith('year_')]
    
    importances = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        "feature": financial_features,
        "importance": importances[financial_indices]
    }).sort_values("importance", ascending=False)
    
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_features["importance"])
    plt.yticks(range(top_n), top_features["feature"].values)
    plt.xlabel("Importance", fontsize=11, fontweight="bold")
    plt.title(f"Top {top_n} Financial Factors - Random Forest", 
              fontsize=12, fontweight="bold", pad=15)
    plt.gca().invert_yaxis()
    
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()
    
    return feature_importance


def analyze_knn_features(model, X_train, y_train, feature_names, top_n=20, save_path=None, random_state=42):
    """
    Analyze feature importance for KNN using permutation importance.
    Excludes horizon indicator features (year_2, year_3, etc.) to focus on financial factors.
    
    KNN doesn't have native feature importance, so we use permutation importance:
    we measure how much the model's performance decreases when a feature is randomly shuffled.
    
    Note: This analysis combines all prediction horizons (see analyze_all_models_by_horizon for horizon-specific analysis)
    """
    # Calculate permutation importance on all features (model was trained with all features)
    # This measures the drop in performance when each feature is randomly permuted
    # n_repeats=10: no clear literature consensus found on best practices; AI recommendation for stability
    perm_importance = permutation_importance(
        model, X_train, y_train,
        scoring='roc_auc',
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Exclude horizon indicator features (year_2, year_3, year_4, year_5) from results as well
    financial_features = [f for f in feature_names if not f.startswith('year_')]
    financial_indices = [i for i, f in enumerate(feature_names) if not f.startswith('year_')]
    
    feature_importance = pd.DataFrame({
        "feature": financial_features,
        "importance": perm_importance.importances_mean[financial_indices],
        "std": perm_importance.importances_std[financial_indices]
    }).sort_values("importance", ascending=False)
    
    top_features = feature_importance.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_features["importance"], color="steelblue", alpha=0.8)
    plt.yticks(range(top_n), top_features["feature"].values)
    plt.xlabel("Permutation Importance (Mean Decrease in ROC-AUC)", fontsize=11, fontweight="bold")
    plt.title(f"Top {top_n} Financial Factors - K-Nearest Neighbors\n(Permutation Importance)", 
              fontsize=12, fontweight="bold", pad=15)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()
    
    return feature_importance


def analyze_global_features(lr_model, rf_model, knn_model, X_train, y_train, feature_names, top_n=15, save_dir="results"):
    """
    Analyze feature importance globally (all horizons combined) for all three models.
    Generates top 15 features for each model for the report and identifies common features across all three,
    in order to determine which ones are robust.
    """
    results = {}
    

    lr_features = analyze_logistic_regression_features(
        lr_model, feature_names, top_n=top_n, 
        save_path=f"{save_dir}/global_top15_lr.png"
    )
    results['lr_features'] = lr_features
    
    rf_features = analyze_random_forest_features(
        rf_model, feature_names, top_n=top_n,
        save_path=f"{save_dir}/global_top15_rf.png"
    )
    results['rf_features'] = rf_features
    
    knn_features = analyze_knn_features(
        knn_model, X_train, y_train, feature_names, top_n=top_n,
        save_path=f"{save_dir}/global_top15_knn.png"
    )
    results['knn_features'] = knn_features
    
    print("Global analysis (all horizons combined) - Top 15 factors:")
    print("-" * 80)
    
    if (lr_features is not None):
        print(f"\nLogistic Regression:")
        for rank, (idx, row) in enumerate(lr_features.head(top_n).iterrows(), 1):
            feature_display = row['feature']
            print(f"  {rank}. {feature_display}: {row['importance']:.4f}")
    
    if (rf_features is not None):
        print(f"\nRandom Forest:")
        for rank, (idx, row) in enumerate(rf_features.head(top_n).iterrows(), 1):
            feature_display = row['feature']
            print(f"  {rank}. {feature_display}: {row['importance']:.4f}")
    
    if (knn_features is not None):
        print(f"\nK-Nearest Neighbors:")
        for rank, (idx, row) in enumerate(knn_features.head(top_n).iterrows(), 1):
            feature_display = row['feature']
            print(f"  {rank}. {feature_display}: {row['importance']:.4f}")
    
    # Find common features across all 3 models
    if (lr_features is not None and rf_features is not None and knn_features is not None):
        lr_top = set(lr_features.head(top_n)['feature'].values)
        rf_top = set(rf_features.head(top_n)['feature'].values)
        knn_top = set(knn_features.head(top_n)['feature'].values)
        
        common_factors = lr_top & rf_top & knn_top
        
        print()
        print("Common factors across all 3 models:")
        print("-" * 80)
        if (common_factors):
            for factor in sorted(common_factors):
                factor_display = factor
                print(f"  • {factor_display}")
            results['common_factors'] = list(common_factors)
        else:
            print("  No common factors found in top 15 across all 3 models")
            results['common_factors'] = []
        print()
    
    return results


def analyze_all_models_by_horizon(dataframe, target_column, top_n=15, save_dir="results"):
    """
    Analyze feature importance for all models (LR, RF, KNN), but this time by separating horizons to see the impact of ratios
    at different horizon levels (short-term, medium-term, or long-term). Also identifies common factors, if any, for the defined horizon.
    Generates a plot for each model and each horizon for the report.
    """

    # Separate features and target for horizon analysis
    features, target = preprocess_data(dataframe, target_column)
    # Remove 'year' column from features (it will be used for filtering, not as a feature)
    if ('year' in features.columns):
        features = features.drop(columns=['year'])
    
    # Get unique years (horizons)
    years = sorted(dataframe['year'].unique())
    all_results = {}
    
    for year in years:
        horizon = 6 - int(year)
        
        # Filter data for this year
        year_mask = dataframe['year'] == year
        X_year = features[year_mask]
        y_year = target[year_mask]
        
        if (len(X_year) < 20):
            print(f"Skipping Year {year}: insufficient data ({len(X_year)} samples)")
            continue
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            # stratify permits to preserve class distribution, essential given the severe imbalance
            X_year, y_year, test_size=0.2, random_state=42, stratify=y_year
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Train and analyze all models for this year
        models_config = [
            ("Logistic Regression", train_logistic_regression, analyze_logistic_regression_features, "lr", None),
            ("Random Forest", train_random_forest, analyze_random_forest_features, "rf", None),
            # n_neighbors=5: standard default that balances local pattern capture with noise smoothing
            ("K-Nearest Neighbors", lambda X, y: train_knn(X, y, n_neighbors=5), 
             analyze_knn_features, "knn", (X_train_scaled, y_train))
        ]
        
        model_features = {}
        for model_name, train_func, analyze_func, model_key, extra_args in models_config:
            model = train_func(X_train_scaled, y_train)
            if (extra_args):
                features_df = analyze_func(model, extra_args[0], extra_args[1], X_train_scaled.columns, top_n=top_n, save_path=None)
            else:
                features_df = analyze_func(model, X_train_scaled.columns, top_n=top_n, save_path=None)
            
            if (features_df is not None):
                model_features[model_key] = features_df
        
        lr_features = model_features.get('lr')
        rf_features = model_features.get('rf')
        knn_features = model_features.get('knn')
        
        # Display the top 15 for each model for this year
        print(f"Year {year} (Horizon: {horizon} year{'s' if horizon > 1 else ''}) - Top {top_n} factors:")
        print("-" * 80)
        
        if (lr_features is not None):
            print(f"\nLogistic Regression:")
            for rank, (idx, row) in enumerate(lr_features.head(top_n).iterrows(), 1):
                feature_display = row['feature']
                print(f"  {rank}. {feature_display}: {row['importance']:.4f}")
        
        if (rf_features is not None):
            print(f"\nRandom Forest:")
            for rank, (idx, row) in enumerate(rf_features.head(top_n).iterrows(), 1):
                feature_display = row['feature']
                print(f"  {rank}. {feature_display}: {row['importance']:.4f}")
        
        if (knn_features is not None):
            print(f"\nK-Nearest Neighbors:")
            for rank, (idx, row) in enumerate(knn_features.head(top_n).iterrows(), 1):
                feature_display = row['feature']
                print(f"  {rank}. {feature_display}: {row['importance']:.4f}")
        
        print()
        print(f"Analyzing Year {year} (Horizon: {horizon} year{'s' if horizon > 1 else ''})...")
        print("-" * 80)
        
        horizon_results = {
            'horizon': horizon,
            'year': year,
            'n_samples': len(X_year)
        }
        
        # Generate plots for each model
        for model_name, train_func, analyze_func, model_key, extra_args in models_config:
            features_df = model_features.get(model_key)
            if (features_df is not None):
                save_path_model = f"{save_dir}/horizon_{horizon}_year_{year}_{model_key}.png"
                plot_horizon_feature_importance(
                    features_df, model_name, horizon, year, top_n=top_n, save_path=save_path_model
                )
                horizon_results[f'{model_key}_features'] = features_df
        
        # Find common factors across all 3 models for this horizon
        if (lr_features is not None and rf_features is not None and knn_features is not None):
            lr_top = set(lr_features.head(top_n)['feature'].values)
            rf_top = set(rf_features.head(top_n)['feature'].values)
            knn_top = set(knn_features.head(top_n)['feature'].values)
            
            common_factors = lr_top & rf_top & knn_top
            
            if (common_factors):
                print(f"Common factors across all 3 models: {len(common_factors)}")
                for factor in sorted(common_factors):
                    factor_display = factor
                    print(f"  • {factor_display}")
                print()
                horizon_results['common_factors'] = list(common_factors)
            else:
                print(f"No common factors found in top {top_n} across all 3 models")
                print()
                horizon_results['common_factors'] = []
        
        all_results[year] = horizon_results
        print()
    
    # Generate comparison heatmap across all horizons, which allows inter-horizon comparison of the top 15
    # This de facto provides the 15 best ratios per horizon
    if (len(all_results) > 1):
        # RF heatmap (best performing model) providing the most reliable top 15
        horizon_results_for_comparison_rf = {}
        for year, results in all_results.items():
            if ('rf_features' in results and results['rf_features'] is not None):
                horizon_results_for_comparison_rf[year] = {
                    'horizon': results['horizon'],
                    'feature_importance': results['rf_features']
                }
        
        if (len(horizon_results_for_comparison_rf) > 1):
            plot_horizon_comparison(horizon_results_for_comparison_rf, top_n=top_n, 
                                   save_path=f"{save_dir}/horizon_comparison_rf.png")
            
            # Collect RF features for LR heatmap (same factors, but with LR coefficients for directionality)
            rf_features_set = set()
            for year, results in horizon_results_for_comparison_rf.items():
                if (results['feature_importance'] is not None):
                    rf_features_set.update(results['feature_importance'].head(top_n)['feature'].values)
            
            # Prepare LR data for same features in order to obtain their best possible interpretation
            horizon_results_for_comparison_lr = {}
            for year, results in all_results.items():
                if ('lr_features' in results and results['lr_features'] is not None):
                    # Filter LR features to only include those in rf_features_set
                    lr_filtered_features = results['lr_features'][
                        results['lr_features']['feature'].isin(rf_features_set)
                    ].copy()
                    
                    # Ensure 'importance' column is present for plotting (absolute coefficient)
                    if ('importance' not in lr_filtered_features.columns):
                        lr_filtered_features['importance'] = np.abs(lr_filtered_features['coefficient'])
                    
                    horizon_results_for_comparison_lr[year] = {
                        'horizon': results['horizon'],
                        'feature_importance': lr_filtered_features
                    }
            
            # Generate LR coefficients heatmap for RF-identified factors
            if (len(horizon_results_for_comparison_lr) > 1):
                plot_horizon_comparison_lr_coefficients(
                    horizon_results_for_comparison_lr, rf_features_set, top_n=top_n,
                    save_path=f"{save_dir}/horizon_comparison_lr_coefficients.png"
                )
    
    return all_results


def plot_horizon_feature_importance(feature_importance_df, model_name, horizon, year, top_n=15, save_path=None):
    """
    Plot feature importance for a specific model and horizon.
    Used in the analyze_all_models_by_horizon method to generate one plot per model for the concerned horizon.
    """
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    
    if ('coefficient' in top_features.columns):
        # Logistic Regression: color by coefficient sign
        colors = ["red" if x < 0 else "green" for x in top_features["coefficient"]]
        plt.barh(range(top_n), top_features["importance"], color=colors)
        plt.xlabel("Absolute Coefficient Value (Importance)", fontsize=11, fontweight="bold")
        subtitle = "(Green: Decreases risk, Red: Increases risk)"
    else:
        # Random Forest or KNN: single color
        if ("K-Nearest" in model_name or "KNN" in model_name):
            color = "purple"
            xlabel = "Permutation Importance"
        else:
            color = "steelblue"
            xlabel = "Feature Importance"
        plt.barh(range(top_n), top_features["importance"], color=color, alpha=0.8)
        plt.xlabel(xlabel, fontsize=11, fontweight="bold")
        subtitle = ""
    
    plt.yticks(range(top_n), top_features["feature"].values, fontsize=9)
    plt.title(f"Top {top_n} Financial Factors - {model_name}\nHorizon: {horizon} year{'s' if horizon > 1 else ''} (Year {year}) {subtitle}", 
              fontsize=12, fontweight="bold", pad=15)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_horizon_comparison(horizon_results, top_n=10, save_path=None):
    """
    Create the RF heatmap by comparing all horizons at once. Allows obtaining the most reliable top 15 per horizon,
    which will be used for the LR heatmap.
    """
    # Collect all unique top features
    all_top_features = set()
    for year, results in horizon_results.items():
        if (results['feature_importance'] is not None):
            all_top_features.update(results['feature_importance'].head(top_n)['feature'].values)
    
    # Create data for heatmap
    years_sorted = sorted(horizon_results.keys())
    features_list = sorted(list(all_top_features))
    
    importance_matrix = []
    for feature in features_list:
        row = []
        for year in years_sorted:
            horizon = horizon_results[year]['horizon']
            feature_importance = horizon_results[year]['feature_importance']
            
            if (feature in feature_importance['feature'].values):
                importance = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
                row.append(importance)
            else:
                row.append(0.0)
        importance_matrix.append(row)
    
    importance_df = pd.DataFrame(
        importance_matrix,
        index=features_list,
        columns=[f'Year {y}\n(Horizon {horizon_results[y]["horizon"]}y)' for y in years_sorted]
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(years_sorted) * 2), max(8, len(features_list) * 0.4)))
    sns.heatmap(importance_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Feature Importance'}, 
                linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_title(f"Feature Importance Comparison Across Prediction Horizons - Random Forest\n(Top {top_n} features per horizon)", 
                fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Prediction Horizon", fontsize=12, fontweight="bold")
    ax.set_ylabel("Financial Factors", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_horizon_comparison_lr_coefficients(lr_horizon_results, rf_features_list, top_n=10, save_path=None):
    """
    Create a heatmap comparing LR coefficients across horizons for factors identified by RF in plot_horizon_comparison.
    Shows the directionality (increase vs decrease risk) of factors identified by the best model (RF).
    
    This allows obtaining a display of the top 15 most reliable ratios with the best interpretation from the three models used.
    """
    if (not rf_features_list):
        return
    
    # Create data for heatmap using only features from RF
    years_sorted = sorted(lr_horizon_results.keys())
    features_list = sorted(list(rf_features_list))
    
    coefficient_matrix = []
    for feature in features_list:
        row = []
        for year in years_sorted:
            horizon = lr_horizon_results[year]['horizon']
            feature_importance = lr_horizon_results[year]['feature_importance']
            
            if (feature_importance is not None and feature in feature_importance['feature'].values):
                # Get the actual coefficient (with sign), not absolute value
                coefficient = feature_importance[feature_importance['feature'] == feature]['coefficient'].iloc[0]
                row.append(coefficient)
            else:
                row.append(0.0)
        coefficient_matrix.append(row)
    
    coefficient_df = pd.DataFrame(
        coefficient_matrix,
        index=features_list,
        columns=[f'Year {y}\n(Horizon {lr_horizon_results[y]["horizon"]}y)' for y in years_sorted]
    )
    
    # Create heatmap with diverging colormap (red for positive, green for negative)
    fig, ax = plt.subplots(figsize=(max(10, len(years_sorted) * 2), max(8, len(features_list) * 0.4)))
    
    # Find center for diverging colormap (0 for coefficients)
    vmax = max(abs(coefficient_df.min().min()), abs(coefficient_df.max().max()))
    vmin = -vmax
    
    sns.heatmap(coefficient_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'LR Coefficient (Red: Increases Risk, Green: Decreases Risk)'}, 
                linewidths=0.5, linecolor='gray', ax=ax, center=0, vmin=vmin, vmax=vmax)
    ax.set_title(f"LR Coefficients for RF-Identified Factors Across Prediction Horizons\n(Shows directionality: positive = increases risk, negative = decreases risk)", 
                fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Prediction Horizon", fontsize=12, fontweight="bold")
    ax.set_ylabel("Financial Factors", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()

