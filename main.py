#!/usr/bin/env python3
"""
Main script to compare ML models on the Polish Companies Bankruptcy dataset
and determine the most impactful financial ratios in bankruptcy prediction.
"""
import argparse
import warnings
# Allows counting most common factors across all horizons
from collections import Counter
warnings.filterwarnings("ignore")

from src.data_loader import load_polish_bankruptcy_data, inspect_data, preprocess_data, split_and_scale, save_to_csv, detect_target_column
from src.data_analysis import analyze_data_quality, summarize_data_quality, plot_bankruptcy_rate_by_horizon
from src.data_cleaning import clean_data
from src.models import train_logistic_regression, train_random_forest, train_knn
from src.evaluation import (evaluate_model, plot_roc_curve, plot_confusion_matrix)
from src.feature_analysis import (
    analyze_all_models_by_horizon, analyze_global_features
)
from src.statistical_tests import compare_all_models

# The goal is to have a clear step-by-step presentation so the reviewer can easily follow the work performed
def main():
    print("=" * 80)
    print("BANKRUPTCY PREDICTION")
    print("Dataset: Polish Companies Bankruptcy (UCI ML Repository, ID: 365)")
    print("Source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data")
    print("=" * 80)
    print()
    
    print("STEP 1: Loading data")
    print("-" * 80)
    dataframe, target_column = load_polish_bankruptcy_data()
    print()
    
    print("STEP 2: Initial data inspection")
    print("-" * 80)
    inspect_data(dataframe, target_column)
    save_to_csv(dataframe, output_path="data/processed/polish_bankruptcy_renamed.csv")
    print()
    
    print("2.4: Data quality analysis")
    print("-" * 80)
    print()
    quality_issues = analyze_data_quality(dataframe, target_column)
    
    # Summarize data quality 
    n_rows = len(dataframe)
    n_cols = len(dataframe.columns)
    quality_summary = summarize_data_quality(quality_issues, n_rows, n_cols, cleaning_report=None)
    
    print("=" * 80)
    print("DATA QUALITY SUMMARY AND CLEANING RULES")
    print("=" * 80)
    print(f"Duplicate rows: {quality_summary['n_duplicate_rows']} rows -> remove them")
    if (quality_summary['n_duplicate_columns'] > 0):
        duplicate_cols_list = quality_issues.get('duplicateColumns', [])
        if (duplicate_cols_list):
            cols_str = ", ".join(duplicate_cols_list)
            print(f"Duplicate columns: {quality_summary['n_duplicate_columns']} columns -> remove {cols_str}")
        else:
            print(f"Duplicate columns: {quality_summary['n_duplicate_columns']} columns -> remove them")
    print(f"Missing values: {quality_summary['n_missing_features']} features -> impute with median (<40%) or remove (>= 40%)")
    if (quality_summary['n_constant_features'] > 0):
        print(f"Constant features: {quality_summary['n_constant_features']} features -> remove them")
    if (quality_summary['n_correlated_pairs'] > 0):
        print(f"Highly correlated pairs: {quality_summary['n_correlated_pairs']} pairs -> remove one feature from each pair")
    if (quality_summary['n_outlier_features'] > 0):
        print(f"Features with outliers: {quality_summary['n_outlier_features']} features -> cap outliers")
    print()
    print(f"Elements requiring intervention: {quality_summary['total_elements_concerned']}/{quality_summary['total_elements']} ({quality_summary['pct_elements_concerned']}%)")
    print("-" * 80)
    dataframe_clean, cleaning_report = clean_data(
        dataframe, target_column,
        missing_threshold=40,
        correlation_threshold=0.95
    )
    
    # Recalculate quality summary with cleaning_report for accurate removed elements count
    quality_summary = summarize_data_quality(quality_issues, n_rows, n_cols, cleaning_report=cleaning_report)
    
    print("=" * 80)
    print("CLEANING PROCESS REPORT")
    print("=" * 80)
    print(f"Original shape: {cleaning_report['original_shape']}")
    print(f"Final shape: {cleaning_report['final_shape']}")
    print(f"Features removed: {cleaning_report['features_removed']}")
    if (cleaning_report['operations']):
        print("\nOperations performed:")
        for i, operation in enumerate(cleaning_report['operations'], 1):
            print(f"  {i}. {operation}")
    print("=" * 80)
    print()
    target_column = detect_target_column(dataframe_clean)
    features, target = preprocess_data(dataframe_clean, target_column)
    
    print("STEP 4: Data preprocessing and train/test split")
    print("-" * 80)
    X_train, X_test, y_train, y_test, _ = split_and_scale(features, target)
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")
    print(f"Features standardized with StandardScaler")
    print()
    print(f"Features separated: {features.shape[1]} variables")
    print(f"Target separated: {target.name}")
    print()
    
    # Train models to make predictions
    # Note: Random Forest and KNN are computationally expensive, the program takes time
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train, n_neighbors=5)
    print()
    
    print("STEP 5: Model evaluation after training (80/20)")
    print("-" * 80)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    knn_metrics = evaluate_model(knn_model, X_test, y_test, "K-Nearest Neighbors")
    
    # Store results for comparison
    results = {
        "Logistic Regression": {
            "Accuracy": lr_metrics["accuracy"],
            "Precision": lr_metrics["precision"],
            "Recall": lr_metrics["recall"],
            "F1-score": lr_metrics["f1"],
            "ROC-AUC": lr_metrics["roc_auc"]
        },
        "Random Forest": {
            "Accuracy": rf_metrics["accuracy"],
            "Precision": rf_metrics["precision"],
            "Recall": rf_metrics["recall"],
            "F1-score": rf_metrics["f1"],
            "ROC-AUC": rf_metrics["roc_auc"]
        },
        "K-Nearest Neighbors": {
            "Accuracy": knn_metrics["accuracy"],
            "Precision": knn_metrics["precision"],
            "Recall": knn_metrics["recall"],
            "F1-score": knn_metrics["f1"],
            "ROC-AUC": knn_metrics["roc_auc"]
        }
    }
    
    print("\nPerformance summary:")
    print("-" * 80)
    for modelName, metrics in results.items():
        print(f"\n{modelName}:")
        for metricName, value in metrics.items():
            print(f"  {metricName}: {value:.4f}")
    
    # Determine best model based on ROC-AUC to answer the question of which is the best model
    winnerScores = {
        "Logistic Regression": results["Logistic Regression"]["ROC-AUC"],
        "Random Forest": results["Random Forest"]["ROC-AUC"],
        "K-Nearest Neighbors": results["K-Nearest Neighbors"]["ROC-AUC"]
    }
    bestModel = max(winnerScores, key=winnerScores.get)
    bestScore = winnerScores[bestModel]
    
    print("\n" + "=" * 80)
    print(f"Best model: {bestModel} (ROC-AUC: {bestScore:.4f})")
    print("=" * 80)
    print()
    
    # Generate bankruptcy rate by horizon plot for the report, justifies our analysis
    plot_bankruptcy_rate_by_horizon(dataframe_clean, target_column,save_path="results/bankruptcy_rate_by_horizon.png")
    
    plot_roc_curve(y_test, lr_metrics["y_pred_proba"], "Logistic Regression",save_path="results/roc_lr.png")
    plot_roc_curve(y_test, rf_metrics["y_pred_proba"], "Random Forest", save_path="results/roc_rf.png")
    plot_roc_curve(y_test, knn_metrics["y_pred_proba"], "K-Nearest Neighbors", save_path="results/roc_knn.png")
    plot_confusion_matrix(lr_metrics["confusion_matrix"], "Logistic Regression",save_path="results/cm_lr.png")
    plot_confusion_matrix(rf_metrics["confusion_matrix"], "Random Forest",save_path="results/cm_rf.png")
    plot_confusion_matrix(knn_metrics["confusion_matrix"], "K-Nearest Neighbors",save_path="results/cm_knn.png")
    
    # Global feature importance analysis (all horizons combined)
    print()
    print("STEP 5.1: Global feature importance analysis (all horizons)")
    print("-" * 80)
    global_results = analyze_global_features(
        lr_model, rf_model, knn_model, X_train, y_train, X_train.columns, 
        top_n=15, save_dir="results"
    )
    
    # Extract common factors from global analysis (top 15 of each model, all horizons combined)
    most_common_factors = []
    if (global_results and global_results.get('common_factors')):
        most_common_factors = global_results['common_factors']
        
        if (most_common_factors):
            print()
            print("Most common factors across all 3 models (global analysis, all horizons):")
            for factor in most_common_factors:
                factor_display = factor.replace("_", " ").title()
                print(f"  • {factor_display}")
            print()
    
    # Feature importance analysis by prediction horizon
    # Since the bankruptcy rate varies significantly by horizon (3.9% at 5 years vs 7% at 1 year),
    # we analyze feature importance separately for each horizon for all models (LR, RF, KNN)
    # This is the chosen approach for the analysis
    if ('year' in dataframe_clean.columns):
        print()
        print("STEP 5.2: Feature importance analysis by prediction horizon")
        print("-" * 80)
        horizon_results_all = analyze_all_models_by_horizon(
            dataframe_clean, target_column, 
            top_n=15,
            save_dir="results"
        )
        print()
    
    print("STEP 6: Statistical tests for model comparison")
    print("-" * 80)
    
    models_dict = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "K-Nearest Neighbors": knn_model
    }
    # Test which is the best model statistically to support our initial test
    statistical_results = compare_all_models(
        models_dict, X_train, y_train, X_test, y_test, cv=5, random_state=42
    )
    
    print()
    print("=" * 80)
    print("SUMMARY OF STATISTICAL TESTS")
    print("=" * 80)
    print()
    print("The statistical tests help determine whether the observed performance")
    print("differences between models are statistically significant or due to chance.")
    print()
    print("Key findings:")
    print("-" * 80)
    
    print("\nPaired t-test results (Cross-Validation):")
    for comparison, results in statistical_results['paired_t_tests'].items():
        model1 = results['model1_name']
        model2 = results['model2_name']
        p_display = f"{results['p_value']:.4f} < 0.05" if (results['p_value'] < 0.05) else f"{results['p_value']:.4f}"
        if (results['significant']):
            if (results['mean_difference'] > 0):
                winner = model1
            else:
                winner = model2
            print(f"  • {model1} vs {model2}: Significant difference (p={p_display})")
            print(f"    → {winner} is significantly better")
        else:
            print(f"  • {model1} vs {model2}: No significant difference (p={p_display})")
            print(f"    → Models have equivalent performance")
    
    print("\nMcNemar's test results (Test Set):")
    for comparison, results in statistical_results['mcnemar_tests'].items():
        model1 = results['model1_name']
        model2 = results['model2_name']
        p_display = f"{results['p_value']:.4f} < 0.05" if (results['p_value'] < 0.05) else f"{results['p_value']:.4f}"
        if (results['significant']):
            if (results['model1_correct_model2_wrong'] > results['model1_wrong_model2_correct']):
                winner = model1
            else:
                winner = model2
            print(f"  • {model1} vs {model2}: Significant difference (p={p_display})")
            print(f"    → {winner} makes significantly more correct predictions")
        else:
            print(f"  • {model1} vs {model2}: No significant difference (p={p_display})")
            print(f"    → Models have equivalent error rates")
    
    print()
    print("=" * 80)
    print()
    
    print("Program ran successfully and  Key financial factors identified and analyzed")
    print("Results saved in the results folder")
    

# Allows executing the program using: python main.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bankruptcy prediction pipeline - Polish Companies Dataset"
    )
    
    args = parser.parse_args()
    main()

