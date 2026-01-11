"""
Data cleaning and preprocessing utilities for the Polish Bankruptcy dataset.
This module applies the cleaning rules used in this project:
duplicate rows, missing values, redundant ratios (from the column mapping), correlation filtering, and outlier capping.
"""
import pandas as pd
import numpy as np
from src.data_column_mapping import detect_duplicate_mappings, COLUMN_MAPPING_SHORT
from src.data_analysis import (
    detect_duplicate_rows,
    detect_missing_values,
    detect_constant_features,
    detect_highly_correlated_features,
    detect_outliers
)
from src.data_loader import preprocess_data


def clean_data(dataframe, target_column, missing_threshold=40, correlation_threshold=0.95):
    # missing_threshold=40, features with >=40% missing are too unreliable to impute meaningfully. 
    # correlation_threshold=0.95, a very high correlation indicates redundancy, removing one feature won't lose information.
    # Upstream analysis showed that 0.95 (see report figure 7) was a good compromise between aggressiveness and redundancy.
    
    # These choices are justified by the literature (see report references).
    """
    Clean the dataset by applying the data-quality rules defined.

    Steps include: removing duplicate rows (same firms), dropping uninformative features (constant/near-constant),
    handling missing values, removing redundant ratios detected from the mapping, filtering highly correlated features, and capping extreme values.
    """
    dataframe_clean = dataframe.copy()
    original_shape = dataframe_clean.shape
    # Dictionary to track all modifications and display them later
    cleaning_report = {
        "original_shape": original_shape,
        "removed_features": [],
        "removed_rows": 0,
        "operations": [],
        "missing_values_imputed": 0,
        "outliers_capped": 0
    }
 
    features, target = preprocess_data(dataframe_clean, target_column)
    
    # Step 1: Remove duplicate rows
    # We assume these are data collection errors (same company, issues with how companies report their ratios, etc.)
    n_duplicates = detect_duplicate_rows(dataframe_clean)
    if (n_duplicates > 0):
        initial_rows = len(dataframe_clean)
        dataframe_clean = dataframe_clean.drop_duplicates()
        removed_rows = initial_rows - len(dataframe_clean)
        cleaning_report["removed_rows"] = removed_rows
        cleaning_report['operations'].append(f"Removed {removed_rows} duplicate rows")
        # Update features and target after removing duplicate rows
        features, target = preprocess_data(dataframe_clean, target_column)
    
    # Step 2: Drop features that share the same short-name mapping (redundant ratios)
    # These features do not provide any additional information.
    duplicates = detect_duplicate_mappings(COLUMN_MAPPING_SHORT)
    
    if (duplicates):
        # Keep one feature per duplicate group.
        # I keep A36 and drop A9 because they have the same short name (explanation can be found at column_mapping).
        for value, keys in duplicates.items():
            # Determine which key to keep and which to exclude
            if (("A9" in keys) and ("A36" in keys)):
                kept_key = "A36" 
                keys_to_exclude = ["A9"]   
            else:
                # Default: keep the first one, exclude the others
                # Do not use, as verification has already been performed. Left for structural purposes.
                kept_key = keys[0]
                keys_to_exclude = keys[1:]
            
            for key_to_exclude in keys_to_exclude:
                # Find the column name in the dataframe (could be original A9 or renamed to "Asset turnover")
                renamed = COLUMN_MAPPING_SHORT.get(key_to_exclude)
                col_name = renamed if renamed else key_to_exclude
                
                # Find all positions where this column name appears
                matching_indices = [i for i, col in enumerate(features.columns) if col == col_name]
                
                if (matching_indices):
                    if (key_to_exclude == "A9"):
                        # A9 should be the first occurrence (earlier in the dataset)
                        indices_to_remove = [matching_indices[0]]
                    else:
                        # For other cases, keep the first occurrence, remove the others
                        indices_to_remove = matching_indices[1:]
                    
                    if (indices_to_remove):
                        # Remove by position using iloc to avoid removing all columns with the same name
                        cols_to_keep = [i for i in range(len(features.columns)) if i not in indices_to_remove]
                        features = features.iloc[:, cols_to_keep]
                        for idx in indices_to_remove:
                            col_to_remove = col_name
                            cleaning_report["removed_features"].append(col_to_remove)
                        cleaning_report["operations"].append(
                            f"Removed {col_to_remove} ({key_to_exclude}): duplicate mapping with {kept_key} "
                            f"(both map to '{value}'), excluded from analysis"
                        )
        # Update dataframe_clean to reflect the removed columns
        dataframe_clean = pd.concat([features, target], axis=1)
    
    # Step 3: Remove constant and low variance features
    # These features are similar for all companies and therefore have little predictive power for bankruptcy
    constant_features = detect_constant_features(features)
    
    if (constant_features):
        features = features.drop(columns=constant_features)
        cleaning_report["removed_features"].extend(constant_features)
        cleaning_report['operations'].append(f"Removed {len(constant_features)} constant/low variance features")
   

    # Step 4: Remove features with high percentage of missing values
    # In this case, their role in bankruptcy prediction should not be considered because they are too rarely present
    # to draw meaningful conclusions.
    missing_data = detect_missing_values(features, threshold=missing_threshold)
    high_missing = list(missing_data['high'].keys())
    
    if (high_missing):
        features = features.drop(columns=high_missing)
        cleaning_report["removed_features"].extend(high_missing)
        cleaning_report["operations"].append(
            f"Removed {len(high_missing)} features with >={missing_threshold}% missing values"
        )
    
    # Step 5: Impute remaining missing values (median for numeric features)
    # According to the literature, below 40%, the variable can still be kept to preserve the dataset.
    missing_cols = features.columns[features.isnull().any()].tolist()
    missing_count_imputed = 0
    if (missing_cols):
        missing_count_imputed = features[missing_cols].isnull().sum().sum()
        for col in missing_cols:
            features[col] = features[col].fillna(features[col].median())
        
        cleaning_report["operations"].append(
            f"Imputed missing values in {len(missing_cols)} features (median)"
        )
    cleaning_report["missing_values_imputed"] = missing_count_imputed
    
    # Step 6: Remove highly correlated features because they provide the same information
    # This is done in two phases:
    # Phase 6a: Remove features correlated with at least 2 others (most redundant first)
    #          to avoid removing a feature that would no longer be correlated with anyone
    # Phase 6b: Remove features from remaining pairs (keep first, remove second)
    #          to clean up high correlations
    
    total_removed_phase_a = 0
    total_removed_phase_b = 0
    
    # Phase 6a
    # Process one by one to ensure we always check for features with ≥2 correlations
    while True:
        high_corr_pairs = detect_highly_correlated_features(features, threshold=correlation_threshold)
        if (not high_corr_pairs):
            break
        
        feature_corr_count = {}
        for feat1, feat2, corr_val in high_corr_pairs:
            feature_corr_count[feat1] = feature_corr_count.get(feat1, 0) + 1
            feature_corr_count[feat2] = feature_corr_count.get(feat2, 0) + 1
     
        features_with_multiple_corr = [feat for feat, count in feature_corr_count.items() if count >= 2]
        
        if (not features_with_multiple_corr):
            # No more features with ≥2 correlations, move to phase 6b
            break
        
        # Remove one feature with multiple correlations (remove the first one alphabetically)
        # Alphabetical order ensures reproducibility and better traceability
        feature_to_remove = sorted(features_with_multiple_corr)[0]
        features = features.drop(columns=[feature_to_remove])
        cleaning_report["removed_features"].append(feature_to_remove)
        total_removed_phase_a += 1
    
    if (total_removed_phase_a > 0):
        cleaning_report["operations"].append(
            f"Removed {total_removed_phase_a} feature{'s' if total_removed_phase_a > 1 else ''} "
            f"with correlation > {correlation_threshold} to at least 2 other features"
        )
    
    # Phase 6b
    # Iterate until no more correlations > threshold remain
    iteration = 0
    while True:
        high_corr_pairs = detect_highly_correlated_features(features, threshold=correlation_threshold)
        if (not high_corr_pairs):
            break
        
        # Find features to remove (keep first, remove second from each pair)
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            if ((feat1 not in features_to_remove) and (feat2 not in features_to_remove)):
                features_to_remove.add(feat2)
        
        if (not features_to_remove):
            break
        
        features = features.drop(columns=list(features_to_remove))
        cleaning_report["removed_features"].extend(list(features_to_remove))
        total_removed_phase_b += len(features_to_remove)
        iteration += 1
    
    if (total_removed_phase_b > 0):
        cleaning_report["operations"].append(
            f"Removed {total_removed_phase_b} feature{'s' if total_removed_phase_b > 1 else ''} "
            f"from highly correlated pairs (threshold: {correlation_threshold}, "
            f"{iteration} iteration{'s' if iteration > 1 else ''})"
        )
    
    # Step 7: Outlier capping using 3*IQR (less aggressive than 1.5*IQR which flags too many normal values)
    # This approach is suggested by the literature
    # Only applied when outliers represent >5% of data, otherwise they're likely genuine rare extreme cases
    outlier_counts = detect_outliers(features, multiplier=3, threshold_pct=5)
    
    if (outlier_counts):
        # Cap outliers for features that have significant outliers
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        capped_features = []
        total_outliers = 0
        
        for col in outlier_counts.keys():
            if (col in numeric_cols):
                Q1 = features[col].quantile(0.25)
                Q3 = features[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Clip values to IQR boundaries (winsorization)
                outliers_before = outlier_counts[col]['count']
                features[col] = features[col].clip(lower=lower_bound, upper=upper_bound)
                capped_features.append(col)
                total_outliers += outliers_before
        
        if (capped_features):
            cleaning_report["operations"].append(
                f"Capped outliers in {len(capped_features)} features using IQR method (3*IQR)"
            )
            cleaning_report["outliers_capped"] = total_outliers
    else:
        cleaning_report["outliers_capped"] = 0
    
    # Recombine cleaned features with target
    dataframe_clean = pd.concat([features, target], axis=1)
    cleaning_report["final_shape"] = dataframe_clean.shape
    cleaning_report["features_removed"] = len(cleaning_report["removed_features"])
    
    print()
    
    return dataframe_clean, cleaning_report

