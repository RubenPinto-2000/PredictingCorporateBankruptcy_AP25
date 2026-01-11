"""
Data quality checks for the UCI Polish Bankruptcy dataset (id=365) and console output display.
These methods are then used in other modules (code simplification).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_loader import preprocess_data
from src.data_column_mapping import detect_duplicate_mappings, COLUMN_MAPPING_SHORT

def detect_duplicate_rows(dataframe):
    """
    Count duplicate rows for cleaning and display.
    """
    return dataframe.duplicated().sum()


def detect_missing_values(features, threshold=None):
    """
    Return missing stats per feature for cleaning and display.
    """
    missing = features.isnull().sum()
    n_total = len(features)
    
    missing_features = missing[missing > 0].sort_values(ascending=False)
    result = {'all': {},'high': {}}
    
    for feat in missing_features.index:
        count_val = missing[feat]
        if (isinstance(count_val, pd.Series)):
            count = int(count_val.iloc[0])
        else:
            count = int(count_val)
        pct = float((count / n_total) * 100)
        result['all'][feat] = {'count': count, 'percentage': pct}
    
    if (threshold is not None):
        for feat in missing_features.index:
            pct = result['all'][feat]['percentage']
            if (pct >= threshold):
                result['high'][feat] = result['all'][feat].copy()
    return result


def detect_constant_features(features):
    """
    List constant / near-constant features for cleaning and display.
    """
    constant_features = []
    for col in features.columns:
        col_data = features[col]
        
        # Handle case where col_data might be a DataFrame (duplicate columns)
        if (isinstance(col_data, pd.DataFrame)):
            col_data = col_data.iloc[:, 0]
        
        nunique_val = col_data.nunique()
        # Convert to int if it's a Series (shouldn't happen but safer)
        if (isinstance(nunique_val, pd.Series)):
            nunique_val = int(nunique_val.iloc[0])
        else:
            nunique_val = int(nunique_val)
        
        if (nunique_val <= 1):
            constant_features.append(col)
        elif (col_data.dtype in ['float64', 'int64']):
            std_val = col_data.std()
            # Handle case where std() might return a Series
            if (isinstance(std_val, pd.Series)):
                std_val = float(std_val.iloc[0])
            else:
                std_val = float(std_val)
            
            if (std_val < 1e-10):
                constant_features.append(col)
    return constant_features


def detect_highly_correlated_features(features, threshold):
    """
    List highly correlated numeric feature pairs (|r| > threshold) for cleaning and display.
    
    The threshold is typically fixed at 0.95, but left as a parameter since this is an arbitrary decision
    that may vary depending on the dataset and analysis requirements.
    """
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    if (len(numeric_cols) == 0):
        return []
    
    corr_matrix = features[numeric_cols].corr().abs()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if (corr_val > threshold):
                high_corr_pairs.append((corr_matrix.columns[i],corr_matrix.columns[j],float(corr_val)))
    return high_corr_pairs


def detect_outliers(features, multiplier=3, threshold_pct=5):
    """
    Flag features with many IQR outliers for cleaning and display.
    
    multiplier=3: uses 3*IQR instead of 1.5*IQR to avoid flagging too many normal values as outliers.
    This is a common practice in the literature for extreme outliers, as mentioned in data_cleaning.py.
    threshold_pct=5: only flags features where outliers represent >5% of data (otherwise they might be genuine rare extremes)
    """
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    
    for col in numeric_cols:
        col_data = features[col]
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        # Convert to int if it's a Series (shouldn't happen but safer)
        if (isinstance(outliers, pd.Series)):
            outliers = int(outliers.iloc[0])
        else:
            outliers = int(outliers)
        
        if (outliers > 0):
            outlier_pct = (outliers / len(features)) * 100
            if (outlier_pct > threshold_pct):
                outlier_counts[col] = {'count': outliers, 'percentage': float(outlier_pct)}
    
    return outlier_counts


def plot_missing_values(missing_data, save_path="results/missing_values.png"):
    """
    Create a bar plot showing missing values percentage per feature for the report.
    """
    if (not missing_data['all']):
        return
    
    # Sort by percentage descending
    sorted_missing = sorted(missing_data['all'].items(), 
                           key=lambda x: x[1]['percentage'], reverse=True)
    
    features = [feat for feat, _ in sorted_missing]
    percentages = [info['percentage'] for _, info in sorted_missing]
    
    plt.figure(figsize=(12, max(8, len(features) * 0.3)))
    plt.barh(range(len(features)), percentages, color='steelblue', alpha=0.7)
    plt.yticks(range(len(features)), features, fontsize=9)
    plt.xlabel("Percentage of Missing Values (%)", fontsize=11, fontweight="bold")
    plt.title("Missing Values Analysis", fontsize=13, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_highly_correlated_features(high_corr_pairs, save_path="results/highly_correlated_features.png"):
    """
    Create a heatmap showing which features are correlated with which for the report.
    """
    if (not high_corr_pairs):
        return
    
    # Take top 20 pairs for visualization
    top_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:20]
    
    # Extract unique features
    all_features = set()
    for pair in top_pairs:
        all_features.add(pair[0])
        all_features.add(pair[1])
    all_features = sorted(list(all_features))
    
    # Create correlation matrix
    corr_matrix = pd.DataFrame(0.0, index=all_features, columns=all_features)
    for feat1, feat2, corr_val in top_pairs:
        corr_matrix.loc[feat1, feat2] = corr_val
        corr_matrix.loc[feat2, feat1] = corr_val
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(all_features) * 0.6), max(10, len(all_features) * 0.6)))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Reds', 
                cbar_kws={'label': 'Correlation Coefficient'}, 
                square=True, linewidths=0.5, linecolor='gray')
    plt.title("Highly Correlated Feature Pairs (Top 20)\nShows which features are correlated with which", 
              fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Feature", fontsize=11, fontweight="bold")
    plt.ylabel("Feature", fontsize=11, fontweight="bold")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_outliers(outlier_counts, save_path="results/outliers.png"):
    """
    Create a bar plot showing outliers count per feature for the report.
    """
    if (not outlier_counts):
        return
    sorted_outliers = sorted(outlier_counts.items(), 
                            key=lambda x: x[1]['percentage'], reverse=True)
    
    features = [feat for feat, _ in sorted_outliers]
    percentages = [info['percentage'] for _, info in sorted_outliers]
    
    plt.figure(figsize=(12, max(8, len(features) * 0.3)))
    plt.barh(range(len(features)), percentages, color='tomato', alpha=0.7)
    plt.yticks(range(len(features)), features, fontsize=9)
    plt.xlabel("Percentage of Outliers (%)", fontsize=11, fontweight="bold")
    plt.title("Outlier Detection Analysis (IQR method, 3*IQR)", fontsize=13, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_data_quality(dataframe, target_column):
    """
    Display results to the console using the previous methods step by step.
    """
    
    features, target = preprocess_data(dataframe, target_column)
    
    # Initialize dictionary to store analysis results
    issues = {
        'missingValues': {},
        'highCorrelation': [],
        'outliers': {},
        'constantFeatures': [],
        'duplicateRows': 0,
        'duplicateColumns': []
    }
    
    print("2.4.1. DUPLICATE ROWS ANALYSIS")
    print("-" * 80)
    duplicates = detect_duplicate_rows(dataframe)
    issues['duplicateRows'] = duplicates
    if (duplicates > 0):
        print(f"Duplicate rows found: {duplicates} ({duplicates/len(dataframe)*100:.2f}%)")
        if ('year' in dataframe.columns):
            duplicated_to_remove = dataframe[dataframe.duplicated(keep='first')]
            
            if (len(duplicated_to_remove) > 0):
                print("By year:")
                year_duplicates = duplicated_to_remove['year'].value_counts().sort_index()
                for year in sorted(year_duplicates.index):
                    count = year_duplicates[year]
                    print(f"  Year {year}: {count} duplicate rows")
    else:
        print("No duplicate rows found")
    print()
    
    print("2.4.2. DUPLICATE COLUMNS ANALYSIS")
    print("-" * 80)
    duplicate_mappings = detect_duplicate_mappings(COLUMN_MAPPING_SHORT)
    duplicate_cols = []
    if (duplicate_mappings):
        # Get the renamed column names from the duplicates
        for value, keys in duplicate_mappings.items():
            duplicate_cols.append(value)
        print(f"Duplicate column names found: {len(duplicate_cols)}")
        issues['duplicateColumns'] = duplicate_cols
    else:
        print("No duplicate column names found")
    print()
    
    # Remove duplicates before continuing analysis
    if (duplicates > 0):
        dataframe = dataframe.drop_duplicates()
        features, target = preprocess_data(dataframe, target_column)
    
    if (duplicate_cols):
        # Keep only first occurrence of each duplicate column name
        seen = set()
        cols_to_keep = []
        for col in features.columns:
            if (col not in seen):
                cols_to_keep.append(col)
                seen.add(col)
        features = features[cols_to_keep]
    
    print("2.4.3. MISSING VALUES ANALYSIS")
    print("-" * 80)
    # features with >=40% missing are too unreliable to impute meaningfully, as indicated in the cleaning module
    missing_data = detect_missing_values(features, threshold=40)
    
    if (missing_data['all']):
        print(f"Features with missing values: {len(missing_data['all'])}")
        issues['missingValues'] = missing_data['all']
        issues['missingValuesHigh'] = missing_data.get('high', {})
        plot_missing_values(missing_data, save_path="results/missing_values.png")
    else:
        print("No missing values found")
    print()
    
    print("2.4.4. CONSTANT AND LOW VARIANCE FEATURES")
    print("-" * 80)
    
    constant_features = detect_constant_features(features)
    
    if (constant_features):
        print(f"Constant/low variance features found: {len(constant_features)}")
        issues['constantFeatures'] = constant_features
    else:
        print("No constant features found")
    print()
    

    print("2.4.5. FEATURE CORRELATION ANALYSIS")
    print("-" * 80)
    
    # threshold=0.95 is a very high correlation indicates redundancy, removing one feature won't lose information, as indicated in data_cleaning
    high_corr_pairs = detect_highly_correlated_features(features, threshold=0.95)
    
    if (high_corr_pairs):
        print(f"Highly correlated feature pairs (correlation > 0.95): {len(high_corr_pairs)}")
        issues['highCorrelation'] = high_corr_pairs
        
        plot_highly_correlated_features(high_corr_pairs, save_path="results/highly_correlated_features.png")
    else:
        print("No highly correlated features found (threshold: 0.95)")
    print()
    
    print("2.4.6. OUTLIER DETECTION (using IQR method)")
    print("-" * 80)

    # See data_cleaning for explanation
    outlier_counts = detect_outliers(features, multiplier=3, threshold_pct=5)
    
    if (outlier_counts):
        print(f"Features with significant outliers (>5%): {len(outlier_counts)}")
        issues['outliers'] = outlier_counts
        
        plot_outliers(outlier_counts, save_path="results/outliers.png")
    else:
        print("No significant outliers detected")
    print()
    
    class_dist = target.value_counts()
    imbalance_ratio = class_dist.min() / class_dist.max()
    
    return issues


def summarize_data_quality(issues, n_rows, n_cols, cleaning_report=None):
    """
    Summarize data quality issues to the console.
    """
    summary = {
        'n_duplicate_rows': issues['duplicateRows'],
        'n_duplicate_columns': len(issues.get('duplicateColumns', [])),
        'n_missing_features': len(issues['missingValues']),
        'n_constant_features': len(issues['constantFeatures']),
        'n_correlated_pairs': len(issues['highCorrelation']),
        'n_outlier_features': len(issues['outliers'])
    }

    cells_duplicate_rows = summary['n_duplicate_rows'] * n_cols
    n_rows_after_duplicate_removal = n_rows - summary['n_duplicate_rows']
    cells_duplicate_columns = summary['n_duplicate_columns'] * n_rows_after_duplicate_removal
    cells_missing = 0
    for feat, info in issues['missingValues'].items():
        cells_missing += info['count']
    cells_constant_features = summary['n_constant_features'] * n_rows_after_duplicate_removal
    cells_correlated_features = summary['n_correlated_pairs'] * n_rows_after_duplicate_removal
    cells_outliers = 0
    for feat, info in issues['outliers'].items():
        cells_outliers += info['count']

    if (cleaning_report is not None):
        original_elements = cleaning_report['original_shape'][0] * cleaning_report['original_shape'][1]
        final_elements = cleaning_report['final_shape'][0] * cleaning_report['final_shape'][1]
        outliers_capped = cleaning_report.get('outliers_capped', 0)
        missing_imputed = cleaning_report.get('missing_values_imputed', 0)
        total_elements_removed = original_elements - final_elements - outliers_capped - missing_imputed
    else:
        total_elements_removed = 0

    total_elements_concerned = (
        cells_duplicate_rows +
        cells_duplicate_columns +
        cells_missing +
        cells_constant_features +
        cells_correlated_features +
        cells_outliers
    )
    
    total_elements = n_rows * n_cols
    
    if (total_elements > 0):
        pct_concerned = (total_elements_concerned / total_elements) * 100
    else:
        pct_concerned = 0
    
    summary['pct_elements_concerned'] = round(pct_concerned, 2)
    summary['total_elements_concerned'] = total_elements_concerned
    summary['total_elements_removed'] = total_elements_removed
    summary['total_elements'] = total_elements
    
    return summary


def plot_bankruptcy_rate_by_horizon(dataframe, target_column, save_path="results/bankruptcy_rate_by_horizon.png"):
    """
    Create a visualization showing the bankruptcy rate by prediction horizon to justify the fact that working by horizon.
    """
    if ('year' not in dataframe.columns):
        return
    
    # Calculate statistics by horizon
    year_stats = []
    for year in sorted(dataframe['year'].unique()):
        horizon = 6 - int(year)
        subset = dataframe[dataframe['year'] == year]
        total = len(subset)
        bankrupt = subset[target_column].sum()
        pct_bankrupt = (bankrupt / total) * 100 if total > 0 else 0
        
        year_stats.append({
            'year': int(year),
            'horizon': horizon,
            'total': total,
            'bankrupt': bankrupt,
            'pct_bankrupt': pct_bankrupt
        })
    
    # Extract data for plotting
    horizons = [s['horizon'] for s in year_stats]
    pct_failures = [s['pct_bankrupt'] for s in year_stats]
    years = [s['year'] for s in year_stats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(horizons)))
    bars = ax1.bar(horizons, pct_failures, color=colors, alpha=0.8, edgecolor='darkred', linewidth=1.5)
    
    for h, pct, year in zip(horizons, pct_failures, years):
        ax1.text(h, pct + 0.1, f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax1.text(h, -0.3, f'Year {year}', ha='center', va='top', fontsize=9, style='italic')
    
    ax1.set_xlabel('Prediction Horizon (years)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bankruptcy Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Bankruptcy Rate by Prediction Horizon', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(horizons)
    ax1.set_xticklabels([f'{h} year{"s" if h > 1 else ""}' for h in horizons])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, max(pct_failures) * 1.15])
    
    ax2.plot(horizons, pct_failures, marker='o', markersize=10, linewidth=2.5, 
             color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markeredgewidth=2)
    
    for h, pct, year in zip(horizons, pct_failures, years):
        ax2.text(h, pct + 0.15, f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax2.text(h, pct - 0.4, f'Year {year}', ha='center', va='top', fontsize=9, style='italic')
    
    # Linear regression to show trend
    z = np.polyfit(horizons, pct_failures, 1)
    p = np.poly1d(z)
    ax2.plot(horizons, p(horizons), "r--", alpha=0.6, linewidth=2, 
             label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    ax2.set_xlabel('Prediction Horizon (years)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bankruptcy Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Evolution of Bankruptcy Rate by Horizon', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(horizons)
    ax2.set_xticklabels([f'{h} year{"s" if h > 1 else ""}' for h in horizons])
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_ylim([min(pct_failures) * 0.9, max(pct_failures) * 1.15])
    
    plt.tight_layout()
    
    # Save plot using the same rule as usual
    if (save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

