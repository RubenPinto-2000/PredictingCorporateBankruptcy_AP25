"""
Load and preprocess the UCI Polish Bankruptcy dataset (id=365).
Loads from a local CSV when available; otherwise downloads from UCI and caches the raw file.
Renames columns in memory when they still use generic labels to use them in the model.
"""
import os
import ssl
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.data_column_mapping import rename_columns

def detect_target_column(dataframe):
    """
    Return the target column name for the Polish Bankruptcy dataset.
    Checks for "Bankruptcy status" (after renaming) or "class" (before renaming).
    """
    # Check for renamed target column first, then original name
    if ("Bankruptcy status" in dataframe.columns):
        return "Bankruptcy status"
    if ("class" in dataframe.columns):
        return "class"
    # This should never happen with the correct Polish Bankruptcy dataset
    raise ValueError(f"Target column not found. Expected 'Bankruptcy status' or 'class', but found columns: {list(dataframe.columns)}")


def load_polish_bankruptcy_data(data_dir="data/raw", dataset_name="polish_bankruptcy.csv", allow_insecure_ssl=False):
    """
    Load the dataset 

    If a local CSV is available, it is loaded from disk. Otherwise the dataset is downloaded from UCI
    and saved locally. Columns are renamed in memory when necessary.
    """
    # Ensure cache directory exists
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, dataset_name)
    
    if (os.path.exists(data_path)):
        print(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
        target_column = detect_target_column(df)
        
        # Rename columns using mapping (handles both generic A1/A2/ columns and "class" -> "Bankruptcy status")
        has_generic_cols = False
        for col in df.columns:
            if (col.startswith('A') and col[1:].isdigit()):
                has_generic_cols = True
                break
        
        # Always rename if we have generic columns OR if target is still "class"
        if (has_generic_cols or target_column == "class"):
            df = rename_columns(df)
            target_column = detect_target_column(df)
        
        return df, target_column
    else:
        # Cache miss: download once from UCI and store the raw CSV locally
        print("Downloading dataset from UCI ML Repository...")
        
        # Some environments have SSL inspection / missing certs; allow an opt-in insecure mode
        original_https_context = ssl._create_default_https_context
        if (allow_insecure_ssl):
            # Opt-in: disable SSL verification (use only if you know your environment requires it)
            ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            # Fetch from UCI dataset(ID: 365)
            polish_dataset = fetch_ucirepo(id=365)
        except Exception as e:
            print(f"Unable to fetch UCI dataset id=365 ({type(e).__name__}: {e}).")
            print("If you are behind a restrictive network or encountering SSL errors,")
            print("retry with allow_insecure_ssl=True.")
            raise RuntimeError(f"Failed to fetch dataset: {e}") from e
        finally:
            # Always restore SSL settings
            ssl._create_default_https_context = original_https_context
        
        # Build a single DataFrame (features + target)
        features = polish_dataset.data.features
        target = polish_dataset.data.targets
        df = pd.concat([features, target], axis=1)
        target_column = detect_target_column(df)
        
        # Save the raw version (before renaming) for reproducibility
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to: {data_path}")
        
        # Rename columns using mapping (handles both generic A1/A2/ columns and "class" -> "Bankruptcy status")
        has_generic_cols = False
        for col in df.columns:
            if (col.startswith('A') and col[1:].isdigit()):
                has_generic_cols = True
                break
        
        # Always rename if we have generic columns OR if target is still "class"
        if (has_generic_cols or target_column == "class"):
            df = rename_columns(df)
            target_column = detect_target_column(df)
        
        return df, target_column


def inspect_data(dataframe, target_column):
    """
    Display an initial analysis of the data before cleaning to the console.
    Allows comparison between before and after cleaning.
    """
    print("2.1. DATASET DIMENSIONS")
    print("-" * 80)
    print(f"DataFrame shape: {dataframe.shape}")
    print(f"Number of observations: {dataframe.shape[0]}")
    print(f"Number of variables: {dataframe.shape[1]}")
    print()
    
    # Display distribution by year to see how data is distributed across prediction horizons
    if ('year' in dataframe.columns):
        print("Distribution by prediction horizon (year):")
        year_counts = dataframe['year'].value_counts().sort_index()
        total = len(dataframe)
        for year in sorted(year_counts.index):
            count = year_counts[year]
            percentage = (count / total) * 100
            horizon = 6 - int(year) 
            year_label = "year" if (horizon == 1) else "years"
            print(f"  Year {year} (horizon {horizon} {year_label}): {count} observations ({percentage:.2f}%)")
        print()
    
    print("2.2. First rows preview")
    print("-" * 80)
    print(dataframe.head())
    print()
    
    print("2.3. Target variable distribution")
    print("-" * 80)
    target_display_name = target_column.replace("_", " ").title()
    print(f"Target column name: {target_display_name}")
    print()
    
    value_counts = dataframe[target_column].value_counts()
    proportions = dataframe[target_column].value_counts(normalize=True) * 100
    label_map = {0: "No Bankruptcy", 1: "Bankruptcy"}
    
    print("Group sizes:")  # visualize how imbalanced the dataset is
    for value, count in value_counts.items():
        label = label_map.get(value, f"Class {value}")
        print(f"  {label}: {count} companies")
    print()
    
    print("Proportions:")
    for value, percentage in proportions.items():
        label = label_map.get(value, f"Class {value}")
        print(f"{label}: {percentage:.2f}%")
    print()


def preprocess_data(dataframe, target_column):
    """
    Separate the target from features so they can be processed independently,
    and to preserve the quality of the target variable.
    """
    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    return features, target


def split_and_scale(features, target, test_size=0.2, random_state=42):
    """
    Split the dataset into train/test sets using the 80/20 split, which is a common standard (no particular reason).
    Apply preprocessing: 'year' is treated as a categorical variable, other features are standardized as they are continuous.
    """
    # Split data into training and test sets
    # test_size=0.2, standard 80/20 split gives enough test data for reliable evaluation while maximizing training data
    # stratify=target permits to preserve the class distribution in both sets, critical for imbalanced datasets like this one
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    # Scaling requires fully numeric data without NaN
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    if (train_missing > 0) or (test_missing > 0):
        raise ValueError(
            f"Cannot scale data with missing values. "
            f"Found {train_missing} missing values in training set and {test_missing} in test set. "
            f"Use preprocess_data with missing_strategy=\"drop\" or \"raise\" to handle missing values first."
        )
    
    # Check if 'year' is present and separate it from other features
    if ('year' in X_train.columns):
        # Get column names
        year_col = ['year']
        numeric_cols = [col for col in X_train.columns if col != 'year']
       
        # One-hot encode 'year' with drop='first': removes one category to avoid perfect multicollinearity
        # (year_1 becomes the reference, other years are encoded relative to it)
        # Standardize numeric features: necessary for distance-based models and helps gradient descent converge
        preprocessor = ColumnTransformer(
            transformers=[
                ('year', OneHotEncoder(drop='first', sparse_output=False), year_col),
                ('numeric', StandardScaler(), numeric_cols)
            ],
            remainder='passthrough'  # Pass through any remaining columns unchanged
        )
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    
        year_feature_names = [f'year_{i}' for i in range(2, 6)]  # year_2, year_3, year_4, year_5
        feature_names = year_feature_names + numeric_cols
        
        # Reconstruct DataFrames
        X_train_scaled = pd.DataFrame(
            X_train_processed, columns=feature_names, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_processed, columns=feature_names, index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor
    else:
        # No 'year' column: standard scaling for all features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=X_test.columns, index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_to_csv(dataframe, output_path="data/processed/polish_bankruptcy_renamed.csv"):
    """
    Save the current in-memory dataset to a CSV file to preserve the processed version.
    """
    output_dir = os.path.dirname(output_path)
    if (output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    dataframe.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")

