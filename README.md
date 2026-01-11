# Bankruptcy Prediction: Polish Companies Dataset

Project for Advanced Programming 2025 course.

## Research Question

“What are the key financial factors that predict corporate bankruptcy, and which classification model best identifies these factors?”

## Dataset

- Source: UCI Machine Learning Repository
- Dataset ID**: 365
- Name**: Polish Companies Bankruptcy Dataset
- Description: Dataset containing financial indicators of Polish companies with their bankruptcy status
- URL: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

## Setup

### Option 1: Conda Environment (Recommended on Nuvolos)

Use the provided environment.yml:

```bash
conda env create -f environment.yml
conda activate bankruptcy-prediction
```

**Note:** If some packages do not install correctly with conda, run this command after activating the environment:
```bash
pip install seaborn pandas numpy scikit-learn matplotlib scipy ucimlrepo
```

### Option 2: Virtual Environment (Local development)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the complete pipeline:
```bash
python main.py
```

### Force re-download dataset:
```bash
python main.py --force-download
```

The script will:
1. Load dataset automatically from UCI (or use local cache)
2. Perform initial inspection
3. Perform data quality analysis
4. Clean data (remove irrelevant features, handle missing values, etc.)
5. Preprocess and split data
6. Train three models (Logistic Regression, Random Forest, and K-Nearest Neighbors)
7. Evaluate performance with multiple metrics
8. Generate visualizations (ROC curves, confusion matrices, feature importance)
9. Identify key financial factors that predict bankruptcy (global and by prediction horizon)
10. Perform statistical tests for model comparison (paired t-test and McNemar's test)

## Project Structure

```
ProjetPython/
├── README.md                # Setup and usage instructions
├── PROPOSAL.md              # Your project proposal
├── AI_USAGE.md              # Explanation of AI tool usage
├── PRESENTATION.md          # Video presentation support
├── environment.yml          # Conda dependencies
├── requirements.txt         # Pip dependencies (alternative)
├── main.py                  # Entry point - THIS MUST WORK
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── data_analysis.py     # Data quality analysis
│   ├── data_cleaning.py     # Data cleaning functions
│   ├── data_column_mapping.py  # Column name mapping utilities
│   ├── models.py            # Model definitions
│   ├── evaluation.py        # Evaluation and visualization
│   ├── feature_analysis.py  # Financial factors identification and analysis
│   └── statistical_tests.py # Statistical model comparison tests
├── data/
│   ├── raw/                 # Original data (downloaded automatically)
│   └── processed/           # Processed data (renamed columns)
├── results/                 # Output figures and metrics
```

## Methodology

### Data Cleaning
- Remove constant/low variance features
- Remove features with >50% missing values
- Impute remaining missing values (median for numeric, mode for categorical)
- Remove duplicate rows
- Handle outliers (IQR capping)
- Remove highly correlated features (threshold: 0.95)

### Preprocessing
- Stratified train/test split (80/20)
- StandardScaler for feature scaling
- Class imbalance handling with `class_weight='balanced'`

### Models
- **Logistic Regression**: `max_iter=1000`, `class_weight='balanced'`
  - Linear model, interpretable, fast training
- **Random Forest**: `n_estimators=300`, `class_weight='balanced_subsample'`
  - Ensemble method (bagging), robust to overfitting
- **K-Nearest Neighbors**: `n_neighbors=5`
  - Instance-based learning, non-parametric, simple and interpretable

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curves
- Feature importance analysis (Logistic Regression coefficients and Random Forest)

### Key Financial Factors Identification

### How Top Factors are Determined

The pipeline identifies the most important financial factors using a two-step approach:

#### 1. Logistic Regression - Top 15 Selection

**Method**: Based on absolute coefficient values

**Process**:
1. Extract coefficients: Each feature has a coefficient (can be positive or negative)
   - Example: `profit_sales_roa = -3.1861`, `operating_margin = 2.9019`
2. **Calculate importance**: Importance = absolute value of coefficient
   - `profit_sales_roa`: | -3.1861 | = 3.1861
   - `operating_margin`: | 2.9019 | = 2.9019
3. **Sort by importance**: All factors are ranked from highest to lowest importance
4. Select top 15: The 15 factors with the highest absolute coefficient values

**Criterion**: The magnitude of the coefficient (absolute value) indicates how strongly the factor influences bankruptcy prediction.

**Interpreting Coefficient Values and Signs**:

The **absolute value** (magnitude) tells us **how important** the factor is, but the **sign** (positive/negative) tells us **in which direction** it affects bankruptcy risk:

- **Negative coefficient** (e.g., `profit_sales_roa = -3.1861`):
  - **Meaning**: As this factor increases, the probability of bankruptcy **decreases**
  - **Interpretation**: Higher values of this factor are associated with **lower bankruptcy risk**
  - **Example**: Higher profit-to-sales ratio → Lower bankruptcy risk (protective factor)

- **Positive coefficient** (e.g., `operating_margin = 2.9019`):
  - **Meaning**: As this factor increases, the probability of bankruptcy **increases**
  - **Interpretation**: Higher values of this factor are associated with **higher bankruptcy risk**
  - **Example**: Higher operating margin (in this context) → Higher bankruptcy risk (risk factor)

**Why use absolute value for ranking?**
- The absolute value measures the **strength of the relationship**, regardless of direction
- A coefficient of -3.0 has the same **impact magnitude** as +3.0, just in opposite directions
- We rank by absolute value to identify the **most influential factors**, then interpret the sign separately to understand whether they increase or decrease risk

**Example Interpretation**:
- `profit_sales_roa: -3.1861` → **Very important protective factor** (large absolute value, negative sign = decreases risk)
- `operating_margin: 2.9019` → **Very important risk factor** (large absolute value, positive sign = increases risk)
- Both are in the top factors because they have high absolute values, but they work in opposite directions

#### 2. Random Forest - Top 15 Selection

**Method**: Based on feature importance scores

**Process**:
1. **Extract importances**: The model automatically calculates importance for each feature
   - Based on average impurity reduction (Gini) across all trees
   - Example: `operating_profit_to_financial_cost = 0.0931`
2. **Sort by importance**: All factors are ranked from highest to lowest importance
3. **Select top 15**: The 15 factors that contribute most to class separation

**Criterion**: The feature importance score indicates how much each factor contributes to distinguishing between bankruptcy and non-bankruptcy cases.

#### 3. Common Factors - Robustness Validation

**Method**: Intersection of both top 15 lists

**Process**:
1. **Compare lists**: 
   - Top 15 from Logistic Regression: `[profit_sales_roa, operating_margin, ...]`
   - Top 15 from Random Forest: `[operating_profit_to_financial_cost, ...]`
2. **Find intersection**: Factors present in BOTH lists
   - Example: If `profit_sales_roa` is in top 15 LR AND top 15 RF → Common factor
3. **Robust factors**: These factors are validated by two different modeling approaches, making them more reliable indicators

**Why this matters**: Factors identified by multiple models are considered more robust and less likely to be model-specific artifacts.

### Why Top 15?

The number 15 is a balance between:
- **Comprehensiveness**: Capturing enough factors to understand the key drivers
- **Readability**: Keeping the analysis focused and interpretable
- **Statistical significance**: Focusing on the most impactful factors

This number can be adjusted by modifying the `top_n` parameter in the analysis functions.

**Additional Analysis**:
- Analysis of Logistic Regression coefficients (identifies factors that increase/decrease bankruptcy risk)
- Random Forest feature importance (identifies most predictive financial indicators)
- Cross-model comparison to find robust factors identified by multiple models
- Economic interpretation of key financial ratios and their meaning
- Statistical comparison of factor values between bankruptcy and non-bankruptcy groups

## Requirements

- Python 3.11
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
- ucimlrepo (for loading UCI dataset)




