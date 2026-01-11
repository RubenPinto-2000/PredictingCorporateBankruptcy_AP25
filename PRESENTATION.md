# Presentation Script: Bankruptcy Prediction Project
**Duration: 10-15 minutes**

----------------------------

## 1. Problem Motivation 

“What are the key financial factors that predict corporate bankruptcy, and which classification model best identifies these factors?”
-----------------------------

## 2. Technical Approach and Methodology 

My pipeline with seven main steps.

### Step 1: Data Loading

### Step 2: Data Inspection and Quality Analysis

### Step 3: Data Cleaning

### Step 4: Preprocessing and Train/Test Split

### Step 5: Model Training and Evaluation

### Step 6: Feature Importance Analysis

### Step 7: Statistical Comparison

--------------------------------------------------------

## 3. Demo of Your Code Running (5-6 minutes)

--------------------------------------------------------

# 4. Results and Learnings (2-3 minutes)

### Key Findings: Best Model

*Random Forest is the best-performing model with the highest ROC-AUC score. 

Interestingly, we found that factor importance varies by prediction horizon. Short-term bankruptcy (1-2 years) is more influenced by liquidity ratios, while long-term bankruptcy (4-5 years) is better predicted by profitability and efficiency ratios."

### Key Findings: Horizon Analysis

- Only one factor across all three models 

- Two factors appear consistently across models for specific horizons

- This heterogeneity justifies the approach by horizon"

### Technical Learnings

1. Data cleaning is crucial

2. Class imbalance matters

3. Correlation threshold choice 

4. Horizon-specific modeling

### Limitations and Future Work

- The dataset is specific 

- historical data,


Future work could include:

- Testing on other countries

- macroeconomic variables

- deep learning approaches

- Developing real-time monitoring systems"

### Conclusion

In conclusion, we successfully built and compared three machine learning models for bankruptcy prediction. Random Forest emerged as the best model, and we identified key financial factors that predict bankruptcy risk. The horizon-specific analysis revealed important temporal patterns in bankruptcy prediction.





Thank you for your attention. Have a nice holidays !!!!!