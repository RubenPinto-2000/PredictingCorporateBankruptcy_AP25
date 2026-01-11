# Project Presentation — Predicting Corporate Bankruptcy (proposal accepted on 26.10)

This project aims to identify the financial factors that can help anticipate whether a company is likely to remain solvent or face bankruptcy. The objective is to use quantitative analysis to detect early signs of financial distress and support more informed corporate decisions.

My data relies on the Polish Companies Bankruptcy Dataset from the UCI Machine Learning Repository (ID: 365), which contains financial information on Polish companies with bankruptcy predictions for horizons ranging from 1 to 5 years ahead. The dataset includes key indicators such as leverage, liquidity, profitability, and market value. After a first data cleaning and standardization process, these variables will be analyzed to reveal patterns distinguishing financially healthy firms from those at risk.

I want to use on my projet three main statistical approaches: Logistic Regression, which provides an interpretable linear model; Random Forest, a non-linear method capable of capturing more complex relationships between those variables; and K-Nearest Neighbors (added as a third model as required). These methods are appropriate. Both Logistic Regression and Random Forest are better than just one - gives me a baseline and a more complex model to compare. I will make sure to use proper train/test split (80/20 or similar), report appropriate metrics (accuracy, precision, recall, F1, ROC-AUC), and handle class imbalance if bankruptcies are rare in my data.

This work aligns closely with my Master's in Finance, combining financial reasoning with data analysis to better understand corporate risk and financial stability.
